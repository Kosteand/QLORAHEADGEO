"""
RewardModel: wraps a base causal LM with a RewardHead.

Key design decisions:
- AutoModel (not AutoModelForCausalLM) for the base; LM head unused.
- last_token_pool extracts the last non-pad token, handling left/right pad.
- forward() returns SequenceClassifierOutputWithPast for TRL compatibility.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from .heads import RewardHead


def last_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool the last non-pad token. Handles both left- and right-padded.
    
    Args:
        hidden_states: (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len), 1 for real tokens, 0 for pad
    Returns:
        pooled: (batch, hidden_size)
    """
    seq_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    
    left_padded = (attention_mask[:, 0] == 0).any()
    
    if left_padded:
        last_idx = torch.full(
            (batch_size,),
            hidden_states.shape[1] - 1,
            device=hidden_states.device,
            dtype=torch.long,
        )
    else:
        last_idx = seq_lengths.long()
    
    pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), last_idx]
    return pooled


class RewardModelConfig(PretrainedConfig):
    """Config for RewardModel. Stores activation_name for reproducibility."""
    model_type = "concave_reward_model"
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        activation_name: str = "bounded_above",
        head_init_scale: float = 0.02,
        head_init_bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.activation_name = activation_name
        self.head_init_scale = head_init_scale
        self.head_init_bias = head_init_bias


class RewardModel(PreTrainedModel):
    """Base LM + RewardHead.
    
    Use:
        cfg = RewardModelConfig(
            base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
            activation_name="bounded_above",  # or "ident" for linear baseline
        )
        model = RewardModel.from_base_model(cfg)
    """
    
    config_class = RewardModelConfig
    
    def __init__(self, config: RewardModelConfig, base_model: Optional[nn.Module] = None):
        super().__init__(config)
        if base_model is None:
            base_model = AutoModel.from_pretrained(config.base_model_name)
        self.model = base_model
        
        hidden_size = self.model.config.hidden_size
        self.reward_head = RewardHead(
            hidden_size=hidden_size,
            activation_name=config.activation_name,
            init_scale=config.head_init_scale,
            init_bias=config.head_init_bias,
        )
        self.hidden_size = hidden_size
    
    @classmethod
    def from_base_model(
        cls,
        config: RewardModelConfig,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "RewardModel":
        base = AutoModel.from_pretrained(
            config.base_model_name,
            torch_dtype=torch_dtype,
        )
        model = cls(config, base_model=base)
        model = model.to(torch_dtype)
        return model
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_preactivation: bool = False,
        **kwargs,
    ) -> SequenceClassifierOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        pooled = last_token_pool(hidden_states, attention_mask)
        
        if return_preactivation:
            z = self.reward_head.preactivation(pooled)
            r = self.reward_head.activation(z)
            return SequenceClassifierOutputWithPast(
                loss=None,
                logits=r,
                hidden_states=z,  # repurposed slot
            )
        else:
            r = self.reward_head(pooled)
            return SequenceClassifierOutputWithPast(loss=None, logits=r)
    
    # PEFT compatibility hooks
    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)
    
    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()
    
    def enable_input_require_grads(self):
        self.model.enable_input_require_grads()
    
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)