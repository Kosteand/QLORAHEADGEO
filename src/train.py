"""
Train a reward model with LoRA on Qwen2.5-0.5B-Instruct (or any base model).

The head is a ConcaveRewardHead: r(h) = phi(w^T h + b) for some fixed phi.
The base model is wrapped in LoRA; the head itself is fully trained
in bf16 (not LoRA-adapted) via PEFT's modules_to_save mechanism.

Usage:
    python -m src.train --config configs/qwen-0.5b-dev.yaml

For development, use a small subset of UltraFeedback (n_train=1000) to
verify the pipeline overfits correctly before scaling up.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer
from trl import RewardConfig, RewardTrainer

from .data import load_ultrafeedback
from .heads import BoundedAbove
from .model import RewardModel, RewardModelConfig


@dataclasses.dataclass
class TrainConfig:
    # Model
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    activation_name: str = "bounded_above"
    head_init_scale: float = 0.02
    head_init_bias: float = 0.0
    head_width: int = 32
    head_intermediate_size: int | None = None  # None -> use base hidden_size
    
    # LoRA
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: list = dataclasses.field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # Data
    max_length: int = 1024
    n_train: int | None = None     # None = full dataset
    n_eval: int = 200
    
    # Training
    output_dir: str = "./outputs/rm-dev"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    bf16: bool = True
    logging_steps: int = 5
    eval_steps: int = 25
    save_strategy: str = "epoch"
    seed: int = 42
    
    # Regularization
    alpha_reg: float = 0.0  # L2 penalty on BoundedAbove's alpha; 0 = disabled

    # Misc
    run_name: str = "rm-dev"
    report_to: str = "none"  # set to "wandb" for real runs


class ConcaveRewardTrainer(RewardTrainer):
    """RewardTrainer + optional L2 penalty on BoundedAbove's alpha.

    Adds cfg.alpha_reg * alpha^2 to the ranking loss at every step.
    When alpha_reg=0 (default) this is identical to RewardTrainer.
    Only fires when a BoundedAbove module is present; other activations
    (ident, bounded, gelu_bounded_above) are unaffected.
    """

    def __init__(self, *args, alpha_reg: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_reg = alpha_reg

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        loss, outputs = out if isinstance(out, tuple) else (out, {})

        if self.alpha_reg > 0.0:
            for module in model.modules():
                if isinstance(module, BoundedAbove):
                    alpha = F.softplus(module.a)
                    loss = loss + self.alpha_reg * (alpha ** 2).sum()
                    break

        return (loss, outputs) if return_outputs else loss


def load_config(path: str) -> TrainConfig:
    """Load YAML config file into a TrainConfig dataclass."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)


def build_model_and_tokenizer(cfg: TrainConfig):
    """Construct the reward model with LoRA and return (model, tokenizer)."""
    print(f"Loading tokenizer for {cfg.base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Building reward model with activation={cfg.activation_name}...")
    rm_config = RewardModelConfig(
        base_model_name=cfg.base_model_name,
        activation_name=cfg.activation_name,
        head_init_scale=cfg.head_init_scale,
        head_init_bias=cfg.head_init_bias,
        head_width=cfg.head_width,
        head_intermediate_size=cfg.head_intermediate_size,
    )
    model = RewardModel.from_base_model(
        rm_config,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Applying LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha})...")
    lora_config = LoraConfig(
        # FEATURE_EXTRACTION because we wrap AutoModel, not AutoModelForSequenceClassification
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        # CRITICAL: makes the entire reward_head submodule fully trainable
        # rather than LoRA-adapted. The head is small and randomly initialized;
        # it must be trained as a full-precision parameter.
        modules_to_save=["reward_head"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters and verify the head is included
    model.print_trainable_parameters()
    head_trainable = any(
        "reward_head" in name and p.requires_grad
        for name, p in model.named_parameters()
    )
    if not head_trainable:
        raise RuntimeError(
            "reward_head is NOT trainable! Check modules_to_save in LoRA config."
        )
    print("[OK] reward_head parameters are trainable")
    
    # Required for gradient checkpointing + LoRA to work correctly
    model.enable_input_require_grads()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--activation_name", default=None,
                        help="Override activation from config (for sweeps)")
    parser.add_argument("--n_train", type=int, default=None,
                        help="Override n_train (e.g., 1000 for overfit test)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    if args.activation_name is not None:
        cfg.activation_name = args.activation_name
    if args.n_train is not None:
        cfg.n_train = args.n_train
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.run_name is not None:
        cfg.run_name = args.run_name
    
    print("=" * 60)
    print("Training config:")
    for field in dataclasses.fields(cfg):
        print(f"  {field.name}: {getattr(cfg, field.name)}")
    print("=" * 60)
    
    # Build model
    model, tokenizer = build_model_and_tokenizer(cfg)
    
    # Load data
    print()
    print(f"Loading UltraFeedback (n_train={cfg.n_train}, n_eval={cfg.n_eval})...")
    train_ds, eval_ds = load_ultrafeedback(
        tokenizer,
        max_length=cfg.max_length,
        n_train=cfg.n_train,
        n_eval=cfg.n_eval,
    )
    print(f"Train: {len(train_ds)} examples, Eval: {len(eval_ds)} examples")
    
    # TRL RewardConfig
    training_args = RewardConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        max_length=cfg.max_length,
        remove_unused_columns=False,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        seed=cfg.seed,
    )
    
    trainer = ConcaveRewardTrainer(
        alpha_reg=cfg.alpha_reg,
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    
    print()
    print("Starting training...")
    trainer.train()
    
    print()
    print("Training done. Saving final model...")
    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)
    
    # Save the full config so evaluate.py can reconstruct the model exactly.
    # Without this, evaluate.py uses RewardModelConfig defaults which may
    # not match what was trained, causing checkpoint shape mismatches.
    import json
    config_path = os.path.join(final_dir, "rm_train_config.json")
    with open(config_path, "w") as f:
        # Don't serialize the lora_target_modules list type if it's anything
        # weird; convert everything via dataclasses.asdict.
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    print(f"Saved model to {final_dir}")
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    main()