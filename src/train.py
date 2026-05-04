"""
Train a reward model with LoRA. Compatible with TRL 0.11.4.

Regularizers:
- alpha_reg: L2 on BoundedAbove's alpha (legacy).
- head_reg_weight: weight on activation .regularization_loss() methods.
- preact_reg: L2 on preactivation magnitude.
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

from .data import load_ultrafeedback, load_gold_labeled
from .heads import BoundedAbove, RewardHead
from .model import RewardModel, RewardModelConfig


@dataclasses.dataclass
class TrainConfig:
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    activation_name: str = "bounded_above"
    head_init_scale: float = 0.02
    head_init_bias: float = 0.0
    head_width: int = 32
    head_intermediate_size: int | None = None

    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: list = dataclasses.field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    dataset_source: str = "ultrafeedback"
    gold_labeled_path: str = "./gold_labeled"
    max_length: int = 1024
    n_train: int | None = None
    n_eval: int = 200

    output_dir: str = "./outputs/rm-dev"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    bf16: bool = True
    logging_steps: int = 5
    eval_steps: int = 25
    save_strategy: str = "epoch"
    seed: int = 42

    alpha_reg: float = 0.0
    head_reg_weight: float = 0.0
    preact_reg: float = 0.0

    run_name: str = "rm-dev"
    report_to: str = "none"


def _module_regularization_loss(model):
    first_param = next(model.parameters())
    total = torch.tensor(0.0, device=first_param.device, dtype=first_param.dtype)
    for submodule in model.modules():
        rfn = getattr(submodule, "regularization_loss", None)
        if rfn is not None and callable(rfn):
            total = total + rfn()
    return total


class ConcaveRewardTrainer(RewardTrainer):
    def __init__(
        self,
        *args,
        alpha_reg: float = 0.0,
        head_reg_weight: float = 0.0,
        preact_reg: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha_reg = alpha_reg
        self.head_reg_weight = head_reg_weight
        self.preact_reg = preact_reg

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        out = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        loss, outputs = out if isinstance(out, tuple) else (out, {})

        if self.alpha_reg > 0.0:
            for module in model.modules():
                if isinstance(module, BoundedAbove):
                    alpha = F.softplus(module.a)
                    loss = loss + self.alpha_reg * (alpha ** 2).sum()
                    break

        if self.head_reg_weight > 0.0:
            reg = _module_regularization_loss(model)
            loss = loss + self.head_reg_weight * reg

        if self.preact_reg > 0.0:
            z_chosen = model(
                input_ids=inputs["input_ids_chosen"],
                attention_mask=inputs["attention_mask_chosen"],
                return_preactivation=True,
            ).hidden_states
            z_rejected = model(
                input_ids=inputs["input_ids_rejected"],
                attention_mask=inputs["attention_mask_rejected"],
                return_preactivation=True,
            ).hidden_states
            preact_penalty = (z_chosen.float().pow(2).mean()
                              + z_rejected.float().pow(2).mean()) * 0.5
            loss = loss + self.preact_reg * preact_penalty

        return (loss, outputs) if return_outputs else loss


def load_config(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)


def build_model_and_tokenizer(cfg):
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
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        modules_to_save=["reward_head"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    head_trainable = any(
        "reward_head" in name and p.requires_grad
        for name, p in model.named_parameters()
    )
    if not head_trainable:
        raise RuntimeError("reward_head is NOT trainable!")
    print("[OK] reward_head parameters are trainable")
    model.enable_input_require_grads()
    return model, tokenizer


def load_data(cfg, tokenizer):
    if cfg.dataset_source == "ultrafeedback":
        return load_ultrafeedback(
            tokenizer,
            max_length=cfg.max_length,
            n_train=cfg.n_train,
            n_eval=cfg.n_eval,
        )
    elif cfg.dataset_source == "gold_labeled":
        return load_gold_labeled(
            tokenizer,
            dataset_path=cfg.gold_labeled_path,
            max_length=cfg.max_length,
            n_train=cfg.n_train,
            n_eval=cfg.n_eval,
        )
    else:
        raise ValueError(f"Unknown dataset_source: {cfg.dataset_source!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--activation_name", default=None)
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--alpha_reg", type=float, default=None)
    parser.add_argument("--head_reg_weight", type=float, default=None)
    parser.add_argument("--preact_reg", type=float, default=None)
    parser.add_argument("--dataset_source", default=None,
                        choices=[None, "ultrafeedback", "gold_labeled"])
    parser.add_argument("--gold_labeled_path", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.activation_name is not None: cfg.activation_name = args.activation_name
    if args.n_train is not None: cfg.n_train = args.n_train
    if args.num_train_epochs is not None: cfg.num_train_epochs = args.num_train_epochs
    if args.output_dir is not None: cfg.output_dir = args.output_dir
    if args.run_name is not None: cfg.run_name = args.run_name
    if args.alpha_reg is not None: cfg.alpha_reg = args.alpha_reg
    if args.head_reg_weight is not None: cfg.head_reg_weight = args.head_reg_weight
    if args.preact_reg is not None: cfg.preact_reg = args.preact_reg
    if args.dataset_source is not None: cfg.dataset_source = args.dataset_source
    if args.gold_labeled_path is not None: cfg.gold_labeled_path = args.gold_labeled_path

    print("=" * 60)
    print("Training config:")
    for field in dataclasses.fields(cfg):
        print(f"  {field.name}: {getattr(cfg, field.name)}")
    print("=" * 60)

    model, tokenizer = build_model_and_tokenizer(cfg)
    train_ds, eval_ds = load_data(cfg, tokenizer)
    print(f"Train: {len(train_ds)} examples, Eval: {len(eval_ds)} examples")

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

    # TRL 0.11.4: uses tokenizer= (not processing_class=)
    trainer = ConcaveRewardTrainer(
        alpha_reg=cfg.alpha_reg,
        head_reg_weight=cfg.head_reg_weight,
        preact_reg=cfg.preact_reg,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)

    import json
    config_path = os.path.join(final_dir, "rm_train_config.json")
    with open(config_path, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    print(f"Saved model to {final_dir}")
    print(f"Saved config to {config_path}")


if __name__ == "__main__":
    main()