"""
PPO training with proxy reward, no/small KL.

Mirrors the PhD's RL figure for LLM RLHF: bounded reward without KL should
behave well, while linear without KL should runaway. Tests this by training
a policy via PPO against a frozen proxy RM, periodically saving checkpoints
for offline gold-RM evaluation.

Run from project root:
    python -m src.ppo_train \
        --policy Qwen/Qwen2.5-0.5B-Instruct \
        --proxy_checkpoint ./outputs/rm-ident-clean/final \
        --proxy_activation ident \
        --kl_coef 0.0 \
        --num_steps 200 \
        --output_dir ./outputs/ppo-ident-no-kl

The output_dir contains:
    - policy_step_K/  policy checkpoints at intervals
    - proxy_log.json  per-step proxy reward stats
    - run_config.json  the config used

Then evaluate with src/ppo_eval_gold.py on the policy_step_K/ checkpoints.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)

from .heads import ACTIVATION_REGISTRY
from .model import RewardModel, RewardModelConfig


@dataclasses.dataclass
class PPORunConfig:
    # Models
    policy_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    proxy_checkpoint: str = ""
    proxy_activation: str = "ident"
    
    # Data
    prompts_dataset: str = "HuggingFaceH4/ultrafeedback_binarized"
    prompts_split: str = "train_prefs"
    n_train_prompts: int = 4000
    n_eval_prompts: int = 100
    max_prompt_length: int = 512
    max_new_tokens: int = 256
    
    # PPO
    num_steps: int = 200
    kl_coef: float = 0.0  # KL penalty coefficient. 0 = no KL.
    learning_rate: float = 1.41e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    ppo_epochs: int = 4
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    
    # Generation
    temperature: float = 1.0
    top_p: float = 1.0
    
    # Logging
    output_dir: str = "./outputs/ppo-dev"
    save_freq: int = 25  # save policy checkpoint every K PPO updates
    log_freq: int = 1
    
    # Misc
    seed: int = 42


def load_proxy_rm(checkpoint_dir: str, base_model_name: str, activation_name: str):
    """Load a trained proxy RM (LoRA + concave/linear/gaussian head)."""
    print(f"Loading proxy RM: {checkpoint_dir} (activation={activation_name})")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_config = RewardModelConfig(
        base_model_name=base_model_name,
        activation_name=activation_name,
    )
    base_rm = RewardModel.from_base_model(base_config, torch_dtype=torch.bfloat16)
    base_rm.config.pad_token_id = tokenizer.pad_token_id
    
    model = PeftModel.from_pretrained(base_rm, checkpoint_dir)
    model = model.cuda().eval()
    # Freeze
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


@torch.no_grad()
def score_with_proxy(proxy_model, proxy_tokenizer, prompts, responses, max_length=1024):
    """Score (prompt, response) pairs with the proxy RM. Returns tensor of rewards."""
    formatted = []
    for prompt, response in zip(prompts, responses):
        msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = proxy_tokenizer.apply_chat_template(msgs, tokenize=False)
        formatted.append(text)
    
    enc = proxy_tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(proxy_model.device)
    
    out = proxy_model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    rewards = out.logits.squeeze(-1).float()
    return rewards


def build_ppo_dataset(tokenizer, dataset_name, split, n_prompts, max_prompt_length):
    """Build a tokenized prompt dataset for PPO. Each example is a tokenized chat-format prompt."""
    print(f"Loading prompts from {dataset_name}:{split}...")
    ds = load_dataset(dataset_name, split=split)
    ds = ds.select(range(min(n_prompts, len(ds))))
    
    def format_prompt(ex):
        msgs = [{"role": "user", "content": ex["prompt"]}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, truncation=True, max_length=max_prompt_length)["input_ids"]
        return {"input_ids": ids, "query": text, "raw_prompt": ex["prompt"]}
    
    ds = ds.map(format_prompt, remove_columns=ds.column_names)
    ds.set_format(type="torch")
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--proxy_checkpoint", required=True)
    parser.add_argument("--proxy_activation", required=True,
                        choices=list(ACTIVATION_REGISTRY.keys()))
    parser.add_argument("--kl_coef", type=float, default=0.0,
                        help="KL penalty coefficient. 0 = no KL.")
    parser.add_argument("--num_steps", type=int, default=200,
                        help="Number of PPO updates.")
    parser.add_argument("--n_train_prompts", type=int, default=4000)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.41e-5)
    parser.add_argument("--save_freq", type=int, default=25)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    cfg = PPORunConfig(
        policy_name=args.policy,
        proxy_checkpoint=args.proxy_checkpoint,
        proxy_activation=args.proxy_activation,
        kl_coef=args.kl_coef,
        num_steps=args.num_steps,
        n_train_prompts=args.n_train_prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        learning_rate=args.learning_rate,
        save_freq=args.save_freq,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    
    # ---- Load policy and ref ----
    print(f"Loading policy: {cfg.policy_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.policy_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_name,
        torch_dtype=torch.bfloat16,
    )
    
    # Reference model (used for KL computation)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_name,
        torch_dtype=torch.bfloat16,
    )
    
    # ---- Load proxy RM ----
    proxy_model, proxy_tokenizer = load_proxy_rm(
        cfg.proxy_checkpoint, cfg.policy_name, cfg.proxy_activation
    )
    
    # ---- Build dataset ----
    train_ds = build_ppo_dataset(
        tokenizer, cfg.prompts_dataset, cfg.prompts_split,
        cfg.n_train_prompts, cfg.max_prompt_length
    )
    
    # ---- PPO config ----
    # Handle TRL API version differences
    ppo_kwargs = {
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "mini_batch_size": cfg.mini_batch_size,
        "kl_penalty": "kl",
        "init_kl_coef": cfg.kl_coef,
        "adap_kl_ctrl": False,
        "cliprange": cfg.cliprange,
        "cliprange_value": cfg.cliprange_value,
        "vf_coef": cfg.vf_coef,
        "seed": cfg.seed,
    }
    # Try newer name first, then older
    import inspect
    sig = inspect.signature(PPOConfig.__init__)
    if "num_ppo_epochs" in sig.parameters:
        ppo_kwargs["num_ppo_epochs"] = cfg.ppo_epochs
    elif "ppo_epochs" in sig.parameters:
        ppo_kwargs["ppo_epochs"] = cfg.ppo_epochs
    
    ppo_config = PPOConfig(**ppo_kwargs)
    
    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}
    
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_ds,
        data_collator=collator,
    )
    
    # ---- PPO loop ----
    proxy_log = []
    gen_kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": True,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    print(f"Starting PPO loop for {cfg.num_steps} updates...")
    t_start = time.time()
    
    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= cfg.num_steps:
            break
        
        query_tensors = [torch.tensor(ids).to(ppo_trainer.accelerator.device)
                         for ids in batch["input_ids"]]
        
        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **gen_kwargs,
        )
        
        # Decode for scoring
        responses_text = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # Score with proxy RM
        rewards = score_with_proxy(
            proxy_model, proxy_tokenizer,
            batch["raw_prompt"], responses_text,
        )
        rewards_list = [r for r in rewards]
        
        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_list)
        
        # Log
        log_entry = {
            "step": step,
            "proxy_mean": float(rewards.mean().item()),
            "proxy_std": float(rewards.std().item()),
            "kl": float(stats.get("objective/kl", 0.0)),
            "policy_loss": float(stats.get("ppo/loss/policy", 0.0)),
            "value_loss": float(stats.get("ppo/loss/value", 0.0)),
            "elapsed_s": time.time() - t_start,
        }
        proxy_log.append(log_entry)
        
        if step % cfg.log_freq == 0:
            print(f"step {step:4d}  proxy={log_entry['proxy_mean']:7.3f} "
                  f"kl={log_entry['kl']:7.3f}  "
                  f"policy_loss={log_entry['policy_loss']:7.3f}  "
                  f"elapsed={log_entry['elapsed_s']:.0f}s")
        
        # Save proxy log
        if step % 5 == 0:
            with open(out_dir / "proxy_log.json", "w") as f:
                json.dump(proxy_log, f, indent=2)
        
        # Save policy checkpoint
        if (step + 1) % cfg.save_freq == 0:
            ckpt_dir = out_dir / f"policy_step_{step + 1}"
            print(f"  Saving policy to {ckpt_dir}")
            policy_model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
    
    # Final save
    with open(out_dir / "proxy_log.json", "w") as f:
        json.dump(proxy_log, f, indent=2)
    
    final_dir = out_dir / "policy_final"
    policy_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    print(f"PPO done. Total time: {time.time() - t_start:.1f}s")
    print(f"Outputs in {out_dir}")


if __name__ == "__main__":
    main()