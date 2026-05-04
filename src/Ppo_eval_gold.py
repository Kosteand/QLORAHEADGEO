"""
PPO training with proxy reward for TRL 0.11.4.

Run from project root:
    python -m src.Ppo_train \
        --proxy_checkpoint ./outputs/rm-ident-cleanv2/final \
        --proxy_activation ident \
        --kl_coef 0.0 \
        --num_steps 200 \
        --output_dir ./outputs/ppo-ident-no-kl
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
from transformers import AutoTokenizer
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)

from .heads import ACTIVATION_REGISTRY
from .model import RewardModel, RewardModelConfig


@dataclasses.dataclass
class PPORunConfig:
    policy_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    proxy_checkpoint: str = ""
    proxy_activation: str = "ident"
    
    prompts_dataset: str = "HuggingFaceH4/ultrafeedback_binarized"
    prompts_split: str = "train_prefs"
    n_train_prompts: int = 4000
    max_prompt_length: int = 512
    max_new_tokens: int = 256
    
    num_steps: int = 200
    kl_coef: float = 0.0
    learning_rate: float = 1.41e-5
    batch_size: int = 16
    mini_batch_size: int = 4
    ppo_epochs: int = 4
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    
    temperature: float = 1.0
    top_p: float = 1.0
    
    output_dir: str = "./outputs/ppo-dev"
    save_freq: int = 25
    log_freq: int = 1
    
    seed: int = 42


def load_proxy_rm(checkpoint_dir, base_model_name, activation_name):
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
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


@torch.no_grad()
def score_with_proxy(proxy_model, proxy_tokenizer, prompts, responses, max_length=1024):
    formatted = []
    for prompt, response in zip(prompts, responses):
        msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = proxy_tokenizer.apply_chat_template(msgs, tokenize=False)
        formatted.append(text)
    
    enc = proxy_tokenizer(
        formatted, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    ).to(proxy_model.device)
    
    out = proxy_model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    rewards = out.logits.squeeze(-1).float()
    return rewards


def build_ppo_dataset(tokenizer, dataset_name, split, n_prompts, max_prompt_length):
    print(f"Loading prompts from {dataset_name}:{split}...")
    ds = load_dataset(dataset_name, split=split)
    ds = ds.select(range(min(n_prompts, len(ds))))
    
    def format_prompt(ex):
        msgs = [{"role": "user", "content": ex["prompt"]}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, truncation=True, max_length=max_prompt_length)["input_ids"]
        return {
            "input_ids": ids,
            "query": text,
            "raw_prompt": ex["prompt"],
        }
    
    ds = ds.map(format_prompt, remove_columns=ds.column_names)
    # Keep raw_prompt and query as strings while input_ids stays as tensor list
    ds.set_format(type="torch", columns=["input_ids"], output_all_columns=True)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--proxy_checkpoint", required=True)
    parser.add_argument("--proxy_activation", required=True,
                        choices=list(ACTIVATION_REGISTRY.keys()))
    parser.add_argument("--kl_coef", type=float, default=0.0)
    parser.add_argument("--num_steps", type=int, default=200)
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
    
    # Load policy and ref
    print(f"Loading policy: {cfg.policy_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.policy_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_name,
        torch_dtype=torch.bfloat16,
    )
    
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        cfg.policy_name,
        torch_dtype=torch.bfloat16,
    )
    
    # Load proxy RM
    proxy_model, proxy_tokenizer = load_proxy_rm(
        cfg.proxy_checkpoint, cfg.policy_name, cfg.proxy_activation
    )
    
    # Build dataset
    train_ds = build_ppo_dataset(
        tokenizer, cfg.prompts_dataset, cfg.prompts_split,
        cfg.n_train_prompts, cfg.max_prompt_length
    )
    
    # PPO config (TRL 0.11.4 API - all args verified from inspect)
    ppo_config = PPOConfig(
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        mini_batch_size=cfg.mini_batch_size,
        ppo_epochs=cfg.ppo_epochs,
        cliprange=cfg.cliprange,
        cliprange_value=cfg.cliprange_value,
        vf_coef=cfg.vf_coef,
        init_kl_coef=cfg.kl_coef,
        adap_kl_ctrl=False,
        kl_penalty="kl",
        seed=cfg.seed,
        log_with=None,  # disable wandb/etc
    )
    
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
        
        # Convert input_ids to list of 1D tensors on device
        query_tensors = []
        for ids in batch["input_ids"]:
            if isinstance(ids, torch.Tensor):
                t = ids.to(ppo_trainer.accelerator.device)
            else:
                t = torch.tensor(ids, device=ppo_trainer.accelerator.device)
            query_tensors.append(t)
        
        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **gen_kwargs,
        )
        
        responses_text = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # Get raw_prompt for scoring
        if "raw_prompt" in batch:
            prompts_for_scoring = batch["raw_prompt"]
        else:
            # Fallback: decode and strip chat template
            prompts_for_scoring = []
            for qt in query_tensors:
                decoded = tokenizer.decode(qt, skip_special_tokens=True)
                if "user\n" in decoded:
                    decoded = decoded.split("user\n", 1)[1]
                    if "assistant" in decoded:
                        decoded = decoded.split("assistant", 1)[0]
                prompts_for_scoring.append(decoded.strip())
        
        # Score with proxy RM
        rewards = score_with_proxy(
            proxy_model, proxy_tokenizer,
            prompts_for_scoring, responses_text,
        )
        rewards_list = [r for r in rewards]
        
        # PPO step
        try:
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards_list)
        except Exception as e:
            print(f"PPO step failed at step {step}: {e}")
            print("Continuing to next batch...")
            continue
        
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
                  f"elapsed={log_entry['elapsed_s']:.0f}s")
        
        if step % 5 == 0:
            with open(out_dir / "proxy_log.json", "w") as f:
                json.dump(proxy_log, f, indent=2)
        
        if (step + 1) % cfg.save_freq == 0:
            ckpt_dir = out_dir / f"policy_step_{step + 1}"
            print(f"  Saving policy to {ckpt_dir}")
            policy_model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
    
    with open(out_dir / "proxy_log.json", "w") as f:
        json.dump(proxy_log, f, indent=2)
    
    final_dir = out_dir / "policy_final"
    policy_model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    
    print(f"PPO done. Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()