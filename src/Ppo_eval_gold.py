"""
Evaluate saved PPO policy checkpoints with the gold RM (Skywork).

PPO training in ppo_train.py periodically saves policy_step_K/ checkpoints.
This script loads each one, generates responses on a held-out prompt set,
scores them with Skywork (gold), and logs the results.

The output_json mirrors proxy_log.json structure: per-step gold reward stats.
Combined with proxy_log.json, you get the full PhD figure: proxy reward
(left panel) and gold reward (right panel) over PPO updates.

Run from project root:
    python -m src.ppo_eval_gold \
        --ppo_dir ./outputs/ppo-ident-no-kl \
        --n_eval_prompts 50 \
        --max_new_tokens 256
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

GOLD_MODEL_NAME = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"


def load_gold_rm(dtype: str = "bf16"):
    print(f"Loading gold RM: {GOLD_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(GOLD_MODEL_NAME)
    
    if tokenizer.chat_template is None:
        from transformers import AutoTokenizer as AT
        llama_tok = AT.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
        tokenizer.chat_template = llama_tok.chat_template
    
    kwargs = {"num_labels": 1, "device_map": "cuda"}
    if dtype == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    try:
        kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForSequenceClassification.from_pretrained(GOLD_MODEL_NAME, **kwargs)
    except (ImportError, ValueError):
        kwargs.pop("attn_implementation", None)
        model = AutoModelForSequenceClassification.from_pretrained(GOLD_MODEL_NAME, **kwargs)
    
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_responses(
    policy_model, policy_tokenizer, prompts,
    max_new_tokens=256, batch_size=4, temperature=1.0, top_p=1.0,
):
    formatted = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = policy_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)
    
    responses = []
    for i in tqdm(range(0, len(formatted), batch_size), desc="Generating"):
        batch = formatted[i : i + batch_size]
        enc = policy_tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        ).to(policy_model.device)
        
        out = policy_model.generate(
            **enc,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=policy_tokenizer.pad_token_id,
        )
        
        input_len = enc.input_ids.shape[1]
        gen_only = out[:, input_len:]
        decoded = policy_tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        responses.extend(decoded)
    
    return responses


@torch.no_grad()
def score_with_gold(gold_model, gold_tokenizer, prompts, responses, batch_size=4):
    formatted = []
    for prompt, response in zip(prompts, responses):
        msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = gold_tokenizer.apply_chat_template(msgs, tokenize=False)
        if gold_tokenizer.bos_token is not None and text.startswith(gold_tokenizer.bos_token):
            text = text[len(gold_tokenizer.bos_token):]
        formatted.append(text)
    
    scores = []
    for i in tqdm(range(0, len(formatted), batch_size), desc="Gold scoring"):
        batch = formatted[i : i + batch_size]
        enc = gold_tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=4096,
        ).to(gold_model.device)
        
        try:
            out = gold_model(**enc)
            batch_scores = out.logits.squeeze(-1).float().cpu().tolist()
        except torch.cuda.OutOfMemoryError:
            batch_scores = []
            for j in range(len(batch)):
                single = {k: v[j:j+1] for k, v in enc.items()}
                out = gold_model(**single)
                batch_scores.append(out.logits[0][0].float().cpu().item())
        
        if not isinstance(batch_scores, list):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)
    
    return scores


def find_checkpoints(ppo_dir: Path) -> list[tuple[int, Path]]:
    """Find all policy_step_K/ and policy_final/ checkpoints, sorted by step."""
    ckpts = []
    for d in ppo_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("policy_step_"):
            step = int(d.name.replace("policy_step_", ""))
            ckpts.append((step, d))
        elif d.name == "policy_final":
            ckpts.append((10**9, d))  # sort last
    return sorted(ckpts, key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo_dir", required=True,
                        help="Output dir from ppo_train.py with policy_step_K/ subdirs.")
    parser.add_argument("--n_eval_prompts", type=int, default=50)
    parser.add_argument("--prompts_dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--prompts_split", default="test_prefs")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--gold_dtype", choices=["bf16", "int4"], default="bf16")
    parser.add_argument("--include_initial", action="store_true",
                        help="Also evaluate the un-trained base policy as step=0.")
    parser.add_argument("--base_policy", default=None,
                        help="Base policy name for include_initial. "
                             "Defaults to value in run_config.json.")
    args = parser.parse_args()
    
    ppo_dir = Path(args.ppo_dir)
    if not ppo_dir.exists():
        raise FileNotFoundError(f"PPO dir not found: {ppo_dir}")
    
    # Load run config
    cfg_path = ppo_dir / "run_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            run_cfg = json.load(f)
        base_policy = run_cfg.get("policy_name", "Qwen/Qwen2.5-0.5B-Instruct")
    else:
        base_policy = args.base_policy or "Qwen/Qwen2.5-0.5B-Instruct"
        run_cfg = {"policy_name": base_policy}
    
    # Find checkpoints
    ckpts = find_checkpoints(ppo_dir)
    if args.include_initial:
        ckpts = [(0, base_policy)] + ckpts
    print(f"Found {len(ckpts)} checkpoints to evaluate.")
    
    # Load eval prompts
    print(f"Loading eval prompts...")
    ds = load_dataset(args.prompts_dataset, split=args.prompts_split)
    ds = ds.select(range(min(args.n_eval_prompts, len(ds))))
    eval_prompts = [ex["prompt"] for ex in ds]
    print(f"  {len(eval_prompts)} prompts.")
    
    # Load gold RM ONCE (it's expensive)
    gold_model, gold_tokenizer = load_gold_rm(dtype=args.gold_dtype)
    
    # Evaluate each checkpoint
    gold_log = []
    
    for step, ckpt in ckpts:
        ckpt_str = str(ckpt) if isinstance(ckpt, Path) else ckpt
        print(f"\n=== Step {step}: {ckpt_str} ===")
        
        # Load policy
        policy_tokenizer = AutoTokenizer.from_pretrained(ckpt_str)
        if policy_tokenizer.pad_token is None:
            policy_tokenizer.pad_token = policy_tokenizer.eos_token
        policy_tokenizer.padding_side = "left"
        
        policy_model = AutoModelForCausalLM.from_pretrained(
            ckpt_str,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        policy_model.eval()
        
        # Generate
        t0 = time.time()
        responses = generate_responses(
            policy_model, policy_tokenizer, eval_prompts,
            max_new_tokens=args.max_new_tokens,
        )
        gen_time = time.time() - t0
        
        # Free policy GPU memory before loading scoring
        del policy_model
        torch.cuda.empty_cache()
        
        # Score with gold (gold is already loaded)
        scores = score_with_gold(gold_model, gold_tokenizer, eval_prompts, responses)
        
        import numpy as np
        scores_arr = np.array(scores)
        log_entry = {
            "step": step,
            "ckpt": ckpt_str,
            "gold_mean": float(scores_arr.mean()),
            "gold_std": float(scores_arr.std()),
            "n_prompts": len(eval_prompts),
            "gen_time_s": gen_time,
        }
        gold_log.append(log_entry)
        print(f"  step={step}  gold_mean={log_entry['gold_mean']:7.3f} "
              f"± {log_entry['gold_std']:.3f}")
        
        # Save after each ckpt
        with open(ppo_dir / "gold_log.json", "w") as f:
            json.dump(gold_log, f, indent=2)
    
    print(f"\nGold evaluation complete. Saved to {ppo_dir / 'gold_log.json'}")


if __name__ == "__main__":
    main()