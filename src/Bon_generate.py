"""
Best-of-N sampling for Goodhart curve experiments.

For each prompt:
1. Generate N responses from a base policy (frozen).
2. Score each response with the proxy RM (the trained linear or concave RM).
3. Score each response with the gold RM (Skywork).

The output is a flat JSON with all N samples per prompt and both scores
each. Plotting / aggregation happens in a separate script (bon_analyze.py).

This separation keeps generation+scoring (slow) decoupled from analysis
(fast iteration). One generate run produces data for all BoN-N values
0 < n <= N.

Run from project root:
    python -m src.bon_generate \
        --proxy_checkpoint ./outputs/rm-bounded-gold5k/final \
        --proxy_activation bounded_above \
        --policy Qwen/Qwen2.5-0.5B-Instruct \
        --output_json ./outputs/bon-bounded.json \
        --n_prompts 200 --n_samples 64

Then analyze:
    python -m src.bon_analyze \
        --proxy_results ./outputs/bon-ident.json ./outputs/bon-bounded.json \
        --output_dir ./outputs/figures

VRAM budget on A10 (24 GB):
    policy 0.5B bf16:    ~1 GB
    proxy RM 0.5B bf16:  ~1 GB
    gold RM 8B bf16:    ~16 GB
    activations + KV:    ~3-5 GB during generation
    -> tight but workable. Reduce batch_size_gen if OOM.
    -> for 3B policy + 3B proxy + 8B gold, you'll need an A100 80GB.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .model import RewardModel, RewardModelConfig

GOLD_MODEL_NAME = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"


# ---------------------------------------------------------------
# Loading models
# ---------------------------------------------------------------

def load_policy(policy_name: str, dtype=torch.bfloat16):
    """Load the base policy for sampling. Frozen; eval mode."""
    print(f"Loading policy: {policy_name}")
    tokenizer = AutoTokenizer.from_pretrained(policy_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # left padding for generation
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        policy_name,
        torch_dtype=dtype,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


def load_proxy_rm(checkpoint_dir: str, base_model_name: str, activation_name: str):
    """Load a trained proxy RM (LoRA + concave/linear head)."""
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
    return model, tokenizer


def load_gold_rm(dtype: str = "bf16"):
    """Load Skywork. Same logic as label_with_gold.py."""
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
        from transformers import BitsAndBytesConfig
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


# ---------------------------------------------------------------
# Generation
# ---------------------------------------------------------------

@torch.no_grad()
def generate_responses(
    prompts: list[str],
    policy_model,
    policy_tokenizer,
    n_samples: int,
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[list[str]]:
    """For each prompt, generate n_samples responses.
    
    Returns list-of-lists: results[i][j] = j-th response to i-th prompt.
    
    Uses temperature=1.0 by default (no scaling) to faithfully sample
    from the policy's distribution. BoN's optimization pressure should
    come from N (the selection ratio), not from temperature scaling.
    """
    results = [[] for _ in prompts]
    
    # Format prompts using chat template
    formatted = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = policy_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)
    
    pbar = tqdm(total=len(prompts) * n_samples, desc="Generating")
    
    # We can either: outer-loop prompts, inner-loop samples; OR
    # batch many (prompt, sample-idx) pairs together. Latter is faster
    # because GPUs prefer larger batches than n_samples likely is.
    # We'll batch across prompts with `num_return_sequences=n_samples`.
    
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = formatted[batch_start : batch_start + batch_size]
        
        inputs = policy_tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(policy_model.device)
        
        # generate n_samples per prompt in one call
        outputs = policy_model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n_samples,
            pad_token_id=policy_tokenizer.pad_token_id,
        )
        
        # outputs shape: (batch_size * n_samples, total_len)
        # decode only the generated portion
        input_len = inputs.input_ids.shape[1]
        gen_only = outputs[:, input_len:]
        
        decoded = policy_tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        
        # unflatten: decoded[i*n_samples + j] -> results[batch_start + i][j]
        for i in range(len(batch_prompts)):
            for j in range(n_samples):
                results[batch_start + i].append(decoded[i * n_samples + j])
        
        pbar.update(len(batch_prompts) * n_samples)
    
    pbar.close()
    return results


# ---------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------

@torch.no_grad()
def score_with_proxy(
    prompt_response_pairs: list[tuple[str, str]],
    proxy_model,
    proxy_tokenizer,
    batch_size: int = 16,
    max_length: int = 1024,
) -> list[float]:
    """Score (prompt, response) pairs with the trained proxy RM.
    
    Returns list of scalar rewards in the same order.
    """
    # Format using the proxy's chat template
    formatted = []
    for prompt, response in prompt_response_pairs:
        msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = proxy_tokenizer.apply_chat_template(msgs, tokenize=False)
        formatted.append(text)
    
    scores = []
    pbar = tqdm(range(0, len(formatted), batch_size), desc="Scoring (proxy)")
    for i in pbar:
        batch = formatted[i : i + batch_size]
        enc = proxy_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(proxy_model.device)
        
        out = proxy_model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        batch_scores = out.logits.squeeze(-1).float().cpu().tolist()
        if not isinstance(batch_scores, list):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)
    
    return scores


@torch.no_grad()
def score_with_gold(
    prompt_response_pairs: list[tuple[str, str]],
    gold_model,
    gold_tokenizer,
    batch_size: int = 4,
    max_length: int = 4096,
) -> list[float]:
    """Score (prompt, response) pairs with the gold RM (Skywork).
    
    Returns list of scalar rewards in the same order.
    """
    formatted = []
    for prompt, response in prompt_response_pairs:
        msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = gold_tokenizer.apply_chat_template(msgs, tokenize=False)
        if gold_tokenizer.bos_token is not None and text.startswith(gold_tokenizer.bos_token):
            text = text[len(gold_tokenizer.bos_token):]
        formatted.append(text)
    
    scores = []
    pbar = tqdm(range(0, len(formatted), batch_size), desc="Scoring (gold)")
    for i in pbar:
        batch = formatted[i : i + batch_size]
        enc = gold_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(gold_model.device)
        
        try:
            out = gold_model(**enc)
            batch_scores = out.logits.squeeze(-1).float().cpu().tolist()
        except torch.cuda.OutOfMemoryError:
            # fall back to one-at-a-time
            batch_scores = []
            for j in range(len(batch)):
                single = {k: v[j:j+1] for k, v in enc.items()}
                out = gold_model(**single)
                batch_scores.append(out.logits[0][0].float().cpu().item())
        
        if not isinstance(batch_scores, list):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)
    
    return scores


# ---------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------

def run_bon(
    proxy_checkpoint: str,
    proxy_base_model: str,
    proxy_activation: str,
    policy_name: str,
    n_prompts: int,
    n_samples: int,
    output_path: str,
    max_new_tokens: int,
    batch_size_gen: int,
    batch_size_proxy: int,
    batch_size_gold: int,
    gold_dtype: str,
    skip_gold: bool,
    prompts_dataset: str,
    prompts_split: str,
):
    """Full pipeline: generate, score with proxy, score with gold, save."""
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    print(f"Loading prompts from {prompts_dataset}:{prompts_split}...")
    ds = load_dataset(prompts_dataset, split=prompts_split)
    ds = ds.select(range(min(n_prompts, len(ds))))
    prompts = [ex["prompt"] for ex in ds]
    print(f"  {len(prompts)} prompts loaded.")
    
    # ==== Stage 1: Generate ====
    # Load policy, generate, then unload to free VRAM for the gold RM.
    policy_model, policy_tokenizer = load_policy(policy_name)
    
    t0 = time.time()
    responses = generate_responses(
        prompts,
        policy_model,
        policy_tokenizer,
        n_samples=n_samples,
        batch_size=batch_size_gen,
        max_new_tokens=max_new_tokens,
    )
    gen_time = time.time() - t0
    print(f"Generation took {gen_time:.1f}s "
          f"({len(prompts) * n_samples / gen_time:.1f} samples/sec)")
    
    # Free policy memory
    del policy_model
    torch.cuda.empty_cache()
    
    # Flatten (prompt, response) for batched scoring
    flat_pairs = []
    flat_meta = []  # parallel: (prompt_idx, sample_idx)
    for i, prompt in enumerate(prompts):
        for j, response in enumerate(responses[i]):
            flat_pairs.append((prompt, response))
            flat_meta.append((i, j))
    
    # ==== Stage 2: Score with proxy ====
    proxy_model, proxy_tokenizer = load_proxy_rm(
        proxy_checkpoint, proxy_base_model, proxy_activation
    )
    t0 = time.time()
    proxy_scores = score_with_proxy(
        flat_pairs, proxy_model, proxy_tokenizer,
        batch_size=batch_size_proxy,
    )
    proxy_time = time.time() - t0
    print(f"Proxy scoring took {proxy_time:.1f}s")
    
    del proxy_model
    torch.cuda.empty_cache()
    
    # ==== Stage 3: Score with gold ====
    if skip_gold:
        print("Skipping gold scoring (skip_gold=True). Gold scores will be None.")
        gold_scores = [None] * len(flat_pairs)
    else:
        gold_model, gold_tokenizer = load_gold_rm(dtype=gold_dtype)
        t0 = time.time()
        gold_scores = score_with_gold(
            flat_pairs, gold_model, gold_tokenizer,
            batch_size=batch_size_gold,
        )
        gold_time = time.time() - t0
        print(f"Gold scoring took {gold_time:.1f}s")
        del gold_model
        torch.cuda.empty_cache()
    
    # ==== Stage 4: Reorganize and save ====
    # Build records: one per (prompt, sample)
    records = []
    for k, (prompt, response) in enumerate(flat_pairs):
        prompt_idx, sample_idx = flat_meta[k]
        records.append({
            "prompt_idx": prompt_idx,
            "sample_idx": sample_idx,
            "prompt": prompt,
            "response": response,
            "proxy_score": proxy_scores[k],
            "gold_score": gold_scores[k],
            "response_length_tokens": len(proxy_tokenizer.encode(response))
                if not skip_gold else None,
        })
    
    output = {
        "config": {
            "proxy_checkpoint": proxy_checkpoint,
            "proxy_activation": proxy_activation,
            "proxy_base_model": proxy_base_model,
            "policy": policy_name,
            "gold_model": GOLD_MODEL_NAME if not skip_gold else None,
            "n_prompts": len(prompts),
            "n_samples": n_samples,
            "max_new_tokens": max_new_tokens,
            "prompts_dataset": prompts_dataset,
            "prompts_split": prompts_split,
        },
        "records": records,
    }
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(records)} records to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy_checkpoint", required=True,
                        help="Path to trained proxy RM checkpoint dir.")
    parser.add_argument("--proxy_base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--proxy_activation", required=True,
                        choices=["ident", "bounded", "bounded_above", "gelu_bounded_above"])
    parser.add_argument("--policy", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base policy for sampling. Should usually match RM base.")
    parser.add_argument("--n_prompts", type=int, default=100,
                        help="Number of prompts to evaluate. Higher=less noise, slower.")
    parser.add_argument("--n_samples", type=int, default=64,
                        help="Max N for BoN. We sample this many; analysis script "
                             "re-uses subsets for smaller N values.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size_gen", type=int, default=4,
                        help="Prompts per generation batch. Each contributes "
                             "n_samples generations, so effective batch is "
                             "batch_size_gen * n_samples. Lower if OOM.")
    parser.add_argument("--batch_size_proxy", type=int, default=16)
    parser.add_argument("--batch_size_gold", type=int, default=4)
    parser.add_argument("--gold_dtype", choices=["bf16", "int4"], default="bf16")
    parser.add_argument("--skip_gold", action="store_true",
                        help="Skip gold scoring; useful for fast iteration when "
                             "you only need proxy scores.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--prompts_dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--prompts_split", default="test_prefs",
                        help="Test split avoids data leakage from RM training.")
    args = parser.parse_args()
    
    run_bon(
        proxy_checkpoint=args.proxy_checkpoint,
        proxy_base_model=args.proxy_base_model,
        proxy_activation=args.proxy_activation,
        policy_name=args.policy,
        n_prompts=args.n_prompts,
        n_samples=args.n_samples,
        output_path=args.output_json,
        max_new_tokens=args.max_new_tokens,
        batch_size_gen=args.batch_size_gen,
        batch_size_proxy=args.batch_size_proxy,
        batch_size_gold=args.batch_size_gold,
        gold_dtype=args.gold_dtype,
        skip_gold=args.skip_gold,
        prompts_dataset=args.prompts_dataset,
        prompts_split=args.prompts_split,
    )


if __name__ == "__main__":
    main()