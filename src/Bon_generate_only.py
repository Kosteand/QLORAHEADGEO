"""
Pure-generation step for BoN experiments, with checkpointing and OOM mitigation.

Key features added vs original:
- Saves progress after every prompt; can resume after crashes.
- Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation.
- Calls torch.cuda.empty_cache() between prompt batches.
- Default --max_new_tokens reduced from 512 to 384 (smaller KV cache).

Resume: pass the same --output_json. Existing prompts are skipped automatically.

Run from project root:
    python -m src.bon_generate_only \
        --policy Qwen/Qwen2.5-0.5B-Instruct \
        --n_prompts 100 \
        --n_samples 512 \
        --output_json ./outputs/bon-gen-512.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Set BEFORE importing torch to take effect on this run.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def save_progress(out_path: Path, output: dict):
    """Atomic save: write to tmp file, then rename. Avoids corruption on crash."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2)
    tmp_path.replace(out_path)


def load_progress(out_path: Path) -> dict | None:
    """Load existing progress file, or None if it doesn't exist or is corrupt."""
    if not out_path.exists():
        return None
    try:
        with open(out_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        print(f"  WARN: existing {out_path} is corrupt; starting fresh.")
        return None


@torch.no_grad()
def generate_for_prompt(
    formatted_prompt: str,
    policy_model,
    policy_tokenizer,
    n_samples: int,
    chunk_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate n_samples responses for ONE prompt, in chunks to bound memory.
    
    Generation memory usage scales with batch_size * sequence_length * model_dim.
    Big chunk_size = OOM. Small chunk_size = many forward passes.
    chunk_size=64 is a reasonable default for 0.5B model on A10.
    """
    inputs = policy_tokenizer(
        [formatted_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(policy_model.device)
    input_len = inputs.input_ids.shape[1]
    
    all_responses = []
    n_remaining = n_samples
    while n_remaining > 0:
        cur = min(chunk_size, n_remaining)
        try:
            outputs = policy_model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=cur,
                pad_token_id=policy_tokenizer.pad_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            # Halve chunk_size and retry
            torch.cuda.empty_cache()
            new_chunk = max(1, cur // 2)
            print(f"  OOM at chunk_size={cur}; halving to {new_chunk}")
            chunk_size = new_chunk
            continue
        
        gen_only = outputs[:, input_len:]
        decoded = policy_tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        all_responses.extend(decoded)
        n_remaining -= cur
        
        # Free memory after each chunk
        del outputs
        del gen_only
    
    return all_responses


def run(
    policy_name: str,
    n_prompts: int,
    n_samples: int,
    output_path: str,
    max_new_tokens: int,
    chunk_size: int,
    temperature: float,
    top_p: float,
    prompts_dataset: str,
    prompts_split: str,
):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing progress to enable resume
    existing = load_progress(out_path)
    completed_prompts = set()
    if existing is not None:
        completed_prompts = {r["prompt_idx"] for r in existing.get("records", [])}
        # Ensure each prompt has the full n_samples; partial prompts re-run
        prompt_counts = {}
        for r in existing["records"]:
            prompt_counts[r["prompt_idx"]] = prompt_counts.get(r["prompt_idx"], 0) + 1
        completed_prompts = {p for p, c in prompt_counts.items() if c == n_samples}
        if completed_prompts:
            print(f"Resuming: {len(completed_prompts)} prompts already done.")
    
    # Load prompts
    print(f"Loading prompts from {prompts_dataset}:{prompts_split}...")
    ds = load_dataset(prompts_dataset, split=prompts_split)
    ds = ds.select(range(min(n_prompts, len(ds))))
    prompts = [ex["prompt"] for ex in ds]
    print(f"  {len(prompts)} prompts loaded; "
          f"{len(prompts) - len(completed_prompts)} remaining.")
    
    if len(completed_prompts) == len(prompts):
        print("All prompts already complete. Nothing to do.")
        return
    
    # Load policy
    print(f"Loading policy: {policy_name}")
    tokenizer = AutoTokenizer.from_pretrained(policy_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        policy_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    
    # Initialize / refresh output structure
    if existing is not None:
        output = existing
        # Drop any partial-prompt records (incomplete prompts will be redone)
        output["records"] = [
            r for r in output["records"]
            if r["prompt_idx"] in completed_prompts
        ]
    else:
        output = {
            "config": {
                "policy": policy_name,
                "n_prompts": len(prompts),
                "n_samples": n_samples,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "prompts_dataset": prompts_dataset,
                "prompts_split": prompts_split,
            },
            "records": [],
            "scores": {},
        }
    
    # Format all prompts
    formatted = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        formatted.append(text)
    
    # Generate per prompt with checkpointing
    n_to_do = len(prompts) - len(completed_prompts)
    pbar = tqdm(total=n_to_do * n_samples, desc="Generating")
    t0 = time.time()
    
    for prompt_idx in range(len(prompts)):
        if prompt_idx in completed_prompts:
            continue
        
        responses = generate_for_prompt(
            formatted[prompt_idx],
            model,
            tokenizer,
            n_samples=n_samples,
            chunk_size=chunk_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Append to records
        for sample_idx, response in enumerate(responses):
            output["records"].append({
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "prompt": prompts[prompt_idx],
                "response": response,
                "response_length_tokens": len(tokenizer.encode(response)),
            })
        
        # Save after each prompt (atomic)
        save_progress(out_path, output)
        
        # Free memory between prompts
        torch.cuda.empty_cache()
        
        pbar.update(n_samples)
    
    pbar.close()
    elapsed = time.time() - t0
    print(f"Generation took {elapsed:.1f}s "
          f"({n_to_do * n_samples / max(elapsed, 1):.1f} samples/sec)")
    print(f"Total records in {out_path}: {len(output['records'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--n_prompts", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=384,
                        help="Lowered from 512 to reduce KV cache memory.")
    parser.add_argument("--chunk_size", type=int, default=64,
                        help="Samples per generation forward pass per prompt. "
                             "Reduced if OOM hit. Effective: chunk_size * 1 = "
                             "GPU batch size.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--prompts_dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--prompts_split", default="test_prefs")
    args = parser.parse_args()

    run(
        policy_name=args.policy,
        n_prompts=args.n_prompts,
        n_samples=args.n_samples,
        output_path=args.output_json,
        max_new_tokens=args.max_new_tokens,
        chunk_size=args.chunk_size,
        temperature=args.temperature,
        top_p=args.top_p,
        prompts_dataset=args.prompts_dataset,
        prompts_split=args.prompts_split,
    )


if __name__ == "__main__":
    main()