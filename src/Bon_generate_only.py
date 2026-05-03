"""
Pure-generation step for BoN experiments.

Generate N responses per prompt from a base policy and save them as JSON.
No reward model scoring is done here -- that's a separate step (bon_score.py)
so the same generations can be reused across multiple RMs.

This is the slow part: generation cost scales with N * n_prompts.
Run this once, then score with as many proxy/gold RMs as you want.

Run from project root:
    python -m src.bon_generate_only \
        --policy Qwen/Qwen2.5-0.5B-Instruct \
        --n_prompts 100 \
        --n_samples 512 \
        --output_json ./outputs/bon-generations-512.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def generate_responses(
    prompts: list[str],
    policy_model,
    policy_tokenizer,
    n_samples: int,
    batch_size: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[list[str]]:
    """For each prompt, generate n_samples responses."""
    results = [[] for _ in prompts]

    formatted = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = policy_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)

    pbar = tqdm(total=len(prompts) * n_samples, desc="Generating")

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = formatted[batch_start : batch_start + batch_size]

        inputs = policy_tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(policy_model.device)

        outputs = policy_model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n_samples,
            pad_token_id=policy_tokenizer.pad_token_id,
        )

        input_len = inputs.input_ids.shape[1]
        gen_only = outputs[:, input_len:]
        decoded = policy_tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for i in range(len(batch_prompts)):
            for j in range(n_samples):
                results[batch_start + i].append(decoded[i * n_samples + j])

        pbar.update(len(batch_prompts) * n_samples)

    pbar.close()
    return results


def run(
    policy_name: str,
    n_prompts: int,
    n_samples: int,
    output_path: str,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    prompts_dataset: str,
    prompts_split: str,
):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading prompts from {prompts_dataset}:{prompts_split}...")
    ds = load_dataset(prompts_dataset, split=prompts_split)
    ds = ds.select(range(min(n_prompts, len(ds))))
    prompts = [ex["prompt"] for ex in ds]
    print(f"  {len(prompts)} prompts loaded.")

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

    t0 = time.time()
    responses = generate_responses(
        prompts,
        model,
        tokenizer,
        n_samples=n_samples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    elapsed = time.time() - t0
    print(f"Generation took {elapsed:.1f}s "
          f"({len(prompts) * n_samples / elapsed:.1f} samples/sec)")

    # Build records, one per (prompt, sample)
    records = []
    for prompt_idx, prompt in enumerate(prompts):
        for sample_idx, response in enumerate(responses[prompt_idx]):
            records.append({
                "prompt_idx": prompt_idx,
                "sample_idx": sample_idx,
                "prompt": prompt,
                "response": response,
                "response_length_tokens": len(tokenizer.encode(response)),
            })

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
        "records": records,
        "scores": {},  # to be filled in by bon_score.py
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(records)} records to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--n_prompts", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
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
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        prompts_dataset=args.prompts_dataset,
        prompts_split=args.prompts_split,
    )


if __name__ == "__main__":
    main()