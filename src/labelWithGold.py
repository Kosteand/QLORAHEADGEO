"""
Generate gold-labeled preference pairs using Skywork-Reward-V2-Llama-3.1-8B.

Workflow:
1. Load UltraFeedback's response pairs (prompt + 2 responses per example).
2. Score both responses with Skywork.
3. The higher-scoring one is "chosen", lower-scoring is "rejected".
4. Save as a Hugging Face dataset on disk for later training.

This is the "Path B" methodology (Gao, Schulman, Hilton 2022): we treat
the gold RM as ground truth, training proxies to mimic its labels.
Goodhart manifests when the proxy's lossy approximation of gold gets
exploited under optimization.

Output format:
    Each example has fields {"chosen", "rejected", "prompt", "gold_score_chosen",
    "gold_score_rejected"}. The chosen and rejected fields are conversation
    lists in OpenAI format, matching UltraFeedback's structure so existing
    tokenization code works unchanged.

Run from project root:
    python -m src.label_with_gold \
        --output_dir ./gold_labeled \
        --n_examples 5000 \
        --gold_dtype bf16

For an A10 (24 GB), use --gold_dtype bf16 (16 GB for weights, ~6 GB headroom).
For tighter VRAM, use --gold_dtype int4 (5 GB for weights via bitsandbytes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

GOLD_MODEL_NAME = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"


def load_gold_model(dtype: str = "bf16"):
    """Load the gold RM. Returns (model, tokenizer)."""
    print(f"Loading gold model: {GOLD_MODEL_NAME} (dtype={dtype})...")
    
    tokenizer = AutoTokenizer.from_pretrained(GOLD_MODEL_NAME)
    if tokenizer.chat_template is None:
        print("  Skywork tokenizer has no chat_template; loading from Llama-3.1-8B-Instruct.")
        from transformers import AutoTokenizer as AT
        llama_tok = AT.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        tokenizer.chat_template = llama_tok.chat_template
    
    kwargs = {"num_labels": 1, "device_map": "cuda"}
    # ... rest unchanged
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
    else:
        raise ValueError(f"Unknown dtype {dtype}; use 'bf16' or 'int4'.")
    
    # Skywork docs recommend flash attention 2 if available; fall back gracefully.
    try:
        kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForSequenceClassification.from_pretrained(GOLD_MODEL_NAME, **kwargs)
    except (ImportError, ValueError) as e:
        print(f"  flash_attention_2 unavailable ({e}); using default attention.")
        kwargs.pop("attn_implementation", None)
        model = AutoModelForSequenceClassification.from_pretrained(GOLD_MODEL_NAME, **kwargs)
    
    model.eval()
    print(f"  Gold model loaded. Hidden size: {model.config.hidden_size}")
    return model, tokenizer


@torch.no_grad()
def score_conversation(conv: list, model, tokenizer, max_length: int = 4096) -> float:
    """Score a single conversation with the gold RM.
    
    Conversation format (OpenAI): [{"role": "user", "content": "..."},
                                    {"role": "assistant", "content": "..."}]
    Returns: scalar reward.
    
    Follows the recommended usage from the Skywork model card:
    apply chat template, tokenize, get logits[0][0].
    """
    formatted = tokenizer.apply_chat_template(conv, tokenize=False)
    # Skywork docs: strip duplicate BOS if present
    if tokenizer.bos_token is not None and formatted.startswith(tokenizer.bos_token):
        formatted = formatted[len(tokenizer.bos_token):]
    
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)
    
    score = model(**inputs).logits[0][0].item()
    return score


@torch.no_grad()
def score_batch(convs: list, model, tokenizer, max_length: int = 4096) -> list:
    """Score multiple conversations as a single batched forward pass.
    
    Padding handles variable-length sequences. Returns list of scalars.
    """
    formatted = []
    for conv in convs:
        text = tokenizer.apply_chat_template(conv, tokenize=False)
        if tokenizer.bos_token is not None and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        formatted.append(text)
    
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(model.device)
    
    scores = model(**inputs).logits.squeeze(-1).float().cpu().tolist()
    if not isinstance(scores, list):
        scores = [scores]
    return scores


def label_pairs(
    n_examples: int,
    batch_size: int,
    output_dir: str,
    gold_dtype: str,
    skip_existing: bool = True,
) -> None:
    """Run the labeling pipeline.
    
    Loads UltraFeedback's chosen+rejected pairs, scores both with the gold RM,
    and saves a new dataset where labels come from gold scores instead of
    human preferences. The chosen and rejected fields can be SWAPPED relative
    to UltraFeedback if the gold disagrees with humans.
    """
    out_path = Path(output_dir)
    if out_path.exists() and skip_existing:
        # Verify it has the expected files
        if (out_path / "dataset_info.json").exists():
            print(f"Output dir {output_dir} already exists. Skipping. "
                  f"Pass --no-skip-existing to regenerate.")
            return
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load source data
    print(f"Loading UltraFeedback (first {n_examples} train_prefs examples)...")
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    ds = ds.select(range(min(n_examples, len(ds))))
    print(f"  Source examples: {len(ds)}")
    
    # Load gold model
    gold_model, gold_tokenizer = load_gold_model(dtype=gold_dtype)
    
    # Score all pairs
    relabeled = {
        "chosen": [],
        "rejected": [],
        "prompt": [],
        "gold_score_chosen": [],
        "gold_score_rejected": [],
        "label_flipped": [],   # True if gold disagreed with UltraFeedback's human label
    }
    
    n_flipped = 0
    
    pbar = tqdm(range(0, len(ds), batch_size), desc="Labeling pairs")
    for batch_start in pbar:
        batch = ds[batch_start : batch_start + batch_size]
        # batch is dict-of-lists; unpack
        ultra_chosen_list = batch["chosen"]
        ultra_rejected_list = batch["rejected"]
        prompt_list = batch["prompt"]
        
        # Build convs: alternating chosen/rejected for batched scoring
        all_convs = []
        for c, r in zip(ultra_chosen_list, ultra_rejected_list):
            all_convs.append(c)
            all_convs.append(r)
        
        # Score in one batched call
        try:
            all_scores = score_batch(all_convs, gold_model, gold_tokenizer)
        except torch.cuda.OutOfMemoryError:
            # Fall back to one-at-a-time if batch is too big
            print(f"  OOM at batch {batch_start}; falling back to per-example.")
            all_scores = [
                score_conversation(c, gold_model, gold_tokenizer)
                for c in all_convs
            ]
        
        # Unpack: scores[0],[2],[4],... are chosen; [1],[3],[5],... are rejected
        for i in range(len(ultra_chosen_list)):
            score_c = all_scores[2 * i]
            score_r = all_scores[2 * i + 1]
            ultra_c = ultra_chosen_list[i]
            ultra_r = ultra_rejected_list[i]
            prompt = prompt_list[i]
            
            # Re-label: gold's higher-scoring response becomes "chosen"
            if score_c >= score_r:
                # Gold agrees with UltraFeedback
                gold_chosen, gold_rejected = ultra_c, ultra_r
                gold_score_chosen, gold_score_rejected = score_c, score_r
                flipped = False
            else:
                # Gold disagrees; swap
                gold_chosen, gold_rejected = ultra_r, ultra_c
                gold_score_chosen, gold_score_rejected = score_r, score_c
                flipped = True
                n_flipped += 1
            
            relabeled["chosen"].append(gold_chosen)
            relabeled["rejected"].append(gold_rejected)
            relabeled["prompt"].append(prompt)
            relabeled["gold_score_chosen"].append(float(gold_score_chosen))
            relabeled["gold_score_rejected"].append(float(gold_score_rejected))
            relabeled["label_flipped"].append(flipped)
        
        pbar.set_postfix(flipped=n_flipped, processed=batch_start + len(ultra_chosen_list))
    
    # Save as HF dataset
    out_dataset = Dataset.from_dict(relabeled)
    out_dataset.save_to_disk(str(out_path))
    
    # Save metadata
    meta = {
        "gold_model": GOLD_MODEL_NAME,
        "n_examples": len(out_dataset),
        "n_flipped_vs_ultrafeedback": n_flipped,
        "flip_rate": n_flipped / len(out_dataset),
        "source_dataset": "HuggingFaceH4/ultrafeedback_binarized:train_prefs",
        "gold_dtype": gold_dtype,
    }
    with open(out_path / "labeling_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print()
    print(f"Saved {len(out_dataset)} gold-labeled examples to {out_path}")
    print(f"Gold disagreed with UltraFeedback humans on {n_flipped} ({100 * n_flipped / len(out_dataset):.1f}%) pairs.")
    print(f"Mean gold score: chosen={sum(relabeled['gold_score_chosen']) / len(out_dataset):.3f}, "
          f"rejected={sum(relabeled['gold_score_rejected']) / len(out_dataset):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./gold_labeled",
                        help="Where to save the relabeled dataset.")
    parser.add_argument("--n_examples", type=int, default=5000,
                        help="How many UltraFeedback pairs to relabel.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Conversations per batched forward pass. "
                             "Note: each pair contributes 2, so effective "
                             "batch is 2 * batch_size. Reduce if OOM.")
    parser.add_argument("--gold_dtype", choices=["bf16", "int4"], default="bf16",
                        help="bf16 needs ~16GB VRAM; int4 needs ~5GB.")
    parser.add_argument("--no_skip_existing", action="store_true",
                        help="Re-label even if output dir already exists.")
    args = parser.parse_args()
    
    label_pairs(
        n_examples=args.n_examples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        gold_dtype=args.gold_dtype,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()