"""
Data preparation for preference-based reward modeling.

Supports two data sources:
- "ultrafeedback": HuggingFaceH4/ultrafeedback_binarized (real human labels)
- "gold_labeled": dataset relabeled by a strong RM (Path B / Gao et al. methodology)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset, load_from_disk


def format_pair(
    example: dict,
    tokenizer,
    max_length: int = 1024,
) -> dict:
    """Format one preference pair into tokenized chosen and rejected.
    
    Expects example to have 'chosen' and 'rejected' fields, each a list
    of message dicts: [{"role": "user", ...}, {"role": "assistant", ...}].
    Both ultrafeedback_binarized and our gold_labeled output use this format.
    """
    chosen_text = tokenizer.apply_chat_template(
        example["chosen"],
        tokenize=False,
        add_generation_prompt=False,
    )
    rejected_text = tokenizer.apply_chat_template(
        example["rejected"],
        tokenize=False,
        add_generation_prompt=False,
    )
    
    chosen_enc = tokenizer(
        chosen_text,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    rejected_enc = tokenizer(
        rejected_text,
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    
    return {
        "input_ids_chosen": chosen_enc["input_ids"],
        "attention_mask_chosen": chosen_enc["attention_mask"],
        "input_ids_rejected": rejected_enc["input_ids"],
        "attention_mask_rejected": rejected_enc["attention_mask"],
    }


def load_ultrafeedback(
    tokenizer,
    max_length: int = 1024,
    n_train: Optional[int] = None,
    n_eval: Optional[int] = None,
) -> tuple[Dataset, Dataset]:
    """Load and tokenize UltraFeedback Binarized (real human labels)."""
    train = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
    )
    eval_ = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="test_prefs",
    )
    
    skip_train = n_train is not None and n_train == 0
    
    if not skip_train and n_train is not None:
        train = train.select(range(min(n_train, len(train))))
    if n_eval is not None:
        eval_ = eval_.select(range(min(n_eval, len(eval_))))
    
    if skip_train:
        train = train.select(range(0))
    else:
        train = train.map(
            lambda x: format_pair(x, tokenizer, max_length),
            remove_columns=train.column_names,
            desc="Tokenizing train",
        )
    
    eval_ = eval_.map(
        lambda x: format_pair(x, tokenizer, max_length),
        remove_columns=eval_.column_names,
        desc="Tokenizing eval",
    )
    
    return train, eval_


def load_gold_labeled(
    tokenizer,
    dataset_path: str,
    max_length: int = 1024,
    n_train: Optional[int] = None,
    n_eval: Optional[int] = None,
    eval_fraction: float = 0.05,
) -> tuple[Dataset, Dataset]:
    """Load a gold-RM-relabeled dataset (produced by label_with_gold.py).
    
    The gold-labeled dataset has a single split. We carve out a held-out
    eval slice from the end of the dataset (eval_fraction of total).
    
    Args:
        tokenizer: HF tokenizer for the proxy base model.
        dataset_path: filesystem path where the gold-labeled dataset lives
            (the directory written by label_with_gold.py).
        max_length: max tokens per response.
        n_train: cap train set size after the eval split is carved off.
        n_eval: cap eval set size.
        eval_fraction: fraction of the gold-labeled dataset to use as eval.
    """
    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"Gold-labeled dataset not found at {dataset_path}. "
            f"Run `python -m src.label_with_gold --output_dir {dataset_path}` first."
        )
    
    full = load_from_disk(dataset_path)
    n_total = len(full)
    n_eval_split = max(1, int(n_total * eval_fraction))
    
    # Last `n_eval_split` examples are held out as eval; the rest is train.
    train = full.select(range(n_total - n_eval_split))
    eval_ = full.select(range(n_total - n_eval_split, n_total))
    
    print(f"  Gold-labeled dataset: {n_total} total, "
          f"{len(train)} train, {len(eval_)} eval (before caps)")
    
    # Apply size caps
    skip_train = n_train is not None and n_train == 0
    if not skip_train and n_train is not None:
        train = train.select(range(min(n_train, len(train))))
    if n_eval is not None:
        eval_ = eval_.select(range(min(n_eval, len(eval_))))
    
    # Drop the metadata fields that would confuse remove_columns
    relevant_cols = train.column_names
    
    if skip_train:
        train = train.select(range(0))
    else:
        train = train.map(
            lambda x: format_pair(x, tokenizer, max_length),
            remove_columns=relevant_cols,
            desc="Tokenizing train (gold)",
        )
    
    eval_ = eval_.map(
        lambda x: format_pair(x, tokenizer, max_length),
        remove_columns=relevant_cols,
        desc="Tokenizing eval (gold)",
    )
    
    return train, eval_


def inspect_example(example: dict, tokenizer) -> None:
    """Print a tokenized example for manual sanity-checking."""
    print("=" * 60)
    print("CHOSEN")
    print("=" * 60)
    print(tokenizer.decode(example["input_ids_chosen"]))
    print()
    print("Last 5 tokens (id, decoded):")
    for tid in example["input_ids_chosen"][-5:]:
        print(f"  {tid:6d}  {repr(tokenizer.decode([tid]))}")
    print(f"Total length: {len(example['input_ids_chosen'])}")
    print()
    print("=" * 60)
    print("REJECTED")
    print("=" * 60)
    print(tokenizer.decode(example["input_ids_rejected"]))
    print()
    print(f"Total length: {len(example['input_ids_rejected'])}")