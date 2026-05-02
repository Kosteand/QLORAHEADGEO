"""
Data preparation for preference-based reward modeling.

Critical details:
- We use tokenizer.apply_chat_template to format prompt+response correctly.
  Skipping this gives out-of-distribution input to the model and reward
  signal collapses.
- We tokenize chosen and rejected SEPARATELY, producing two (input_ids,
  attention_mask) pairs per example.
- We do NOT pad here. Padding happens in the data collator at batch time.
"""

from __future__ import annotations

from typing import Optional

from datasets import Dataset, load_dataset


def format_pair(
    example: dict,
    tokenizer,
    max_length: int = 1024,
) -> dict:
    """Format one preference pair into tokenized chosen and rejected.
    
    Expects example to have 'chosen' and 'rejected' fields, each a list
    of message dicts in the format used by HF chat templates:
        [{"role": "user", "content": "..."},
         {"role": "assistant", "content": "..."}]
    
    This is the format used by HuggingFaceH4/ultrafeedback_binarized.
    Other datasets may need adapting.
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
    """Load and tokenize UltraFeedback Binarized.
    
    Args:
        tokenizer: HF tokenizer for the base model.
        max_length: max tokens per response (chosen or rejected).
        n_train: if set, use only first n examples (for fast development).
        n_eval: if set, use only first n eval examples.
    """
    train = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
    )
    eval_ = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="test_prefs",
    )
    
    if n_train is not None:
        train = train.select(range(min(n_train, len(train))))
    if n_eval is not None:
        eval_ = eval_.select(range(min(n_eval, len(eval_))))
    
    # Map: each example becomes (chosen_ids, chosen_mask, rejected_ids, rejected_mask)
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


def inspect_example(example: dict, tokenizer) -> None:
    """Print a tokenized example for manual sanity-checking.
    
    Look for: special tokens at start/end, no weird truncation,
    chosen and rejected formatted identically except for content.
    """
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