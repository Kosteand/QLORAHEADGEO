"""
Score pre-generated BoN responses with a reward model.

Takes a JSON file produced by bon_generate_only.py and adds a new score
column under output["scores"][label]. Multiple scoring runs accumulate
in the same file -- proxy_linear, proxy_bounded, proxy_gaussian, gold_skywork.

Run from project root:
    # Score with a proxy RM
    python -m src.bon_score \
        --input_json ./outputs/bon-generations-512.json \
        --score_label proxy_gaussian \
        --rm_type proxy \
        --proxy_checkpoint ./outputs/rm-gaussian-gold5k/final \
        --proxy_activation gaussian

    # Score with the gold RM (Skywork)
    python -m src.bon_score \
        --input_json ./outputs/bon-generations-512.json \
        --score_label gold \
        --rm_type gold

The same input_json is updated in-place with the new score column.
Run scoring multiple times for different RMs; each run adds another column.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .model import RewardModel, RewardModelConfig

GOLD_MODEL_NAME = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"


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
    return model, tokenizer


def load_gold_rm(dtype: str = "bf16"):
    """Load Skywork."""
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


@torch.no_grad()
def score_with_proxy(
    prompt_response_pairs: list[tuple[str, str]],
    proxy_model,
    proxy_tokenizer,
    batch_size: int = 16,
    max_length: int = 1024,
) -> list[float]:
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
            batch_scores = []
            for j in range(len(batch)):
                single = {k: v[j:j+1] for k, v in enc.items()}
                out = gold_model(**single)
                batch_scores.append(out.logits[0][0].float().cpu().item())

        if not isinstance(batch_scores, list):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)

    return scores


def run(
    input_json: str,
    score_label: str,
    rm_type: str,
    proxy_checkpoint: str | None,
    proxy_base_model: str,
    proxy_activation: str | None,
    gold_dtype: str,
    batch_size: int,
    overwrite: bool,
):
    """Score the records in input_json with the specified RM, save in-place."""

    in_path = Path(input_json)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_json}")

    print(f"Loading {input_json}...")
    with open(in_path) as f:
        data = json.load(f)
    records = data["records"]
    if "scores" not in data:
        data["scores"] = {}

    if score_label in data["scores"] and not overwrite:
        print(f"Score label '{score_label}' already exists. "
              f"Pass --overwrite to recompute. Skipping.")
        return

    # Build (prompt, response) pairs in record order
    pairs = [(r["prompt"], r["response"]) for r in records]

    if rm_type == "proxy":
        if proxy_checkpoint is None or proxy_activation is None:
            raise ValueError("rm_type=proxy requires --proxy_checkpoint and --proxy_activation")
        model, tokenizer = load_proxy_rm(proxy_checkpoint, proxy_base_model, proxy_activation)

        t0 = time.time()
        scores = score_with_proxy(pairs, model, tokenizer, batch_size=batch_size)
        elapsed = time.time() - t0
        print(f"Proxy scoring took {elapsed:.1f}s")

        meta = {
            "type": "proxy",
            "checkpoint": proxy_checkpoint,
            "activation": proxy_activation,
            "base_model": proxy_base_model,
        }

    elif rm_type == "gold":
        model, tokenizer = load_gold_rm(dtype=gold_dtype)

        t0 = time.time()
        scores = score_with_gold(pairs, model, tokenizer, batch_size=batch_size)
        elapsed = time.time() - t0
        print(f"Gold scoring took {elapsed:.1f}s")

        meta = {
            "type": "gold",
            "model_name": GOLD_MODEL_NAME,
            "dtype": gold_dtype,
        }

    else:
        raise ValueError(f"Unknown rm_type: {rm_type}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Save scores
    data["scores"][score_label] = {
        "values": scores,
        "meta": meta,
    }

    with open(in_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(scores)} '{score_label}' scores to {in_path}")
    print(f"  mean: {sum(scores) / len(scores):.4f}")
    print(f"  min:  {min(scores):.4f}")
    print(f"  max:  {max(scores):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--score_label", required=True,
                        help="Identifier for this score column (e.g. 'proxy_gaussian', 'gold').")
    parser.add_argument("--rm_type", required=True, choices=["proxy", "gold"])
    parser.add_argument("--proxy_checkpoint", default=None)
    parser.add_argument("--proxy_base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--proxy_activation", default=None,
                        choices=[None, "ident", "bounded", "bounded_above",
                                 "gelu_bounded_above", "gaussian"])
    parser.add_argument("--gold_dtype", choices=["bf16", "int4"], default="bf16")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size; defaults to 16 for proxy, 4 for gold.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = 16 if args.rm_type == "proxy" else 4

    run(
        input_json=args.input_json,
        score_label=args.score_label,
        rm_type=args.rm_type,
        proxy_checkpoint=args.proxy_checkpoint,
        proxy_base_model=args.proxy_base_model,
        proxy_activation=args.proxy_activation,
        gold_dtype=args.gold_dtype,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()