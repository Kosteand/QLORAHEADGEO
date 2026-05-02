"""
Evaluation harness with both standard RM metrics and runaway-suppression
diagnostics specific to the concave-head paper.

Standard metrics:
- Pairwise accuracy on held-out preferences
- Mean reward for chosen and rejected
- Reward margin (chosen - rejected)
- Length correlation (does the RM just learn 'longer = better'?)

Runaway diagnostics:
- Reward profile along w: r(h_train + t*w/||w||) for t in [-T, +T]
  Linear head: straight line. Concave head: turns over.
- Reward profile along random orthogonal directions: should be ~flat
  (the head only depends on h via w, so orthogonal directions don't move r)
- Empirical curvature: second derivative of r along w on training data
- Effective gradient norm: how much |dr/dh| shrinks as we move along +w

Run:
    python -m src.evaluate --checkpoint outputs/rm-qwen-0.5b/final \
                            --base_model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from .data import load_ultrafeedback
from .heads import RewardHead
from .model import RewardModel, RewardModelConfig


def load_trained_rm(checkpoint_dir: str, base_model_name: str, activation_name: str):
    """Reconstruct the trained RewardModel from a saved LoRA checkpoint.
    
    If a saved rm_train_config.json is present in the checkpoint dir,
    we use it to reconstruct the head with the exact architecture used
    at training time (head_width, intermediate_size, etc.). Without this,
    we fall back to RewardModelConfig defaults which may not match.
    """
    # Try to load saved training config so we get head_width etc. right.
    config_path = Path(checkpoint_dir) / "rm_train_config.json"
    if config_path.exists():
        with open(config_path) as f:
            saved = json.load(f)
        print(f"Loaded saved training config from {config_path}")
        config = RewardModelConfig(
            base_model_name=saved.get("base_model_name", base_model_name),
            activation_name=saved.get("activation_name", activation_name),
            head_init_scale=saved.get("head_init_scale", 0.02),
            head_init_bias=saved.get("head_init_bias", 0.0),
            head_width=saved.get("head_width", 32),
            head_intermediate_size=saved.get("head_intermediate_size", None),
        )
    else:
        print(
            f"WARNING: no rm_train_config.json found at {config_path}. "
            f"Using RewardModelConfig defaults; this may cause shape "
            f"mismatches if the saved checkpoint used non-default values."
        )
        config = RewardModelConfig(
            base_model_name=base_model_name,
            activation_name=activation_name,
        )
    
    base_rm = RewardModel.from_base_model(config, torch_dtype=torch.bfloat16)
    base_rm.config.pad_token_id = AutoTokenizer.from_pretrained(base_model_name).pad_token_id
    
    # Load LoRA adapter + saved head
    model = PeftModel.from_pretrained(base_rm, checkpoint_dir)
    model.eval()
    return model


@torch.no_grad()
def standard_eval(model, tokenizer, dataset, batch_size: int = 8) -> dict:
    """Compute pairwise accuracy and reward statistics."""
    device = next(model.parameters()).device
    
    rewards_chosen = []
    rewards_rejected = []
    lengths_chosen = []
    lengths_rejected = []
    preacts_chosen = []
    preacts_rejected = []
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        
        for key_ids, key_mask, r_list, len_list, pre_list in [
            ("input_ids_chosen", "attention_mask_chosen",
             rewards_chosen, lengths_chosen, preacts_chosen),
            ("input_ids_rejected", "attention_mask_rejected",
             rewards_rejected, lengths_rejected, preacts_rejected),
        ]:
            padded = tokenizer.pad(
                {"input_ids": batch[key_ids], "attention_mask": batch[key_mask]},
                return_tensors="pt",
            ).to(device)
            
            out = model(
                input_ids=padded["input_ids"],
                attention_mask=padded["attention_mask"],
                return_preactivation=True,
            )
            r = out.logits.squeeze(-1).float().cpu().numpy()
            z = out.hidden_states.squeeze(-1).float().cpu().numpy()
            r_list.extend(r.tolist())
            pre_list.extend(z.tolist())
            len_list.extend(padded["attention_mask"].sum(dim=1).cpu().tolist())
    
    rc = np.array(rewards_chosen)
    rr = np.array(rewards_rejected)
    pc = np.array(preacts_chosen)
    pr = np.array(preacts_rejected)
    lc = np.array(lengths_chosen)
    lr = np.array(lengths_rejected)
    
    accuracy = (rc > rr).mean()
    
    # Length correlation: do longer responses get higher reward?
    # We want this near zero; values >0.5 indicate severe length bias.
    all_r = np.concatenate([rc, rr])
    all_l = np.concatenate([lc, lr])
    length_corr = np.corrcoef(all_r, all_l)[0, 1]
    
    return {
        "pairwise_accuracy": float(accuracy),
        "reward_chosen_mean": float(rc.mean()),
        "reward_chosen_std": float(rc.std()),
        "reward_rejected_mean": float(rr.mean()),
        "reward_rejected_std": float(rr.std()),
        "reward_margin_mean": float((rc - rr).mean()),
        "reward_margin_std": float((rc - rr).std()),
        "preact_chosen_mean": float(pc.mean()),
        "preact_chosen_std": float(pc.std()),
        "preact_rejected_mean": float(pr.mean()),
        "preact_rejected_std": float(pr.std()),
        "length_correlation": float(length_corr),
        "n_examples": len(rc),
    }


def runaway_diagnostic(
    model,
    tokenizer,
    dataset,
    n_examples: int = 50,
    t_range: tuple[float, float] = (-5.0, 20.0),
    n_steps: int = 50,
) -> dict:
    """Probe reward along the steepest-ascent direction in hidden space.

    For each example we:
    1. Compute its pooled hidden state h.
    2. Compute g = ∇_h r(h) (the steepest-ascent direction).
    3. Probe r(h + t * g/||g||) for t in [t_min, t_max].
    4. Aggregate the reward curves.

    For the linear head, g = w everywhere, so this reduces to the
    original w-direction probe. For the MLP head, g varies per example
    but still measures how far the reward can grow as we move in the
    most reward-increasing direction from each training point.

    THIS IS THE PAPER'S HEADLINE FIGURE.
    """
    device = next(model.parameters()).device

    # Find the RewardHead instance anywhere in the PEFT-wrapped model.
    head = None
    for _, module in model.named_modules():
        if isinstance(module, RewardHead):
            head = module
            break
    if head is None:
        raise RuntimeError("Could not find RewardHead in model")

    indices = np.random.RandomState(0).choice(
        len(dataset), size=min(n_examples, len(dataset)), replace=False
    )

    # Collect pooled hidden states from the base transformer.
    pooled_hs = []
    with torch.no_grad():
        for idx in indices:
            ex = dataset[int(idx)]
            ids = torch.tensor([ex["input_ids_chosen"]], device=device)
            mask = torch.tensor([ex["attention_mask_chosen"]], device=device)
            base = model.base_model.model.model  # peft → RewardModel → AutoModel
            out = base(input_ids=ids, attention_mask=mask, return_dict=True)
            from .model import last_token_pool
            h = last_token_pool(out.last_hidden_state, mask).squeeze(0)
            pooled_hs.append(h.detach())

    pooled_hs = torch.stack(pooled_hs)  # (n, hidden)

    # Compute the steepest-ascent direction g = ∇_h r(h) for each h.
    # torch.enable_grad re-enables grad inside this no-grad context.
    with torch.enable_grad():
        h_g = pooled_hs.detach().clone().requires_grad_(True)
        r_val = head(h_g)  # (n, 1)
        grads = torch.autograd.grad(r_val.sum(), h_g)[0].detach()  # (n, hidden)

    g_norms = grads.norm(dim=-1, keepdim=True)          # (n, 1)
    grad_unit = grads / (g_norms + 1e-8)               # (n, hidden)

    print(f"  gradient norm: mean={g_norms.mean().item():.4f}, "
          f"std={g_norms.std().item():.4f}")

    # Sweep t and evaluate r(h + t * g_unit) for each example.
    ts = torch.linspace(t_range[0], t_range[1], n_steps, device=device)
    r_curves = []
    with torch.no_grad():
        for t in ts:
            h_t = pooled_hs + t.item() * grad_unit  # (n, hidden)
            r_t = head(h_t).squeeze(-1)             # (n,)
            r_curves.append(r_t.float())

    r_curves = torch.stack(r_curves, dim=1)  # (n, n_steps)
    r_mean = r_curves.mean(dim=0).cpu().numpy()
    r_std = r_curves.std(dim=0).cpu().numpy()

    return {
        "ts": ts.cpu().numpy().tolist(),
        "reward_mean": r_mean.tolist(),
        "reward_std": r_std.tolist(),
        "grad_norm_mean": float(g_norms.mean().item()),
        "grad_norm_std": float(g_norms.std().item()),
    }


def runaway_summary(diagnostic: dict) -> dict:
    """Summarize the runaway curve into a few scalars.
    
    Key metrics:
    - reward_at_t_max: reward at the largest t we tested. For a linear head
      this is huge; for a concave head it's bounded.
    - asymptote_estimate: the highest reward observed across all t. For
      concave heads this approximates the supremum sup_h r(h).
    - linearity_score: R^2 of a linear fit to the reward curve. Linear
      heads score 1.0; concave heads score lower the more they bend.
    - concavity_index: the largest second difference / first difference.
      Larger negative values indicate stronger concavity.
    """
    ts = np.array(diagnostic["ts"])
    r = np.array(diagnostic["reward_mean"])
    
    # Linearity: how well does a straight line fit r as a function of t?
    p = np.polyfit(ts, r, 1)
    r_pred = np.polyval(p, ts)
    ss_res = ((r - r_pred) ** 2).sum()
    ss_tot = ((r - r.mean()) ** 2).sum()
    linearity_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Concavity index: average second difference, normalized
    second_diff = np.diff(r, n=2)
    concavity_index = float(second_diff.mean())
    
    return {
        "reward_at_t_max": float(r[-1]),
        "reward_at_t_min": float(r[0]),
        "asymptote_estimate": float(r.max()),
        "linearity_r2": float(linearity_r2),
        "concavity_index": concavity_index,  # negative = concave
        "linear_fit_slope": float(p[0]),
        "linear_fit_intercept": float(p[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--activation_name", default="bounded_above")
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    model = load_trained_rm(args.checkpoint, args.base_model, args.activation_name)
    model = model.cuda() if torch.cuda.is_available() else model
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading eval data ({args.n_eval} examples)...")
    # n_train=0 since evaluate.py doesn't use the train split.
    # We pass None to skip train-split tokenization entirely.
    _, eval_ds = load_ultrafeedback(
        tokenizer, max_length=1024, n_train=0, n_eval=args.n_eval
    )
    
    print()
    print("Running standard evaluation...")
    standard = standard_eval(model, tokenizer, eval_ds)
    print(json.dumps(standard, indent=2))
    
    print()
    print("Running runaway diagnostic...")
    runaway = runaway_diagnostic(model, tokenizer, eval_ds, t_range=(-50.0, 500.0), n_steps=100)
    summary = runaway_summary(runaway)
    print("Runaway summary:")
    print(json.dumps(summary, indent=2))
    
    results = {
        "standard": standard,
        "runaway": runaway,
        "runaway_summary": summary,
        "activation_name": args.activation_name,
        "checkpoint": args.checkpoint,
    }
    
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()