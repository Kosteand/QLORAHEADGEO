"""
Evaluation harness with both standard RM metrics and runaway-suppression
diagnostics specific to the concave-head paper.

Standard metrics:
- Pairwise accuracy on held-out preferences
- Mean reward for chosen and rejected
- Reward margin (chosen - rejected)
- Length correlation (does the RM just learn 'longer = better'?)

Runaway diagnostics:
- Reward profile along grad direction: r(h + t * g/||g||) for t in [-T, +T]
  Linear head: straight line, unbounded.
  Bounded-above head: rises and saturates at a ceiling (asymptote).
  Gaussian head: rises, peaks, and falls (no monotonic direction).
- The diagnostic adapts: monotonic curves get linearity/asymptote metrics;
  peaked curves get peak/falloff metrics.

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
    """Reconstruct the trained RewardModel from a saved LoRA checkpoint."""
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
            f"Using RewardModelConfig defaults."
        )
        config = RewardModelConfig(
            base_model_name=base_model_name,
            activation_name=activation_name,
        )

    base_rm = RewardModel.from_base_model(config, torch_dtype=torch.bfloat16)
    base_rm.config.pad_token_id = AutoTokenizer.from_pretrained(base_model_name).pad_token_id

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

    all_r = np.concatenate([rc, rr])
    all_l = np.concatenate([lc, lr])
    length_corr = np.corrcoef(all_r, all_l)[0, 1]

    # For Gaussian-head models, |z| (distance from peak) is the meaningful
    # quantity, not z itself. Report both.
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
        "preact_chosen_abs_mean": float(np.abs(pc).mean()),
        "preact_rejected_mean": float(pr.mean()),
        "preact_rejected_std": float(pr.std()),
        "preact_rejected_abs_mean": float(np.abs(pr).mean()),
        "length_correlation": float(length_corr),
        "n_examples": len(rc),
    }


def runaway_diagnostic(
    model,
    tokenizer,
    dataset,
    n_examples: int = 50,
    t_range: tuple[float, float] = (-50.0, 500.0),
    n_steps: int = 100,
) -> dict:
    """Probe reward along the steepest-ascent direction in hidden space.

    For each example: compute g = ∇_h r(h), then probe r(h + t * g/||g||)
    for t over the given range. Aggregate the resulting curves.

    Works for any activation shape:
    - Linear: straight line, unbounded.
    - Bounded-above: rises, asymptotes (ceiling).
    - Gaussian: rises, peaks, falls (no monotonic direction).
    """
    device = next(model.parameters()).device

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

    pooled_hs = []
    with torch.no_grad():
        for idx in indices:
            ex = dataset[int(idx)]
            ids = torch.tensor([ex["input_ids_chosen"]], device=device)
            mask = torch.tensor([ex["attention_mask_chosen"]], device=device)
            base = model.base_model.model.model
            out = base(input_ids=ids, attention_mask=mask, return_dict=True)
            from .model import last_token_pool
            h = last_token_pool(out.last_hidden_state, mask).squeeze(0)
            pooled_hs.append(h.detach())

    pooled_hs = torch.stack(pooled_hs)

    with torch.enable_grad():
        h_g = pooled_hs.detach().clone().requires_grad_(True)
        r_val = head(h_g)
        grads = torch.autograd.grad(r_val.sum(), h_g)[0].detach()

    g_norms = grads.norm(dim=-1, keepdim=True)
    grad_unit = grads / (g_norms + 1e-8)

    print(f"  gradient norm: mean={g_norms.mean().item():.4f}, "
          f"std={g_norms.std().item():.4f}")

    ts = torch.linspace(t_range[0], t_range[1], n_steps, device=device)
    r_curves = []
    with torch.no_grad():
        for t in ts:
            h_t = pooled_hs + t.item() * grad_unit
            r_t = head(h_t).squeeze(-1)
            r_curves.append(r_t.float())

    r_curves = torch.stack(r_curves, dim=1)
    r_mean = r_curves.mean(dim=0).cpu().numpy()
    r_std = r_curves.std(dim=0).cpu().numpy()

    return {
        "ts": ts.cpu().numpy().tolist(),
        "reward_mean": r_mean.tolist(),
        "reward_std": r_std.tolist(),
        "grad_norm_mean": float(g_norms.mean().item()),
        "grad_norm_std": float(g_norms.std().item()),
    }


def runaway_summary(diagnostic: dict, activation_name: str = None) -> dict:
    """Summarize the runaway curve into scalars.
    
    Adapts to curve shape:
    - Monotonic curves (ident, bounded_above): linearity_r2, asymptote_estimate.
    - Peaked curves (gaussian): peak_position, reward_falloff_at_endpoint.
    
    Always reports:
    - reward_at_t_max, reward_at_t_min, reward_at_peak: actual values.
    - is_peaked: whether the curve has an interior maximum (suggests Gaussian-like).
    - linearity_r2: R^2 of linear fit. Linear heads ~1.0, peaked heads ~0.
    - concavity_index: average second difference. Negative = concave somewhere.
    
    Args:
        diagnostic: output of runaway_diagnostic().
        activation_name: optional, for default-monotonic vs default-peaked metrics.
    """
    ts = np.array(diagnostic["ts"])
    r = np.array(diagnostic["reward_mean"])
    
    # ---- Linear fit: how line-like is the curve? ----
    p = np.polyfit(ts, r, 1)
    r_pred = np.polyval(p, ts)
    ss_res = ((r - r_pred) ** 2).sum()
    ss_tot = ((r - r.mean()) ** 2).sum()
    linearity_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # ---- Concavity index ----
    second_diff = np.diff(r, n=2)
    concavity_index = float(second_diff.mean())
    
    # ---- Peak detection ----
    # Find argmax. If it's strictly interior (not at boundary), curve is peaked.
    peak_idx = int(np.argmax(r))
    peak_position = float(ts[peak_idx])
    reward_at_peak = float(r[peak_idx])
    
    # "Strictly interior" means at least one step away from each end.
    is_peaked = (peak_idx > 0) and (peak_idx < len(r) - 1)
    
    # If peaked: how much does reward fall off from peak to either end?
    # reward_falloff_left = drop from peak to leftmost t.
    # reward_falloff_right = drop from peak to rightmost t.
    reward_falloff_left = float(reward_at_peak - r[0])
    reward_falloff_right = float(reward_at_peak - r[-1])
    # Total drop from peak to the worse of the two endpoints
    reward_falloff_max = max(reward_falloff_left, reward_falloff_right)
    
    summary = {
        "reward_at_t_max": float(r[-1]),
        "reward_at_t_min": float(r[0]),
        "reward_at_peak": reward_at_peak,
        "peak_position_t": peak_position,
        "is_peaked": bool(is_peaked),
        "reward_falloff_left": reward_falloff_left,
        "reward_falloff_right": reward_falloff_right,
        "reward_falloff_max": reward_falloff_max,
        "linearity_r2": float(linearity_r2),
        "concavity_index": concavity_index,
        "linear_fit_slope": float(p[0]),
        "linear_fit_intercept": float(p[1]),
    }
    
    # For monotonic-style activations, the asymptote estimate is meaningful.
    # For peaked, it's the peak instead. Report under shape-appropriate key.
    monotonic_activations = {"ident", "bounded_above", "bounded"}
    peaked_activations = {"gaussian", "quadratic", "p_linear_bounded_above", "gelu_bounded_above"}
    
    if activation_name in monotonic_activations or (activation_name is None and not is_peaked):
        summary["asymptote_estimate"] = float(r.max())
        summary["curve_shape"] = "monotonic"
    elif activation_name in peaked_activations or is_peaked:
        summary["curve_shape"] = "peaked"
    else:
        summary["curve_shape"] = "unknown"
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--activation_name", default="bounded_above")
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--t_min", type=float, default=-50.0,
                        help="Minimum t for runaway diagnostic.")
    parser.add_argument("--t_max", type=float, default=500.0,
                        help="Maximum t for runaway diagnostic. For Gaussian, "
                             "may want to reduce (e.g. 50) to focus on peak; "
                             "or keep large to see falloff.")
    parser.add_argument("--n_steps", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}...")
    model = load_trained_rm(args.checkpoint, args.base_model, args.activation_name)
    model = model.cuda() if torch.cuda.is_available() else model

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading eval data ({args.n_eval} examples)...")
    _, eval_ds = load_ultrafeedback(
        tokenizer, max_length=1024, n_train=0, n_eval=args.n_eval
    )

    print()
    print("Running standard evaluation...")
    standard = standard_eval(model, tokenizer, eval_ds)
    print(json.dumps(standard, indent=2))

    print()
    print("Running runaway diagnostic...")
    runaway = runaway_diagnostic(
        model, tokenizer, eval_ds,
        t_range=(args.t_min, args.t_max),
        n_steps=args.n_steps,
    )
    summary = runaway_summary(runaway, activation_name=args.activation_name)
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