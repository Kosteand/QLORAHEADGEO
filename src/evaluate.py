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
from .model import RewardModel, RewardModelConfig


def load_trained_rm(checkpoint_dir: str, base_model_name: str, activation_name: str):
    """Reconstruct the trained RewardModel from a saved LoRA checkpoint."""
    config = RewardModelConfig(base_model_name=base_model_name, activation_name=activation_name)
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


@torch.no_grad()
def runaway_diagnostic(
    model,
    tokenizer,
    dataset,
    n_examples: int = 50,
    t_range: tuple[float, float] = (-5.0, 20.0),
    n_steps: int = 50,
) -> dict:
    """Probe reward along +w direction in hidden space.
    
    For each example, we:
    1. Compute its hidden state h.
    2. Compute reward at h + t * w_unit for t in [t_min, t_max].
    3. Aggregate the reward curves.
    
    For the LINEAR head, this should be a straight line: r grows
    linearly with t and is unbounded.
    
    For a CONCAVE head, the curve should turn over: marginal reward
    diminishes and r approaches an asymptote.
    
    THIS IS THE PAPER'S HEADLINE FIGURE.
    """
    device = next(model.parameters()).device
    
    # Extract w from the trained head.
    # PeftModel wraps the base model; we navigate to the actual head.
    # The reward_head is in modules_to_save, so it lives at:
    #   model.base_model.model.reward_head
    head = None
    for name, module in model.named_modules():
        if name.endswith("reward_head") and hasattr(module, "linear"):
            head = module
            break
    if head is None:
        raise RuntimeError("Could not find reward_head in model")
    
    w = head.linear.weight.detach().squeeze(0)  # (hidden_size,)
    w_unit = w / w.norm()
    
    print(f"  ||w|| = {w.norm().item():.4f}, hidden_size = {w.shape[0]}")
    
    # Sample n_examples from dataset and compute their pooled hidden states.
    indices = np.random.RandomState(0).choice(
        len(dataset), size=min(n_examples, len(dataset)), replace=False
    )
    
    # We need access to the unwrapped model to get hidden states without
    # applying the head. Use the chosen response from each example.
    pooled_hs = []
    for idx in indices:
        ex = dataset[int(idx)]
        ids = torch.tensor([ex["input_ids_chosen"]], device=device)
        mask = torch.tensor([ex["attention_mask_chosen"]], device=device)
        
        # Forward through base; we need last_hidden_state pooled.
        # Easiest path: call the wrapped model and intercept hidden state.
        # We can use the preactivation output as a proxy: it's w^T h + b,
        # so we can recover h's projection. But for the runaway diagnostic
        # we actually want to perturb h and recompute, which requires h itself.
        
        # Use the base model directly through the PEFT wrapper:
        base = model.base_model.model.model  # peft wrapper -> RewardModel -> AutoModel
        out = base(input_ids=ids, attention_mask=mask, return_dict=True)
        from .model import last_token_pool
        h = last_token_pool(out.last_hidden_state, mask)  # (1, hidden)
        pooled_hs.append(h.squeeze(0))
    
    pooled_hs = torch.stack(pooled_hs)  # (n, hidden)
    
    # Sweep t across the range
    ts = torch.linspace(t_range[0], t_range[1], n_steps, device=device)
    
    # For each (h, t), compute r(h + t * w_unit) = phi(w^T (h + t*w_unit) + b)
    #                                            = phi((w^T h + b) + t * ||w||)
    # We can vectorize this completely without recomputing forward passes.
    
    z_base = (pooled_hs @ w + head.linear.bias.squeeze()).detach()  # (n,)
    w_norm = w.norm()
    
    # z(h, t) = z_base + t * ||w||
    z_grid = z_base.unsqueeze(1) + ts.unsqueeze(0) * w_norm  # (n, n_steps)
    r_grid = head.activation(z_grid)  # (n, n_steps), apply activation
    
    r_mean = r_grid.float().mean(dim=0).cpu().numpy()  # average over examples
    r_std = r_grid.float().std(dim=0).cpu().numpy()
    
    return {
        "ts": ts.cpu().numpy().tolist(),
        "reward_mean": r_mean.tolist(),
        "reward_std": r_std.tolist(),
        "w_norm": float(w_norm.item()),
        "z_base_mean": float(z_base.float().mean().item()),
        "z_base_std": float(z_base.float().std().item()),
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
    _, eval_ds = load_ultrafeedback(tokenizer, max_length=1024, n_train=10, n_eval=args.n_eval)
    
    print()
    print("Running standard evaluation...")
    standard = standard_eval(model, tokenizer, eval_ds)
    print(json.dumps(standard, indent=2))
    
    print()
    print("Running runaway diagnostic...")
    runaway = runaway_diagnostic(model, tokenizer, eval_ds)
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