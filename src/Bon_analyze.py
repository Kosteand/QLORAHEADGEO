"""
Analyze BoN results from a generation+scoring file.

Takes a file produced by bon_generate_only.py and scored with bon_score.py
(which adds entries to data["scores"][label]). For each proxy_label, BoN-N
selection picks the highest-proxy-score sample from the first N samples,
and we report the proxy and gold scores at that selection.

This is the headline analysis: proxy goes up with N (optimization succeeding),
but gold should peak then drop for un-bounded RMs (Goodhart) and stay flat
for properly-bounded RMs.

Run from project root:
    python -m src.bon_analyze \
        --input_json ./outputs/bon-generations-512.json \
        --proxy_labels proxy_linear proxy_bounded proxy_gaussian \
        --gold_label gold \
        --output_dir ./outputs/figures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_scored_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def organize_by_prompt(records: list, scores: dict) -> dict:
    """Group records by prompt_idx, attaching score columns to each."""
    by_prompt = defaultdict(list)
    
    for i, rec in enumerate(records):
        # Attach all available scores to the record
        rec_with_scores = dict(rec)
        for label, score_data in scores.items():
            rec_with_scores[f"score_{label}"] = score_data["values"][i]
        by_prompt[rec["prompt_idx"]].append(rec_with_scores)
    
    for p in by_prompt:
        by_prompt[p].sort(key=lambda x: x["sample_idx"])
    return dict(by_prompt)


def compute_bon_curve(
    by_prompt: dict,
    n_values: list[int],
    proxy_label: str,
    gold_label: str | None = None,
    length_penalty: float = 0.0,
) -> dict:
    """For each N, pick best-by-proxy from first N samples, record proxy + gold.
    
    Args:
        by_prompt: prompts -> list of scored records.
        n_values: which N to evaluate.
        proxy_label: which proxy score to use for ranking.
        gold_label: which gold score to record (optional).
        length_penalty: subtract length_penalty * length_tokens from proxy
            during selection. 0.0 = standard BoN. >0 controls for length bias.
    """
    proxy_key = f"score_{proxy_label}"
    gold_key = f"score_{gold_label}" if gold_label else None
    
    results = {
        "N": [],
        "proxy_mean": [],
        "proxy_std": [],
        "gold_mean": [],
        "gold_std": [],
        "selected_length_mean": [],
        "selected_length_std": [],
    }
    
    for N in n_values:
        proxy_picks = []
        gold_picks = []
        len_picks = []
        
        for samples in by_prompt.values():
            pool = samples[:N]
            if len(pool) == 0:
                continue
            
            # Pick by proxy score (with optional length penalty)
            def selection_score(r):
                base = r[proxy_key]
                if length_penalty > 0:
                    base -= length_penalty * r.get("response_length_tokens", 0)
                return base
            
            best = max(pool, key=selection_score)
            
            proxy_picks.append(best[proxy_key])
            if gold_key is not None and gold_key in best:
                gold_picks.append(best[gold_key])
            if "response_length_tokens" in best:
                len_picks.append(best["response_length_tokens"])
        
        results["N"].append(N)
        results["proxy_mean"].append(float(np.mean(proxy_picks)))
        results["proxy_std"].append(float(np.std(proxy_picks)))
        if gold_picks:
            results["gold_mean"].append(float(np.mean(gold_picks)))
            results["gold_std"].append(float(np.std(gold_picks)))
        else:
            results["gold_mean"].append(None)
            results["gold_std"].append(None)
        if len_picks:
            results["selected_length_mean"].append(float(np.mean(len_picks)))
            results["selected_length_std"].append(float(np.std(len_picks)))
        else:
            results["selected_length_mean"].append(None)
            results["selected_length_std"].append(None)
    
    return results


def print_curve(label: str, curve: dict):
    print(f"\n=== {label} ===")
    has_gold = curve["gold_mean"][0] is not None
    has_len = curve["selected_length_mean"][0] is not None
    
    header = f"{'N':>6}  {'proxy':>10} ± {'std':<8}"
    if has_gold:
        header += f"  {'gold':>10} ± {'std':<8}"
    if has_len:
        header += f"  {'len_tok':>8}"
    print(header)
    print("-" * len(header))
    
    for i, N in enumerate(curve["N"]):
        row = f"{N:>6}  {curve['proxy_mean'][i]:>10.4f} ± {curve['proxy_std'][i]:<8.4f}"
        if has_gold:
            row += f"  {curve['gold_mean'][i]:>10.4f} ± {curve['gold_std'][i]:<8.4f}"
        if has_len:
            row += f"  {curve['selected_length_mean'][i]:>8.1f}"
        print(row)


def plot_curves(curves: list, output_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    has_gold = any(c["gold_mean"][0] is not None for _, c in curves)
    
    # Plot 1: proxy reward vs N
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, c in curves:
        ax.errorbar(c["N"], c["proxy_mean"], yerr=c["proxy_std"],
                    label=label, marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Proxy reward (selected sample)")
    ax.set_title("Proxy reward vs. BoN-N")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "bon_proxy.png", dpi=150)
    plt.close(fig)
    
    if has_gold:
        # Plot 2: gold reward vs N (HEADLINE)
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, c in curves:
            if c["gold_mean"][0] is None:
                continue
            ax.errorbar(c["N"], c["gold_mean"], yerr=c["gold_std"],
                        label=label, marker="o", capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel("Gold reward (Skywork)")
        ax.set_title("Gold reward vs. BoN-N (Goodhart curve)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "bon_gold.png", dpi=150)
        plt.close(fig)
        
        # Plot 3: combined
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for label, c in curves:
            axes[0].errorbar(c["N"], c["proxy_mean"], yerr=c["proxy_std"],
                             label=label, marker="o", capsize=3)
            if c["gold_mean"][0] is not None:
                axes[1].errorbar(c["N"], c["gold_mean"], yerr=c["gold_std"],
                                 label=label, marker="o", capsize=3)
        axes[0].set(xscale="log", xlabel="N", ylabel="Proxy reward",
                    title="Proxy reward (the optimization target)")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].set(xscale="log", xlabel="N", ylabel="Gold reward",
                    title="Gold reward (true quality)")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "bon_combined.png", dpi=150)
        plt.close(fig)
    
    # Length plot - useful for diagnosing length-bias gaming
    has_len = any(c["selected_length_mean"][0] is not None for _, c in curves)
    if has_len:
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, c in curves:
            if c["selected_length_mean"][0] is None:
                continue
            ax.errorbar(c["N"], c["selected_length_mean"],
                        yerr=c["selected_length_std"],
                        label=label, marker="o", capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel("Selected response length (tokens)")
        ax.set_title("Length of BoN-selected responses")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "bon_length.png", dpi=150)
        plt.close(fig)
    
    print(f"\nSaved plots to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True,
                        help="Generation file scored with bon_score.py")
    parser.add_argument("--proxy_labels", nargs="+", required=True,
                        help="Score labels to compare (e.g., proxy_linear proxy_gaussian).")
    parser.add_argument("--display_labels", nargs="+", default=None,
                        help="Display labels for each proxy_label.")
    parser.add_argument("--gold_label", default="gold",
                        help="Score label of the gold RM (default: 'gold').")
    parser.add_argument("--output_dir", default="./outputs/figures")
    parser.add_argument("--n_values", nargs="+", type=int, default=None,
                        help="N values to evaluate. Defaults to dyadic up to max.")
    parser.add_argument("--length_penalty", type=float, default=0.0,
                        help="Subtract length_penalty * length from proxy "
                             "during selection. Tests length-bias-controlled BoN.")
    parser.add_argument("--save_curves_json", default=None)
    args = parser.parse_args()
    
    if args.display_labels is None:
        args.display_labels = list(args.proxy_labels)
    if len(args.display_labels) != len(args.proxy_labels):
        raise ValueError("display_labels and proxy_labels must have the same length.")
    
    print(f"Loading {args.input_json}...")
    data = load_scored_data(args.input_json)
    records = data["records"]
    scores = data.get("scores", {})
    
    # Sanity-check requested labels exist
    available = set(scores.keys())
    requested = set(args.proxy_labels) | {args.gold_label}
    missing = requested - available
    if missing:
        raise ValueError(f"Score labels not found in input file: {missing}. "
                         f"Available: {available}. "
                         f"Run bon_score.py first.")
    
    by_prompt = organize_by_prompt(records, scores)
    max_samples = data["config"]["n_samples"]
    
    if args.n_values is None:
        n_values = [1]
        n = 2
        while n <= max_samples:
            n_values.append(n)
            n *= 2
    else:
        n_values = sorted(set(args.n_values))
        n_values = [n for n in n_values if n <= max_samples]
    
    if args.length_penalty > 0:
        print(f"Using length-controlled BoN selection (penalty={args.length_penalty})")
    
    curves = []
    for proxy_label, display_label in zip(args.proxy_labels, args.display_labels):
        curve = compute_bon_curve(
            by_prompt,
            n_values,
            proxy_label=proxy_label,
            gold_label=args.gold_label,
            length_penalty=args.length_penalty,
        )
        print_curve(display_label, curve)
        curves.append((display_label, curve))
    
    plot_curves(curves, args.output_dir)
    
    if args.save_curves_json:
        out = {label: curve for label, curve in curves}
        with open(args.save_curves_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved aggregated curves to {args.save_curves_json}")


if __name__ == "__main__":
    main()