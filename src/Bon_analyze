"""
Analyze BoN results and produce Goodhart curves.

Takes one or more JSON files produced by bon_generate.py and computes,
for each value of N in {1, 2, 4, 8, ..., max}, the average proxy and
gold scores under BoN-N.

BoN-N from a pool of N_max samples: for each prompt, take the first N
samples and pick the one with highest proxy score; record both its
proxy and gold scores. Average across prompts. This gives us BoN-N's
proxy and gold reward curves as functions of N.

The headline figure of your paper:
- x-axis: N (log scale, 1 to N_max)
- y-axis: gold reward
- two lines: linear-RM vs concave-RM proxy
- expected pattern: linear's gold drops at high N (Goodhart);
                   concave's gold stays flat or grows (no Goodhart).

Run from project root:
    python -m src.bon_analyze \
        --inputs ./outputs/bon-ident.json ./outputs/bon-bounded.json \
        --labels "Linear RM" "Concave RM" \
        --output_dir ./outputs/figures

If matplotlib isn't available, the script just prints the curves as text.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_records(path: str) -> tuple[dict, list]:
    """Load BoN output JSON. Returns (config, list of per-sample records)."""
    with open(path) as f:
        data = json.load(f)
    return data["config"], data["records"]


def organize_by_prompt(records: list) -> dict[int, list[dict]]:
    """Group records by prompt_idx."""
    by_prompt = defaultdict(list)
    for rec in records:
        by_prompt[rec["prompt_idx"]].append(rec)
    # ensure within each prompt, samples are ordered by sample_idx
    for p in by_prompt:
        by_prompt[p].sort(key=lambda x: x["sample_idx"])
    return dict(by_prompt)


def compute_bon_curve(
    by_prompt: dict[int, list[dict]],
    n_values: list[int],
) -> dict:
    """For each N in n_values, compute mean proxy and gold under BoN-N.
    
    BoN-N: from each prompt's pool of samples, take the first N samples,
    pick the one with highest proxy score, record its proxy and gold.
    Average across prompts.
    
    Returns a dict with keys 'N', 'proxy_mean', 'proxy_std', 'gold_mean',
    'gold_std', and aggregate stats per N.
    """
    n_prompts = len(by_prompt)
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
        
        for prompt_idx, samples in by_prompt.items():
            # use first N samples (or all if fewer)
            pool = samples[:N]
            if len(pool) == 0:
                continue
            # pick highest proxy score
            best = max(pool, key=lambda r: r["proxy_score"])
            proxy_picks.append(best["proxy_score"])
            if best.get("gold_score") is not None:
                gold_picks.append(best["gold_score"])
            if best.get("response_length_tokens") is not None:
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
    """Print a BoN curve as a text table."""
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


def plot_curves(curves: list[tuple[str, dict]], output_dir: str):
    """Generate Goodhart curve plots. Skips if matplotlib unavailable."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots. "
              "Install with: pip install matplotlib")
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
    ax.set_xlabel("N (BoN samples per prompt)")
    ax.set_ylabel("Proxy reward (selected sample)")
    ax.set_title("Proxy reward vs. BoN-N")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "bon_proxy.png", dpi=150)
    plt.close(fig)
    
    if has_gold:
        # Plot 2: gold reward vs N (THE HEADLINE FIGURE)
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, c in curves:
            if c["gold_mean"][0] is None:
                continue
            ax.errorbar(c["N"], c["gold_mean"], yerr=c["gold_std"],
                        label=label, marker="o", capsize=3)
        ax.set_xscale("log")
        ax.set_xlabel("N (BoN samples per prompt)")
        ax.set_ylabel("Gold reward (selected sample, scored by Skywork)")
        ax.set_title("Gold reward vs. BoN-N (Goodhart curve)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "bon_gold.png", dpi=150)
        plt.close(fig)
        
        # Plot 3: proxy and gold together (side by side or two lines per RM)
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
    
    print(f"\nSaved plots to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths to bon_generate.py output JSONs.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display labels for each input. Defaults to "
                             "the proxy_activation field.")
    parser.add_argument("--output_dir", default="./outputs/figures")
    parser.add_argument("--n_values", nargs="+", type=int, default=None,
                        help="N values to evaluate. Defaults to log spacing "
                             "from 1 to max samples.")
    parser.add_argument("--save_curves_json", default=None,
                        help="If set, save aggregated curves as JSON for "
                             "downstream plotting.")
    args = parser.parse_args()
    
    if args.labels is None:
        args.labels = [None] * len(args.inputs)
    if len(args.labels) != len(args.inputs):
        raise ValueError("--inputs and --labels must have the same length.")
    
    curves = []
    for path, label in zip(args.inputs, args.labels):
        config, records = load_records(path)
        if label is None:
            label = f"{config['proxy_activation']}"
        
        by_prompt = organize_by_prompt(records)
        max_samples = config["n_samples"]
        
        # Default N grid: dyadic, ~log spacing
        if args.n_values is None:
            n_values = [1]
            n = 2
            while n <= max_samples:
                n_values.append(n)
                n *= 2
        else:
            n_values = sorted(set(args.n_values))
            n_values = [n for n in n_values if n <= max_samples]
        
        curve = compute_bon_curve(by_prompt, n_values)
        print_curve(label, curve)
        curves.append((label, curve))
    
    plot_curves(curves, args.output_dir)
    
    if args.save_curves_json:
        out = {label: curve for label, curve in curves}
        with open(args.save_curves_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved aggregated curves to {args.save_curves_json}")


if __name__ == "__main__":
    main()