"""
Plot PPO results comparing head types and KL settings.

Run from project root:
    python -m src.ppo_analyze \
        --ppo_dirs ./outputs/ppo-ident-no-kl ./outputs/ppo-ident-kl \
                   ./outputs/ppo-gaussian-no-kl \
        --labels "Linear no-KL" "Linear KL" "Gaussian no-KL" \
        --output_dir ./outputs/figures-ppo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_logs(ppo_dir: Path):
    proxy_path = ppo_dir / "proxy_log.json"
    gold_path = ppo_dir / "gold_log.json"
    
    proxy_log = []
    if proxy_path.exists():
        with open(proxy_path) as f:
            proxy_log = json.load(f)
    
    gold_log = []
    if gold_path.exists():
        with open(gold_path) as f:
            gold_log = json.load(f)
    
    return proxy_log, gold_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo_dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output_dir", default="./outputs/figures-ppo")
    parser.add_argument("--smooth_window", type=int, default=10)
    args = parser.parse_args()
    
    if args.labels is None:
        args.labels = [Path(d).name for d in args.ppo_dirs]
    if len(args.labels) != len(args.ppo_dirs):
        raise ValueError("labels and ppo_dirs must have same length")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    for label, ppo_dir in zip(args.labels, args.ppo_dirs):
        proxy_log, gold_log = load_logs(Path(ppo_dir))
        
        if proxy_log:
            steps = np.array([e["step"] for e in proxy_log])
            proxy = np.array([e["proxy_mean"] for e in proxy_log])
            
            if args.smooth_window > 1 and len(proxy) >= args.smooth_window:
                kernel = np.ones(args.smooth_window) / args.smooth_window
                proxy_smooth = np.convolve(proxy, kernel, mode="valid")
                steps_smooth = steps[args.smooth_window - 1:]
                axes[0].plot(steps, proxy, alpha=0.3)
                axes[0].plot(steps_smooth, proxy_smooth, label=label, linewidth=2)
            else:
                axes[0].plot(steps, proxy, label=label, linewidth=2)
        
        if gold_log:
            g_steps = np.array([e["step"] for e in gold_log])
            g_mean = np.array([e["gold_mean"] for e in gold_log])
            g_std = np.array([e["gold_std"] for e in gold_log])
            n = np.array([e.get("n_prompts", 1) for e in gold_log])
            sem = g_std / np.sqrt(np.maximum(n, 1))
            
            axes[1].errorbar(g_steps, g_mean, yerr=sem, label=label,
                             marker="o", capsize=3, linewidth=2)
    
    axes[0].set(xlabel="PPO updates", ylabel="Proxy reward (training signal)",
                title="Proxy reward over PPO updates")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    
    axes[1].set(xlabel="PPO updates", ylabel="Gold reward (Skywork)",
                title="Gold reward over PPO updates")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_dir / "ppo_combined.png", dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_dir / 'ppo_combined.png'}")


if __name__ == "__main__":
    main()