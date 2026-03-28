#!/usr/bin/env python3
"""Analysis and visualization tools for MultiGrid training runs.

Usage:
    # Plot a single run
    python -m minigrid.multigrid.examples.analysis runs/empty_20260327_120000

    # Compare multiple runs
    python -m minigrid.multigrid.examples.analysis runs/empty_* --compare

    # Plot specific metrics
    python -m minigrid.multigrid.examples.analysis runs/empty_20260327_120000 \
        --metrics mean_return policy_loss entropy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Exponential moving average smoothing."""
    if len(values) < 2:
        return values
    weights = np.exp(np.linspace(-1.0, 0.0, window))
    weights /= weights.sum()
    smoothed = np.convolve(values, weights, mode="valid")
    # Pad start with original values
    pad = np.full(len(values) - len(smoothed), values[0])
    return np.concatenate([pad, smoothed])


def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load metrics and config from a training run."""
    metrics = pd.read_csv(run_dir / "metrics.csv")
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    return metrics, config


def plot_single_run(
    run_dir: Path,
    metrics_to_plot: list[str] | None = None,
    window: int = 10,
    save: bool = True,
):
    """Plot learning curves for a single training run."""
    df, config = load_run(run_dir)

    if metrics_to_plot is None:
        # Auto-detect agent columns
        agent_cols = [c for c in df.columns if c.startswith("return_agent_")]
        metrics_to_plot = ["mean_return"] + agent_cols + ["policy_loss", "value_loss", "entropy"]

    metrics_to_plot = [m for m in metrics_to_plot if m in df.columns]

    n_plots = len(metrics_to_plot)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    x = df["total_steps"].values

    for ax, metric in zip(axes, metrics_to_plot):
        values = df[metric].values
        ax.plot(x, values, alpha=0.3, color="steelblue")
        ax.plot(x, smooth(values, window), color="steelblue", linewidth=2)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

        # Add eval points if available
        eval_col = f"eval_{metric}" if f"eval_{metric}" in df.columns else None
        if eval_col is None and metric == "mean_return":
            eval_col = "eval_return_mean"
        if eval_col and eval_col in df.columns:
            eval_mask = df[eval_col].notna()
            if eval_mask.any():
                ax.scatter(
                    x[eval_mask],
                    df[eval_col][eval_mask],
                    color="red",
                    s=30,
                    zorder=5,
                    label="eval",
                )
                ax.legend()

    axes[-1].set_xlabel("Environment Steps")

    env_name = config.get("env", "unknown")
    n_agents = config.get("num_agents", "?")
    shared = " (shared)" if config.get("shared_policy") else ""
    fig.suptitle(f"IPPO on {env_name} ({n_agents} agents{shared})", fontsize=14)
    fig.tight_layout()

    if save:
        fig.savefig(run_dir / "learning_curves.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {run_dir / 'learning_curves.png'}")

    return fig


def plot_comparison(
    run_dirs: list[Path],
    metric: str = "mean_return",
    window: int = 10,
    save_path: str | None = None,
):
    """Compare learning curves across multiple runs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for run_dir in run_dirs:
        df, config = load_run(run_dir)
        if metric not in df.columns:
            continue

        x = df["total_steps"].values
        values = df[metric].values
        label = f"{config.get('env', '?')} n={config.get('num_agents', '?')}"
        if config.get("shared_policy"):
            label += " (shared)"

        ax.plot(x, smooth(values, window), linewidth=2, label=label)
        ax.fill_between(
            x,
            smooth(np.maximum(values - np.std(values) * 0.5, 0), window),
            smooth(values + np.std(values) * 0.5, window),
            alpha=0.1,
        )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Comparison: {metric.replace('_', ' ').title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def print_summary(run_dir: Path):
    """Print a text summary of a training run."""
    df, config = load_run(run_dir)

    print(f"\n{'=' * 60}")
    print(f"Run: {run_dir.name}")
    print(f"{'=' * 60}")
    print(f"Environment: {config.get('env')}")
    print(f"Agents: {config.get('num_agents')}")
    print(f"Grid size: {config.get('grid_size')}")
    print(f"Shared policy: {config.get('shared_policy', False)}")
    print(f"Total steps: {df['total_steps'].iloc[-1]:,}")
    print(f"Iterations: {len(df)}")

    # Final metrics
    last_n = min(20, len(df))
    recent = df.tail(last_n)

    print(f"\nFinal {last_n} iterations:")
    print(f"  Mean return:  {recent['mean_return'].mean():.4f} +/- {recent['mean_return'].std():.4f}")
    print(f"  Policy loss:  {recent['policy_loss'].mean():.4f}")
    print(f"  Value loss:   {recent['value_loss'].mean():.4f}")
    print(f"  Entropy:      {recent['entropy'].mean():.4f}")

    # Per-agent returns
    agent_cols = [c for c in df.columns if c.startswith("return_agent_")]
    if agent_cols:
        print(f"\n  Per-agent returns (last {last_n} iters):")
        for col in agent_cols:
            name = col.replace("return_", "")
            print(f"    {name}: {recent[col].mean():.4f}")

    # Eval metrics
    eval_cols = [c for c in df.columns if c.startswith("eval_")]
    if eval_cols:
        last_eval = df.dropna(subset=eval_cols).tail(1)
        if not last_eval.empty:
            print(f"\n  Last evaluation:")
            for col in eval_cols:
                print(f"    {col}: {last_eval[col].values[0]:.4f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze MultiGrid training runs")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run directories")
    parser.add_argument("--compare", action="store_true", help="Compare multiple runs")
    parser.add_argument("--metrics", nargs="+", default=None, help="Metrics to plot")
    parser.add_argument("--window", type=int, default=10, help="Smoothing window")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)

    args = parser.parse_args()

    run_dirs = [d for d in args.run_dirs if d.is_dir() and (d / "metrics.csv").exists()]

    if not run_dirs:
        print("No valid run directories found.")
        return

    if args.compare and len(run_dirs) > 1:
        save_path = args.save_path or "comparison.png"
        plot_comparison(
            run_dirs,
            metric=args.metrics[0] if args.metrics else "mean_return",
            window=args.window,
            save_path=None if args.no_save else save_path,
        )
    else:
        for run_dir in run_dirs:
            print_summary(run_dir)
            plot_single_run(
                run_dir,
                metrics_to_plot=args.metrics,
                window=args.window,
                save=not args.no_save,
            )

    if not args.no_save:
        plt.show()


if __name__ == "__main__":
    main()
