#!/usr/bin/env python3
"""Train IPPO agents on multi-agent grid environments.

Usage:
    python -m minigrid.multigrid.examples.train --env empty --num-agents 2 --total-steps 200000
    python -m minigrid.multigrid.examples.train --env adversarial --num-agents 2 --total-steps 200000
    python -m minigrid.multigrid.examples.train --env doorkey --num-agents 2 --total-steps 500000

Results are saved to runs/<env>_<timestamp>/ with:
    - metrics.csv: per-iteration training metrics
    - config.json: training configuration
    - model.pt: final policy weights
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from minigrid.multigrid.envs import (
    MultiGridAdversarialEnv,
    MultiGridDoorKeyEnv,
    MultiGridEmptyEnv,
)
from minigrid.multigrid.examples.ippo import IPPOTrainer

ENV_MAP = {
    "empty": MultiGridEmptyEnv,
    "doorkey": MultiGridDoorKeyEnv,
    "adversarial": MultiGridAdversarialEnv,
}


def make_env(env_name: str, num_agents: int, size: int, full_obs: bool):
    env_cls = ENV_MAP[env_name]
    return env_cls(size=size, num_agents=num_agents, full_obs=full_obs)


def evaluate(trainer: IPPOTrainer, env, n_episodes: int = 10) -> dict[str, float]:
    """Run evaluation episodes with greedy actions."""
    all_returns = {name: [] for name in env.possible_agents}

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_rewards = {name: 0.0 for name in env.possible_agents}

        while env.agents:
            actions = {}
            for name in env.agents:
                actions[name] = trainer.get_action(name, obs[name])
            obs, rewards, _, _, _ = env.step(actions)
            for name, r in rewards.items():
                episode_rewards[name] += r

        for name, r in episode_rewards.items():
            all_returns[name].append(r)

    result = {}
    for name in env.possible_agents:
        result[f"eval_return_{name}"] = np.mean(all_returns[name])
    result["eval_return_mean"] = np.mean(
        [np.mean(all_returns[n]) for n in env.possible_agents]
    )
    return result


def train(args):
    # Setup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.env}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create environments
    env = make_env(args.env, args.num_agents, args.grid_size, args.full_obs)
    eval_env = make_env(args.env, args.num_agents, args.grid_size, args.full_obs)

    # Create trainer
    trainer = IPPOTrainer(
        env=env,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=args.device,
        shared_policy=args.shared_policy,
    )

    # Save config
    config = vars(args)
    config["run_dir"] = str(run_dir)
    config["possible_agents"] = env.possible_agents
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    total_steps = 0
    iteration = 0
    all_metrics = []

    print(f"Training IPPO on {args.env} with {args.num_agents} agents")
    print(f"Output: {run_dir}")
    print(f"{'Iter':>5} {'Steps':>10} {'MeanReturn':>12} {'PolicyLoss':>12} {'Entropy':>10}")
    print("-" * 55)

    while total_steps < args.total_steps:
        iteration += 1

        # Collect rollout
        rollout_info = trainer.collect_rollout(args.rollout_steps)
        total_steps += rollout_info["steps"]

        # PPO update
        losses = trainer.update()

        # Compute training episode returns
        episodes = rollout_info["episodes"]
        if episodes:
            mean_returns = {
                name: np.mean([ep.get(name, 0.0) for ep in episodes])
                for name in env.possible_agents
            }
            mean_return = np.mean(list(mean_returns.values()))
        else:
            mean_returns = {name: 0.0 for name in env.possible_agents}
            mean_return = 0.0

        # Periodic evaluation
        eval_metrics = {}
        if iteration % args.eval_interval == 0:
            eval_metrics = evaluate(trainer, eval_env, n_episodes=args.eval_episodes)

        # Log
        metrics = {
            "iteration": iteration,
            "total_steps": total_steps,
            "mean_return": mean_return,
            "n_episodes": len(episodes),
            **{f"return_{name}": mean_returns.get(name, 0.0) for name in env.possible_agents},
            **losses,
            **eval_metrics,
        }
        all_metrics.append(metrics)

        if iteration % args.log_interval == 0:
            eval_str = (
                f"  eval={eval_metrics['eval_return_mean']:.3f}"
                if eval_metrics
                else ""
            )
            print(
                f"{iteration:5d} {total_steps:10d} {mean_return:12.4f} "
                f"{losses['policy_loss']:12.4f} {losses['entropy']:10.4f}{eval_str}"
            )

    # Save final model and metrics
    trainer.save(str(run_dir / "model.pt"))
    df = pd.DataFrame(all_metrics)
    df.to_csv(run_dir / "metrics.csv", index=False)

    # Final evaluation
    final_eval = evaluate(trainer, eval_env, n_episodes=20)
    print(f"\nFinal evaluation (20 episodes):")
    for k, v in final_eval.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nResults saved to {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train IPPO on MultiGrid environments")

    # Environment
    parser.add_argument("--env", type=str, default="empty", choices=list(ENV_MAP.keys()))
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--full-obs", action="store_true")

    # Training
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shared-policy", action="store_true", help="Share policy across agents")
    parser.add_argument("--device", type=str, default="cpu")

    # Logging
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="runs")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
