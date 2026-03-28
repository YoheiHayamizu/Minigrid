#!/usr/bin/env python3
"""Manual control for multi-agent grid environments.

Control agent_0 with keyboard, other agents act randomly.

Usage:
    python -m minigrid.multigrid.examples.manual_control --env empty --num-agents 2
    python -m minigrid.multigrid.examples.manual_control --env adversarial --num-agents 3

Controls:
    Arrow keys: move/turn agent_0
    Space: toggle
    P: pickup
    D: drop
    Enter: done action
    Backspace: reset environment
    Escape: quit
"""
from __future__ import annotations

import argparse

import numpy as np
import pygame

from minigrid.core.actions import Actions
from minigrid.multigrid.envs import (
    MultiGridAdversarialEnv,
    MultiGridDoorKeyEnv,
    MultiGridEmptyEnv,
)

ENV_MAP = {
    "empty": MultiGridEmptyEnv,
    "doorkey": MultiGridDoorKeyEnv,
    "adversarial": MultiGridAdversarialEnv,
}

KEY_TO_ACTION = {
    pygame.K_LEFT: Actions.left,
    pygame.K_RIGHT: Actions.right,
    pygame.K_UP: Actions.forward,
    pygame.K_SPACE: Actions.toggle,
    pygame.K_p: Actions.pickup,
    pygame.K_d: Actions.drop,
    pygame.K_RETURN: Actions.done,
}


def run(args):
    env_cls = ENV_MAP[args.env]
    env = env_cls(
        size=args.grid_size,
        num_agents=args.num_agents,
        render_mode="human",
        screen_size=args.screen_size,
    )

    obs, _ = env.reset(seed=args.seed)
    env.render()

    step_count = 0
    running = True

    print(f"\nManual control: {args.env} ({args.num_agents} agents)")
    print("You control agent_0 (red). Others act randomly.")
    print("Arrow keys=move, P=pickup, D=drop, Space=toggle, Backspace=reset, Esc=quit\n")

    while running:
        human_action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_BACKSPACE:
                    obs, _ = env.reset()
                    env.render()
                    step_count = 0
                    print("--- Reset ---")
                elif event.key in KEY_TO_ACTION:
                    human_action = KEY_TO_ACTION[event.key]

        if human_action is not None and env.agents:
            # Build action dict
            actions = {}
            for name in env.agents:
                if name == "agent_0":
                    actions[name] = human_action
                else:
                    actions[name] = env.action_space(name).sample()

            obs, rewards, terminations, truncations, infos = env.step(actions)
            step_count += 1

            # Print step info
            parts = [f"step={step_count}"]
            for name in env.possible_agents:
                if name in rewards:
                    r = rewards[name]
                    t = terminations.get(name, False)
                    if r != 0 or t:
                        parts.append(f"{name}: r={r:.2f} done={t}")
            if len(parts) > 1:
                print("  ".join(parts))

            if not env.agents:
                print(f"\n--- Episode done after {step_count} steps ---")
                print("Press Backspace to reset or Escape to quit.\n")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Manual control for MultiGrid")
    parser.add_argument("--env", type=str, default="empty", choices=list(ENV_MAP.keys()))
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--screen-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
