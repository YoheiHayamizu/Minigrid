"""Independent PPO (IPPO) for multi-agent grid environments.

Each agent has its own policy network and is trained independently
using PPO, treating other agents as part of the environment.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Shared CNN backbone with separate actor (policy) and critic (value) heads."""

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int):
        super().__init__()
        h, w, c = obs_shape  # (view_size, view_size, 3)

        self.image_conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size
        dummy = torch.zeros(1, c, h, w)
        conv_out = self.image_conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out + 4, 128),  # +4 for direction one-hot
            nn.ReLU(),
        )

        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(
        self, image: torch.Tensor, direction: torch.Tensor
    ) -> tuple[Categorical, torch.Tensor]:
        # image: (batch, H, W, C) -> (batch, C, H, W)
        x = image.float() / 255.0
        if x.dim() == 4 and x.shape[-1] <= 4:
            # (batch, H, W, C) -> (batch, C, H, W)
            x = x.permute(0, 3, 1, 2)
        x = self.image_conv(x)

        # direction one-hot
        direction = direction.long()
        if direction.dim() == 0:
            direction = direction.unsqueeze(0)
        dir_onehot = torch.zeros(direction.shape[0], 4, device=image.device)
        dir_onehot.scatter_(1, direction.unsqueeze(1) if direction.dim() == 1 else direction, 1.0)

        x = torch.cat([x, dir_onehot], dim=1)
        x = self.fc(x)

        logits = self.actor(x)
        value = self.critic(x)

        return Categorical(logits=logits), value.squeeze(-1)


class RolloutBuffer:
    """Stores transitions for one agent over a rollout."""

    def __init__(self):
        self.images: list[np.ndarray] = []
        self.directions: list[int] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.dones: list[bool] = []

    def add(
        self,
        image: np.ndarray,
        direction: int,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        self.images.append(image)
        self.directions.append(direction)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(
        self, last_value: float, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns."""
        n = len(self.rewards)
        advantages = torch.zeros(n)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            advantages[t] = last_gae = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae
            )

        returns = advantages + torch.tensor(self.values)
        return returns, advantages

    def to_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "images": torch.tensor(np.array(self.images), device=device),
            "directions": torch.tensor(self.directions, device=device),
            "actions": torch.tensor(self.actions, device=device),
            "log_probs": torch.tensor(self.log_probs, device=device),
        }

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


class IPPOTrainer:
    """Independent PPO trainer for multi-agent environments.

    Each agent gets its own ActorCritic network and optimizer.
    Training is fully independent — each agent treats others as
    part of the environment.

    Args:
        env: A PettingZoo ParallelEnv (e.g., MultiGridEmptyEnv).
        lr: Learning rate.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_eps: PPO clipping epsilon.
        entropy_coef: Entropy bonus coefficient.
        value_coef: Value loss coefficient.
        n_epochs: PPO update epochs per rollout.
        batch_size: Mini-batch size for PPO updates.
        device: Torch device.
        shared_policy: If True, all agents share a single policy network.
    """

    def __init__(
        self,
        env,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
        shared_policy: bool = False,
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Get observation/action shapes from first agent
        sample_agent = env.possible_agents[0]
        obs_space = env.observation_space(sample_agent)
        obs_shape = obs_space["image"].shape
        n_actions = env.action_space(sample_agent).n

        # Create per-agent networks (or shared)
        self.shared_policy = shared_policy
        if shared_policy:
            shared_net = ActorCritic(obs_shape, n_actions).to(self.device)
            shared_opt = torch.optim.Adam(shared_net.parameters(), lr=lr)
            self.policies = {name: shared_net for name in env.possible_agents}
            self.optimizers = {name: shared_opt for name in env.possible_agents}
        else:
            self.policies = {}
            self.optimizers = {}
            for name in env.possible_agents:
                net = ActorCritic(obs_shape, n_actions).to(self.device)
                self.policies[name] = net
                self.optimizers[name] = torch.optim.Adam(net.parameters(), lr=lr)

        self.buffers = {name: RolloutBuffer() for name in env.possible_agents}

    @torch.no_grad()
    def collect_rollout(self, n_steps: int) -> dict[str, Any]:
        """Collect n_steps of experience from the environment.

        Returns metrics dict with episode returns.
        """
        obs, _ = self.env.reset()
        episode_rewards = {name: 0.0 for name in self.env.possible_agents}
        completed_episodes: list[dict[str, float]] = []
        steps_collected = 0

        for buf in self.buffers.values():
            buf.clear()

        while steps_collected < n_steps:
            if not self.env.agents:
                # Episode ended, record and reset
                completed_episodes.append(dict(episode_rewards))
                episode_rewards = {name: 0.0 for name in self.env.possible_agents}
                obs, _ = self.env.reset()

            # Get actions from each agent's policy
            actions = {}
            for name in self.env.agents:
                policy = self.policies[name]
                image = torch.tensor(
                    obs[name]["image"], device=self.device
                ).unsqueeze(0)
                direction = torch.tensor(
                    [obs[name]["direction"]], device=self.device
                )

                dist, value = policy(image, direction)
                action = dist.sample()

                actions[name] = action.item()
                self.buffers[name].add(
                    image=obs[name]["image"],
                    direction=obs[name]["direction"],
                    action=action.item(),
                    log_prob=dist.log_prob(action).item(),
                    reward=0.0,  # filled after step
                    value=value.item(),
                    done=False,
                )

            # Step environment
            next_obs, rewards, terminations, truncations, infos = self.env.step(
                actions
            )

            # Fill in rewards and dones
            for name in actions:
                buf = self.buffers[name]
                buf.rewards[-1] = rewards.get(name, 0.0)
                buf.dones[-1] = terminations.get(name, False) or truncations.get(
                    name, False
                )
                episode_rewards[name] += rewards.get(name, 0.0)

            obs = next_obs
            steps_collected += 1

        # Record final partial episode
        if any(episode_rewards[n] != 0.0 for n in self.env.possible_agents):
            completed_episodes.append(dict(episode_rewards))

        return {"episodes": completed_episodes, "steps": steps_collected}

    def update(self) -> dict[str, float]:
        """Run PPO updates on collected rollouts.

        Returns dict of loss metrics.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        agents_to_update = (
            [self.env.possible_agents[0]]
            if self.shared_policy
            else self.env.possible_agents
        )

        for name in agents_to_update:
            buf = self.buffers[name]
            if len(buf) == 0:
                continue

            # Compute last value for bootstrapping
            policy = self.policies[name]
            with torch.no_grad():
                if self.env.agents and name in self.env.agents:
                    obs, _ = self.env.reset()
                    last_img = torch.tensor(
                        obs[name]["image"], device=self.device
                    ).unsqueeze(0)
                    last_dir = torch.tensor(
                        [obs[name]["direction"]], device=self.device
                    )
                    _, last_value = policy(last_img, last_dir)
                    last_val = last_value.item()
                else:
                    last_val = 0.0

            returns, advantages = buf.compute_returns(
                last_val, self.gamma, self.gae_lambda
            )
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                advantages = torch.zeros_like(advantages)

            tensors = buf.to_tensors(self.device)
            returns = returns.to(self.device)
            advantages = advantages.to(self.device)

            n_samples = len(buf)
            indices = np.arange(n_samples)

            for _ in range(self.n_epochs):
                np.random.shuffle(indices)

                for start in range(0, n_samples, self.batch_size):
                    end = min(start + self.batch_size, n_samples)
                    batch_idx = indices[start:end]

                    batch_images = tensors["images"][batch_idx]
                    batch_dirs = tensors["directions"][batch_idx]
                    batch_actions = tensors["actions"][batch_idx]
                    batch_old_log_probs = tensors["log_probs"][batch_idx]
                    batch_returns = returns[batch_idx]
                    batch_advantages = advantages[batch_idx]

                    dist, values = policy(batch_images, batch_dirs)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()

                    # PPO clipped objective
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                        * batch_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = nn.functional.mse_loss(values, batch_returns)

                    # Total loss
                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy
                    )

                    optimizer = self.optimizers[name]
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    n_updates += 1

        n_updates = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def get_action(self, agent_name: str, obs: dict) -> int:
        """Get a greedy action for evaluation."""
        policy = self.policies[agent_name]
        policy.eval()
        with torch.no_grad():
            image = torch.tensor(obs["image"], device=self.device).unsqueeze(0)
            direction = torch.tensor([obs["direction"]], device=self.device)
            dist, _ = policy(image, direction)
            return dist.probs.argmax(dim=-1).item()

    def save(self, path: str):
        """Save all policy weights."""
        state = {}
        for name, policy in self.policies.items():
            state[name] = policy.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        """Load policy weights."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        for name, policy in self.policies.items():
            if name in state:
                policy.load_state_dict(state[name])
