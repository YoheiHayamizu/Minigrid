from __future__ import annotations

import hashlib
import math
from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Iterable, TypeVar

import gymnasium as gym
import numpy as np
import pygame
import pygame.freetype
from gymnasium import spaces
from pettingzoo import ParallelEnv

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Point, WorldObj
from minigrid.multigrid.agent import AGENT_COLORS, AgentState

T = TypeVar("T")


class MultiGridEnv(ParallelEnv):
    """Multi-agent 2D grid world environment.

    Extends PettingZoo's ParallelEnv for simultaneous multi-agent interaction
    on a shared Minigrid grid.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
        "name": "multigrid_v0",
    }

    def __init__(
        self,
        mission_space: MissionSpace,
        num_agents: int = 2,
        grid_size: int | None = None,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: str | None = None,
        screen_size: int | None = 640,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        full_obs: bool = False,
        agent_colors: list[str] | None = None,
    ):
        super().__init__()

        # Mission
        self.mission = mission_space.sample()
        self._mission_space = mission_space

        # Grid dimensions
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size
        assert width is not None and height is not None
        self.width = width
        self.height = height

        # Actions
        self.actions = Actions
        self._action_space = spaces.Discrete(len(self.actions))

        # Agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observation spaces (depends on full_obs mode)
        self.full_obs = full_obs
        if full_obs:
            image_obs_shape = (self.width, self.height, 3)
        else:
            image_obs_shape = (self.agent_view_size, self.agent_view_size, 3)
        self._obs_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=image_obs_shape, dtype="uint8"
                ),
                "direction": spaces.Discrete(4),
                "mission": mission_space,
            }
        )

        # Agent setup
        assert num_agents >= 1, "Must have at least 1 agent"
        self._num_agents = num_agents
        colors = agent_colors or AGENT_COLORS
        assert len(colors) >= num_agents, (
            f"Need at least {num_agents} colors, got {len(colors)}"
        )

        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = []

        # Agent states (ordered dict for deterministic iteration)
        self.agent_states: OrderedDict[str, AgentState] = OrderedDict()
        for i, name in enumerate(self.possible_agents):
            self.agent_states[name] = AgentState(color=colors[i])

        # PettingZoo dicts
        self.observation_spaces = {
            name: self._obs_space for name in self.possible_agents
        }
        self.action_spaces = {
            name: self._action_space for name in self.possible_agents
        }

        # Steps
        assert isinstance(max_steps, int), (
            f"max_steps must be int, got: {type(max_steps)}"
        )
        self.max_steps = max_steps
        self.step_count = 0

        # Environment config
        self.see_through_walls = see_through_walls

        # Grid (initialized properly in reset)
        self.grid = Grid(width, height)

        # Rendering
        self.render_mode = render_mode
        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None
        self.highlight = highlight
        self.tile_size = tile_size

        # Compatibility shim: set by step() during toggle processing
        self.carrying = None

        # RNG (set by reset)
        self.np_random = None

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        """Return the observation space for an agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Return the action space for an agent."""
        return self.action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment and return initial observations."""
        # Seed RNG
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif self.np_random is None:
            self.np_random = np.random.default_rng()

        # Reset all agent states
        colors = [self.agent_states[name].color for name in self.possible_agents]
        self.agent_states = OrderedDict()
        for i, name in enumerate(self.possible_agents):
            self.agent_states[name] = AgentState(color=colors[i])

        # All agents are active at start
        self.agents = list(self.possible_agents)

        # Create fresh grid
        self.grid = Grid(self.width, self.height)

        # Subclass fills the grid and places agents
        self._gen_grid(self.width, self.height)

        # Validate all agents are placed
        for name in self.possible_agents:
            state = self.agent_states[name]
            assert state.pos != (-1, -1), (
                f"{name} was not placed by _gen_grid(). "
                "Use self.place_agent(agent_index) in your _gen_grid() implementation."
            )
            assert 0 <= state.pos[0] < self.width and 0 <= state.pos[1] < self.height
            # Check agent doesn't overlap with a non-overlappable object
            cell = self.grid.get(*state.pos)
            assert cell is None or cell.can_overlap(), (
                f"{name} placed on non-overlappable object at {state.pos}"
            )

        # Reset step count
        self.step_count = 0
        self.carrying = None

        if self.render_mode == "human":
            self.render()

        # Generate observations
        observations = {name: self.gen_obs(name) for name in self.agents}
        infos = {name: {} for name in self.agents}

        return observations, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Execute simultaneous actions for all active agents.

        Processing order:
        1. Rotation actions (left/right) — no conflicts possible
        2. Forward movement with collision resolution
        3. Object interactions (pickup/drop/toggle) in agent index order
        4. Termination/truncation checks
        5. Observation generation
        """
        self.step_count += 1

        rewards = {name: 0.0 for name in self.agents}
        terminations = {name: False for name in self.agents}
        truncations = {name: False for name in self.agents}
        infos = {name: {} for name in self.agents}

        # Collect current agent positions for occupancy checks
        agent_positions = {
            name: self.agent_states[name].pos for name in self.agents
        }

        # --- Phase 1: Rotations (no conflicts) ---
        for name in self.agents:
            action = actions.get(name)
            state = self.agent_states[name]
            if action == self.actions.left:
                state.dir = (state.dir - 1) % 4
            elif action == self.actions.right:
                state.dir = (state.dir + 1) % 4

        # --- Phase 2: Forward movement with collision resolution ---
        intended = {}  # agent_name -> intended new position
        for name in self.agents:
            action = actions.get(name)
            state = self.agent_states[name]
            if action == self.actions.forward:
                fwd_pos = state.front_pos
                # Check grid bounds
                if not (0 <= fwd_pos[0] < self.width and 0 <= fwd_pos[1] < self.height):
                    intended[name] = state.pos  # stay
                    continue
                fwd_cell = self.grid.get(*fwd_pos)
                # Check if cell is passable
                if fwd_cell is not None and not fwd_cell.can_overlap():
                    intended[name] = state.pos  # blocked by object
                    continue
                intended[name] = fwd_pos
            else:
                intended[name] = state.pos  # not moving

        # Resolve collisions
        resolved = self._resolve_movements(intended, agent_positions)

        # Commit movements
        for name in self.agents:
            self.agent_states[name].pos = resolved[name]

        # Check goal/lava after movement
        for name in self.agents:
            state = self.agent_states[name]
            cell = self.grid.get(*state.pos)
            if cell is not None and cell.type == "goal":
                terminations[name] = True
                rewards[name] = self._reward(name)
            elif cell is not None and cell.type == "lava":
                terminations[name] = True
                rewards[name] = 0.0

        # --- Phase 3: Object interactions (deterministic order) ---
        # Build set of positions occupied by agents (after movement)
        occupied = {self.agent_states[n].pos for n in self.agents}

        for name in self.agents:
            action = actions.get(name)
            state = self.agent_states[name]

            if action not in (
                self.actions.pickup,
                self.actions.drop,
                self.actions.toggle,
            ):
                continue

            fwd_pos = state.front_pos
            # Bounds check
            if not (0 <= fwd_pos[0] < self.width and 0 <= fwd_pos[1] < self.height):
                continue

            # Check if another agent is in the forward cell
            agent_in_front = fwd_pos in occupied and fwd_pos != state.pos

            fwd_cell = self.grid.get(*fwd_pos)

            if action == self.actions.pickup:
                if (
                    fwd_cell
                    and fwd_cell.can_pickup()
                    and state.carrying is None
                    and not agent_in_front
                ):
                    state.carrying = fwd_cell
                    state.carrying.cur_pos = (-1, -1)
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

            elif action == self.actions.drop:
                if (
                    state.carrying
                    and fwd_cell is None
                    and not agent_in_front
                ):
                    self.grid.set(fwd_pos[0], fwd_pos[1], state.carrying)
                    state.carrying.cur_pos = fwd_pos
                    state.carrying = None

            elif action == self.actions.toggle:
                if fwd_cell:
                    # Compatibility shim for Door.toggle which reads env.carrying
                    self.carrying = state.carrying
                    fwd_cell.toggle(self, fwd_pos)
                    self.carrying = None

        # --- Phase 4: Truncation ---
        if self.step_count >= self.max_steps:
            for name in self.agents:
                truncations[name] = True

        # Apply termination/truncation to agent states
        for name in list(self.agents):
            if terminations[name]:
                self.agent_states[name].terminated = True
            if truncations[name]:
                self.agent_states[name].truncated = True

        if self.render_mode == "human":
            self.render()

        # Generate observations for still-active agents
        observations = {}
        for name in self.agents:
            observations[name] = self.gen_obs(name)

        # Remove terminated/truncated agents (PettingZoo convention)
        self.agents = [
            name for name in self.agents
            if self.agent_states[name].is_active
        ]

        return observations, rewards, terminations, truncations, infos

    def _resolve_movements(
        self,
        intended: dict[str, tuple[int, int]],
        current: dict[str, tuple[int, int]],
    ) -> dict[str, tuple[int, int]]:
        """Resolve simultaneous movement conflicts.

        Rules:
        1. If two+ agents intend to move to the same cell, none of them move.
        2. If an agent intends to move to a cell occupied by another agent
           that is NOT moving away, the moving agent stays.
        3. If two agents would swap positions (A->B and B->A), neither moves.

        Returns resolved positions for all agents.
        """
        resolved = dict(intended)
        changed = True

        # Iterate until stable (conflicts can cascade)
        while changed:
            changed = False

            # Rule 1: Multiple agents targeting the same cell
            target_counts: dict[tuple[int, int], list[str]] = {}
            for name, pos in resolved.items():
                target_counts.setdefault(pos, []).append(name)

            for pos, agents_targeting in target_counts.items():
                # Only conflict if multiple agents are MOVING to the same cell
                movers = [n for n in agents_targeting if resolved[n] != current[n]]
                if len(movers) >= 2:
                    for name in movers:
                        if resolved[name] != current[name]:
                            resolved[name] = current[name]
                            changed = True

            # Rule 2: Moving into a cell occupied by a stationary agent
            for name, target in resolved.items():
                if target == current[name]:
                    continue  # not moving
                # Is there another agent whose resolved position is our target?
                for other_name, other_target in resolved.items():
                    if other_name == name:
                        continue
                    if other_target == target:
                        # Other agent is (or will be) at our target
                        resolved[name] = current[name]
                        changed = True
                        break

            # Rule 3: Swap detection (A->B and B->A)
            for name_a, target_a in resolved.items():
                if target_a == current[name_a]:
                    continue
                for name_b, target_b in resolved.items():
                    if name_b == name_a:
                        continue
                    if target_b == current[name_b]:
                        continue
                    if target_a == current[name_b] and target_b == current[name_a]:
                        resolved[name_a] = current[name_a]
                        resolved[name_b] = current[name_b]
                        changed = True

        return resolved

    def render(self) -> np.ndarray | None:
        """Render the environment. Full implementation in issue #6."""
        return None

    def close(self):
        """Close the rendering window."""
        if self.window:
            pygame.quit()
            self.window = None

    # ------------------------------------------------------------------
    # Abstract method — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _gen_grid(self, width: int, height: int):
        """Generate the grid layout and place agents.

        Subclasses must:
        1. Create self.grid = Grid(width, height)
        2. Add walls, objects, etc.
        3. Call self.place_agent(i) for each agent index
        4. Set self.mission
        """
        pass

    # ------------------------------------------------------------------
    # Observation generation (placeholder — implemented in issue #5)
    # ------------------------------------------------------------------

    def gen_obs(self, agent_name: str) -> dict[str, Any]:
        """Generate observation for a specific agent.

        Full implementation (POMDP/MDP, visibility, agent encoding)
        is in issue #5. This returns a minimal valid observation.
        """
        state = self.agent_states[agent_name]

        if self.full_obs:
            image = self.grid.encode()
        else:
            image = np.zeros(
                (self.agent_view_size, self.agent_view_size, 3), dtype="uint8"
            )

        return {
            "image": image,
            "direction": state.dir,
            "mission": self.mission,
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _reward(self, agent_name: str) -> float:
        """Compute the reward to be given upon success.

        Override in subclasses for custom reward logic.
        """
        return 1 - 0.9 * (self.step_count / self.max_steps)

    # ------------------------------------------------------------------
    # Placement helpers
    # ------------------------------------------------------------------

    def place_obj(
        self,
        obj: WorldObj | None,
        top: Point | None = None,
        size: tuple[int, int] | None = None,
        reject_fn=None,
        max_tries=math.inf,
    ) -> tuple[int, int]:
        """Place an object at an empty position in the grid.

        Avoids positions occupied by any agent.
        """
        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        # Collect all agent positions for rejection
        agent_positions = {
            state.pos for state in self.agent_states.values()
            if state.pos != (-1, -1)
        }

        num_tries = 0
        while True:
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")
            num_tries += 1

            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            # Don't place on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place where any agent is
            if pos in agent_positions:
                continue

            # Custom rejection
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj: WorldObj, i: int, j: int):
        """Put an object at a specific position in the grid."""
        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
        self,
        agent_index: int,
        top: Point | None = None,
        size: tuple[int, int] | None = None,
        rand_dir: bool = True,
        max_tries=math.inf,
    ) -> tuple[int, int]:
        """Place an agent at an empty position in the grid.

        Args:
            agent_index: Index of the agent to place (0 to num_agents-1).
            top: Top-left corner of the placement region.
            size: Size of the placement region.
            rand_dir: If True, randomize the agent's direction.
            max_tries: Maximum placement attempts.

        Returns:
            The position where the agent was placed.
        """
        name = self.possible_agents[agent_index]
        state = self.agent_states[name]

        # Temporarily mark as unplaced so place_obj doesn't reject its own pos
        old_pos = state.pos
        state.pos = (-1, -1)

        pos = self.place_obj(None, top, size, max_tries=max_tries)
        state.pos = pos

        if rand_dir:
            state.dir = self._rand_int(0, 4)

        return pos

    # ------------------------------------------------------------------
    # Hash
    # ------------------------------------------------------------------

    def hash(self, size: int = 16) -> str:
        """Compute a hash that uniquely identifies the current state."""
        sample_hash = hashlib.sha256()
        to_encode = [
            self.grid.encode().tolist(),
            {name: (s.pos, s.dir) for name, s in self.agent_states.items()},
        ]
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))
        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self) -> int:
        return self.max_steps - self.step_count

    # ------------------------------------------------------------------
    # Random utilities (ported from MiniGridEnv)
    # ------------------------------------------------------------------

    def _rand_int(self, low: int, high: int) -> int:
        """Generate random integer in [low, high)."""
        return self.np_random.integers(low, high)

    def _rand_float(self, low: float, high: float) -> float:
        """Generate random float in [low, high)."""
        return self.np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        """Generate random boolean value."""
        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """Pick a random element in a list."""
        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> list[T]:
        """Sample a random subset of distinct elements of a list."""
        lst = list(iterable)
        assert num_elems <= len(lst)

        out: list[T] = []
        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self) -> str:
        """Generate a random color name (string)."""
        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(
        self, x_low: int, x_high: int, y_low: int, y_high: int
    ) -> tuple[int, int]:
        """Generate a random (x, y) position tuple."""
        return (
            self.np_random.integers(x_low, x_high),
            self.np_random.integers(y_low, y_high),
        )
