"""Tests for the minigrid.multigrid multi-agent extension."""
from __future__ import annotations

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Door, Goal, Key, Lava
from minigrid.multigrid.agent import AGENT_COLORS, AgentState
from minigrid.multigrid.envs import (
    MultiGridAdversarialEnv,
    MultiGridDoorKeyEnv,
    MultiGridEmptyEnv,
)
from minigrid.multigrid.multigrid_env import MultiGridEnv


# ---------------------------------------------------------------------------
# Helper: controllable test environment
# ---------------------------------------------------------------------------
class FixedEnv(MultiGridEnv):
    """Test environment with manually controlled agent placement."""

    def __init__(self, num_agents=2, grid_size=8, **kwargs):
        mission_space = MissionSpace(mission_func=lambda: "test")
        super().__init__(
            mission_space=mission_space,
            num_agents=num_agents,
            grid_size=grid_size,
            max_steps=50,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        for i in range(len(self.possible_agents)):
            self.place_agent(i)


# ---------------------------------------------------------------------------
# PettingZoo API conformance
# ---------------------------------------------------------------------------
class TestPettingZooConformance:
    def test_empty_2_agents(self):
        env = MultiGridEmptyEnv(size=8, num_agents=2)
        parallel_api_test(env, num_cycles=20)

    def test_empty_3_agents(self):
        env = MultiGridEmptyEnv(size=8, num_agents=3)
        parallel_api_test(env, num_cycles=20)

    def test_empty_4_agents(self):
        env = MultiGridEmptyEnv(size=10, num_agents=4)
        parallel_api_test(env, num_cycles=20)

    def test_doorkey_2_agents(self):
        env = MultiGridDoorKeyEnv(size=8, num_agents=2)
        parallel_api_test(env, num_cycles=20)

    def test_adversarial_2_agents(self):
        env = MultiGridAdversarialEnv(size=8, num_agents=2)
        parallel_api_test(env, num_cycles=20)

    def test_adversarial_3_agents(self):
        env = MultiGridAdversarialEnv(size=8, num_agents=3)
        parallel_api_test(env, num_cycles=20)


# ---------------------------------------------------------------------------
# Collision resolution
# ---------------------------------------------------------------------------
class TestCollisionResolution:
    def test_same_target(self):
        """Two agents moving to the same cell: neither moves."""
        env = FixedEnv()
        env.reset(seed=42)
        env.agent_states["agent_0"].pos = (2, 3)
        env.agent_states["agent_0"].dir = 0  # right -> (3,3)
        env.agent_states["agent_1"].pos = (4, 3)
        env.agent_states["agent_1"].dir = 2  # left -> (3,3)

        env.step({"agent_0": Actions.forward, "agent_1": Actions.forward})
        assert env.agent_states["agent_0"].pos == (2, 3)
        assert env.agent_states["agent_1"].pos == (4, 3)

    def test_swap_positions(self):
        """Two agents swapping positions: neither moves."""
        env = FixedEnv()
        env.reset(seed=42)
        env.agent_states["agent_0"].pos = (2, 3)
        env.agent_states["agent_0"].dir = 0  # right -> (3,3)
        env.agent_states["agent_1"].pos = (3, 3)
        env.agent_states["agent_1"].dir = 2  # left -> (2,3)

        env.step({"agent_0": Actions.forward, "agent_1": Actions.forward})
        assert env.agent_states["agent_0"].pos == (2, 3)
        assert env.agent_states["agent_1"].pos == (3, 3)

    def test_blocked_by_stationary(self):
        """Agent moving into occupied cell (stationary agent): doesn't move."""
        env = FixedEnv()
        env.reset(seed=42)
        env.agent_states["agent_0"].pos = (2, 3)
        env.agent_states["agent_0"].dir = 0  # right -> (3,3)
        env.agent_states["agent_1"].pos = (3, 3)
        env.agent_states["agent_1"].dir = 0

        env.step({"agent_0": Actions.forward, "agent_1": Actions.left})
        assert env.agent_states["agent_0"].pos == (2, 3)

    def test_non_conflicting_moves(self):
        """Non-conflicting simultaneous moves succeed."""
        env = FixedEnv()
        env.reset(seed=42)
        env.agent_states["agent_0"].pos = (2, 2)
        env.agent_states["agent_0"].dir = 0  # right -> (3,2)
        env.agent_states["agent_1"].pos = (4, 4)
        env.agent_states["agent_1"].dir = 1  # down -> (4,5)

        env.step({"agent_0": Actions.forward, "agent_1": Actions.forward})
        assert env.agent_states["agent_0"].pos == (3, 2)
        assert env.agent_states["agent_1"].pos == (4, 5)


# ---------------------------------------------------------------------------
# Object interactions
# ---------------------------------------------------------------------------
class TestObjectInteractions:
    def test_pickup(self):
        env = FixedEnv()
        env.reset(seed=42)
        env.put_obj(Ball("blue"), 3, 3)
        env.agent_states["agent_0"].pos = (2, 3)
        env.agent_states["agent_0"].dir = 0
        env.agent_states["agent_1"].pos = (5, 5)
        env.agent_states["agent_1"].dir = 0

        env.step({"agent_0": Actions.pickup, "agent_1": Actions.left})
        assert env.agent_states["agent_0"].carrying is not None
        assert env.agent_states["agent_0"].carrying.type == "ball"
        assert env.grid.get(3, 3) is None

    def test_drop(self):
        env = FixedEnv()
        env.reset(seed=42)
        env.agent_states["agent_0"].pos = (2, 3)
        env.agent_states["agent_0"].dir = 0
        env.agent_states["agent_0"].carrying = Ball("blue")
        env.agent_states["agent_1"].pos = (5, 5)
        env.agent_states["agent_1"].dir = 0

        env.step({"agent_0": Actions.drop, "agent_1": Actions.left})
        assert env.agent_states["agent_0"].carrying is None
        assert env.grid.get(3, 3) is not None
        assert env.grid.get(3, 3).type == "ball"

    def test_competing_pickup(self):
        """Two agents competing for same pickup: lower-index wins."""
        env = FixedEnv()
        env.reset(seed=42)
        env.put_obj(Key("yellow"), 3, 3)
        env.agent_states["agent_0"].pos = (2, 3)
        env.agent_states["agent_0"].dir = 0  # facing (3,3)
        env.agent_states["agent_1"].pos = (3, 2)
        env.agent_states["agent_1"].dir = 1  # facing (3,3)

        env.step({"agent_0": Actions.pickup, "agent_1": Actions.pickup})
        assert env.agent_states["agent_0"].carrying is not None
        assert env.agent_states["agent_1"].carrying is None

    def test_door_toggle_with_key(self):
        """Agent with key can toggle a locked door via compat shim."""
        env = FixedEnv()
        env.reset(seed=42)
        door = Door("yellow", is_locked=True)
        env.put_obj(door, 3, 3)
        env.agent_states["agent_0"].pos = (2, 3)
        env.agent_states["agent_0"].dir = 0
        env.agent_states["agent_0"].carrying = Key("yellow")
        env.agent_states["agent_1"].pos = (5, 5)
        env.agent_states["agent_1"].dir = 0

        env.step({"agent_0": Actions.toggle, "agent_1": Actions.left})
        assert door.is_open
        assert not door.is_locked


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
class TestObservations:
    def test_pomdp_shape(self):
        env = FixedEnv()
        obs, _ = env.reset(seed=42)
        assert obs["agent_0"]["image"].shape == (7, 7, 3)

    def test_mdp_shape(self):
        env = FixedEnv(full_obs=True)
        obs, _ = env.reset(seed=42)
        assert obs["agent_0"]["image"].shape == (8, 8, 3)

    def test_other_agent_visible_in_mdp(self):
        env = FixedEnv(full_obs=True)
        obs, _ = env.reset(seed=42)
        img = obs["agent_0"]["image"]
        pos = env.agent_states["agent_1"].pos
        assert img[pos[0], pos[1], 0] == OBJECT_TO_IDX["agent"]

    def test_no_grid_corruption(self):
        """Grid should not retain AgentObj after observation."""
        env = FixedEnv()
        env.reset(seed=42)
        env.gen_obs("agent_0")
        for j in range(env.height):
            for i in range(env.width):
                cell = env.grid.get(i, j)
                assert cell is None or cell.type != "agent"


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------
class TestTermination:
    def test_goal_terminates_agent(self):
        env = FixedEnv()
        env.reset(seed=42)
        env.put_obj(Goal(), 3, 3)
        env.agent_states["agent_0"].pos = (2, 3)
        env.agent_states["agent_0"].dir = 0
        env.agent_states["agent_1"].pos = (5, 5)
        env.agent_states["agent_1"].dir = 0

        _, rew, term, _, _ = env.step(
            {"agent_0": Actions.forward, "agent_1": Actions.left}
        )
        assert term["agent_0"]
        assert rew["agent_0"] > 0
        assert "agent_0" not in env.agents
        assert "agent_1" in env.agents

    def test_max_steps_truncates_all(self):
        env = FixedEnv()
        env.reset(seed=42)
        env.step_count = 49  # next step = 50 = max_steps

        _, _, _, trunc, _ = env.step(
            {"agent_0": Actions.left, "agent_1": Actions.left}
        )
        assert trunc["agent_0"]
        assert trunc["agent_1"]
        assert len(env.agents) == 0

    def test_terminated_agents_removed(self):
        env = FixedEnv()
        env.reset(seed=42)
        env.put_obj(Goal(), 3, 3)
        env.agent_states["agent_0"].pos = (2, 3)
        env.agent_states["agent_0"].dir = 0
        env.agent_states["agent_1"].pos = (5, 5)
        env.agent_states["agent_1"].dir = 0

        env.step({"agent_0": Actions.forward, "agent_1": Actions.left})
        assert "agent_0" not in env.agents


# ---------------------------------------------------------------------------
# Example environments smoke tests
# ---------------------------------------------------------------------------
class TestExampleEnvs:
    def test_empty_runs(self):
        env = MultiGridEmptyEnv(size=6, num_agents=2, render_mode="rgb_array")
        obs, _ = env.reset(seed=1)
        for _ in range(10):
            if not env.agents:
                break
            actions = {name: env.action_space(name).sample() for name in env.agents}
            env.step(actions)
        img = env.render()
        assert img is not None

    def test_doorkey_runs(self):
        env = MultiGridDoorKeyEnv(size=8, num_agents=2, render_mode="rgb_array")
        obs, _ = env.reset(seed=1)
        for _ in range(20):
            if not env.agents:
                break
            actions = {name: env.action_space(name).sample() for name in env.agents}
            env.step(actions)
        img = env.render()
        assert img is not None

    def test_adversarial_runs(self):
        env = MultiGridAdversarialEnv(size=8, num_agents=2, render_mode="rgb_array")
        obs, _ = env.reset(seed=1)
        for _ in range(20):
            if not env.agents:
                break
            actions = {name: env.action_space(name).sample() for name in env.agents}
            env.step(actions)
        img = env.render()
        assert img is not None
