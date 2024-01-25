from __future__ import annotations

import numpy as np

from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


class MemoryEnv(MiniGridEnv):

    """
    ## Description

    This environment is a memory test. The agent starts in a small room where it
    sees an object. It then has to go through a narrow hallway which ends in a
    split. At each end of the split there is an object, one of which is the same
    as the object in the starting room. The agent has to remember the initial
    object, and go to the matching object at split.

    ## Mission Space

    "go to the matching object at the end of the hallway"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the correct matching object.
    2. The agent reaches the wrong matching object.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    S: size of map SxS.

    - `MiniGrid-MemoryS17Random-v0`
    - `MiniGrid-MemoryS13Random-v0`
    - `MiniGrid-MemoryS13-v0`
    - `MiniGrid-MemoryS11-v0`

    """

    def __init__(
        self, size=8, random_length=False, max_steps: int | None = None, **kwargs
    ):
        self.size = size
        self.random_length = random_length

        if max_steps is None:
            max_steps = 5 * size**2

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "go to the matching object at the end of the hallway"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert height % 2 == 1
        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2
        if self.random_length:
            hallway_end = self._rand_int(4, width - 2)
        else:
            hallway_end = width - 3

        # Start room
        for i in range(1, 5):
            self.grid.set(i, upper_room_wall, Wall())
            self.grid.set(i, lower_room_wall, Wall())
        self.grid.set(4, upper_room_wall + 1, Wall())
        self.grid.set(4, lower_room_wall - 1, Wall())

        # Horizontal hallway
        for i in range(5, hallway_end):
            self.grid.set(i, upper_room_wall + 1, Wall())
            self.grid.set(i, lower_room_wall - 1, Wall())

        # Vertical hallway
        for j in range(0, height):
            if j != height // 2:
                self.grid.set(hallway_end, j, Wall())
            self.grid.set(hallway_end + 2, j, Wall())

        # Fix the player's start position and orientation
        for agent in self.agents.values():
            agent.pos = (self._rand_int(1, hallway_end + 1), height // 2)
            agent.dir = 0

        # Place objects
        start_room_obj = self._rand_elem([Key, Ball])
        self.grid.set(1, height // 2 - 1, start_room_obj("green"))

        other_objs = self._rand_elem([[Ball, Key], [Key, Ball]])
        pos0 = (hallway_end + 1, height // 2 - 2)
        pos1 = (hallway_end + 1, height // 2 + 2)
        self.grid.set(*pos0, other_objs[0]("green"))
        self.grid.set(*pos1, other_objs[1]("green"))

        # Choose the target objects
        if start_room_obj == other_objs[0]:
            self.success_pos = (pos0[0], pos0[1] + 1)
            self.failure_pos = (pos1[0], pos1[1] - 1)
        else:
            self.success_pos = (pos1[0], pos1[1] - 1)
            self.failure_pos = (pos0[0], pos0[1] + 1)

        self.mission = "go to the matching object at the end of the hallway"

    def step(self, action):
        for agent in self.agents.values():
            if action[agent.id] == Actions.pickup:
                action[agent.id] = Actions.toggle
        obs, reward, terminated, truncated, info = super().step(action)

        reward = info["rewards"]
        terminated = info["terminated"]
        truncated = info["truncated"]

        for agent in self.agents.values():
            if agent.pos == self.success_pos:
                reward[agent.id] = self._reward()
                terminated[agent.id] = True
            if agent.pos == self.failure_pos:
                reward[agent.id] = 0
                terminated[agent.id] = True

        info["rewards"] = reward
        info["terminated"] = terminated
        info["truncated"] = truncated

        # The reward from environment is the sum of rewards of all agents
        reward = sum(reward.values())
        terminated = any(terminated.values()) if self.is_competitive_env else all(terminated.values())
        truncated = any(truncated.values())

        return obs, reward, terminated, truncated, info
