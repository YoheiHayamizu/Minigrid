from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Tuple, Optional, Iterable, Union, List

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.agent import Agent
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv


class EmptyEnv(MiniGridEnv):
    """
    ## Description

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

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

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Empty-5x5-v0`
    - `MiniGrid-Empty-Random-5x5-v0`
    - `MiniGrid-Empty-6x6-v0`
    - `MiniGrid-Empty-Random-6x6-v0`
    - `MiniGrid-Empty-8x8-v0`
    - `MiniGrid-Empty-16x16-v0`

    """

    def __init__(
        self,
        size: int = 8,
        agents: int | Iterable[Agent] = 1,
        agents_start_pos: List[Tuple[int, int]] = [(1, 1), ],
        agents_start_dir: List[int] = [0, ],
        max_steps: int | None = None,
        **kwargs,
    ):
        num_agents = agents if isinstance(agents, int) else len(agents)
        if agents_start_pos is not None:
            assert len(agents_start_pos) == num_agents, "Number of agents and starting positions must match"
            assert len(agents_start_dir) == num_agents, "Number of agents and starting directions must match"
        self.agents_start_pos = agents_start_pos
        self.agents_start_dir = agents_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            agents=agents,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agents_start_pos is not None:
            for i in range(len(self.agents_start_pos)):
                self.agents[i].pos = self.agents_start_pos[i]
                self.agents[i].dir = self.agents_start_dir[i]
        else:
            self.place_agent()

        self.mission = "get to the green goal square"
