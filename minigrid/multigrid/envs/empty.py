from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.multigrid.multigrid_env import MultiGridEnv


class MultiGridEmptyEnv(MultiGridEnv):
    """Multi-agent empty room environment.

    N agents in an empty room, each with their own colored goal.
    Each agent must navigate to its assigned goal independently.
    """

    def __init__(
        self,
        size: int = 8,
        num_agents: int = 2,
        max_steps: int | None = None,
        **kwargs,
    ):
        if max_steps is None:
            max_steps = 4 * size * size

        mission_space = MissionSpace(mission_func=lambda: "get to your goal")

        super().__init__(
            mission_space=mission_space,
            num_agents=num_agents,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width: int, height: int):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Place a colored goal for each agent
        for i, name in enumerate(self.possible_agents):
            color = self.agent_states[name].color
            self.place_obj(Goal(color))

        # Place agents at random positions
        for i in range(len(self.possible_agents)):
            self.place_agent(i)

        self.mission = "get to your goal"

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = super().step(actions)

        # Check if each agent reached their colored goal
        for name in list(self.agents):
            state = self.agent_states[name]
            cell = self.grid.get(*state.pos)
            if cell is not None and cell.type == "goal" and cell.color == state.color:
                rewards[name] = self._reward(name)
                terminations[name] = True
                state.terminated = True

        # Remove newly terminated agents
        self.agents = [
            name for name in self.agents if self.agent_states[name].is_active
        ]

        return obs, rewards, terminations, truncations, infos
