from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.multigrid.multigrid_env import MultiGridEnv


class MultiGridAdversarialEnv(MultiGridEnv):
    """Competitive multi-agent environment.

    Agents race to reach a single goal. The first agent to reach the
    goal gets a positive reward; all other agents get zero reward
    and the episode ends for everyone.
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

        mission_space = MissionSpace(mission_func=lambda: "race to the goal")

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

        # Single goal
        self.place_obj(Goal())

        # Place agents at random positions
        for i in range(len(self.possible_agents)):
            self.place_agent(i)

        self.mission = "race to the goal"

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = super().step(actions)

        # Check if any agent reached the goal
        winner = None
        for name in list(self.agents):
            state = self.agent_states[name]
            cell = self.grid.get(*state.pos)
            if cell is not None and cell.type == "goal":
                winner = name
                break

        if winner is not None:
            # Winner gets reward, all agents terminate
            rewards[winner] = self._reward(winner)
            for name in self.agents:
                terminations[name] = True
                self.agent_states[name].terminated = True
            self.agents = []

        return obs, rewards, terminations, truncations, infos
