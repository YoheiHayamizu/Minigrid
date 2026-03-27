from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.multigrid.multigrid_env import MultiGridEnv


class MultiGridDoorKeyEnv(MultiGridEnv):
    """Cooperative multi-agent door-key environment.

    Agents must cooperate: one picks up the key and unlocks the door,
    allowing another to pass through and reach the goal.
    When any agent reaches the goal, all agents receive a shared reward.
    """

    def __init__(
        self,
        size: int = 8,
        num_agents: int = 2,
        max_steps: int | None = None,
        **kwargs,
    ):
        if max_steps is None:
            max_steps = 10 * size * size

        mission_space = MissionSpace(
            mission_func=lambda: "cooperate to unlock the door and reach the goal"
        )

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

        # Vertical wall splitting the room
        split_x = width // 2
        self.grid.vert_wall(split_x, 0, height)

        # Locked door in the wall
        door_y = self._rand_int(1, height - 1)
        self.door = Door("yellow", is_locked=True)
        self.grid.set(split_x, door_y, self.door)

        # Key in the left room
        self.place_obj(
            Key("yellow"), top=(1, 1), size=(split_x - 1, height - 2)
        )

        # Goal in the right room
        self.place_obj(
            Goal(), top=(split_x + 1, 1), size=(width - split_x - 2, height - 2)
        )

        # Place all agents in the left room
        for i in range(len(self.possible_agents)):
            self.place_agent(i, top=(1, 1), size=(split_x - 1, height - 2))

        self.mission = "cooperate to unlock the door and reach the goal"

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = super().step(actions)

        # Cooperative: when any agent reaches the goal, all get rewarded
        goal_reached = False
        for name in list(self.agents) + [
            n for n in self.possible_agents if self.agent_states[n].terminated
        ]:
            state = self.agent_states[name]
            cell = self.grid.get(*state.pos)
            if cell is not None and cell.type == "goal" and not goal_reached:
                goal_reached = True

        if goal_reached:
            reward = self._reward(self.possible_agents[0])
            for name in self.agents:
                rewards[name] = reward
                terminations[name] = True
                self.agent_states[name].terminated = True
            self.agents = []

        return obs, rewards, terminations, truncations, infos
