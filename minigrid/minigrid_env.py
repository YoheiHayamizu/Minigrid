from __future__ import annotations

from abc import abstractmethod
import hashlib
import math
import numpy as np
import pygame
import pygame.freetype
from typing import Any, Iterable, SupportsFloat, TypeVar, Optional, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from minigrid.core.actions import Actions
from minigrid.core.agent import Agent
from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, TILE_PIXELS
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import WorldObj

T = TypeVar("T")
AgentID = int


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        mission_space: MissionSpace,
        agents: int | Iterable[Agent] = 1,
        grid_size: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        max_steps: int = 100,
        see_through_walls: bool = False,
        agent_view_size: int = 7,
        render_mode: Optional[str] = None,
        screen_size: Optional[int] = 1,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
        is_competitive_env: bool = True,
    ):
        """
        MiniGrid environment initialization

        Args:
            mission_space (MissionSpace): The mission space for the agent
            agents (int | Iterable[Agent], optional): Number of agents in the environment. Defaults to 1.
            grid_size (Optional[int], optional): Size of the environment grid (width and height). Defaults to None.
            width (Optional[int], optional): Width of the environment grid (overrides grid_size). Defaults to None.
            height (Optional[int], optional): Height of the environment grid (overrides grid_size). Defaults to None.
            max_steps (int, optional): Maximum number of steps per episode. Defaults to 100.
            see_through_walls (bool, optional): Whether agents can see through walls. Defaults to False.
            agent_view_size (int, optional): Size of agent view (must be odd). Defaults to 7.
            render_mode (Optional[str], optional): Rendering mode (human or rgb_array). Defaults to None.
            screen_size (Optional[int], optional): Size of the screen (in tiles). Defaults to 1.
            highlight (bool, optional): Whether to highlight the agent's field of view. Defaults to True.
            tile_size (int, optional): Size of the tiles when rendering. Defaults to TILE_PIXELS.
            agent_pov (bool, optional): Whether to render the agent's point of view. Defaults to False.
        """
        # Initialize mission
        self.mission = mission_space.sample()
        self.is_competitive_env = is_competitive_env

        # Initialize grid
        width, height = (grid_size, grid_size) if grid_size else (width, height)
        assert width is not None and height is not None
        self.width, self.height = width, height
        self.grid = Grid(width, height)

        # Initialize agents
        if isinstance(agents, int):
            self.agents: dict[AgentID, Agent] = {}
            for agent_id in range(agents):
                agent = Agent(
                    agent_id,
                    mission_space,
                    view_size=agent_view_size,
                    see_through_walls=see_through_walls,
                )
                self.agents[agent_id] = agent
        elif isinstance(agents, Iterable):
            self.agents: dict[AgentID, Agent] = {agent.id: agent for agent in agents}
        else:
            raise ValueError(f"Invalid argument for agents: {agents}")

        # Action enumeration for this environment
        self.actions = Actions

        # Set joint action space
        self.action_space = spaces.Dict({
            agent.id: agent.action_space
            for agent in self.agents.values()
        })

        # Set joint observation space
        self.observation_space = spaces.Dict({
            agent.id: agent.observation_space
            for agent in self.agents.values()
        })

        # Range of possible rewards
        self.reward_range = (0, 1)

        assert isinstance(
            max_steps, int
        ), f"The argument max_steps must be an integer, got: {type(max_steps)}"
        self.max_steps = max_steps

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov

        self.screen_size = screen_size
        self.render_size = None
        self.window = None
        self.clock = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: dict[str, Any] = {},
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        super().reset(seed=seed)

        # Agents update
        for agent in self.agents.values():
            agent.reset()

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # Set the agents' mission
        for agent in self.agents.values():
            agent.mission = self.mission

        # These fields should be defined by _gen_grid
        for agent in self.agents.values():
            assert (
                agent.pos >= (0, 0)
                if isinstance(agent.pos, tuple)
                else all(agent.pos >= 0) and agent.dir >= 0
            )

        # Check that the agent doesn't overlap with an object
        for agent in self.agents.values():
            start_cell = self.grid.get(*agent.pos)
            assert start_cell is None or start_cell.can_overlap()

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {}

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [
            self.grid.encode().tolist(),
            *[agent.pos for agent in self.agents.values()],
            *[agent.dir for agent in self.agents.values()],
        ]
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        """
        Number of steps remaining in the episode (until truncation).
        """
        return self.max_steps - self.step_count

    def pprint_grid(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """
        for agent in self.agents.values():
            if agent.pos == (-1, -1) or agent.dir == -1 or self.grid is None:
                raise ValueError("The environment hasn't been `reset` therefore the `agent_pos`, `agent_dir` or `grid` are unknown.")

        # Map of object types to short string
        OBJECT_TO_STR = {
            "wall": "W",
            "floor": "F",
            "door": "D",
            "key": "K",
            "ball": "A",
            "box": "B",
            "goal": "G",
            "lava": "V",
        }

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        output = ""
        grid = Grid.from_grid(self.grid)
        for j in range(grid.height):
            for i in range(grid.width):
                tile = grid.get(i, j)
                if tile is None:
                    output += "  "
                    continue

                if tile.type == 'agent':
                    output += 2 * AGENT_DIR_TO_STR[tile.dir]
                    continue

                if tile.type == "door":
                    if tile.is_open:
                        output += "__"
                    elif tile.is_locked:
                        output += "L" + tile.color[0].upper()
                    else:
                        output += "D" + tile.color[0].upper()
                    continue

                output += OBJECT_TO_STR[tile.type] + tile.color[0].upper()

            if j < grid.height - 1:
                output += "\n"

        return output

    @abstractmethod
    def _gen_grid(self, width, height):
        pass

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low: int, high: int) -> int:
        """
        Generate random integer in [low,high]
        """

        return self.np_random.integers(low, high)

    def _rand_float(self, low: float, high: float) -> float:
        """
        Generate random float in [low,high]
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self) -> bool:
        """
        Generate random boolean value
        """

        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable: Iterable[T]) -> T:
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable: Iterable[T], num_elems: int) -> list[T]:
        """
        Sample a random subset of distinct elements of a list.
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out: list[T] = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self) -> str:
        """
        Generate a random color name (string).
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(
        self, x_low: int, x_high: int, y_low: int, y_high: int
    ) -> tuple[int, int]:
        """
        Generate a random (x, y) position tuple.
        """

        return (
            self.np_random.integers(x_low, x_high),
            self.np_random.integers(y_low, y_high),
        )

    def place_obj(
        self,
        obj: WorldObj | None,
        top: tuple[int, int] = None,
        size: tuple[int, int] = None,
        reject_fn=None,
        max_tries=math.inf,
    ):
        """
        Place an object at an empty position in the grid.

        Parameters
        ----------
        obj: WorldObj
            Object to place in the grid
        top: tuple[int, int]
            Top-left position of the rectangular area where to place the object
        size: tuple[int, int]
            Width and height of the rectangular area where to place the object
        reject_fn: Callable(env, pos) -> bool
            Function to filter out potential positions
        max_tries: int
            Maximum number of attempts to place the object
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = (
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
            )

            # print(self.grid.get(*pos), self.agent.pos)

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None:
                continue

            # Don't place the object where the agent is
            if any(np.array_equal(pos, agent.pos) for agent in self.agents.values()):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj: WorldObj, i: int, j: int):
        """
        Put an object at a specific position in the grid
        """
        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        """
        Set agents' starting points at empty positions in the grid.
        """
        for agent in self.agents.values():
            agent.pos = (-1, -1)
            pos = self.place_obj(None, top, size, max_tries=max_tries)
            agent.pos = pos

            if rand_dir:
                agent.dir = self._rand_int(0, 4)

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """
        assert len(self.agents) == 1, "This property is deprecated for multi-agent environments."
        return self.agents[0].in_view(x, y)

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """
        assert len(self.agents) == 1, "This property is deprecated for multi-agent environments."
        return self.agents[0].sees(x, y, self.grid_with_agents())

    def step(
        self, actions: Dict[AgentID, ActType]  # type: ignore
    ) -> Tuple[Dict[AgentID, ObsType], Dict[AgentID, SupportsFloat], Dict[AgentID, bool], Dict[AgentID, bool], Dict[str, Any]]:  # type: ignore
        """
        Run one timestep of the environment's dynamics for all agents.

        Args:
            actions (ActType): A set of actions for each agent.

        Returns:
            obs (Dict[AgentID, ObsType]): Agent observations.
            reward (Dict[AgentID, SupportsFloat]):
                Amount of reward returned after previous action for each agent.
                The reward from environment is the sum of rewards of all agents. Access individual rewards by `info["rewards"]`.
            terminated (Dict[AgentID, bool]):
                Whether the episode has ended for each agent (success or failure).
                    The episode ends if any one of the following conditions is met:
                    1. The agent reaches the goal when `is_competitive_env` is True.
                    2. The agent reaches the goal when `is_competitive_env` is False.
                Access individual terminations by `info["terminated"]`.
            truncated (Dict[AgentID, bool]):
                Whether the episode has ended due to the time limit for each agent (max steps reached).
                Access individual truncations by `info["truncated"]`.
        """
        self.step_count += 1

        reward = {agent.id: 0 for agent in self.agents.values()}
        agent_locations = {agent.id: agent.pos for agent in self.agents.values()}

        # Randomize the order in which the agents act each timestep
        acting_agent_ids = list(actions.keys())
        np.random.shuffle(acting_agent_ids)

        for agent_id in acting_agent_ids:
            action, agent = actions[agent_id], self.agents[agent_id]

            if agent.terminated:
                continue

            # Rotate left
            if action == self.actions.left:
                agent.dir = (agent.dir - 1) % 4

            # Rotate right
            elif action == self.actions.right:
                agent.dir = (agent.dir + 1) % 4

            # Move forward
            elif action == self.actions.forward:
                # Get the contents of the cell in front of the agent
                fwd_pos = agent.front_pos
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_cell is None or fwd_cell.can_overlap():
                    if fwd_pos not in agent_locations.values():
                        agent.pos = fwd_pos
                        agent_locations[agent.id] = fwd_pos
                if fwd_cell is not None and fwd_cell.type == "goal":
                    agent.terminated = True
                    reward[agent.id] = self._reward()
                if fwd_cell is not None and fwd_cell.type == "lava":
                    agent.terminated = True

            # Pick up an object
            elif action == self.actions.pickup:
                # Get the contents of the cell in front of the agent
                fwd_pos = agent.front_pos
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_cell and fwd_cell.can_pickup():
                    if agent.carrying is None:
                        agent.carrying = fwd_cell
                        agent.carrying.cur_pos = np.array([-1, -1])
                        self.grid.set(*fwd_pos, None)

            # Drop an object
            elif action == self.actions.drop:
                # Get the contents of the cell in front of the agent
                fwd_pos = agent.front_pos
                fwd_cell = self.grid.get(*fwd_pos)

                if not fwd_cell and agent.carrying:
                    self.grid.set(*fwd_pos, agent.carrying)
                    agent.carrying.cur_pos = fwd_pos
                    agent.carrying = None

            # Toggle/activate an object
            elif action == self.actions.toggle:
                # Get the contents of the cell in front of the agent
                fwd_pos = agent.front_pos
                fwd_cell = self.grid.get(*fwd_pos)

                if fwd_cell:
                    fwd_cell.toggle(self, agent, fwd_pos)

            # Done action (not used by default)
            elif action == self.actions.done:
                pass

            else:
                raise ValueError(f"Unknown action: {action}")

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        truncated = {agent.id: (self.step_count >= self.max_steps) for agent in self.agents.values()}
        terminated = {agent.id: agent.terminated for agent in self.agents.values()}

        info = {}
        info["rewards"] = reward
        info["terminated"] = terminated
        info["truncated"] = truncated

        # The reward from environment is the sum of rewards of all agents
        reward = sum(reward.values())
        terminated = any(terminated.values()) if self.is_competitive_env else all(terminated.values())
        truncated = any(truncated.values())

        return obs, reward, terminated, truncated, info

    def grid_with_agents(self):
        """
        Return a copy of the grid with the agents on it.
        """
        grid = Grid.from_grid(self.grid)
        for agent in self.agents.values():
            grid.set(*agent.pos, agent)
        return grid

    def gen_obs(self) -> Dict[AgentID, ObsType]:  # type: ignore
        """
        Generate observations for each agent (partially observable, low-resolution encoding).
        """
        return {agent.id: agent.gen_obs(self.grid_with_agents()) for agent in self.agents.values()}

    def get_pov_render(self, tile_size, agent_id: int = 0):
        """
        Render an agent's POV observation for visualization
        """
        img = {}
        # Compute agent visibility masks
        for agent in self.agents.values():

            # Keep the original agent directions to restore them later
            tmp_agent_dirs = [a.dir for a in self.agents.values()]
            # Set all other agents' directions to ones from the current agent POV
            for a in self.agents.values():
                if a.id != agent.id:
                    a.dir = (a.dir - agent.dir - 1) % 4

            grid, vis_mask = agent.gen_obs_grid(self.grid_with_agents())

            # Render the whole grid
            img[agent.id] = grid.render(
                tile_size,
                highlight_mask=vis_mask,
            )

            # Restore the original agent directions
            for a in self.agents.values():
                if a.id != agent.id:
                    a.dir = tmp_agent_dirs[a.id]

        print(img.keys())
        return img[agent_id]

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Compute agent visibility masks
        grid_with_agents = self.grid_with_agents()
        vis_masks = {
            agent.id: agent.gen_obs_grid(grid_with_agents)[1]
            for agent in self.agents.values()
        }

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        for agent in self.agents.values():
            # Compute the world coordinates of the bottom-left corner
            # of the agent's view area
            f_vec = agent.dir_vec
            r_vec = agent.right_vec
            top_left = (
                agent.pos
                + f_vec * (agent.view_size - 1)
                - r_vec * (agent.view_size // 2)
            )

            # For each cell in the visibility mask
            for vis_j in range(0, agent.view_size):
                for vis_i in range(0, agent.view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_masks[agent.id][vis_i, vis_j]:
                        continue

                    # Compute the world coordinates of this cell
                    abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                    if abs_i < 0 or abs_i >= self.width:
                        continue
                    if abs_j < 0 or abs_j >= self.height:
                        continue

                    # Mark this cell to be highlighted
                    highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = grid_with_agents.render(
            tile_size,
            highlight_mask=highlight_mask if highlight else None,
        )
        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """
        Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """

        if agent_pov:
            return self.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)

    def render(self):
        """
        Render the environment.
        """
        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if self.render_mode == "human":
            img = np.transpose(img, axes=(1, 0, 2))
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption("minigrid")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            surf = pygame.surfarray.make_surface(img)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            # offset = 32 if self.agent_pov else 64
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            font_size = 22
            text = self.mission
            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            pygame.quit()
