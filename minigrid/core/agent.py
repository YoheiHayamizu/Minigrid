from __future__ import annotations
from typing import Optional, TYPE_CHECKING, Tuple, Optional
import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from minigrid.core.mission import MissionSpace
from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.constants import COLORS, IDX_TO_COLOR, DIR_TO_VEC
from minigrid.core.world_object import WorldObj
from minigrid.utils.rendering import (
    fill_coords,
    point_in_triangle,
    rotate_fn
)


class Agent(WorldObj):
    """
    Class representing an agent in the environment.
    """

    def __init__(
        self,
        index: int,
        mission_space: MissionSpace,
        view_size: int = 7,
        see_through_walls: bool = False,
    ):
        """
        Parameters
        ----------
        index : int
            The index of the agent in the environment
        mission_space : MissionSpace
            The mission space for the agent
        view_size : int
            The size of the agent's view (must be odd)
        see_through_walls : bool
            Whether the agent can see through walls
        """
        color = IDX_TO_COLOR[index % (max(IDX_TO_COLOR) + 1)]
        super().__init__('agent', color)
        self.index = index

        # Number of cells (width and height) in the agent view
        assert view_size % 2 == 1, "view_size must be odd"
        assert view_size >= 3, "view_size must be at least 3"
        self.view_size = view_size

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(Actions))

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(view_size, view_size, 3),
                dtype='uint8',
            ),
            'direction': spaces.Discrete(len(DIR_TO_VEC)),
            'mission': mission_space,
        })
        self.see_through_walls = see_through_walls

        # Current position and direction of the agent
        self.pos: Tuple[int, int] = None
        self.dir: int = None

        # Current mission and carrying
        self.mission: str = None
        self.carrying: Optional[WorldObj] = None
        self.terminated = False

    def reset(self):
        """
        Reset the agent before environment episode.
        """
        self.pos = (-1, -1)
        self.dir = -1

    @property
    def mission(self) -> str:
        """
        Get the mission string for the agent.
        """
        return self._mission

    @mission.setter
    def mission(self, mission: str):
        """
        Set the mission string for the agent.
        """
        self._mission = mission

    @property
    def carrying(self) -> WorldObj:
        """
        Get the object that the agent is currently carrying.
        Alias for `contains`.
        """
        return self.contains

    @carrying.setter
    def carrying(self, obj: WorldObj):
        """
        Set the object that the agent is currently carrying.
        """
        self.contains = obj

    @property
    def dir_vec(self) -> int:
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        assert (0 <= self.dir < 4), f"Invalid direction: {self.dir} is not within range(0, 4)"
        return DIR_TO_VEC[self.dir]

    @property
    def right_vec(self) -> np.ndarray[int]:
        """
        Get the vector pointing to the right of the agent.
        """
        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self) -> Tuple[int, int]:
        """
        Get the position of the cell that is right in front of the agent.
        """
        return self.pos + self.dir_vec

    def get_view_coords(self, i, j) -> Tuple[int, int]:
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """
        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self) -> Tuple[int, int, int, int]:
        """
        Get the extents of the square set of tiles visible to an agent.
        Note: the bottom extent indices are not included in the set.
        """
        # Facing right
        if self.dir == 0:
            topX = self.pos[0]
            topY = self.pos[1] - self.view_size // 2
        # Facing down
        elif self.dir == 1:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1]
        # Facing left
        elif self.dir == 2:
            topX = self.pos[0] - self.view_size + 1
            topY = self.pos[1] - self.view_size // 2
        # Facing up
        elif self.dir == 3:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.view_size
        botY = topY + self.view_size

        return topX, topY, botX, botY

    def relative_coords(self, x, y) -> Tuple[int, int]:
        """
        Check if a grid position belongs to the agent's field of view,
        and return the corresponding coordinates.
        """
        vx, vy = self.get_view_coords(x, y)

        if not (0 <= vx < self.view_size) or not (0 <= vy < self.view_size):
            return None

        return vx, vy

    def in_view(self, x: int, y: int) -> bool:
        """
        Check if a grid position is visible to the agent.
        """
        return self.relative_coords(x, y) is not None

    def sees(self, x: int, y: int, grid: Grid) -> bool:
        """
        Check if a non-empty grid postion is visible to the agent.
        """
        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs(grid)

        obs_grid, _ = Grid.decode(obs["image"])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = grid.get(x, y)

        assert world_cell is not None
        return obs_cell is not None and obs_cell.type == world_cell.type

    def gen_obs_grid(self, grid: Grid) -> Tuple[Grid, np.ndarray[bool]]:
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.

        Returns
        -------
        grid : Grid
            The sub-grid observed by the agent
        vis_mask : array
            Mask telling us which grid cells are visible to the agent
        """
        topX, topY, botX, botY = self.get_view_exts()

        grid = grid.slice(topX, topY, self.view_size, self.view_size)

        for i in range(self.dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(self.view_size // 2, self.view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self, grid: Grid):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        # grid, vis_mask = self.gen_obs_grid()
        grid, vis_mask = self.gen_obs_grid(grid)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {"image": image, "direction": self.dir, "mission": self.mission}

        return obs

    def encode(self):
        return (
            self.type,  # type
            self.color,  # color
            self.dir,  # state
        )

    def render(self, img: np.ndarray[int]):
        """
        Draw the agent.
        """
        c = COLORS[self.color]

        if self.dir == -1:
            return

        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * np.pi * self.dir)
        fill_coords(img, tri_fn, c)
