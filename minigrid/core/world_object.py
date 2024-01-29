from __future__ import annotations

from typing import TYPE_CHECKING, Tuple, Optional

import numpy as np

from minigrid.core.constants import (
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
    ASTATE_TO_IDX,
    WOSTATE_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    IDX_TO_ASTATE,
    IDX_TO_WOSTATE
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
)

if TYPE_CHECKING:
    from minigrid.core.agent import Agent
    from minigrid.minigrid_env import MiniGridEnv


class WorldObj:
    """
    Base class for grid world objects

    Attributes:
    ----------
    type : str
        String representing the type of this object
    color : str
        String representing the color of this object
    state : int
        State of the object, can be any of the WOSTATE_TO_IDX values
    indicator : int
        Object indicator variable (i.e. this agent is me)
    contains : Optional[WorldObj]
        Object contained inside this object, if any
    """

    def __init__(self, type: str, color: str):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self._type = type
        self._color = color
        self._state = 0
        self._indicator = 0

        self.contains = None

        # Initial position of the object
        self.init_pos: Tuple[int, int] | None = None

        # Current position of the object
        self.cur_pos: Tuple[int, int] | None = None

    @property
    def type(self) -> str:
        return self._type

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, color: str):
        self._color = color

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, state: int):
        self._state = state

    def can_overlap(self) -> bool:
        """
        Can the agent overlap with this?
        """
        return False

    def can_pickup(self) -> bool:
        """
        Can the agent pick this up?
        """
        return False

    def can_contain(self) -> bool:
        """
        Can this contain another object?
        """
        return False

    def see_behind(self) -> bool:
        """
        Can the agent see behind this object?
        """
        return True

    def toggle(self, env: MiniGridEnv, agent: 'Agent', pos: tuple[int, int]) -> bool:
        """
        Method to trigger/toggle an action this object performs
        """
        return False

    def encode(self) -> tuple[int, int, int]:
        """
        Encode a description of this object as a 3-tuple of integers
        """
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.state)

    @staticmethod
    def decode(type_idx: int, color_idx: int, state: int) -> Optional[WorldObj]:
        """
        Create an object from a 3-tuple state description
        """

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == "empty" or obj_type == "unseen" or obj_type == "agent":
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == "wall":
            v = Wall(color)
        elif obj_type == "floor":
            v = Floor(color)
        elif obj_type == "ball":
            v = Ball(color)
        elif obj_type == "key":
            v = Key(color)
        elif obj_type == "box":
            v = Box(color)
        elif obj_type == "door":
            v = Door(color, is_open, is_locked)
        elif obj_type == "goal":
            v = Goal()
        elif obj_type == "lava":
            v = Lava()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r: np.ndarray) -> np.ndarray:
        """
        Draw this object with the given renderer
        """
        raise NotImplementedError


class Goal(WorldObj):
    def __init__(self):
        super().__init__("goal", "green")

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color: str = "blue"):
        super().__init__("floor", color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    """
    Lava tile the agent can fall into
    """

    def __init__(self):
        super().__init__("lava", "red")

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(WorldObj):
    """
    Wall that the agent can't walk through
    """

    def __init__(self, color: str = "grey"):
        super().__init__("wall", color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Door(WorldObj):
    """
    A door that can be opened with the corresponding colored key.
    """

    def __init__(self, color: str, is_open: bool = False, is_locked: bool = False):
        super().__init__("door", color)
        self.is_open = is_open
        self.is_locked = is_locked

    @property
    def is_open(self) -> bool:
        """
        Whether the door is open.
        """
        return self.state == WOSTATE_TO_IDX['open']

    @is_open.setter
    def is_open(self, value: bool):
        """
        Set the door to be open or closed.
        """
        if value:
            self.state = WOSTATE_TO_IDX['open']  # set state to open
        elif not self.is_locked:
            self.state = WOSTATE_TO_IDX['closed']  # closed (unless already locked)

    @property
    def is_locked(self) -> bool:
        """
        Whether the door is locked.
        """
        return self.state == WOSTATE_TO_IDX['locked']

    @is_locked.setter
    def is_locked(self, value: bool):
        """
        Set the door to be locked or unlocked.
        """
        if value:
            self.state = WOSTATE_TO_IDX['locked']  # set state to locked
        elif not self.is_open:
            self.state = WOSTATE_TO_IDX['closed']  # closed (unless already open)

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env: 'MiniGridEnv', agent: 'Agent', pos: tuple[int, int]) -> bool:
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(agent.carrying, Key) and agent.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False
        self.is_open = not self.is_open
        return True

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):
    """
    Colored key that can unlock doors and be picked up by the agent
    """

    def __init__(self, color: str = "blue"):
        super().__init__("key", color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    """
    Ball object that the agent can pick up
    """

    def __init__(self, color="blue"):
        super().__init__("ball", color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(WorldObj):
    """
    Box object that the agent can pick up or toggle
    """

    def __init__(self, color, contains: Optional[WorldObj] = None):
        super().__init__("box", color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env: 'MiniGridEnv', agent: 'Agent', pos: Tuple[int, int]) -> bool:
        # Replace the box by its contents
        env.grid.set(pos[0], pos[1], self.contains)
        return True
