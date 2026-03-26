from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from minigrid.core.constants import DIR_TO_VEC

if TYPE_CHECKING:
    from minigrid.core.world_object import WorldObj

# Default color palette for agents (up to 6 agents)
AGENT_COLORS = ["red", "blue", "green", "purple", "yellow", "grey"]


@dataclass
class AgentState:
    """Per-agent state for multi-agent grid environments."""

    pos: tuple[int, int] = (-1, -1)
    dir: int = 0
    color: str = "red"
    carrying: WorldObj | None = field(default=None, repr=False)
    terminated: bool = False
    truncated: bool = False

    @property
    def dir_vec(self) -> tuple[int, int]:
        """Get the direction vector for this agent."""
        assert 0 <= self.dir < 4, f"Invalid dir: {self.dir}"
        vec = DIR_TO_VEC[self.dir]
        return (int(vec[0]), int(vec[1]))

    @property
    def front_pos(self) -> tuple[int, int]:
        """Get the position of the cell directly in front of this agent."""
        dx, dy = self.dir_vec
        return (self.pos[0] + dx, self.pos[1] + dy)

    @property
    def is_active(self) -> bool:
        """Whether this agent is still active (not terminated or truncated)."""
        return not self.terminated and not self.truncated
