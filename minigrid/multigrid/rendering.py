from __future__ import annotations

import math

import numpy as np

from minigrid.core.constants import COLORS, TILE_PIXELS
from minigrid.utils.rendering import (
    downsample,
    fill_coords,
    point_in_triangle,
    rotate_fn,
)


def render_agent_tile(
    img: np.ndarray,
    pos: tuple[int, int],
    direction: int,
    color: str,
    tile_size: int = TILE_PIXELS,
    subdivs: int = 3,
):
    """Render a colored agent triangle onto an existing image.

    Args:
        img: The full rendered grid image to overlay onto.
        pos: (x, y) grid position of the agent.
        direction: Agent direction (0=right, 1=down, 2=left, 3=up).
        color: Agent color name.
        tile_size: Pixel size of each tile.
        subdivs: Supersampling factor for anti-aliasing.
    """
    tri_fn = point_in_triangle(
        (0.12, 0.19),
        (0.87, 0.50),
        (0.12, 0.81),
    )
    tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * direction)

    # Render with supersampling
    big = np.zeros((tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)
    fill_coords(big, tri_fn, COLORS[color])
    tile_img = downsample(big, subdivs)

    # Overlay onto image (only non-black pixels)
    x, y = pos
    ymin = y * tile_size
    ymax = (y + 1) * tile_size
    xmin = x * tile_size
    xmax = (x + 1) * tile_size

    mask = tile_img.any(axis=2)
    img[ymin:ymax, xmin:xmax][mask] = tile_img[mask]
