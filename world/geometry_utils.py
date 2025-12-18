"""
geometry_utils.py

Geometry Utility helpers
"""

from __future__ import annotations

import math
import numpy as np
from world.schema import Box, Vec3
from world.grid_world import GridWorld


def lerp(a, b, t: float):
    return tuple((1 - t) * x + t * y for x, y in zip(a, b))


def aabb_point_distance(p: Vec3, box: Box) -> float:
    """
    Distance from point p to axis-aligned box (0 if inside).
    """
    px, py, pz = p
    x0, y0, z0 = box.min_corner
    x1, y1, z1 = box.max_corner

    # clamp point to box
    cx = min(max(px, x0), x1)
    cy = min(max(py, y0), y1)
    cz = min(max(pz, z0), z1)

    dx = px - cx
    dy = py - cy
    dz = pz - cz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def mark_box_on_mask(grid: GridWorld, mask: np.ndarray, box: Box, inflation_cells: int = 0) -> None:
    """
    Mark cells overlapped by a Box as occupied in an external boolean mask.
    Uses the same indexing logic as GridWorld.mark_box_obstacle, but does not
    mutate grid. Optional inflation in grid cells.
    """
    xmin, ymin, zmin = grid.bounds_min
    res = grid.resolution

    bx0, by0, bz0 = box.min_corner
    bx1, by1, bz1 = box.max_corner

    ix0 = int(math.floor((bx0 - xmin) / res)) - inflation_cells
    iy0 = int(math.floor((by0 - ymin) / res)) - inflation_cells
    iz0 = int(math.floor((bz0 - zmin) / res)) - inflation_cells

    ix1 = int(math.floor((bx1 - xmin) / res)) + inflation_cells
    iy1 = int(math.floor((by1 - ymin) / res)) + inflation_cells
    iz1 = int(math.floor((bz1 - zmin) / res)) + inflation_cells

    ix0 = max(ix0, 0)
    iy0 = max(iy0, 0)
    iz0 = max(iz0, 0)

    ix1 = min(ix1, grid.nx - 1)
    iy1 = min(iy1, grid.ny - 1)
    iz1 = min(iz1, grid.nz - 1)

    if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
        return

    mask[ix0:ix1 + 1, iy0:iy1 + 1, iz0:iz1 + 1] = True


def make_grid_with_mask(static_grid: GridWorld, dyn_mask: np.ndarray) -> GridWorld:
    """
    Create a planning grid with occupancy = static_occ OR dyn_mask.
    Does NOT mutate static_grid.
    """
    g = GridWorld(static_grid.bounds_min, static_grid.bounds_max, resolution=static_grid.resolution)
    g.occ = np.array(static_grid.occ | dyn_mask, copy=True)
    return g