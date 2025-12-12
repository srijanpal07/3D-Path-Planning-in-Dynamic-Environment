import math
from typing import Tuple, List, Sequence, Mapping, Any

import numpy as np

from world.schema import Vec3, Box, Obstacle

Idx3 = Tuple[int, int, int]


class GridWorld:
    """
    3D occupancy grid built from continuous boxes.

    - Coordinates: world (x,y,z) in meters
    - Grid: indices (ix,iy,iz) with resolution [m/cell]
    - occ[ix,iy,iz] == True  => cell is occupied
    """

    def __init__(self, bounds_min: Vec3, bounds_max: Vec3, resolution: float = 0.25):
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max
        self.resolution = float(resolution)

        dx = bounds_max[0] - bounds_min[0]
        dy = bounds_max[1] - bounds_min[1]
        dz = bounds_max[2] - bounds_min[2]

        # Number of cells in each dimension
        self.nx = int(math.floor(dx / self.resolution)) + 1
        self.ny = int(math.floor(dy / self.resolution)) + 1
        self.nz = int(math.floor(dz / self.resolution)) + 1

        self.occ = np.zeros((self.nx, self.ny, self.nz), dtype=bool)

    # -------------------------
    # Coordinate conversions
    # -------------------------
    def world_to_grid(self, p: Vec3) -> Idx3:
        """
        Convert world coordinates (x,y,z) to grid indices (ix,iy,iz).

        Raises ValueError if p lies outside the grid bounds.
        """
        x, y, z = p
        xmin, ymin, zmin = self.bounds_min

        ix = int(round((x - xmin) / self.resolution))
        iy = int(round((y - ymin) / self.resolution))
        iz = int(round((z - zmin) / self.resolution))

        idx = (ix, iy, iz)
        if not self.in_bounds_idx(idx):
            raise ValueError(f"Point {p} maps to out-of-bounds grid index {idx}")
        return idx

    def grid_to_world(self, idx: Idx3) -> Vec3:
        """
        Convert grid indices (ix,iy,iz) to the center of the corresponding cell.
        """
        ix, iy, iz = idx
        xmin, ymin, zmin = self.bounds_min
        x = xmin + ix * self.resolution
        y = ymin + iy * self.resolution
        z = zmin + iz * self.resolution
        return (float(x), float(y), float(z))

    # -------------------------
    # Bounds / occupancy
    # -------------------------
    def in_bounds_idx(self, idx: Idx3) -> bool:
        ix, iy, iz = idx
        return (
            0 <= ix < self.nx
            and 0 <= iy < self.ny
            and 0 <= iz < self.nz
        )

    def is_free(self, idx: Idx3) -> bool:
        """
        True if the cell is inside bounds and not occupied.
        """
        if not self.in_bounds_idx(idx):
            return False
        ix, iy, iz = idx
        return not bool(self.occ[ix, iy, iz])

    def mark_box_obstacle(self, box: Box) -> None:
        """
        Mark all grid cells overlapped by this Box as occupied.

        We do this by marking any cell whose center falls within [min_corner, max_corner].
        """
        xmin, ymin, zmin = self.bounds_min

        bx0, by0, bz0 = box.min_corner
        bx1, by1, bz1 = box.max_corner

        # Convert box extents to index ranges
        # floor for min, floor for max (cell centers <= max will be included)
        ix0 = int(math.floor((bx0 - xmin) / self.resolution))
        iy0 = int(math.floor((by0 - ymin) / self.resolution))
        iz0 = int(math.floor((bz0 - zmin) / self.resolution))

        ix1 = int(math.floor((bx1 - xmin) / self.resolution))
        iy1 = int(math.floor((by1 - ymin) / self.resolution))
        iz1 = int(math.floor((bz1 - zmin) / self.resolution))

        # Clamp to grid bounds
        ix0 = max(ix0, 0)
        iy0 = max(iy0, 0)
        iz0 = max(iz0, 0)

        ix1 = min(ix1, self.nx - 1)
        iy1 = min(iy1, self.ny - 1)
        iz1 = min(iz1, self.nz - 1)

        if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
            # Box is completely outside the grid; nothing to do
            return

        self.occ[ix0 : ix1 + 1, iy0 : iy1 + 1, iz0 : iz1 + 1] = True

    # -------------------------
    # Neighbors (6-connected)
    # -------------------------
    def neighbors6(self, idx: Idx3) -> List[Idx3]:
        """
        Return free 6-connected neighbors of idx.
        Moves: +/-x, +/-y, +/-z (no diagonals).
        """
        ix, iy, iz = idx
        nbrs: List[Idx3] = []

        candidates = [
            (ix + 1, iy, iz),
            (ix - 1, iy, iz),
            (ix, iy + 1, iz),
            (ix, iy - 1, iz),
            (ix, iy, iz + 1),
            (ix, iy, iz - 1),
        ]
        for j in candidates:
            if self.is_free(j):
                nbrs.append(j)
        return nbrs


# --------------------------------------------------------
# Helper: build a GridWorld from static + dynamic boxes
# --------------------------------------------------------
def build_grid_from_env(
    bounds_min: Vec3,
    bounds_max: Vec3,
    static_obstacles: Sequence[Obstacle],
    dyn_specs: Sequence[Mapping[str, Any]],
    resolution: float = 0.25,
) -> GridWorld:
    """
    Build a GridWorld for the baseline (fully known offline) scenario.

    - static_obstacles: list of Obstacle (kind='box') with Box params
    - dyn_specs: list of dicts, each with keys:
        'min0', 'max0', 'min1', 'max1'
      matching the structure used in demo_viz.py.

    For the baseline, dynamic obstacles are approximated by a single
    bounding box that covers the whole motion from (min0,max0) to (min1,max1),
    and that entire swept volume is treated as occupied.
    """
    grid = GridWorld(bounds_min, bounds_max, resolution=resolution)

    # Mark static obstacles
    for ob in static_obstacles:
        if isinstance(ob.params, Box):
            grid.mark_box_obstacle(ob.params)

    # Mark dynamic swept volumes (conservative bounding box per dyn_spec)
    for spec in dyn_specs:
        min0 = spec["min0"]
        max0 = spec["max0"]
        min1 = spec["min1"]
        max1 = spec["max1"]

        # Build a bounding box that contains all positions
        dyn_min = tuple(min(a, b) for a, b in zip(min0, min1))  # type: ignore[arg-type]
        dyn_max = tuple(max(a, b) for a, b in zip(max0, max1))  # type: ignore[arg-type]

        dyn_box = Box(min_corner=dyn_min, max_corner=dyn_max)
        grid.mark_box_obstacle(dyn_box)

    return grid
