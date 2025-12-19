"""
planners/dstar_lite.py

Online D* Lite planner on a 3D grid.
- Static obstacles known globally.
- Dynamic obstacles are discovered locally via a sensor radius and represented as a temporary mask.
- D* Lite updates its solution incrementally as the sensed occupancy changes over time.

This file is intentionally separate so your baseline/offline and online A* submissions stay unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
import heapq
import json
import os
import time

import numpy as np

from world.env_io import load_env_config
from world.schema import Obstacle, Box, Scenario, Frame, Vec3
from world.grid_world import build_grid_from_env, GridWorld
from world.geometry_utils import (
    aabb_point_distance,
    lerp,
    mark_box_on_mask,
)
from viz.plotly_viz import visualize_scenario


# -------- CONFIGURABLE PARAMETERS --------
DT = 0.1                  # Frame timestep (seconds)
N_FRAMES = 230            # Number of frames to generate
INFLATION_CELLS = 1       # Inflation (grid cells) for sensed dynamic obstacles
MOVE_EVERY = 3            # move once every N frames (increase to slow down)
RESOLUTION = 0.25         # Grid resolution (m)
NEIGHBORS = 26            # 6 or 26 (D* Lite supports either)
# ----------------------------------------


Idx = Tuple[int, int, int]


@dataclass
class PlannerMetrics:
    planner: str
    success: bool
    failure_reason: Optional[str]
    planning_time_s: float
    path_length_m: float
    num_replans: int


def path_length(world_path: List[Vec3]) -> float:
    if not world_path or len(world_path) < 2:
        return 0.0
    dist = 0.0
    for a, b in zip(world_path[:-1], world_path[1:]):
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        dist += (dx * dx + dy * dy + dz * dz) ** 0.5
    return dist


# -------------------------
# Grid adapter for D* Lite
# -------------------------
class DStarGrid:
    """
    Wraps your GridWorld + a dynamic mask (known local obstacles) to provide:
    - in_bounds
    - is_free
    - neighbors (6 or 26)
    - step_cost (meters)
    """

    def __init__(self, static_grid: GridWorld, dyn_mask: np.ndarray, neighbors: int = 26):
        self.g = static_grid
        self.dyn_mask = dyn_mask  # bool occupancy mask (True means blocked)
        self.neighbors_mode = neighbors

    @property
    def shape(self):
        return self.g.occ.shape

    def in_bounds(self, s: Idx) -> bool:
        x, y, z = s
        nx, ny, nz = self.shape
        return (0 <= x < nx) and (0 <= y < ny) and (0 <= z < nz)

    def is_free(self, s: Idx) -> bool:
        x, y, z = s
        # static occupancy is in g.occ; dynamic is dyn_mask
        return (not self.g.occ[x, y, z]) and (not self.dyn_mask[x, y, z])

    def neighbors(self, s: Idx) -> List[Idx]:
        x, y, z = s
        out = []

        if self.neighbors_mode == 6:
            deltas = [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ]
        else:
            deltas = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        deltas.append((dx, dy, dz))

        for dx, dy, dz in deltas:
            nb = (x + dx, y + dy, z + dz)
            if self.in_bounds(nb):
                out.append(nb)
        return out

    def step_cost(self, a: Idx, b: Idx) -> float:
        # If destination is blocked, treat as impassable
        if not self.is_free(b):
            return float("inf")

        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dz = b[2] - a[2]
        # distance in meters
        return self.g.resolution * ((dx * dx + dy * dy + dz * dz) ** 0.5)


# -------------------------
# D* Lite core
# -------------------------
def heuristic(a: Idx, b: Idx, resolution: float) -> float:
    dx, dy, dz = a[0] - b[0], a[1] - b[1], a[2] - b[2]
    return resolution * ((dx * dx + dy * dy + dz * dz) ** 0.5)


def key_less(k1, k2) -> bool:
    # Lexicographic compare on (k1_0, k1_1)
    return k1[0] < k2[0] or (k1[0] == k2[0] and k1[1] < k2[1])


class DStarLite:
    def __init__(self, grid: DStarGrid, s_start: Idx, s_goal: Idx):
        self.grid = grid
        self.s_start = s_start
        self.s_goal = s_goal

        self.g: Dict[Idx, float] = {}
        self.rhs: Dict[Idx, float] = {}
        self.U: List[Tuple[Tuple[float, float], Idx]] = []  # heap of (key, state)
        self.km = 0.0

        self._set_rhs(self.s_goal, 0.0)
        heapq.heappush(self.U, (self.calculate_key(self.s_goal), self.s_goal))

    def _get_g(self, s: Idx) -> float:
        return self.g.get(s, float("inf"))

    def _set_g(self, s: Idx, v: float) -> None:
        self.g[s] = v

    def _get_rhs(self, s: Idx) -> float:
        return self.rhs.get(s, float("inf"))

    def _set_rhs(self, s: Idx, v: float) -> None:
        self.rhs[s] = v

    def calculate_key(self, s: Idx) -> Tuple[float, float]:
        g_rhs = min(self._get_g(s), self._get_rhs(s))
        return (
            g_rhs + heuristic(self.s_start, s, self.grid.g.resolution) + self.km,
            g_rhs,
        )

    def update_vertex(self, u: Idx) -> None:
        if u != self.s_goal:
            # rhs(u) = min_{s in succ(u)} (c(u,s) + g(s))
            best = float("inf")
            for s in self.grid.neighbors(u):
                c = self.grid.step_cost(u, s)
                if c == float("inf"):
                    continue
                best = min(best, c + self._get_g(s))
            self._set_rhs(u, best)

        # remove u from queue lazily by pushing new key if inconsistent
        if self._get_g(u) != self._get_rhs(u):
            heapq.heappush(self.U, (self.calculate_key(u), u))

    def compute_shortest_path(self, max_iters: int = 2_000_000) -> None:
        iters = 0
        while self.U:
            iters += 1
            if iters > max_iters:
                raise RuntimeError("D* Lite: exceeded max iterations.")

            k_old, u = heapq.heappop(self.U)
            k_new = self.calculate_key(u)

            # Skip stale heap entries
            if k_old != k_new:
                continue

            # Standard termination condition:
            # while topKey < calculate_key(start) OR rhs(start) != g(start)
            top_key = self.U[0][0] if self.U else (float("inf"), float("inf"))
            if (not key_less(top_key, self.calculate_key(self.s_start))) and (
                self._get_rhs(self.s_start) == self._get_g(self.s_start)
            ):
                break

            g_u = self._get_g(u)
            rhs_u = self._get_rhs(u)

            if g_u > rhs_u:
                self._set_g(u, rhs_u)
                for p in self.grid.neighbors(u):
                    self.update_vertex(p)
            else:
                self._set_g(u, float("inf"))
                self.update_vertex(u)
                for p in self.grid.neighbors(u):
                    self.update_vertex(p)


    def get_next_step(self) -> Optional[Idx]:
        """
        Return the next state to move to from s_start based on current g-values.
        """
        if not self.grid.is_free(self.s_start):
            return None

        best_s = None
        best_val = float("inf")
        for s in self.grid.neighbors(self.s_start):
            c = self.grid.step_cost(self.s_start, s)
            if c == float("inf"):
                continue
            val = c + self._get_g(s)
            if val < best_val:
                best_val = val
                best_s = s
        return best_s

    def notify_start_moved(self, s_new: Idx) -> None:
        # Update km for consistent keys after moving start
        self.km += heuristic(self.s_start, s_new, self.grid.g.resolution)
        self.s_start = s_new

    def notify_edge_costs_changed(self, changed_cells: Set[Idx]) -> None:
        """
        When occupancy changes, affected vertices should be updated.
        A safe (slightly conservative) update is to update each changed cell and its neighbors.
        """
        for u in changed_cells:
            self.update_vertex(u)
            for p in self.grid.neighbors(u):
                self.update_vertex(p)


# -------------------------
# Public runner
# -------------------------
def run_online_dstar(env_path: str, outfile: Optional[str], sensor_radius: float = 0.5) -> None:
    print("Evironment used for planning:", env_path)
    print("Running online D* Lite with known static obstacles and local sensing based replanning for dynamic obstacles...")
    print(f"Sensor radius is set to: {sensor_radius} m")

    env = load_env_config(env_path)
    env_name = (env_path.split("/")[-1]).split(".json")[0]

    dt = DT
    n_frames = N_FRAMES
    inflation_cells = INFLATION_CELLS
    move_every = MOVE_EVERY

    if outfile is None:
        outfile = f"viz_online_dstar_{env_name}.html"

    total_planning_time_s = 0.0
    n_replans = 0
    failure_reason = None
    success = False

    bounds_min = env.bounds_min
    bounds_max = env.bounds_max
    start = env.start
    goal = env.goal

    static_boxes = [
        Obstacle(kind="box", params=box, dynamic=False, name=f"static_{i}")
        for i, box in enumerate(env.static_boxes)
    ]

    dyn_specs = [
        {"name": d.name, "min0": d.min0, "max0": d.max0, "min1": d.min1, "max1": d.max1}
        for d in env.dyn_boxes
    ]

    # IMPORTANT: static-only grid (dynamic unknown globally)
    static_grid = build_grid_from_env(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        static_obstacles=static_boxes,
        dyn_specs=[],
        resolution=RESOLUTION,
    )

    probe_idx = static_grid.world_to_grid(start)
    goal_idx = static_grid.world_to_grid(goal)

    executed_world: List[Vec3] = []
    frames: List[Frame] = []

    # D* state: we maintain a dynamic mask that changes over time
    dyn_mask = np.zeros_like(static_grid.occ, dtype=bool)
    dgrid = DStarGrid(static_grid, dyn_mask, neighbors=NEIGHBORS)
    dstar = DStarLite(dgrid, probe_idx, goal_idx)

    # Keep a copy to compute "changed cells"
    prev_dyn_mask = dyn_mask.copy()

    # --- INITIAL PLAN (must do once) ---
    n_replans += 1
    t0 = time.perf_counter()
    dstar.compute_shortest_path()
    total_planning_time_s += (time.perf_counter() - t0)


    try:
        for k in range(n_frames):
            t = 0.0 if n_frames <= 1 else k / (n_frames - 1)

            # True dynamic boxes at this time (for visualization + sensing)
            dyn_boxes: List[Obstacle] = []
            true_dyn_boxes_geom: List[Box] = []
            for spec in dyn_specs:
                minc = lerp(spec["min0"], spec["min1"], t)
                maxc = lerp(spec["max0"], spec["max1"], t)
                b = Box(min_corner=minc, max_corner=maxc)
                true_dyn_boxes_geom.append(b)
                dyn_boxes.append(Obstacle(kind="box", params=b, dynamic=True, name=spec["name"]))

            probe_world = static_grid.grid_to_world(probe_idx)

            # Rebuild current dyn_mask from what the probe can sense
            dyn_mask[:, :, :] = False
            for b in true_dyn_boxes_geom:
                if aabb_point_distance(probe_world, b) <= sensor_radius:
                    mark_box_on_mask(static_grid, dyn_mask, b, inflation_cells=inflation_cells)

            # Never block probe/goal cells (prevents contact deadlocks)
            dyn_mask[probe_idx] = False
            dyn_mask[goal_idx] = False

            # Detect changes in sensed occupancy and inform D*
            changed = set(map(tuple, np.argwhere(dyn_mask != prev_dyn_mask)))
            if changed:
                n_replans += 1
                t0 = time.perf_counter()
                dstar.notify_edge_costs_changed(changed)
                dstar.compute_shortest_path()
                total_planning_time_s += (time.perf_counter() - t0)

            prev_dyn_mask[:, :, :] = dyn_mask

            # Compute a short "predicted" path for visualization (greedy rollout)
            predicted_idx: List[Idx] = [probe_idx]
            cur = probe_idx
            # roll out up to 200 steps max (or until goal)
            for _ in range(200):
                dstar.s_start = cur
                nxt = dstar.get_next_step()
                if nxt is None:
                    break
                predicted_idx.append(nxt)
                cur = nxt
                if cur == goal_idx:
                    break

            planned_world = [static_grid.grid_to_world(idx) for idx in predicted_idx]

            # Move the probe
            if k % move_every == 0 and probe_idx != goal_idx:
                dstar.s_start = probe_idx
                nxt = dstar.get_next_step()
                if nxt is None:
                    continue
                    # raise RuntimeError("D* Lite: no next step (no path).")
                dstar.notify_start_moved(nxt)
                probe_idx = nxt

            executed_world.append(static_grid.grid_to_world(probe_idx))

            frames.append(
                Frame(
                    t=round(k * dt, 2),
                    obstacles=static_boxes + dyn_boxes,
                    planned_path=planned_world,  # magenta
                    # executed_path=executed_world,  # blue (if your schema supports it)
                    probe=static_grid.grid_to_world(probe_idx),
                )
            )

            if probe_idx == goal_idx:
                success = True
                break

        if not success and probe_idx == goal_idx:
            success = True

    except Exception as e:
        failure_reason = f"Replanner crashed: {repr(e)}"
        success = False

    # Scenario + visualization
    scn = Scenario(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        start=start,
        goal=goal,
        frames=frames,
        title="Online D* Lite: static known + local sensing",
    )

    out = visualize_scenario(scn, outfile=outfile, sensor_radius=sensor_radius)

    metrics = PlannerMetrics(
        planner="online_dstar_lite",
        success=success,
        failure_reason=failure_reason,
        planning_time_s=total_planning_time_s,
        path_length_m=path_length(executed_world),
        num_replans=n_replans,
    )

    print("\n======= PLANNER METRICS =======")
    print(f"Planner           : Online D* Lite (incremental replanning)")
    print(f"Success           : {metrics.success}")
    print(f"Failure reason    : {metrics.failure_reason}")
    print(f"Planning time     : {metrics.planning_time_s:.6f} s")
    print(f"Path length       : {metrics.path_length_m:.4f} m")
    print(f"Number of replans : {metrics.num_replans}")
    print("================================\n")

    # Save metrics next to others
    metrics_file = f"metrics_online_dstar_{env_name}.json"
    with open(metrics_file, "w") as f:
        json.dump(asdict(metrics), f, indent=2)
    print(f'Saved metric scores to "{metrics_file}"')

    print("\n======== VISUALIZATION =========")
    print(f'Saved "{out}" in the root directory.')
    print(f'To visualize simulation, open "{out}" in your browser.')
    print("================================\n")
