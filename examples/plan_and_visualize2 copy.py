"""
plan_and_visualize.py

Modes:
  1) baseline: Fully known (static + conservative dynamic union) offline A*
  2) online:   Static-known + dynamic sensed locally, online replanning using A*

Notes:
- Baseline code path is preserved (uses build_grid_from_env with full dyn_specs).
- Online replanning does NOT modify baseline A* or baseline grid building.
- Online mode uses the same astar_3d() but replans repeatedly with a dynamic mask.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from world.env_io import load_env_config
from world.schema import Obstacle, Box, Scenario, Frame, Vec3
from world.grid_world import build_grid_from_env, GridWorld
from planners.astar3d import astar_3d
from viz.plotly_viz import visualize_scenario


# ----------------------------
# Utility helpers
# ----------------------------
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


def path_blocked(planning_grid: GridWorld, idx_path: List[Tuple[int, int, int]], start_i: int, lookahead: int) -> bool:
    """
    Check if any of the next `lookahead` cells on a path are occupied.
    """
    end = min(start_i + lookahead, len(idx_path))
    for j in range(start_i, end):
        if not planning_grid.is_free(idx_path[j]):
            return True
    return False


def normalize_path(idx_path, start_idx, goal_idx):
    if not idx_path:
        return idx_path
    # If it's reversed (goal -> start), flip it
    if idx_path[0] == goal_idx and idx_path[-1] == start_idx:
        return list(reversed(idx_path))
    # If neither end matches perfectly, pick the orientation that starts closer to start_idx
    if idx_path[0] != start_idx and idx_path[-1] == start_idx:
        return list(reversed(idx_path))
    return idx_path


# ----------------------------
# Baseline (unchanged behavior)
# ----------------------------
def run_baseline(env_path: str, outfile: str, dt: float, n_frames: int) -> None:
    env = load_env_config(env_path)

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

    # Baseline conservative: static + swept dynamic union
    grid = build_grid_from_env(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        static_obstacles=static_boxes,
        dyn_specs=dyn_specs,
        resolution=0.25,
    )

    start_idx = grid.world_to_grid(start)
    goal_idx = grid.world_to_grid(goal)

    idx_path = astar_3d(grid, start_idx, goal_idx)
    world_path = [grid.grid_to_world(idx) for idx in idx_path]

    frames: List[Frame] = []
    for k in range(n_frames):
        t = 0.0 if n_frames <= 1 else k / (n_frames - 1)

        dyn_boxes: List[Obstacle] = []
        for spec in dyn_specs:
            minc = lerp(spec["min0"], spec["min1"], t)
            maxc = lerp(spec["max0"], spec["max1"], t)
            dyn_boxes.append(
                Obstacle(
                    kind="box",
                    params=Box(min_corner=minc, max_corner=maxc),
                    dynamic=True,
                    name=spec["name"],
                )
            )

        # Follow the precomputed path for visualization
        path_pos = min(int(t * (len(world_path) - 1)), len(world_path) - 1)
        probe = world_path[path_pos]
        planned = world_path

        frames.append(Frame(
            t=round(k * dt, 2),
            obstacles=static_boxes + dyn_boxes,
            planned_path=planned,
            probe=probe,
        ))

    scn = Scenario(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        start=start,
        goal=goal,
        frames=frames,
        title="Baseline: Offline A* (conservative dynamic swept-volume)",
    )

    out = visualize_scenario(scn, outfile=outfile)
    print(f"Wrote {out} (open it in your browser).")


# ----------------------------
# Online replanning (Option B)
# ----------------------------
def run_online_astar(env_path: str,outfile: str, dt: float, n_frames: int, sensor_radius: float, inflation_cells: int, lookahead_cells: int) -> None:
    print("RUNNING ONLINE REPLANNING MODE")
    
    env = load_env_config(env_path)

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

    # Static-only grid
    static_grid = build_grid_from_env(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        static_obstacles=static_boxes,
        dyn_specs=[],              # key difference vs baseline
        resolution=0.25,
    )

    goal_idx = static_grid.world_to_grid(goal)

    # Probe state
    probe_idx = static_grid.world_to_grid(start)
    executed_world: List[Vec3] = []

    # Current plan in grid space
    current_idx_path: Optional[List[Tuple[int, int, int]]] = None
    path_cursor = 0

    frames: List[Frame] = []

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

        # Current probe world position
        probe_world = static_grid.grid_to_world(probe_idx)

        # "Forget/update dynamically": only obstacles currently within sensor range are known
        dyn_mask = np.zeros_like(static_grid.occ, dtype=bool)

        for b in true_dyn_boxes_geom:
            if aabb_point_distance(probe_world, b) <= sensor_radius:
                mark_box_on_mask(static_grid, dyn_mask, b, inflation_cells=inflation_cells)

        # IMPORTANT: clear after marking so we never “trap” the planner at the start
        dyn_mask[probe_idx] = False
        dyn_mask[goal_idx] = False

        planning_grid = make_grid_with_mask(static_grid, dyn_mask)

        # If goal becomes blocked (rare), stop early
        if not planning_grid.is_free(goal_idx):
            print("Goal became blocked under current sensed obstacles; stopping.")
            break

        # Replan conditions
        need_plan = (current_idx_path is None)
        if k < 10:
            print(f"[k={k}] probe_idx={probe_idx} goal_idx={goal_idx}")
            print(f"       have_path={current_idx_path is not None}, path_len={0 if current_idx_path is None else len(current_idx_path)}")

        if not need_plan:
            # If we ran out of path
            if path_cursor >= len(current_idx_path):
                need_plan = True
            else:
                # If upcoming segment blocked
                need_plan = path_blocked(planning_grid, current_idx_path, path_cursor, lookahead_cells)

        if need_plan:
            try:
                if not planning_grid.is_free(probe_idx):
                    print("DEBUG: probe cell is occupied under current sensed mask!", probe_idx)

                if not planning_grid.is_free(goal_idx):
                    print("DEBUG: goal cell is occupied under current sensed mask!", goal_idx)

                current_idx_path = astar_3d(planning_grid, probe_idx, goal_idx)
                current_idx_path = normalize_path(current_idx_path, probe_idx, goal_idx)
                path_cursor = 0
            except Exception as e:
                print("Online replanning: replanner crashed:", repr(e))
                print("  probe_idx free?", planning_grid.is_free(probe_idx), "probe_idx=", probe_idx)
                print("  goal_idx  free?", planning_grid.is_free(goal_idx),  "goal_idx=", goal_idx)
                break


        if k < 10:
            print(f"       replanned_len={len(current_idx_path)} start={current_idx_path[0]} next={current_idx_path[1] if len(current_idx_path)>1 else None}")


        # Planned path for visualization (world coords)
        planned_world = [static_grid.grid_to_world(idx) for idx in current_idx_path[path_cursor:]]

        # Move one step along the planned path (if already at goal, remain)
        executed_world.append(probe_world)

        if probe_idx == goal_idx:
            # Already reached goal; keep drawing frames with empty planned path
            frames.append(Frame(
                t=round(k * dt, 2),
                obstacles=static_boxes + dyn_boxes,
                planned_path=[],
                probe=probe_world,
            ))
            break

        if current_idx_path is None or len(current_idx_path) == 0:
            print("No path returned; stopping.")
            break

        # Move one step forward along the current plan
        # (current_idx_path[path_cursor] is where we are supposed to be now)
        if current_idx_path is None or len(current_idx_path) == 0:
            print("No path returned; stopping.")
            break

        # Ensure cursor is in range
        path_cursor = max(0, min(path_cursor, len(current_idx_path) - 1))

        move_every = 3  # move once every 3 frames (increase to slow more)

        if k % move_every == 0:
            if path_cursor < len(current_idx_path) - 1:
                probe_idx = current_idx_path[path_cursor + 1]
                path_cursor += 1
            else:
                # end of path (should be at goal)
                probe_idx = current_idx_path[-1]

        frames.append(Frame(
            t=round(k * dt, 2),
            obstacles=static_boxes + dyn_boxes,
            planned_path=planned_world,
            probe=probe_world,
        ))

    scn = Scenario(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        start=start,
        goal=goal,
        frames=frames,
        title="Online replanning A*: static-known + dynamic sensed locally (forget/update)",
    )

    # Show sensor in viz (you already added this support in plotly_viz.py)
    out = visualize_scenario(scn, outfile=outfile, sensor_radius=sensor_radius)
    print(f"Wrote {out} (open it in your browser).")


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="environments/baseline_env2.json", help="Path to environment JSON (EnvironmentConfig).")
    parser.add_argument("--outfile", type=str, default="viz2.html",help="Output HTML file.")
    parser.add_argument("--planner", type=str, default="baseline", choices=["baseline", "online"], help="Planner mode: baseline (offline conservative) or online (replanning).")

    parser.add_argument("--dt", type=float, default=0.1, help="Frame timestep (seconds).")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to generate.")

    # Online replanning params
    parser.add_argument("--sensor-radius", type=float, default=0.5, help="Sensor radius (m) for discovering dynamic obstacles.")
    parser.add_argument("--inflation-cells", type=int, default=1, help="Inflation (grid cells) for sensed dynamic obstacles.")
    parser.add_argument("--lookahead-cells", type=int, default=10, help="Replan if any of next L path cells become blocked.")

    args = parser.parse_args()

    if args.planner == "baseline":
        run_baseline(args.env, args.outfile, dt=args.dt, n_frames=args.frames)
    else:
        run_online_astar(
            args.env, args.outfile,
            dt=args.dt, n_frames=args.frames,
            sensor_radius=args.sensor_radius,
            inflation_cells=args.inflation_cells,
            lookahead_cells=args.lookahead_cells,
        )


if __name__ == "__main__":
    main()
