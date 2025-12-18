from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from world.env_io import load_env_config
from world.schema import Obstacle, Box, Scenario, Frame, Vec3
from world.grid_world import build_grid_from_env, GridWorld
from planners.astar3d import astar_3d
from viz.plotly_viz import visualize_scenario
from examples.geometry_utils import mark_box_on_mask, aabb_point_distance, lerp, make_grid_with_mask


# -------- CONFIGURABLE PARAMETERS --------
DT = 0.1                  # Frame timestep (seconds)
N_FRAMES = 120            # Number of frames to generate
INFLATION_CELLS = 1       # Inflation (grid cells) for sensed dynamic obstacles.
LOOKAHEAD_CELLS = 10      # Replan if any of next L path cells become blocked.
MOVE_EVERY = 3            # how often to move the probe along the planned path (move once every 3 frames, increase to slow more)
RESOLUTION = 0.25         # Grid resolution (m)
# ----------------------------------------


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


def run_online_astar(env_path: str,outfile: str, sensor_radius: float=0.5) -> None:
    print("Running online A* replanning with local sensing...")    
    env = load_env_config(env_path)
    dt= DT
    n_frames = N_FRAMES
    sensor_radius = sensor_radius
    inflation_cells = INFLATION_CELLS
    lookahead_cells = LOOKAHEAD_CELLS
    move_every = MOVE_EVERY
    resolution = RESOLUTION

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

        # only obstacles currently within sensor range are known
        dyn_mask = np.zeros_like(static_grid.occ, dtype=bool)

        for b in true_dyn_boxes_geom:
            if aabb_point_distance(probe_world, b) <= sensor_radius:
                mark_box_on_mask(static_grid, dyn_mask, b, inflation_cells=inflation_cells)

        # IMPORTANT: clear after marking so we never “trap” the planner at the start
        dyn_mask[probe_idx] = False
        dyn_mask[goal_idx] = False

        planning_grid = make_grid_with_mask(static_grid, dyn_mask)

        # If goal becomes blocked, stop early
        if not planning_grid.is_free(goal_idx):
            print("Goal became blocked under current sensed obstacles; stopping.")
            break

        # Replan conditions
        need_plan = (current_idx_path is None)

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
                print("probe_idx free?", planning_grid.is_free(probe_idx), "probe_idx=", probe_idx)
                print("goal_idx  free?", planning_grid.is_free(goal_idx),  "goal_idx=", goal_idx)
                break

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
        title="Online replanning A*: Known static obstacles + dynamic obstacles sensed locally",
    )

    # Show sensor in viz (as a sphere around the probe)
    out = visualize_scenario(scn, outfile=outfile, sensor_radius=sensor_radius)
    print(f"Wrote {out} (open it in your browser).")
