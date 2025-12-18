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
from typing import List, Tuple, Optional
import numpy as np
from world.env_io import load_env_config
from world.schema import Obstacle, Box, Scenario, Frame, Vec3
from world.grid_world import build_grid_from_env, GridWorld
from planners.astar3d import astar_3d
from planners.online_astar import run_online_astar
from viz.plotly_viz import visualize_scenario
from examples.geometry_utils import lerp


# -------- CONFIGURABLE PARAMETERS --------
DT = 0.1                  # Frame timestep (seconds)
N_FRAMES = 120            # Number of frames to generate
INFLATION_CELLS = 1       # Inflation (grid cells) for sensed dynamic obstacles.
LOOKAHEAD_CELLS = 10      # Replan if any of next L path cells become blocked.
MOVE_EVERY = 3            # how often to move the probe along the planned path (move once every 3 frames, increase to slow more)
RESOLUTION = 0.25         # Grid resolution (m)
# ----------------------------------------


# ----------------------------
# Baseline: Offline A* with conservative dynamic swept-volume
# ----------------------------
def run_baseline(env_path: str, outfile: str) -> None:
    print("Running baseline A* with fully known static + dynamic obstacles...")
    env = load_env_config(env_path)
    dt = DT
    n_frames = N_FRAMES

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

    # Baseline: static + swept dynamic union
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="environments/baseline_env2.json", help="Path to environment JSON (EnvironmentConfig).")
    parser.add_argument("--outfile", type=str, default="viz2.html",help="Output HTML file.")
    parser.add_argument("--planner", type=str, default="baseline", choices=["baseline", "online"], help="Planner mode: baseline (offline conservative) or online (replanning).")
    parser.add_argument("--sensor-radius", type=float, default=0.5, help="Sensor radius (m) for discovering dynamic obstacles -- only for online replanning.") 
    args = parser.parse_args()

    if args.planner == "baseline":
        run_baseline(args.env, args.outfile)
    else:
        run_online_astar(
            args.env, args.outfile,
            sensor_radius=args.sensor_radius,
            )


if __name__ == "__main__":
    main()
