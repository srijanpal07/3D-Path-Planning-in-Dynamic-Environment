"""
plan_and_visualize.py

Modes:
  1) (offline_astar) Offline A* 6/26 neighbors: Fully known environment (static + dynamic obstacles) with each node having 6or 26 neighbors.
  2) (online_astar) Online A* (26 neighbors): Static obstacles known and dynamic sensed locally, online replanning using A*
  3) (online_dstar) Online D* : Not implemeted yet.
  4) (rrtstar) RRT*: Not implemented yet.

Notes:
- Offline A* uses conservative swept-volume of dynamic obstacles.
- Online A* replans when any of the next L path cells become blocked by sensed dynamic obstacles.
- Visualization saved as HTML using Plotly.
"""

from __future__ import annotations
import argparse
from typing import List, Optional
import time
from dataclasses import dataclass, asdict
import json

from world.env_io import load_env_config
from world.schema import Obstacle, Box, Scenario, Frame
from world.grid_world import build_grid_from_env
from world.geometry_utils import lerp
from planners.astar3d import astar_3d
from planners.online_astar import run_online_astar
from viz.plotly_viz import visualize_scenario


# -------- CONFIGURABLE PARAMETERS --------
DT = 0.1                  # Frame timestep (seconds)
N_FRAMES = 120            # Number of frames to generate
INFLATION_CELLS = 1       # Inflation (grid cells) for sensed dynamic obstacles.
LOOKAHEAD_CELLS = 10      # Replan if any of next L path cells become blocked.
MOVE_EVERY = 3            # how often to move the probe along the planned path (move once every 3 frames, increase to slow more)
RESOLUTION = 0.25         # Grid resolution (m)
# ----------------------------------------


# ------------------------------
# Metrics dataclass and utility
# ------------------------------
@dataclass
class PlannerMetrics:
    planner: str
    success: bool
    failure_reason: Optional[str]
    planning_time_s: float
    path_length_m: float


def path_length(world_path) -> float:
    """Sum of Euclidean distances along a list of 3D points."""
    if not world_path or len(world_path) < 2:
        return 0.0
    dist = 0.0
    for a, b in zip(world_path[:-1], world_path[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dz = b[2] - a[2]
        dist += (dx*dx + dy*dy + dz*dz) ** 0.5
    return dist


# -------------------------------------------------
# Offline A* with conservative dynamic swept-volume
# -------------------------------------------------
def run_offline_astar(env_path: str, outfile: str, neighbors: int = 26) -> None:
    print("Evironment used for planning:", env_path)
    print(f"Running offline A* ({neighbors} neighbors) on fully known static and dynamic obstacles...")
    
    env = load_env_config(env_path)
    env_name = (env_path.split('/')[-1]).split('.json')[0]
    dt = DT
    n_frames = N_FRAMES
    if outfile is None:
        outfile = f"viz_offline_astar{neighbors}n_{env_name}.html"

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

    # offline astar: static + swept dynamic union
    grid = build_grid_from_env(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        static_obstacles=static_boxes,
        dyn_specs=dyn_specs,
        resolution=0.25,
    )

    start_idx = grid.world_to_grid(start)
    goal_idx = grid.world_to_grid(goal)

    # idx_path = astar_3d(grid, start_idx, goal_idx)
    # world_path = [grid.grid_to_world(idx) for idx in idx_path]

    t0 = time.perf_counter()    # METRICS: planning time + success/failure
    idx_path = None
    failure_reason = None   # METRICS: success/failure reason

    try:
        idx_path = astar_3d(grid, start_idx, goal_idx, neighbors=neighbors)  # offline A* :contentReference[oaicite:4]{index=4}
    except Exception as e:
        failure_reason = repr(e)
    finally:
        planning_time_s = time.perf_counter() - t0

    if idx_path is None:
        metrics = PlannerMetrics(
            planner=f"Offline A* ({neighbors} neighbors)",
            success=False,
            failure_reason=failure_reason,
            planning_time_s=planning_time_s,
            path_length_m=0.0,
        )
        print("\nMETRICS:", asdict(metrics))
        print(f"Offline A* ({neighbors} neighbors) planner failed; not generating visualization.")
        return

    world_path = [grid.grid_to_world(idx) for idx in idx_path]

    dist_m = path_length(world_path)  # METRICS: distance traveled 

    metrics = PlannerMetrics(
        planner=f"Offline A* ({neighbors} neighbors)",
        success=True,
        failure_reason=None,
        planning_time_s=planning_time_s,
        path_length_m=dist_m,
    )

    print("\n======= PLANNER METRICS =======")
    print(f"Planner           : {metrics.planner}")
    print(f"Success           : {metrics.success}")
    if not metrics.success:
        print(f"Failure reason    : {metrics.failure_reason}")
    else:
        print("Failure reason    : None")
    print(f"Planning time     : {metrics.planning_time_s:.6f} s")
    print(f"Path length       : {metrics.path_length_m:.4f} m")
    print("================================\n")

    # Save metrics to JSON
    metric_scores_file = f"metrics_offline_astar{neighbors}n_{env_name}.json"
    with open(metric_scores_file, "w") as f:
        json.dump(asdict(metrics), f, indent=2)
    print(f"Saved metric scores to \"{metric_scores_file}\"")

    # Visualization frames
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
        title=f"Offline A* ({neighbors} neighbors): Fully known static and dynamic obstacles",
    )

    print("\n======== VISUALIZATION =========")
    out = visualize_scenario(scn, outfile=outfile)
    print(f"Saved \"{out}\" in the root directory.")
    print(f"To visualize simulation, open \"{out}\" in your browser.")
    print("================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="environments/env3.json", help="Path to environment JSON (EnvironmentConfig).")
    parser.add_argument("--outfile", type=str, default=None,help="Output HTML file.")
    parser.add_argument("--planner", type=str, default="offline_astar", choices=["offline_astar", "online_astar"], help="Planner mode: offline_astar (fully known) or online_astar (partially known).")
    parser.add_argument("--neighbors", type=int, default=26, choices=[6, 26], help="Number of neighbors for A* (26 or 6) [only for offline planning].")
    parser.add_argument("--sensor-radius", type=float, default=0.5, help="Sensor radius (m) for discovering dynamic obstacles [only for online replanning].") 
    args = parser.parse_args()

    if args.planner == "offline_astar":
        run_offline_astar(args.env, args.outfile, neighbors=args.neighbors)
    else:
        run_online_astar(
            args.env, args.outfile,
            sensor_radius=args.sensor_radius,
            )


if __name__ == "__main__":
    main()
