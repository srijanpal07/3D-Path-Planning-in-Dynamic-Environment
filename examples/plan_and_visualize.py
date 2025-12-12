"""
plan_and_visualize.py

Load Environment -> Run Planner -> Visualize

This script loads a saved environment configuration (EnvironmentConfig),
reconstructs the static and dynamic obstacles, and builds a GridWorld
used for planning.

It then:
  - runs a chosen planner (currently 3D A* for the baseline)
  - converts the resulting grid-level path into world coordinates
  - generates a time-indexed Scenario with:
        * moving dynamic obstacles
        * the static environment
        * the precomputed A* trajectory
        * the probe moving along this trajectory

Finally, the script produces an interactive HTML visualization
(showing obstacles, the probe, and the planned path).

This script is intended for evaluating path planning algorithms on
previously saved environments.

Outputs:
  - baseline_viz.html
"""

import numpy as np
from world.env_io import load_env_config
from world.schema import Obstacle, Box
from world.env_config import DynBoxSpec
from world.grid_world import build_grid_from_env
from world.astar3d import astar_3d
from viz.plotly_viz import visualize_scenario
from world.schema import Scenario, Frame

ENVIRONMENT_PATH = "environments/baseline_env.json"
VIZ_PATH = "baseline_viz.html"


def lerp(a, b, t):
    return tuple((1 - t) * np.array(a) + t * np.array(b))


def main():
    env = load_env_config(ENVIRONMENT_PATH)

    bounds_min = env.bounds_min
    bounds_max = env.bounds_max
    start = env.start
    goal = env.goal

    # Rebuild static Obstacles for visualization/planning
    static_boxes = [
        Obstacle(
            kind="box",
            params=box,
            dynamic=False,
            name=f"static_{i}",
        )
        for i, box in enumerate(env.static_boxes)
    ]

    # Rebuild dyn_specs in the same format used before (dicts)
    dyn_specs = [
        {
            "name": d.name,
            "min0": d.min0,
            "max0": d.max0,
            "min1": d.min1,
            "max1": d.max1,
        }
        for d in env.dyn_boxes
    ]

    # Build a 3D grid and precompute offline A* path
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


    T = 40  # number of frames for the animation
    frames = []

    for k in range(T):
        t = k / (T - 1)

        # dynamic obstacles at this time
        dyn_boxes = []
        for spec in dyn_specs:
            min_corner = lerp(spec["min0"], spec["min1"], t)
            max_corner = lerp(spec["max0"], spec["max1"], t)
            dyn_boxes.append(
                Obstacle(
                    kind="box",
                    params=Box(min_corner=min_corner, max_corner=max_corner),
                    dynamic=True,
                    name=spec["name"],
                )
            )
        
        # Follow the precomputed A* path (offline baseline)
        path_pos = min(int(t * (len(world_path) - 1)), len(world_path) - 1)
        probe = world_path[path_pos]
        planned = world_path  # show the full A* path


        frames.append(
            Frame(
                t=round(k * 0.2, 2),
                obstacles=static_boxes + dyn_boxes,
                planned_path=planned,
                probe=probe,
            )
        )

    scn = Scenario(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        start=start,
        goal=goal,
        frames=frames,
        title="Baseline: 3D A* Environment with Static (blue) and Dynamic (orange) Boxes",
    )

    out = visualize_scenario(scn, outfile=VIZ_PATH)
    print(f"Wrote {out} (open it in your browser).")


if __name__ == "__main__":
    main()
