"""
generate_env.py

Environment Creation + Optional Visualization

This script constructs a 3D environment containing:
  - world bounds
  - start and goal locations
  - static obstacles (axis-aligned 3D boxes)
  - dynamic obstacles defined by linear motion between two 3D box states

It then saves this environment to a reusable JSON file using EnvironmentConfig.

Optionally, the script also:
  - builds a 3D occupancy grid
  - computes a baseline 3D A* path
  - generates a Scenario with interpolated dynamic obstacles and the
    computed path for visualization

Output:
  - scenes/baseline_env.json       (environment only, no frames or path)
  - viz.html                       (optional preview visualization)

This script is mainly used only for designing or editing environments.
It does NOT need to be run when evaluating planners.
"""

import numpy as np
from world.schema import Scenario, Frame, Obstacle, Box
from world.grid_world import build_grid_from_env
from world.astar3d import astar_3d
from viz.plotly_viz import visualize_scenario
from world.env_config import EnvironmentConfig, DynBoxSpec
from world.env_io import save_env_config

ENV_SAVE_PATH = "scenes/baseline_env.json"
SAVE_VIZ = True  # set to False to skip visualization step
VIZ_SAVE_PATH = "baseline_viz.html"


def lerp(a, b, t):
    return tuple((1 - t) * np.array(a) + t * np.array(b))


def main():
    # ---------- WORLD BOUNDS ----------
    bounds_min = (0.0, 0.0, 0.0)
    bounds_max = (10.0, 10.0, 5.0)

    # ---------- START & GOAL ----------
    start = (1.0, 1.0, 1.0)
    goal  = (9.0, 9.0, 3.0)

    # ---------- STATIC BOXES ----------
    static_boxes = [
        # tall pillar near center
        Obstacle(
            kind="box",
            params=Box(min_corner=(4.0, 4.0, 0.0), max_corner=(5.0, 5.0, 4.0)),
            dynamic=False,
            name="static_center_tall",
        ),
        # low block near start
        Obstacle(
            kind="box",
            params=Box(min_corner=(2.0, 1.5, 0.0), max_corner=(3.5, 3.5, 1.5)),
            dynamic=False,
            name="static_near_start",
        ),
        # mid-height block near goal
        Obstacle(
            kind="box",
            params=Box(min_corner=(7.0, 7.0, 0.0), max_corner=(8.5, 8.5, 2.5)),
            dynamic=False,
            name="static_near_goal",
        ),
        # flat “roof” in the middle
        Obstacle(
            kind="box",
            params=Box(min_corner=(5.0, 2.0, 2.0), max_corner=(8.0, 4.0, 3.0)),
            dynamic=False,
            name="static_roof",
        ),
    ]

    # ---------- DYNAMIC BOX TEMPLATES ----------
    # Each dynamic box has a start/end box (min/max in 3D).
    dyn_specs = [
        # moves diagonally up-right
        dict(
            name="dyn_diag_up",
            min0=(1.0, 6.0, 0.5), max0=(2.0, 7.0, 1.5),
            min1=(4.0, 9.0, 3.0), max1=(5.0,10.0, 4.0),
        ),
        # moves mostly in -x,+z
        dict(
            name="dyn_left_up",
            min0=(8.5, 3.0, 0.5), max0=(9.5, 4.0, 1.5),
            min1=(5.5, 3.0, 3.0), max1=(6.5, 4.0, 4.0),
        ),
        # moves mostly in +y,+z
        dict(
            name="dyn_forward_up",
            min0=(3.0, 1.0, 0.5), max0=(4.0, 2.0, 1.5),
            min1=(3.0, 5.0, 3.0), max1=(4.0, 6.0, 4.0),
        ),
        # moves downwards in z while sliding in y
        dict(
            name="dyn_down",
            min0=(6.0, 8.5, 3.5), max0=(7.0, 9.5, 4.5),
            min1=(6.0, 5.5, 0.5), max1=(7.0, 6.5, 1.5),
        ),
        # small box orbiting around center in y-z plane (approx, via lerp)
        dict(
            name="dyn_center1",
            min0=(4.5, 4.0, 0.5), max0=(5.0, 4.5, 1.0),
            min1=(4.5, 6.0, 3.0), max1=(5.0, 6.5, 3.5),
        ),
        dict(
            name="dyn_center2",
            min0=(5.5, 6.0, 3.0), max0=(6.0, 6.5, 3.5),
            min1=(5.5, 4.0, 0.5), max1=(6.0, 4.5, 1.0),
        ),
    ]

    # Build EnvironmentConfig and save to file
    env = EnvironmentConfig(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        start=start,
        goal=goal,
        static_boxes=[ob.params for ob in static_boxes],  # just the Box geometry
        dyn_boxes=[
            DynBoxSpec(
                name=spec["name"],
                min0=spec["min0"],
                max0=spec["max0"],
                min1=spec["min1"],
                max1=spec["max1"],
            )
            for spec in dyn_specs
        ],
    )

    # Save the environment config (no frames or path)
    save_env_config(env, ENV_SAVE_PATH)
    print(f"Saved environment config to {ENV_SAVE_PATH}.")

    if SAVE_VIZ:
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
            path_pos = int(t * (len(world_path) - 1))
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


        out = visualize_scenario(scn, outfile=VIZ_SAVE_PATH)
        print(f"Wrote {out} (open it in your browser).")


if __name__ == "__main__":
    main()
