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
  - computes a offline 3D A* path
  - generates a Scenario with obstacles and the
    computed path for visualization

Output:
  - environments/baseline_env.json       (environment only, no frames or path)
  - viz.html                       (optional preview visualization)

This script is mainly used only for designing or editing environments.
It does NOT need to be run when evaluating planners.
"""

import numpy as np
from world.schema import Scenario, Frame, Obstacle, Box
from world.grid_world import build_grid_from_env
from world.env_config import EnvironmentConfig, DynBoxSpec
from world.env_io import save_env_config
from planners.astar3d import astar_3d
from viz.plotly_viz import visualize_scenario


#---------------------- Parameters ----------------------
START = (0.5, 0.5, 0.5)
GOAL  = (19.0, 14.0, 4.0)
ENV_SAVE_PATH = "environments/env_complex.json"
VIZ_TITLE = "Complex Environment: 3D A* Environment with Static (blue) and Dynamic (orange) Boxes"
SAVE_VIZ = True  # set to False to skip visualization step
VIZ_SAVE_PATH = "baseline_viz.html"
#--------------------------------------------------------

RNG_SEED = 3

# If bounds become 20x20, these work well
N_STATIC = 11
N_DYNAMIC = 11

STATIC_SIZE_RANGE = ((0.8, 0.8, 1.0), (2.5, 2.5, 5.0))  # min/max size (dx,dy,dz)
DYN_SIZE_RANGE    = ((0.6, 0.6, 1.0), (3.5, 3.5, 2.0))

CLEARANCE_START_GOAL = 1.0  # meters, keep obstacles away



def rand_box(rng, bounds_min, bounds_max, size_lo, size_hi):
    dx = rng.uniform(size_lo[0], size_hi[0])
    dy = rng.uniform(size_lo[1], size_hi[1])
    dz = rng.uniform(size_lo[2], size_hi[2])

    x0 = rng.uniform(bounds_min[0], bounds_max[0] - dx)
    y0 = rng.uniform(bounds_min[1], bounds_max[1] - dy)
    z0 = rng.uniform(bounds_min[2], bounds_max[2] - dz)

    min_corner = (x0, y0, z0)
    max_corner = (x0 + dx, y0 + dy, z0 + dz)
    return min_corner, max_corner


def dist(p, q):
    return float(np.linalg.norm(np.array(p) - np.array(q)))


def aabb_overlap(minA, maxA, minB, maxB):
    # True if overlapping (including touching)
    return (
        (minA[0] <= maxB[0] and maxA[0] >= minB[0]) and
        (minA[1] <= maxB[1] and maxA[1] >= minB[1]) and
        (minA[2] <= maxB[2] and maxA[2] >= minB[2])
    )




def lerp(a, b, t):
    return tuple((1 - t) * np.array(a) + t * np.array(b))


def main():
    # ---------- WORLD BOUNDS ----------
    bounds_min = (0.0, 0.0, 0.0)
    bounds_max = (20.0, 20.0, 6.0)

    # ---------- START & GOAL ----------
    start = START
    goal  = GOAL

    # ---------- STATIC BOXES ----------
    rng = np.random.default_rng(RNG_SEED)

    static_boxes = []
    static_geom = []  # keep (min,max) for overlap checks

    while len(static_boxes) < N_STATIC:
        mn, mx = rand_box(rng, bounds_min, bounds_max, *STATIC_SIZE_RANGE)

        # keep away from start/goal
        center = tuple((np.array(mn) + np.array(mx)) / 2.0)
        if dist(center, start) < CLEARANCE_START_GOAL or dist(center, goal) < CLEARANCE_START_GOAL:
            continue

        # avoid overlapping existing statics (optional)
        overlaps = any(aabb_overlap(mn, mx, mn2, mx2) for mn2, mx2 in static_geom)
        if overlaps:
            continue

        name = f"static_{len(static_boxes)}"
        static_geom.append((mn, mx))
        static_boxes.append(
            Obstacle(kind="box", params=Box(min_corner=mn, max_corner=mx), dynamic=False, name=name)
        )


    # ---------- DYNAMIC BOX TEMPLATES ----------
    dyn_specs = []
    for i in range(N_DYNAMIC):
        mn0, mx0 = rand_box(rng, bounds_min, bounds_max, *DYN_SIZE_RANGE)
        mn1, mx1 = rand_box(rng, bounds_min, bounds_max, *DYN_SIZE_RANGE)

        dyn_specs.append(dict(
            name=f"dyn_{i}",
            min0=mn0, max0=mx0,
            min1=mn1, max1=mx1,
        ))


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
            
            # Follow the precomputed A* path
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
            title=VIZ_TITLE,
        )

        print("\nSaving an example visualization on the saved environment with offline astar planner...")
        out = visualize_scenario(scn, outfile=VIZ_SAVE_PATH)
        print(f"Saved \"{out}\" in the root directory.")
        print(f"To visualize simulation, open \"{out}\" in your browser.\n")


if __name__ == "__main__":
    main()
