import numpy as np
from world.schema import Scenario, Frame, Obstacle, Box

def lerp(a, b, t):
    return tuple((1 - t) * np.array(a) + t * np.array(b))

def main():
    bounds_min = (0.0, 0.0, 0.0)
    bounds_max = (10.0, 10.0, 5.0)

    start = (1.0, 1.0, 0.5)
    goal  = (9.0, 9.0, 0.5)

    # ----- Static boxes (buildings / obstacles) -----
    static_box_1 = Obstacle(
        kind="box",
        params=Box(min_corner=(3.0, 3.0, 0.0), max_corner=(4.5, 5.5, 2.5)),
        dynamic=False,
        name="static_1"
    )

    static_box_2 = Obstacle(
        kind="box",
        params=Box(min_corner=(6.5, 2.0, 0.0), max_corner=(8.0, 3.0, 2.0)),
        dynamic=False,
        name="static_2"
    )

    # ----- Dynamic boxes (moving obstacles) -----
    # Dynamic box 1 moves along +x direction
    dyn1_min_0 = (2.0, 7.0, 0.0)
    dyn1_max_0 = (3.0, 8.5, 2.0)
    dyn1_min_1 = (7.0, 7.0, 0.0)
    dyn1_max_1 = (8.0, 8.5, 2.0)

    # Dynamic box 2 moves diagonally in xy-plane
    dyn2_min_0 = (1.0, 4.0, 0.0)
    dyn2_max_0 = (2.0, 5.0, 1.5)
    dyn2_min_1 = (4.0, 7.0, 0.0)
    dyn2_max_1 = (5.0, 8.0, 1.5)

    T = 25
    frames = []

    for k in range(T):
        t = k / (T - 1)

        # Dynamic box positions (simple linear interpolation)
        dyn1_min = lerp(dyn1_min_0, dyn1_min_1, t)
        dyn1_max = lerp(dyn1_max_0, dyn1_max_1, t)
        dyn_box_1 = Obstacle(
            kind="box",
            params=Box(min_corner=dyn1_min, max_corner=dyn1_max),
            dynamic=True,
            name="dynamic_1"
        )

        dyn2_min = lerp(dyn2_min_0, dyn2_min_1, t)
        dyn2_max = lerp(dyn2_max_0, dyn2_max_1, t)
        dyn_box_2 = Obstacle(
            kind="box",
            params=Box(min_corner=dyn2_min, max_corner=dyn2_max),
            dynamic=True,
            name="dynamic_2"
        )

        # Probe motion: simple straight-line for now
        probe = lerp(start, goal, t)

        # Planned path for this frame: a little "arc" to goal via a mid waypoint
        mid = ((probe[0] + goal[0]) / 2.0,
               (probe[1] + goal[1]) / 2.0,
               probe[2] + 0.7)
        planned = [probe, mid, goal]

        frames.append(Frame(
            t=round(k * 0.2, 2),
            obstacles=[static_box_1, static_box_2, dyn_box_1, dyn_box_2],
            planned_path=planned,
            probe=probe
        ))

    scn = Scenario(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        start=start,
        goal=goal,
        frames=frames,
        title="Demo: Only Box Obstacles (Static = Blue, Dynamic = Orange)"
    )

    from viz.plotly_viz import visualize_scenario
    out = visualize_scenario(scn, outfile="viz.html")
    print(f"Wrote {out} (open it in your browser).")

if __name__ == "__main__":
    main()
