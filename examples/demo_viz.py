import numpy as np
from world.schema import Scenario, Frame
from viz.plotly_viz import visualize_scenario

def lerp(a, b, t):
    return tuple((1-t)*np.array(a) + t*np.array(b))

def main():
    bounds_min = (0.0, 0.0, 0.0)
    bounds_max = (10.0, 10.0, 5.0)

    # Static obstacle (box)
    static_box = ((3.0, 3.0, 0.0), (4.5, 5.0, 2.5))

    # Dynamic obstacle (starts near y=6, moves in +x)
    dyn_start = (1.5, 6.0, 0.0)
    dyn_end   = (7.5, 6.0, 0.0)  # slide along x
    dyn_size  = (1.0, 1.0, 2.0)

    # Probe start/goal
    start = (1.0, 1.0, 0.5)
    goal  = (9.0, 9.0, 0.5)

    T = 25
    frames = []
    for k in range(T):
        t = k / (T-1)

        # Dynamic box position at frame k
        dyn_center = lerp(dyn_start, dyn_end, t)
        dyn_min = (dyn_center[0] - dyn_size[0]/2,
                   dyn_center[1] - dyn_size[1]/2,
                   dyn_center[2])
        dyn_max = (dyn_center[0] + dyn_size[0]/2,
                   dyn_center[1] + dyn_size[1]/2,
                   dyn_center[2] + dyn_size[2])

        # Fake probe motion (just interpolate start->goal)
        probe = lerp(start, goal, t)

        # Fake “planned path” polyline (current probe -> goal with a gentle two-point bend)
        mid = ((probe[0] + goal[0])/2, probe[1], probe[2] + 0.5)
        path = [probe, mid, goal]

        # Boxes present this frame (static + dynamic)
        boxes = [static_box, (dyn_min, dyn_max)]

        frames.append(Frame(
            t=round(k*0.2, 2),  # seconds
            boxes=boxes,
            path=path,
            probe=probe
        ))

    scn = Scenario(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        frames=frames,
        title="Demo: Dynamic Box + Moving Probe"
    )

    out = visualize_scenario(scn, outfile="viz.html")
    print(f"Wrote {out} (open it in your browser).")

if __name__ == "__main__":
    main()

