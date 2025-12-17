import numpy as np
import plotly.graph_objects as go
from typing import List
from world.schema import Scenario, Obstacle, Box, Vec3

STATIC_COLOR   = "rgb(0, 150, 255)"    # static obstacles: blue
DYNAMIC_COLOR  = "rgb(255, 140, 0)"    # dynamic obstacles: orange
BOUNDS_COLOR   = "rgba(120,120,120,0.5)"

START_COLOR    = "rgb(0, 200, 0)"      # green
GOAL_COLOR     = "rgb(200, 0, 0)"      # red
PROBE_COLOR    = "rgb(90, 0, 200)"     # purple
PATH_COLOR     = "rgb(220, 0, 220)"    # magenta
TAIL_COLOR     = "rgb(0, 0, 0)"    # blue
SENSOR_COLOR = "rgb(200,200,160,0.5)" # light grey-ish
SENSOR_WIDTH = 2


# ---------------------- bounds wireframe ----------------------

def _bounds_wireframe(minc: Vec3, maxc: Vec3) -> go.Scatter3d:
    x0, y0, z0 = minc
    x1, y1, z1 = maxc
    verts = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ])
    edges = np.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ])
    lines = []
    for a,b in edges:
        lines.append(verts[a])
        lines.append(verts[b])
        lines.append([np.nan, np.nan, np.nan])
    L = np.array(lines)
    return go.Scatter3d(
        x=L[:,0], y=L[:,1], z=L[:,2],
        mode="lines",
        line=dict(width=2, color=BOUNDS_COLOR),
        name="bounds",
        showlegend=True,
    )


# ---------------------- obstacle wireframes ----------------------

def _box_edges(box: Box, color: str) -> go.Scatter3d:
    x0,y0,z0 = box.min_corner
    x1,y1,z1 = box.max_corner
    verts = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ])
    edges = np.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ])
    lines = []
    for a,b in edges:
        lines.append(verts[a])
        lines.append(verts[b])
        lines.append([np.nan, np.nan, np.nan])
    L = np.array(lines)
    return go.Scatter3d(
        x=L[:,0], y=L[:,1], z=L[:,2],
        mode="lines",
        line=dict(width=2, color=color),
        showlegend=False,
    )


def _obstacle_traces(ob: Obstacle) -> List:
    """
    Only box obstacles are visualized.
    Static = blue, Dynamic = orange.
    """
    color = DYNAMIC_COLOR if ob.dynamic else STATIC_COLOR

    if isinstance(ob.params, Box):
        return [_box_edges(ob.params, color)]

    # If later you accidentally create a sphere/cylinder, this will remind you.
    raise ValueError(f"Only Box obstacles are supported in the Plotly viz for now, got {type(ob.params)}.")


# ---------------------- markers & paths ----------------------

def _start_goal_marker(p: Vec3, color: str, label: str) -> go.Scatter3d:
    x, y, z = p

    # Use symbols that are valid for 3D markers in Plotly
    if label.upper() == "START":
        symbol = "square-open"    # outlined square
    else:
        symbol = "diamond-open"   # outlined diamond for GOAL

    return go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode="markers+text",
        name=label,
        marker=dict(
            size=5,
            symbol=symbol,
            color=color,
            line=dict(width=3, color="black"),
            opacity=1.0,
        ),
        text=[label],
        textposition="top center",
        textfont=dict(color=color, size=18),
        showlegend=True,
    )


def _probe_marker(p: Vec3) -> go.Scatter3d:
    # Brighter, larger probe with a thin white outline
    return go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]],
        mode="markers",
        name="Probe",
        marker=dict(
            size=9,
            symbol="circle",
            color=PROBE_COLOR,
            line=dict(width=2, color="white"),  # thin outline
            opacity=1.0,
        ),
        showlegend=True, # False,
    )


def _path_trace(path: List[Vec3], color: str, name: str,
                width: int = 2) -> go.Scatter3d:
    P = np.array(path)
    return go.Scatter3d(
        x=P[:,0], y=P[:,1], z=P[:,2],
        mode="lines",
        line=dict(width=width, color=color),
        name=name,
        showlegend=False,   # legend handled by dummy
    )


def _legend_line(color: str, name: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode="lines",
        line=dict(width=3, color=color),
        name=name,
        showlegend=True,
        visible="legendonly",
    )


def _sensor_wire_sphere(center: tuple[float, float, float],
                        radius: float,
                        n_theta: int = 24,
                        n_phi: int = 12) -> list[go.Scatter3d]:
    """
    Returns a set of 3D line traces approximating a wireframe sphere.

    - n_theta controls resolution around the circle (longitude)
    - n_phi controls number of latitude rings
    """
    cx, cy, cz = center
    traces: list[go.Scatter3d] = []

    # Latitude rings (phi from 0..pi)
    phis = np.linspace(0.15*np.pi, 0.85*np.pi, n_phi)  # skip poles to avoid clutter
    thetas = np.linspace(0, 2*np.pi, n_theta)

    for phi in phis:
        x = cx + radius * np.sin(phi) * np.cos(thetas)
        y = cy + radius * np.sin(phi) * np.sin(thetas)
        z = cz + radius * np.cos(phi) * np.ones_like(thetas)

        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=SENSOR_COLOR, width=SENSOR_WIDTH),
            name="sensor",
            showlegend=False,
            hoverinfo="skip",
        ))

    # A few longitude rings (vertical great circles)
    longitudes = [0.0, 0.5*np.pi, np.pi/4, 3*np.pi/4]
    phis2 = np.linspace(0, np.pi, n_theta)

    for theta0 in longitudes:
        x = cx + radius * np.sin(phis2) * np.cos(theta0)
        y = cy + radius * np.sin(phis2) * np.sin(theta0)
        z = cz + radius * np.cos(phis2)

        traces.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            line=dict(color=SENSOR_COLOR, width=SENSOR_WIDTH),
            name="sensor",
            showlegend=False,
            hoverinfo="skip",
        ))

    return traces


# ---------------------- main entry ----------------------

def visualize_scenario(scn: Scenario, outfile: str = "viz.html", sensor_radius: float | None = 0.25) -> str:
    frames = []
    probe_tail: List[Vec3] = []

    # Build all frames
    for fr in scn.frames:
        frame_traces: List = []

        # Bounds box
        frame_traces.append(_bounds_wireframe(scn.bounds_min, scn.bounds_max))

        # Obstacles for this frame
        for ob in fr.obstacles:
            frame_traces.extend(_obstacle_traces(ob))

        # START & GOAL (in every frame so they are always visible)
        frame_traces.append(_start_goal_marker(scn.start, START_COLOR, "START"))
        frame_traces.append(_start_goal_marker(scn.goal,  GOAL_COLOR,  "GOAL"))

        # Planned path from this frame
        if fr.planned_path:
            frame_traces.append(_path_trace(fr.planned_path, PATH_COLOR, "Planned Trajectory"))

        # Probe + tail (executed path)
        if fr.probe:
            probe_tail.append(fr.probe)
            frame_traces.append(_path_trace(probe_tail, TAIL_COLOR, "Traversed Trajectory", width=2))
            frame_traces.append(_probe_marker(fr.probe))
            if sensor_radius is not None and fr.probe is not None:
                frame_traces.extend(_sensor_wire_sphere(fr.probe, sensor_radius))

        frames.append(go.Frame(data=frame_traces, name=f"{fr.t:.2f}s"))

    # If no frames, build an empty figure
    if not frames:
        fig = go.Figure()
    else:
        # Use the first frame's data as the initial visible data
        fig = go.Figure(
            data=frames[0].data,
            frames=frames,
        )

    # Simple legend dummies
    fig.add_trace(_legend_line(PATH_COLOR, "Planned Trajectory"))
    fig.add_trace(_legend_line(TAIL_COLOR, "Traversed Trajectory"))

    fig.update_layout(
        title=scn.title,
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
            aspectmode="data",
            bgcolor="white",
        ),
        paper_bgcolor="white",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.05, x=0, xanchor="left", yanchor="top",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=200, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                    )],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
        sliders=[dict(
            steps=[
                dict(
                    method="animate",
                    args=[[fr.name], dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    )],
                    label=fr.name,
                )
                for fr in frames
            ],
            x=0.05, y=-0.05, len=0.9,
        )] if frames else [],
    )

    fig.write_html(outfile, include_plotlyjs="cdn", auto_open=False)
    return outfile

