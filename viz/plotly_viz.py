from typing import Tuple, List
import numpy as np
import plotly.graph_objects as go
from world.schema import Scenario, Frame, Box3, Vec3

def _box_edges(minc: Vec3, maxc: Vec3) -> np.ndarray:
    """Return 24x3 array of the 12 edges (as segments) for a rectangular box."""
    x0,y0,z0 = minc
    x1,y1,z1 = maxc
    # 8 vertices
    V = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],  # bottom
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],  # top
    ])
    # 12 edges (pairs of indices)
    E = np.array([
        [0,1],[1,2],[2,3],[3,0],  # bottom
        [4,5],[5,6],[6,7],[7,4],  # top
        [0,4],[1,5],[2,6],[3,7],  # verticals
    ])
    # Convert into polyline segments with NaN separators for Plotly Scatter3d
    lines = []
    for a,b in E:
        lines.append(V[a])
        lines.append(V[b])
        lines.append([np.nan, np.nan, np.nan])  # break
    return np.array(lines)

def _scatter_edges_for_boxes(boxes: List[Box3]) -> go.Scatter3d:
    if not boxes:
        # empty dummy trace
        return go.Scatter3d(x=[], y=[], z=[], mode="lines", name="obstacles")
    all_lines = []
    for (mn, mx) in boxes:
        all_lines.append(_box_edges(mn, mx))
    L = np.vstack(all_lines)
    return go.Scatter3d(
        x=L[:,0], y=L[:,1], z=L[:,2],
        mode="lines",
        name="obstacles",
        line=dict(width=3),
        showlegend=True
    )

def _scatter_path(path: List[Vec3]) -> go.Scatter3d:
    P = np.array(path)
    return go.Scatter3d(
        x=P[:,0], y=P[:,1], z=P[:,2],
        mode="lines+markers",
        name="planned path",
        marker=dict(size=3),
        line=dict(width=5),
        showlegend=True
    )

def _scatter_probe(p: Vec3) -> go.Scatter3d:
    return go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]],
        mode="markers",
        name="probe",
        marker=dict(size=6, symbol="circle"),
        showlegend=True
    )

def _bounds_wireframe(bounds_min: Vec3, bounds_max: Vec3) -> go.Scatter3d:
    return _scatter_edges_for_boxes([(bounds_min, bounds_max)]).update(name="bounds")

def visualize_scenario(scn: Scenario, outfile: str = "viz.html") -> str:
    """
    Build an animated 3D figure for the provided Scenario and write it to HTML.
    Returns the path to the written HTML.
    """
    # Base traces (frame 0)
    f0: Frame = scn.frames[0]
    base_traces = []

    # Bounds box
    base_traces.append(_bounds_wireframe(scn.bounds_min, scn.bounds_max))

    # Obstacles, path, probe at t0
    base_traces.append(_scatter_edges_for_boxes(f0.boxes))
    if f0.path:
        base_traces.append(_scatter_path(f0.path))
    if f0.probe:
        base_traces.append(_scatter_probe(f0.probe))

    # Frames (animation)
    frames = []
    for fr in scn.frames:
        traces = []
        traces.append(_scatter_edges_for_boxes(fr.boxes))
        if fr.path:
            traces.append(_scatter_path(fr.path))
        if fr.probe:
            traces.append(_scatter_probe(fr.probe))
        frames.append(go.Frame(
            data=traces,
            name=f"{fr.t:.2f}s"
        ))

    # Layout + sliders
    fig = go.Figure(
        data=base_traces,
        frames=frames,
        layout=go.Layout(
            title=scn.title,
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                aspectmode="data"
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.05,
                x=0,
                xanchor="left",
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=200, redraw=True),
                                          fromcurrent=True, transition=dict(duration=0))]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate",
                                            transition=dict(duration=0))])
                ]
            )],
            sliders=[dict(
                steps=[dict(method="animate",
                            args=[[fr.name], dict(mode="immediate",
                                                  frame=dict(duration=0, redraw=True),
                                                  transition=dict(duration=0))],
                            label=fr.name) for fr in frames],
                x=0.05, y=-0.05, len=0.9
            )]
        )
    )

    fig.write_html(outfile, include_plotlyjs="cdn", auto_open=False)
    return outfile

