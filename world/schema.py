from dataclasses import dataclass
from typing import List, Tuple

Vec3 = Tuple[float, float, float]


@dataclass
class Box:
    min_corner: Vec3
    max_corner: Vec3


@dataclass
class Obstacle:
    # We still keep a 'kind' string for readability / debugging, but it's always "box".
    kind: str
    params: Box
    dynamic: bool = False   # False => static obstacle (blue), True => dynamic (orange)
    name: str = ""          # optional label


@dataclass
class Frame:
    t: float
    obstacles: List[Obstacle]
    # Current planned path (polyline) from current probe position to goal
    planned_path: List[Vec3] | None = None
    # Current probe position
    probe: Vec3 | None = None


@dataclass
class Scenario:
    bounds_min: Vec3
    bounds_max: Vec3
    start: Vec3
    goal: Vec3
    frames: List[Frame]
    title: str = "3D Dynamic Planning"
