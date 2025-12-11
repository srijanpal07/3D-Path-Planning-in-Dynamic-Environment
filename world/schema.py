from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal, Union

Vec3 = Tuple[float, float, float]

@dataclass
class Box:
    min_corner: Vec3
    max_corner: Vec3

@dataclass
class Sphere:
    center: Vec3
    radius: float

@dataclass
class Cylinder:
    base_center: Vec3   # center of the bottom face
    axis: Vec3          # direction vector (will be normalized)
    radius: float
    height: float

ObstacleKind = Literal["box", "sphere", "cylinder"]
ObstacleParams = Union[Box, Sphere, Cylinder]

@dataclass
class Obstacle:
    kind: ObstacleKind
    params: ObstacleParams
    dynamic: bool = False           # False => static color, True => dynamic color
    name: str = ""                  # optional label

@dataclass
class Frame:
    t: float
    obstacles: List[Obstacle]
    # Current planned path (polyline) from current probe position to goal
    planned_path: Optional[List[Vec3]] = None
    # Current probe position
    probe: Optional[Vec3] = None

@dataclass
class Scenario:
    bounds_min: Vec3
    bounds_max: Vec3
    start: Vec3
    goal: Vec3
    frames: List[Frame]
    title: str = "3D Dynamic Planning"
