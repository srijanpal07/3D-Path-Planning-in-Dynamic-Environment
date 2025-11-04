from dataclasses import dataclass
from typing import List, Tuple, Optional

Vec3 = Tuple[float, float, float]
Box3 = Tuple[Vec3, Vec3]  # (min_corner, max_corner)

@dataclass
class Frame:
    t: float
    # Obstacles present at this time (both static+dynamic already “resolved” for this frame)
    boxes: List[Box3]
    # Optional: current planned path (polyline) at time t
    path: Optional[List[Vec3]] = None
    # Probe position at time t
    probe: Optional[Vec3] = None

@dataclass
class Scenario:
    bounds_min: Vec3    # e.g., (0,0,0)
    bounds_max: Vec3    # e.g., (10,10,5)
    frames: List[Frame]
    title: str = "3D Dynamic Planning"

