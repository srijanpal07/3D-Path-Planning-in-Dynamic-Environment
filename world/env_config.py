from dataclasses import dataclass
from typing import List

from world.schema import Vec3, Box


@dataclass
class DynBoxSpec:
    """
    Motion of one dynamic box: linearly interpolated from (min0,max0) to (min1,max1)
    over normalized time t in [0,1].
    """
    name: str
    min0: Vec3
    max0: Vec3
    min1: Vec3
    max1: Vec3


@dataclass
class EnvironmentConfig:
    """
    Environment/scene description WITHOUT frames or probe path.
    - bounds
    - start, goal
    - static obstacles (boxes)
    - dynamic obstacle motion specs
    """
    bounds_min: Vec3
    bounds_max: Vec3
    start: Vec3
    goal: Vec3
    static_boxes: List[Box]
    dyn_boxes: List[DynBoxSpec]
