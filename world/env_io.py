import json
from dataclasses import asdict
from typing import List

from world.schema import Box, Vec3
from world.env_config import EnvironmentConfig, DynBoxSpec


def save_env_config(env: EnvironmentConfig, path: str) -> None:
    """
    Save only the environment description (no frames, no probe path).
    """
    data = {
        "bounds_min": env.bounds_min,
        "bounds_max": env.bounds_max,
        "start": env.start,
        "goal": env.goal,
        "static_boxes": [
            {
                "min_corner": box.min_corner,
                "max_corner": box.max_corner,
            }
            for box in env.static_boxes
        ],
        "dyn_boxes": [
            {
                "name": d.name,
                "min0": d.min0,
                "max0": d.max0,
                "min1": d.min1,
                "max1": d.max1,
            }
            for d in env.dyn_boxes
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_env_config(path: str) -> EnvironmentConfig:
    """
    Load an EnvironmentConfig from JSON saved by save_env_config.
    """
    with open(path, "r") as f:
        data = json.load(f)

    bounds_min: Vec3 = tuple(data["bounds_min"])
    bounds_max: Vec3 = tuple(data["bounds_max"])
    start: Vec3 = tuple(data["start"])
    goal: Vec3 = tuple(data["goal"])

    static_boxes: List[Box] = []
    for bd in data["static_boxes"]:
        static_boxes.append(
        Box(
            min_corner=tuple(bd["min_corner"]),
            max_corner=tuple(bd["max_corner"]),
        )
    )

    dyn_boxes: List[DynBoxSpec] = []
    for dd in data["dyn_boxes"]:
        dyn_boxes.append(
            DynBoxSpec(
                name=dd["name"],
                min0=tuple(dd["min0"]),
                max0=tuple(dd["max0"]),
                min1=tuple(dd["min1"]),
                max1=tuple(dd["max1"]),
            )
        )

    return EnvironmentConfig(
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        start=start,
        goal=goal,
        static_boxes=static_boxes,
        dyn_boxes=dyn_boxes,
    )
