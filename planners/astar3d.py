import heapq
import math
from typing import Tuple, List, Dict, Set
from world.grid_world import GridWorld, Idx3


def _heuristic(idx: Idx3, goal: Idx3, grid: GridWorld) -> float:
    """
    Euclidean distance in 3D between cell centers, used as A* heuristic.
    """
    dx = (idx[0] - goal[0]) * grid.resolution
    dy = (idx[1] - goal[1]) * grid.resolution
    dz = (idx[2] - goal[2]) * grid.resolution
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def astar_3d(grid: GridWorld, start: Idx3, goal: Idx3) -> List[Idx3]:
    """
    3D A* on a GridWorld with 6-connected or 26-connected neighbors.

    Returns:
        List of grid indices from start to goal (inclusive).

    Raises:
        ValueError if start or goal are not free.
        RuntimeError if no path is found.
    """
    if not grid.is_free(start):
        raise ValueError(f"Start cell {start} is not free.")
    if not grid.is_free(goal):
        raise ValueError(f"Goal cell {goal} is not free.")

    # Open set as a min-heap of (f, g, idx)
    open_heap: List[Tuple[float, float, Idx3]] = []

    g_cost: Dict[Idx3, float] = {}
    parent: Dict[Idx3, Idx3] = {}
    closed: Set[Idx3] = set()

    g_cost[start] = 0.0
    f0 = _heuristic(start, goal, grid)
    heapq.heappush(open_heap, (f0, 0.0, start))

    found = False

    while open_heap:
        f, g, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            found = True
            break

        # Explore 6-connected neighbors (Uncomment to use 6-connected)
        # for nbr in grid.neighbors6(current):
        #     step_cost = grid.resolution  # all 6 moves are axis-aligned
        #     tentative_g = g + step_cost

        #     if tentative_g < g_cost.get(nbr, float("inf")):
        #         g_cost[nbr] = tentative_g
        #         parent[nbr] = current
        #         f_new = tentative_g + _heuristic(nbr, goal, grid)
        #         heapq.heappush(open_heap, (f_new, tentative_g, nbr))

        # Explore 26-connected neighbors
        for nbr in grid.neighbors26(current):
            step_cost = grid.resolution  # all 26 moves are axis-aligned
            tentative_g = g + step_cost

            if tentative_g < g_cost.get(nbr, float("inf")):
                g_cost[nbr] = tentative_g
                parent[nbr] = current
                f_new = tentative_g + _heuristic(nbr, goal, grid)
                heapq.heappush(open_heap, (f_new, tentative_g, nbr))

    if not found:
        raise RuntimeError("A*: no path found from start to goal.")

    # Reconstruct path from goal back to start
    path: List[Idx3] = []
    cur = goal
    path.append(cur)
    while cur != start:
        cur = parent[cur]
        path.append(cur)

    path.reverse()
    return path
