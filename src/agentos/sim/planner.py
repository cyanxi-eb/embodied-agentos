"""A* global planner on the occupancy grid, plus path simplification."""

from __future__ import annotations

import heapq
import math
from typing import Iterable

from agentos.sim.world import OccupancyGrid

Point = tuple[float, float]


def _octile(ax: int, ay: int, bx: int, by: int) -> float:
    dx = abs(ax - bx)
    dy = abs(ay - by)
    return (dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy)


def astar(
    grid: OccupancyGrid,
    start: Point,
    goal: Point,
    max_expansions: int = 40000,
) -> list[Point] | None:
    """8-connected A* with corner-cut prevention.

    Returns a list of world points from (a free cell near) start to (a free cell
    near) goal, or ``None`` when no path exists.
    """
    start_free = grid.nearest_free(*start, max_radius=0.8)
    goal_free = grid.nearest_free(*goal, max_radius=0.8)
    if start_free is None or goal_free is None:
        return None
    si = grid.world_to_index(*start_free)
    gi = grid.world_to_index(*goal_free)
    if si is None or gi is None:
        return None
    if si == gi:
        return [start_free, goal_free]

    is_free_index = grid.is_free_index
    nx = grid.nx
    moves = (
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (-1, -1, math.sqrt(2.0)),
    )
    open_heap: list[tuple[float, float, int, int]] = []
    counter = 0
    heapq.heappush(open_heap, (_octile(si[0], si[1], gi[0], gi[1]), 0.0, si[0], si[1]))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {si: 0.0}
    closed: set[tuple[int, int]] = set()
    expansions = 0

    while open_heap:
        _, g, i, j = heapq.heappop(open_heap)
        if (i, j) in closed:
            continue
        closed.add((i, j))
        expansions += 1
        if expansions > max_expansions:
            return None
        if (i, j) == gi:
            return _reconstruct(grid, came_from, si, gi)
        for di, dj, cost in moves:
            ni, nj = i + di, j + dj
            if not is_free_index(ni, nj):
                continue
            if di and dj:
                # no corner cutting: both orthogonal neighbours must be free
                if not (is_free_index(i + di, j) and is_free_index(i, j + dj)):
                    continue
            ng = g + cost
            key = (ni, nj)
            if ng < g_score.get(key, math.inf) - 1e-12:
                g_score[key] = ng
                came_from[key] = (i, j)
                counter += 1
                heapq.heappush(open_heap, (ng + _octile(ni, nj, gi[0], gi[1]), ng, ni, nj))
    return None


def _reconstruct(
    grid: OccupancyGrid,
    came_from: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[Point]:
    cells = [goal]
    current = goal
    while current != start:
        current = came_from[current]
        cells.append(current)
    cells.reverse()
    path = [grid.index_to_world(i, j) for i, j in cells]
    return path


def simplify_path(
    grid: OccupancyGrid,
    path: list[Point],
    clearance_check: float | None = None,
    max_segment_m: float = 1.6,
) -> list[Point]:
    """Greedy line-of-sight simplification (keeps the path geometrically safe).

    ``max_segment_m`` caps how long a simplified segment may be: pure pursuit aims
    at path vertices, so very long segments would make the robot cut corners
    through inflated obstacle cells.
    """
    if len(path) <= 2:
        return list(path)
    keep = [path[0]]
    anchor = 0
    index = 2
    while index < len(path):
        blocked = False
        segment = math.hypot(path[index][0] - path[anchor][0], path[index][1] - path[anchor][1])
        if segment > max_segment_m:
            blocked = True
        if not blocked and clearance_check is not None:
            for k in range(anchor + 1, index + 1):
                if grid.clearance(path[k][0], path[k][1], max_clearance=clearance_check) < clearance_check:
                    blocked = True
                    break
        if not blocked and grid.raycast(*path[anchor], *path[index]):
            index += 1
            continue
        keep.append(path[index - 1])
        anchor = index - 1
        index = anchor + 2
    keep.append(path[-1])
    # de-duplicate consecutive points
    dedup: list[Point] = []
    for point in keep:
        if not dedup or math.hypot(point[0] - dedup[-1][0], point[1] - dedup[-1][1]) > 1e-6:
            dedup.append(point)
    return dedup


def path_length(path: Iterable[Point]) -> float:
    points = list(path)
    return sum(
        math.hypot(points[k + 1][0] - points[k][0], points[k + 1][1] - points[k][1])
        for k in range(len(points) - 1)
    )
