"""2D world geometry: rectangles, occupancy grid, ray casting.

The simulator is deterministic and pure Python: no rendering is required for
physics, and the grid is small (12 m / 0.1 m -> 120x120 cells) so A* is cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in metres."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"invalid rect: {self}")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            self.x0 - margin <= x <= self.x1 + margin
            and self.y0 - margin <= y <= self.y1 + margin
        )

    def inflate(self, margin: float) -> "Rect":
        return Rect(self.x0 - margin, self.y0 - margin, self.x1 + margin, self.y1 + margin)

    def to_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


def point_in_polygon(x: float, y: float, polygon: Iterable[tuple[float, float]]) -> bool:
    """Ray casting point-in-polygon; used by the safety geofence."""
    points = list(polygon)
    if len(points) < 3:
        return False
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


class OccupancyGrid:
    """Boolean occupancy grid with configurable inflation.

    Two grids are normally built per scene:

    * planning grid (inflation ~= robot radius + margin) for A*
    * collision grid (inflation ~= robot radius - margin of tolerance) for physics

    keeping them distinct avoids counting pure-pursuit tracking error as a crash.
    """

    def __init__(
        self,
        width: float,
        height: float,
        resolution: float,
        obstacles: Iterable[Rect],
        inflation: float = 0.0,
        wall_thickness: float = 0.15,
    ) -> None:
        self.width = float(width)
        self.height = float(height)
        self.resolution = float(resolution)
        self.inflation = float(inflation)
        self.nx = max(1, int(round(width / resolution)))
        self.ny = max(1, int(round(height / resolution)))
        self.blocked = bytearray(self.nx * self.ny)
        # Boundary walls
        wall = Rect(0.0, 0.0, wall_thickness, height)
        wall_r = Rect(width - wall_thickness, 0.0, width, height)
        wall_t = Rect(0.0, 0.0, width, wall_thickness)
        wall_b = Rect(0.0, height - wall_thickness, width, height)
        all_obstacles = [wall, wall_r, wall_t, wall_b, *obstacles]
        for rect in all_obstacles:
            self._mark_rect(rect.inflate(self.inflation))

    # ------------------------------------------------------------- indexing
    def world_to_index(self, x: float, y: float) -> tuple[int, int] | None:
        i = int(math.floor(x / self.resolution))
        j = int(math.floor(y / self.resolution))
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i, j
        return None

    def index_to_world(self, i: int, j: int) -> tuple[float, float]:
        return ((i + 0.5) * self.resolution, (j + 0.5) * self.resolution)

    def _mark_rect(self, rect: Rect) -> None:
        i0 = max(0, int(math.floor(rect.x0 / self.resolution)))
        i1 = min(self.nx - 1, int(math.floor(rect.x1 / self.resolution)))
        j0 = max(0, int(math.floor(rect.y0 / self.resolution)))
        j1 = min(self.ny - 1, int(math.floor(rect.y1 / self.resolution)))
        for j in range(j0, j1 + 1):
            base = j * self.nx
            for i in range(i0, i1 + 1):
                self.blocked[base + i] = 1

    # ---------------------------------------------------------------- query
    def is_free_index(self, i: int, j: int) -> bool:
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return self.blocked[j * self.nx + i] == 0
        return False

    def is_free(self, x: float, y: float) -> bool:
        idx = self.world_to_index(x, y)
        return self.is_free_index(*idx) if idx else False

    def clearance(self, x: float, y: float, max_clearance: float = 1.5) -> float:
        """Approximate distance to the nearest blocked cell (metres, capped)."""
        if not self.is_free(x, y):
            return 0.0
        steps = int(math.ceil(max_clearance / self.resolution))
        best = max_clearance
        i0, j0 = self.world_to_index(x, y) or (0, 0)
        for dj in range(-steps, steps + 1):
            j = j0 + dj
            if not 0 <= j < self.ny:
                continue
            for di in range(-steps, steps + 1):
                i = i0 + di
                if not 0 <= i < self.nx:
                    continue
                if self.blocked[j * self.nx + i]:
                    dist = math.hypot(di, dj) * self.resolution
                    if dist < best:
                        best = dist
        return best

    def nearest_free(self, x: float, y: float, max_radius: float = 1.2) -> tuple[float, float] | None:
        idx = self.world_to_index(x, y)
        if idx and self.is_free_index(*idx):
            return (x, y)
        i0, j0 = idx if idx else (int(x / self.resolution), int(y / self.resolution))
        max_r = int(math.ceil(max_radius / self.resolution))
        for radius in range(1, max_r + 1):
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    if max(abs(di), abs(dj)) != radius:
                        continue
                    i, j = i0 + di, j0 + dj
                    if self.is_free_index(i, j):
                        return self.index_to_world(i, j)
        return None

    def raycast(self, x0: float, y0: float, x1: float, y1: float, step: float | None = None) -> bool:
        """True if the straight segment is collision-free on this grid."""
        step = step or self.resolution * 0.5
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(dist / step)))
        for k in range(n + 1):
            t = k / n
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            if not self.is_free(x, y):
                return False
        return True

    def free_ratio(self) -> float:
        return 1.0 - sum(self.blocked) / float(len(self.blocked))

    def to_ascii(self, robot_xy: tuple[float, float] | None = None) -> str:
        """Small diagnostic rendering, useful when a test fails on the layout."""
        rx = ry = -1.0
        if robot_xy:
            rx, ry = robot_xy
        lines = []
        for j in range(self.ny - 1, -1, -1):
            row = []
            for i in range(self.nx):
                x, y = self.index_to_world(i, j)
                if math.hypot(x - rx, y - ry) < self.resolution * 0.7:
                    row.append("R")
                else:
                    row.append("#" if self.blocked[j * self.nx + i] else ".")
            lines.append("".join(row))
        return "\n".join(lines)
