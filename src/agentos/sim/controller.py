"""Pure-pursuit path tracking for a differential-drive base.

The carrot point is computed **along the path** (project the robot onto the
nearest polyline segment, then walk the configured lookahead distance forward).
A naive implementation that aims directly at a distant path vertex cuts corners
through inflated obstacle cells at sharp turns - that failure mode was observed
during development and is documented in ``docs/failure_cases.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from agentos.types import Pose, normalize_angle

Point = tuple[float, float]


@dataclass
class PurePursuitConfig:
    lookahead_m: float = 0.6
    max_linear: float = 0.45
    max_angular: float = 1.0
    goal_tolerance_m: float = 0.3
    slow_radius_m: float = 1.1
    min_linear: float = 0.16
    max_cross_track_m: float = 0.9


class PurePursuit:
    """Stateful tracker: keeps the nearest path segment index between ticks."""

    def __init__(self, config: PurePursuitConfig | None = None) -> None:
        self.config = config or PurePursuitConfig()
        self._index = 0
        self.last_cross_track = 0.0

    def reset(self) -> None:
        self._index = 0
        self.last_cross_track = 0.0

    @property
    def index(self) -> int:
        return self._index

    # ------------------------------------------------------------- projection
    def _project(self, pose: Pose, path: Sequence[Point]) -> tuple[int, float, float, float]:
        """Nearest point on the polyline: ``(segment_index, t, x, y)``."""
        if len(path) == 1:
            return 0, 0.0, path[0][0], path[0][1]
        start = max(0, self._index - 2)
        end = min(len(path) - 2, self._index + 80)
        best: tuple[float, int, float, float, float] | None = None
        for k in range(start, end + 1):
            ax, ay = path[k]
            bx, by = path[k + 1]
            seg_dx, seg_dy = bx - ax, by - ay
            seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
            if seg_len_sq < 1e-12:
                continue
            t = ((pose.x - ax) * seg_dx + (pose.y - ay) * seg_dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            px, py = ax + seg_dx * t, ay + seg_dy * t
            distance = math.hypot(px - pose.x, py - pose.y)
            if best is None or distance < best[0]:
                best = (distance, k, t, px, py)
        if best is None:
            return 0, 0.0, path[0][0], path[0][1]
        distance, k, t, px, py = best
        self._index = k
        self.last_cross_track = distance
        return k, t, px, py

    def _carrot(
        self, px: float, py: float, segment_index: int, path: Sequence[Point], lookahead: float | None = None
    ) -> Point:
        remaining = self.config.lookahead_m if lookahead is None else lookahead
        k = segment_index
        while k < len(path) - 1:
            bx, by = path[k + 1]
            segment = math.hypot(bx - px, by - py)
            if segment <= 1e-9:
                k += 1
                continue
            if segment >= remaining:
                ratio = remaining / segment
                return (px + (bx - px) * ratio, py + (by - py) * ratio)
            remaining -= segment
            px, py = bx, by
            k += 1
        return path[-1]

    # ----------------------------------------------------------------- command
    def command(self, pose: Pose, path: Sequence[Point]) -> tuple[float, float, bool]:
        """Return ``(v, w, reached)`` for the current pose."""
        if not path:
            return 0.0, 0.0, True
        goal = path[-1]
        goal_dist = math.hypot(goal[0] - pose.x, goal[1] - pose.y)
        if goal_dist <= self.config.goal_tolerance_m:
            self._index = max(0, len(path) - 1)
            return 0.0, 0.0, True

        segment_index, _t, px, py = self._project(pose, path)
        # adaptive lookahead: shorten it when the carrot is far off-heading so the
        # vehicle tracks tight corners instead of cutting through inflated cells
        lookahead = self.config.lookahead_m
        carrot = self._carrot(px, py, segment_index, path, lookahead)
        alpha = normalize_angle(math.atan2(carrot[1] - pose.y, carrot[0] - pose.x) - pose.yaw)
        for _shrink in range(2):
            if abs(alpha) <= 0.35:
                break
            lookahead = max(0.18, lookahead * 0.55)
            carrot = self._carrot(px, py, segment_index, path, lookahead)
            alpha = normalize_angle(math.atan2(carrot[1] - pose.y, carrot[0] - pose.x) - pose.yaw)
        carrot_dist = max(0.05, math.hypot(carrot[0] - pose.x, carrot[1] - pose.y))
        curvature = 2.0 * math.sin(alpha) / carrot_dist

        if abs(alpha) > 1.1:
            # target is too far off-heading: rotate in place first
            v = 0.0
            omega = max(-self.config.max_angular, min(self.config.max_angular, 1.8 * math.sin(alpha)))
            return v, omega, False

        # slow down in curves and near the goal
        v = self.config.max_linear / (1.0 + 1.1 * abs(curvature))
        if goal_dist < self.config.slow_radius_m:
            v *= max(0.4, goal_dist / self.config.slow_radius_m)
        v = max(self.config.min_linear, min(self.config.max_linear, v))
        omega = max(-self.config.max_angular, min(self.config.max_angular, v * curvature))
        return v, omega, False

