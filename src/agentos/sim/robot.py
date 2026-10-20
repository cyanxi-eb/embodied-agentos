"""Differential-drive robot model (unicycle, velocity/omega limited)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from agentos.types import Pose, normalize_angle


@dataclass
class DiffDriveRobot:
    pose: Pose
    radius_m: float = 0.24
    max_linear: float = 0.5
    max_angular: float = 1.0
    min_linear: float = -0.2

    def clamp(self, v: float, w: float) -> tuple[float, float]:
        """Defense in depth: the gateway already clamps, the model re-checks."""
        v = max(self.min_linear, min(self.max_linear, v))
        w = max(-self.max_angular, min(self.max_angular, w))
        return v, w

    def step(self, v: float, w: float, dt: float) -> Pose:
        v, w = self.clamp(v, w)
        self.pose = Pose(
            x=self.pose.x + v * math.cos(self.pose.yaw) * dt,
            y=self.pose.y + v * math.sin(self.pose.yaw) * dt,
            yaw=normalize_angle(self.pose.yaw + w * dt),
        )
        return self.pose
