"""Deterministic geometric perception for the simulator.

Visibility model: an object is visible when it is inside the perception radius
and the straight segment robot->object is collision-free on the *physical*
occupancy grid (so shelves occlude objects behind them).  This is a cheap but
honest stand-in for a depth camera + detector, and it makes ``NOT_FOUND`` a real
geometric outcome instead of a scripted one.
"""

from __future__ import annotations

import math
import random
from typing import Any

from agentos.perception.base import PerceptionBackend
from agentos.sim.world import OccupancyGrid
from agentos.types import Detection, WorldState


class SimPerception(PerceptionBackend):
    def __init__(
        self,
        grid: OccupancyGrid,
        radius_m: float = 3.2,
        seed: int = 0,
        noise_m: float = 0.0,
        dropout_p: float = 0.0,
    ) -> None:
        self.grid = grid
        self.radius_m = float(radius_m)
        self.noise_m = float(noise_m)
        self.dropout_p = float(dropout_p)
        self._rng = random.Random(seed * 7919 + 17)

    # ------------------------------------------------------------------ core
    def _visible(self, world: WorldState, object_id: str, x: float, y: float) -> float | None:
        distance = math.hypot(x - world.robot.x, y - world.robot.y)
        if distance > self.radius_m:
            return None
        if not self.grid.raycast(world.robot.x, world.robot.y, x, y):
            return None
        return distance

    def detect_all(self, world: WorldState) -> list[Detection]:
        detections: list[Detection] = []
        for obj in world.free_objects():
            distance = self._visible(world, obj.object_id, obj.x, obj.y)
            if distance is None:
                continue
            if self.dropout_p and self._rng.random() < self.dropout_p:
                continue  # injected detection dropout
            x, y = obj.x, obj.y
            if self.noise_m:
                x += self._rng.gauss(0.0, self.noise_m)
                y += self._rng.gauss(0.0, self.noise_m)
            detections.append(
                Detection(
                    object_id=obj.object_id,
                    color=obj.color,
                    category=obj.category,
                    x=x,
                    y=y,
                    confidence=max(0.2, round(1.0 - 0.12 * distance, 3)),
                    source="sim-camera",
                    distance=round(distance, 3),
                )
            )
        detections.sort(key=lambda d: d.distance or 0.0)
        return detections

    def locate(self, query: dict[str, Any], world: WorldState) -> list[Detection]:
        matched = [d for d in self.detect_all(world) if _matches(d, query)]
        return matched

    def describe(self, world: WorldState) -> dict[str, Any]:
        detected = self.detect_all(world)
        return {
            "robot": world.robot.to_dict(),
            "holding": world.holding,
            "battery": round(world.battery, 2),
            "visible": [d.to_dict() for d in detected],
            "visible_colors": sorted({d.color for d in detected}),
            "visible_categories": sorted({d.category for d in detected}),
            "object_count": len(detected),
        }


def _matches(detection: Detection, query: dict[str, Any]) -> bool:
    color = query.get("color")
    category = query.get("category")
    if color and str(color).lower() != detection.color.lower():
        return False
    if category and str(category).lower() != detection.category.lower():
        return False
    return True
