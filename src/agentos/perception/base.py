"""Abstract interfaces for perception and vision-language backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentos.types import Detection, WorldState


class PerceptionBackend(ABC):
    """Geometric perception (what the robot can see right now)."""

    @abstractmethod
    def detect_all(self, world: WorldState) -> list[Detection]:
        """All currently visible detections, nearest first."""

    @abstractmethod
    def locate(self, query: dict[str, Any], world: WorldState) -> list[Detection]:
        """Visible detections matching ``{color?, category?}``, nearest first."""

    @abstractmethod
    def describe(self, world: WorldState) -> dict[str, Any]:
        """Compact structured scene description for traces / prompts."""


class VLMBackend(ABC):
    """Vision-language backend (Qwen2.5-VL style).

    ``image`` is an optional image reference (path/bytes/base64).  The offline
    simulator backend ignores it and reads ground truth; a real backend would
    consume the RGB frame from the ROS2 camera bridge.
    """

    name: str = "base"

    @abstractmethod
    def locate(self, query: dict[str, Any], world: WorldState, image: Any = None) -> list[Detection]:
        """Grounded object locations for a natural-language-ish query."""

    @abstractmethod
    def answer(self, question: str, world: WorldState, image: Any = None) -> str:
        """Natural-language scene question answering."""

    @abstractmethod
    def describe_scene(self, world: WorldState, image: Any = None) -> dict[str, Any]:
        """Structured scene description (for traces and for the planner prompt)."""
