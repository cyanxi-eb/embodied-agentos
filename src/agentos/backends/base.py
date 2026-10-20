"""Robot backend interface.

A backend is the only component allowed to command hardware / the simulator.
The agent loop never calls ``cmd_vel`` directly: it selects a skill, the skill
calls the backend, and the backend's motion commands go through the safety
gateway.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from agentos.types import SkillResult, WorldState


class RobotBackend(ABC):
    """Common interface implemented by ``SimSkillBackend`` and ``Ros2SkillBackend``."""

    # --------------------------------------------------------------- world
    @property
    @abstractmethod
    def world(self) -> WorldState: ...

    @property
    @abstractmethod
    def sim_time(self) -> float: ...

    @property
    def active_goal(self) -> Any:
        return None

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Serializable world snapshot (used for checkpoint resume + traces)."""

    @abstractmethod
    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore from a snapshot produced by :meth:`snapshot`."""

    # ------------------------------------------------------------- commands
    @abstractmethod
    def navigate_to(self, x: float, y: float, yaw: float | None = None, timeout_s: float = 45.0) -> SkillResult: ...

    @abstractmethod
    def approach_object(self, object_id: str, standoff_m: float = 0.6, timeout_s: float = 45.0) -> SkillResult: ...

    @abstractmethod
    def pick(self, object_id: str, timeout_s: float = 20.0) -> SkillResult: ...

    @abstractmethod
    def place(self, x: float, y: float, zone: str = "", timeout_s: float = 20.0) -> SkillResult: ...

    @abstractmethod
    def speak(self, text: str) -> SkillResult: ...

    @abstractmethod
    def stop(self) -> SkillResult: ...

    @abstractmethod
    def dock(self, timeout_s: float = 60.0) -> SkillResult: ...

    @abstractmethod
    def retreat(self, distance_m: float = 0.35) -> SkillResult: ...

    # --------------------------------------------------------------- cancel
    @abstractmethod
    def cancel_active_goal(self) -> dict[str, Any]:
        """Cancel the in-flight ROS2 action / sim goal and clear its handle."""

    def set_frame_hook(self, hook: Callable[[WorldState, dict[str, Any]], None] | None) -> None:
        """Demo/eval hook called while the simulation steps (no-op for real robots)."""
        return None
