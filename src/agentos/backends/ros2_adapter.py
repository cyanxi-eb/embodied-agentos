"""Shared behaviour for the ROS2 body adapters.

The three concrete bodies (``wheeled_humanoid`` / ``mobile_manipulator`` /
``arm6``) implement ``RobotBackend`` by translating the *same* skill calls into
different ROS2 interfaces, frames and units.  Everything that does not depend on
the body family lives here:

* the transport handle and the one-active-goal bookkeeping
* the world mirror that skill postconditions are evaluated against
* object approach / dock / retreat primitives built on ``_move_base``
* cancellation, snapshot/restore and the TTS/stop plumbing

Subclasses provide only the body-specific pieces:

``_move_base``  - how a world pose becomes this body's navigation goal
``_pick``       - how a grasp is expressed (body-relative pose, arm service,
                  joint trajectory, ...)
``_place``      - same for placement

That split is the code behind "换设备只换适配层": the task graph, planner,
skills, behavior tree and safety gateway never import these modules.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

from agentos.backends.base import RobotBackend
from agentos.backends.ros2_transport import (
    InterfaceSpec,
    RclpyTransport,
    TransportResult,
)
from agentos.types import FailureCode, GameObject, Pose, SkillResult, WorldState, normalize_angle


def adapter_success(skill: str, detail: str = "", **data: Any) -> SkillResult:
    """Successful skill result carrying a structured payload dict.

    ``SkillResult.success(..., data=...)`` folds keyword arguments into the
    payload, so an explicit dict has to go through the constructor.  Keeping that
    detail here makes the body adapters read as plain translations.
    """
    return SkillResult(skill=skill, ok=True, code=FailureCode.OK, detail=detail, data=dict(data))


def adapter_failure(skill: str, code: FailureCode, detail: str = "", **data: Any) -> SkillResult:
    """Failed skill result carrying a structured payload dict."""
    return SkillResult(skill=skill, ok=False, code=code, detail=detail, data=dict(data))


class Ros2AdapterBackend(RobotBackend):
    """Base class for the three body adapters."""

    #: Registry name; subclasses override.
    device_name = "generic_ros2"
    #: Human-readable body description (logs, docs, API).
    display_name = "generic ROS2 body"
    #: Logical ROS2 interfaces this body uses; deployment resolves the types.
    interfaces: tuple[InterfaceSpec, ...] = ()
    supports_navigation = True
    supports_manipulation = True
    tts_topic = "/agentos/tts"
    cmd_vel_topic = "/cmd_vel"

    def __init__(
        self,
        transport: Any = None,
        world: WorldState | None = None,
        *,
        dock: tuple[float, float] = (0.0, 0.0),
        clock: Callable[[], float] | None = None,
    ) -> None:
        # ``transport=None`` means "the real robot": construct the guarded rclpy
        # transport, which raises a readable error when ROS2 is unavailable.
        self.transport = transport if transport is not None else RclpyTransport(interfaces=self.interfaces)
        self._world = world if world is not None else WorldState(dock=tuple(dock))
        self._active_goal: dict[str, Any] | None = None
        self._active_kind: str | None = None
        self._last_feedback: dict[str, Any] = {}
        self._clock = clock or time.monotonic

    # --------------------------------------------------------------- world API
    @property
    def world(self) -> WorldState:
        return self._world

    @property
    def sim_time(self) -> float:
        return float(self._clock())

    @property
    def active_goal(self) -> Any:
        return self._active_goal

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": self.device_name,
            "world": self._world.to_dict(),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """Restore the local world mirror.

        The authoritative pose still lives on the robot; a real deployment should
        re-seed this mirror from ``/odom`` + TF after a restart.  For the offline
        fake transport the mirror is the whole state.
        """
        world_data = snapshot.get("world") if isinstance(snapshot, dict) else None
        if world_data:
            self._world = WorldState.from_dict(world_data)
        self._active_goal = None
        self._active_kind = None

    # ------------------------------------------------------------ dispatch
    def _dispatch_action(self, skill: str, name: str, goal: dict[str, Any], timeout_s: float) -> SkillResult:
        self._active_goal = {"interface": name, "kind": "action", "goal": dict(goal)}
        self._active_kind = name
        try:
            result = self.transport.send_action(name, goal, timeout_s)
        except Exception as error:  # a broken transport must not crash the agent loop
            return adapter_failure(
                skill,
                FailureCode.BACKEND_ERROR,
                f"{self.device_name}: transport raised {type(error).__name__}: {error}",
            )
        finally:
            self._active_goal = None
            self._active_kind = None
        return self._skill_result(skill, result)

    def _dispatch_service(
        self, skill: str, name: str, request: dict[str, Any], timeout_s: float
    ) -> SkillResult:
        self._active_goal = {"interface": name, "kind": "service", "goal": dict(request)}
        self._active_kind = name
        try:
            result = self.transport.call_service(name, request, timeout_s)
        except Exception as error:
            return adapter_failure(
                skill,
                FailureCode.BACKEND_ERROR,
                f"{self.device_name}: transport raised {type(error).__name__}: {error}",
            )
        finally:
            self._active_goal = None
            self._active_kind = None
        return self._skill_result(skill, result)

    def _skill_result(self, skill: str, result: TransportResult) -> SkillResult:
        data = {"backend": self.device_name, **dict(result.data)}
        if result.ok:
            return adapter_success(skill, result.detail or f"{self.device_name}: {skill} ok", **data)
        return adapter_failure(skill, result.code, result.detail, **data)

    def _unsupported(self, skill: str, capability: str) -> SkillResult:
        return adapter_failure(
            skill,
            FailureCode.UNSUPPORTED,
            f"{self.display_name} has no {capability}; this skill needs a body that provides it "
            "(the task graph is unchanged, only the adapter layer differs)",
            backend=self.device_name,
            capability=capability,
            supported=False,
        )

    # ----------------------------------------------------------- geometry
    def _body_delta(self, x: float, y: float, yaw: float | None) -> tuple[float, float, float]:
        """World target -> (forward, lateral, dyaw) in the robot's body frame."""
        robot = self._world.robot
        dx = float(x) - robot.x
        dy = float(y) - robot.y
        cos_yaw, sin_yaw = math.cos(robot.yaw), math.sin(robot.yaw)
        forward = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy
        dyaw = 0.0 if yaw is None else normalize_angle(float(yaw) - robot.yaw)
        return forward, lateral, dyaw

    def _world_to_base(self, x: float, y: float, base: Pose | None = None) -> tuple[float, float]:
        """World metres -> body/base-frame metres (2D, no z).

        ``base=None`` means the body itself (the AMR's ``base_link``); a fixed
        arm passes its configured base pose explicitly.
        """
        origin = base if base is not None else self._world.robot
        dx = float(x) - origin.x
        dy = float(y) - origin.y
        cos_yaw, sin_yaw = math.cos(origin.yaw), math.sin(origin.yaw)
        return cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy

    def _commit_pose(self, x: float, y: float, yaw: float | None) -> None:
        robot = self._world.robot
        self._world.robot = Pose(
            float(x),
            float(y),
            robot.yaw if yaw is None else normalize_angle(float(yaw)),
        )

    @staticmethod
    def _make_placed_object(
        object_id: str, meta: dict[str, Any], x: float, y: float, zone: str
    ) -> GameObject:
        return GameObject(
            object_id=object_id,
            color=str(meta.get("color", "unknown")),
            category=str(meta.get("category", "unknown")),
            x=round(float(x), 3),
            y=round(float(y), 3),
            zone=zone,
        )

    # --------------------------------------------------------- primitives
    def _move_base(
        self,
        x: float,
        y: float,
        yaw: float | None,
        timeout_s: float,
        *,
        allow_reverse: bool = False,
    ) -> SkillResult:
        raise NotImplementedError

    def _pick(self, object_id: str, timeout_s: float) -> SkillResult:
        raise NotImplementedError

    def _place(self, x: float, y: float, zone: str, timeout_s: float) -> SkillResult:
        raise NotImplementedError

    # ---------------------------------------------------------- navigation
    def navigate_to(self, x: float, y: float, yaw: float | None = None, timeout_s: float = 45.0) -> SkillResult:
        if not self.supports_navigation:
            return self._unsupported("navigate_to", "mobile base / navigation")
        return self._move_base(float(x), float(y), None if yaw is None else float(yaw), float(timeout_s))

    def approach_object(self, object_id: str, standoff_m: float = 0.6, timeout_s: float = 45.0) -> SkillResult:
        if not self.supports_navigation:
            return self._unsupported("approach_object", "mobile base / approach navigation")
        obj = self._world.objects.get(object_id)
        if obj is None:
            return adapter_failure(
                "approach_object", FailureCode.NOT_FOUND, f"object {object_id!r} not in world model"
            )
        robot = self._world.robot
        dx, dy = robot.x - obj.x, robot.y - obj.y
        bearing = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-6 else 0.0
        target_x = obj.x + standoff_m * math.cos(bearing)
        target_y = obj.y + standoff_m * math.sin(bearing)
        approach_yaw = math.atan2(obj.y - target_y, obj.x - target_x)
        # The approach pose is a docking manoeuvre: reversing is a legitimate
        # strategy for the non-holonomic body, so it is explicitly allowed here.
        navigation = self._move_base(
            target_x, target_y, approach_yaw, float(timeout_s), allow_reverse=True
        )
        if not navigation.ok:
            return adapter_failure(
                "approach_object",
                navigation.code,
                f"approach navigation failed: {navigation.detail}",
                nav=navigation.to_dict(),
            )
        distance = math.hypot(self._world.robot.x - obj.x, self._world.robot.y - obj.y)
        self._world.last_approached = {
            "object_id": object_id,
            "color": obj.color,
            "category": obj.category,
            "x": round(obj.x, 3),
            "y": round(obj.y, 3),
            "distance": round(distance, 3),
        }
        return adapter_success(
            "approach_object",
            f"approached {object_id} at {distance:.2f}m ({self.device_name})",
            object_id=object_id,
            distance=round(distance, 3),
            standoff=standoff_m,
        )

    # -------------------------------------------------------- manipulation
    def pick(self, object_id: str, timeout_s: float = 20.0) -> SkillResult:
        if not self.supports_manipulation:
            return self._unsupported("pick", "manipulation")
        return self._pick(str(object_id), float(timeout_s))

    def place(self, x: float, y: float, zone: str = "", timeout_s: float = 20.0) -> SkillResult:
        if not self.supports_manipulation:
            return self._unsupported("place", "manipulation")
        return self._place(float(x), float(y), str(zone), float(timeout_s))

    # -------------------------------------------------------------- skills
    def speak(self, text: str) -> SkillResult:
        payload = {"data": str(text)}
        try:
            self.transport.publish(self.tts_topic, payload)
        except Exception as error:
            return adapter_failure(
                "speak",
                FailureCode.BACKEND_ERROR,
                f"{self.device_name}: TTS publish failed: {type(error).__name__}: {error}",
            )
        self._world.last_spoken = str(text)
        return adapter_success(
            "speak", f"published {len(text)} chars to {self.tts_topic}", text=text
        )

    def stop(self) -> SkillResult:
        cleanup = self.cancel_active_goal()
        try:
            self.transport.publish(
                self.cmd_vel_topic,
                {
                    "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
            )
        except Exception as error:
            return adapter_failure(
                "stop",
                FailureCode.BACKEND_ERROR,
                f"{self.device_name}: zero-velocity publish failed: {type(error).__name__}: {error}",
                cleanup=cleanup,
            )
        self._world.v = 0.0
        self._world.w = 0.0
        return adapter_success(
            "stop", "zero velocity published and active goal cancelled", cleanup=cleanup
        )

    def dock(self, timeout_s: float = 60.0) -> SkillResult:
        if not self.supports_navigation:
            return self._unsupported("dock", "mobile base / docking")
        if self._world.holding:
            return adapter_failure(
                "dock", FailureCode.PRECONDITION_FAILED, "cannot dock while holding an object"
            )
        dock_x, dock_y = self._world.dock
        navigation = self._move_base(
            float(dock_x), float(dock_y), None, float(timeout_s), allow_reverse=True
        )
        if not navigation.ok:
            return adapter_failure(
                "dock",
                navigation.code,
                f"dock navigation failed: {navigation.detail}",
                nav=navigation.to_dict(),
            )
        self._world.docked = True
        self._world.battery = 100.0
        return adapter_success(
            "dock", "docked and charging", pose=self._world.robot.to_dict()
        )

    def retreat(self, distance_m: float = 0.35) -> SkillResult:
        if not self.supports_navigation:
            return self._unsupported("retreat", "mobile base / retreat")
        robot = self._world.robot
        target_x = robot.x - float(distance_m) * math.cos(robot.yaw)
        target_y = robot.y - float(distance_m) * math.sin(robot.yaw)
        navigation = self._move_base(target_x, target_y, robot.yaw, 10.0, allow_reverse=True)
        if not navigation.ok:
            return adapter_failure(
                "retreat",
                navigation.code,
                f"retreat failed: {navigation.detail}",
                nav=navigation.to_dict(),
            )
        return adapter_success(
            "retreat", f"retreated {distance_m:.2f}m", moved=round(float(distance_m), 3)
        )

    # ------------------------------------------------------------- cancel
    def cancel_active_goal(self) -> dict[str, Any]:
        goal = self._active_goal
        self._active_goal = None
        self._active_kind = None
        if goal is None:
            return {"cancelled": False, "was_active": False, "handle_cleared": True}
        try:
            response = self.transport.cancel(str(goal["interface"]))
        except Exception as error:
            response = {"cancelled": False, "error": f"{type(error).__name__}: {error}"}
        return {
            "cancelled": bool(response.get("cancelled", False)),
            "was_active": True,
            "interface": goal["interface"],
            "goal": goal,
            "handle_cleared": bool(response.get("handle_cleared", True)),
        }
