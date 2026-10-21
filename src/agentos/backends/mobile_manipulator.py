"""Mobile manipulator adapter (non-holonomic AMR + collaborative arm).

What makes this body different from the other two:

* **Communication**: a navigation *action* ``/amr/navigate`` **and** an arm
  *service* ``/arm/move`` - two different ROS2 interaction patterns in one body.
* **Action semantics**: the base is non-holonomic (no strafing), so the adapter
  emits a world pose plus an explicit ``reverse`` flag and an ``align_first``
  hint.  A backward target is rejected unless the caller allows reverse: the
  adapter refuses to guess when the camera should face the travel direction.
* **Units / frame**: the AMR consumes **millimetres**; the skill layer and the
  world model speak metres.  The conversion happens exactly here, in both
  directions (goal out, ``reached_pose_mm`` feedback back in).

The arm commands are expressed in the AMR's ``base_link`` frame in millimetres,
because the arm is mounted on the moving base.
"""

from __future__ import annotations

import math
from typing import Any

from agentos.backends.ros2_adapter import (
    Ros2AdapterBackend,
    adapter_failure,
    adapter_success,
)
from agentos.backends.ros2_transport import InterfaceSpec
from agentos.types import FailureCode, Pose, SkillResult, WorldState, normalize_angle


def mm(value_m: float) -> float:
    """Metres -> millimetres, rounded so the wire payload is stable/testable."""
    return round(float(value_m) * 1000.0, 3)


def m(value_mm: float) -> float:
    """Millimetres -> metres (the direction of the robot's feedback)."""
    return float(value_mm) / 1000.0


class MobileManipulatorBackend(Ros2AdapterBackend):
    """AMR + arm: non-holonomic world-pose navigation in mm, arm via service."""

    device_name = "mobile_manipulator"
    display_name = "复合移动协作机器人（非全向 AMR + 协作臂）"
    nav_action = "/amr/navigate"
    arm_service = "/arm/move"
    supports_navigation = True
    supports_manipulation = True
    interfaces = (
        InterfaceSpec("action", nav_action),
        InterfaceSpec("service", arm_service),
    )

    def __init__(
        self,
        transport: Any = None,
        world: WorldState | None = None,
        *,
        dock: tuple[float, float] = (0.0, 0.0),
        arm_table_height_mm: float = 900.0,
        align_threshold_rad: float = 0.35,
        forward_tolerance_m: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(transport=transport, world=world, dock=dock, **kwargs)
        self.arm_table_height_mm = float(arm_table_height_mm)
        self.align_threshold_rad = float(align_threshold_rad)
        self.forward_tolerance_m = float(forward_tolerance_m)

    # ------------------------------------------------------------ navigation
    def navigate_to(
        self,
        x: float,
        y: float,
        yaw: float | None = None,
        timeout_s: float = 45.0,
        allow_reverse: bool = True,
    ) -> SkillResult:
        """World-pose navigation; backward targets need ``allow_reverse``.

        The public default allows reverse (a docking manoeuvre needs it), while
        callers that want the strict "always face the travel direction" policy
        pass ``allow_reverse=False`` and get a readable precondition failure.
        """
        if not self.supports_navigation:
            return self._unsupported("navigate_to", "mobile base / navigation")
        return self._move_base(
            float(x),
            float(y),
            None if yaw is None else float(yaw),
            float(timeout_s),
            allow_reverse=allow_reverse,
        )

    def _move_base(
        self,
        x: float,
        y: float,
        yaw: float | None,
        timeout_s: float,
        *,
        allow_reverse: bool = True,
    ) -> SkillResult:
        robot = self._world.robot
        dx = float(x) - robot.x
        dy = float(y) - robot.y
        distance = math.hypot(dx, dy)
        forward, lateral, dyaw = self._body_delta(x, y, yaw)
        reverse = forward < -self.forward_tolerance_m
        if reverse and not allow_reverse:
            return adapter_failure(
                "navigate_to",
                FailureCode.PRECONDITION_FAILED,
                "target is behind the non-holonomic base "
                f"(body_forward={forward:.2f}m, lateral={lateral:.2f}m); "
                "pass allow_reverse=True to authorise an explicit reverse approach",
                backend=self.device_name,
                body_forward_m=round(forward, 4),
                body_lateral_m=round(lateral, 4),
                requires_reverse=True,
            )

        heading = math.atan2(dy, dx) if distance > 1e-6 else robot.yaw
        align_first = (
            not reverse
            and abs(normalize_angle(heading - robot.yaw)) > self.align_threshold_rad
        )
        goal = {
            "frame_id": "map",
            "x_mm": mm(x),
            "y_mm": mm(y),
            "yaw_rad": None if yaw is None else round(normalize_angle(float(yaw)), 6),
            "reverse": reverse,
            "align_first": align_first,
            "holonomic": False,
        }
        result = self._dispatch_action("navigate_to", self.nav_action, goal, timeout_s)
        if not result.ok:
            return result

        # Robot -> adapter: the AMR reports its reached pose in millimetres.
        reached = result.data.get("reached_pose_mm")
        if isinstance(reached, dict) and "x_mm" in reached and "y_mm" in reached:
            reached_yaw = reached.get("yaw_rad", yaw if yaw is not None else robot.yaw)
            self._world.robot = Pose(
                m(reached["x_mm"]), m(reached["y_mm"]), normalize_angle(float(reached_yaw))
            )
        else:
            self._commit_pose(x, y, yaw)
        self._last_feedback = {
            "goal": goal,
            "final_pose": self._world.robot.to_dict(),
            "reverse": reverse,
        }
        return result

    # -------------------------------------------------------- manipulation
    def _arm_request(self, command: str, **fields: Any) -> dict[str, Any]:
        request = {
            "command": command,
            "frame_id": "base_link",
            "arm": "right",
        }
        request.update(fields)
        return request

    def _pick(self, object_id: str, timeout_s: float) -> SkillResult:
        obj = self._world.objects.get(object_id)
        if obj is None:
            return adapter_failure("pick", FailureCode.NOT_FOUND, f"object {object_id!r} not in world model")
        base_x, base_y = self._world_to_base(obj.x, obj.y)
        request = self._arm_request(
            "pick",
            object_id=object_id,
            x_mm=mm(base_x),
            y_mm=mm(base_y),
            z_mm=self.arm_table_height_mm,
            color=obj.color,
            category=obj.category,
        )
        result = self._dispatch_service("pick", self.arm_service, request, timeout_s)
        if not result.ok:
            return result
        self._world.remove_object(object_id)
        self._world.holding = object_id
        self._world.held_meta = obj.to_dict()
        self._world.last_detection = None
        return adapter_success(
            "pick",
            f"grasped {object_id} with the arm service at base frame ({base_x:.2f}, {base_y:.2f})m",
            object_id=object_id,
            color=obj.color,
            category=obj.category,
            base_target_mm=[mm(base_x), mm(base_y), self.arm_table_height_mm],
            backend=self.device_name,
        )

    def _place(self, x: float, y: float, zone: str, timeout_s: float) -> SkillResult:
        if not self._world.holding:
            return adapter_failure("place", FailureCode.NOT_HOLDING, "no object in gripper")
        meta = dict(self._world.held_meta or {})
        base_x, base_y = self._world_to_base(x, y)
        request = self._arm_request(
            "place",
            x_mm=mm(base_x),
            y_mm=mm(base_y),
            z_mm=self.arm_table_height_mm,
            zone=zone,
        )
        result = self._dispatch_service("place", self.arm_service, request, timeout_s)
        if not result.ok:
            return result
        object_id = str(meta.get("object_id", self._world.holding))
        self._world.add_object(self._make_placed_object(object_id, meta, x, y, zone))
        self._world.holding = None
        self._world.held_meta = None
        self._world.last_placed = {
            "object_id": object_id,
            "color": str(meta.get("color", "unknown")),
            "category": str(meta.get("category", "unknown")),
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "zone": zone,
        }
        return adapter_success(
            "place",
            f"placed {object_id} in zone {zone or 'n/a'}",
            object_id=object_id,
            drop=[round(float(x), 3), round(float(y), 3)],
            zone=zone,
            backend=self.device_name,
        )
