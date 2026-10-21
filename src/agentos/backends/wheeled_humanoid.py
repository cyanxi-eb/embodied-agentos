"""Wheeled humanoid adapter (X1-class body).

What makes this body different from the other two:

* **Communication**: one navigation action ``/humanoid/nav_to`` plus one
  manipulation action ``/humanoid/manipulate``.
* **Action semantics**: the base is omnidirectional and can rotate in place, so
  the action takes a *relative body pose* ``(dx, dy, dyaw)`` instead of a world
  pose.  It may strafe, which the non-holonomic AMR in
  :mod:`agentos.backends.mobile_manipulator` cannot express.
* **Units / frame**: metres and radians in ``base_link`` - no unit conversion,
  only a frame transform from the world pose the skill layer speaks in.

The transform is intentionally the only geometry in this file: if a vendor's
action signature changes, this adapter changes and nothing above it does.
"""

from __future__ import annotations

from typing import Any

from agentos.backends.ros2_adapter import (
    Ros2AdapterBackend,
    adapter_failure,
    adapter_success,
)
from agentos.backends.ros2_transport import InterfaceSpec
from agentos.types import FailureCode, SkillResult, WorldState


class WheeledHumanoidBackend(Ros2AdapterBackend):
    """Omnidirectional wheeled humanoid (relative-body commands, m + rad)."""

    device_name = "wheeled_humanoid"
    display_name = "轮式人形（全向底盘 + 双臂，X1 类）"
    nav_action = "/humanoid/nav_to"
    manipulate_action = "/humanoid/manipulate"
    supports_navigation = True
    supports_manipulation = True
    interfaces = (
        InterfaceSpec("action", nav_action),
        InterfaceSpec("action", manipulate_action),
    )

    def __init__(
        self,
        transport: Any = None,
        world: WorldState | None = None,
        *,
        dock: tuple[float, float] = (0.0, 0.0),
        pick_height_m: float = 0.95,
        place_height_m: float = 0.85,
        **kwargs: Any,
    ) -> None:
        super().__init__(transport=transport, world=world, dock=dock, **kwargs)
        self.pick_height_m = float(pick_height_m)
        self.place_height_m = float(place_height_m)

    # ------------------------------------------------------------ navigation
    def _move_base(
        self,
        x: float,
        y: float,
        yaw: float | None,
        timeout_s: float,
        *,
        allow_reverse: bool = False,
    ) -> SkillResult:
        # The humanoid does not need a reverse strategy: it can strafe or rotate
        # in place, so ``allow_reverse`` is accepted and intentionally ignored.
        dx, dy, dyaw = self._body_delta(x, y, yaw)
        rotate_only = abs(dx) < 1e-3 and abs(dy) < 1e-3 and abs(dyaw) > 1e-3
        goal = {
            "frame_id": "base_link",
            "dx_m": round(dx, 6),
            "dy_m": round(dy, 6),
            "dyaw_rad": round(dyaw, 6),
            "rotate_in_place": rotate_only,
            "holonomic": True,
        }
        result = self._dispatch_action("navigate_to", self.nav_action, goal, timeout_s)
        if result.ok:
            self._commit_pose(x, y, yaw)
            self._last_feedback = {"goal": goal, "final_pose": self._world.robot.to_dict()}
        return result

    # -------------------------------------------------------- manipulation
    def _pick(self, object_id: str, timeout_s: float) -> SkillResult:
        obj = self._world.objects.get(object_id)
        if obj is None:
            return adapter_failure("pick", FailureCode.NOT_FOUND, f"object {object_id!r} not in world model")
        body_dx, body_dy, _ = self._body_delta(obj.x, obj.y, None)
        goal = {
            "command": "pick",
            "object_id": object_id,
            "frame_id": "base_link",
            "target_body_m": {"x": round(body_dx, 6), "y": round(body_dy, 6)},
            "height_m": self.pick_height_m,
            "color": obj.color,
            "category": obj.category,
        }
        result = self._dispatch_action("pick", self.manipulate_action, goal, timeout_s)
        if not result.ok:
            return result
        self._world.remove_object(object_id)
        self._world.holding = object_id
        self._world.held_meta = obj.to_dict()
        self._world.last_detection = None
        return adapter_success(
            "pick",
            f"grasped {object_id} at body frame ({body_dx:.2f}, {body_dy:.2f})m",
            object_id=object_id,
            color=obj.color,
            category=obj.category,
            target_body_m=[round(body_dx, 4), round(body_dy, 4)],
            backend=self.device_name,
        )

    def _place(self, x: float, y: float, zone: str, timeout_s: float) -> SkillResult:
        if not self._world.holding:
            return adapter_failure("place", FailureCode.NOT_HOLDING, "no object in gripper")
        meta = dict(self._world.held_meta or {})
        body_dx, body_dy, _ = self._body_delta(x, y, None)
        goal = {
            "command": "place",
            "frame_id": "base_link",
            "target_body_m": {"x": round(body_dx, 6), "y": round(body_dy, 6)},
            "height_m": self.place_height_m,
            "zone": zone,
        }
        result = self._dispatch_action("place", self.manipulate_action, goal, timeout_s)
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
        )
