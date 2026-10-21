"""Fixed-base 6-axis collaborative arm adapter.

What makes this body different from the other two:

* **Communication**: a joint-trajectory action ``/arm/follow_joint_trajectory``
  plus a gripper action ``/arm/gripper``.  There is no mobile base at all.
* **Action semantics**: navigation-shaped skills (``navigate_to`` / ``dock`` /
  ``retreat`` / ``approach_object``) are not approximated - they return a
  readable ``UNSUPPORTED`` failure so the graph can report it instead of
  pretending the arm drove somewhere.
* **Units / frame**: trajectories are **joint space in radians**; pick/place
  targets are expressed in the arm's **base frame** and checked against the
  configured reach.  There is no IK here: a deployed vendor action server does
  that, and this adapter sends the configured joint waypoints plus the
  base-frame target it validated.

The point of the file is the *negative* capability: "same task graph, only the
adapter differs" must also mean "a skill the body does not have fails cleanly".
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from agentos.backends.ros2_adapter import (
    Ros2AdapterBackend,
    adapter_failure,
    adapter_success,
)
from agentos.backends.ros2_transport import InterfaceSpec
from agentos.types import FailureCode, Pose, SkillResult, WorldState


class Arm6Backend(Ros2AdapterBackend):
    """6-axis arm on a fixed base: joint-space motion, no navigation."""

    device_name = "arm6"
    display_name = "六轴协作臂（固定基座）"
    joint_action = "/arm/follow_joint_trajectory"
    gripper_action = "/arm/gripper"
    supports_navigation = False
    supports_manipulation = True
    joint_names = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    home_joints = (0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0)
    pick_joints = (0.0, -1.15, 0.65, -1.10, 0.0, 0.0)
    place_joints = (-0.65, -1.05, 0.85, -1.35, 0.0, 0.0)
    interfaces = (
        InterfaceSpec("action", joint_action, "control_msgs.action.FollowJointTrajectory"),
        InterfaceSpec("action", gripper_action, "control_msgs.action.GripperCommand"),
    )

    def __init__(
        self,
        transport: Any = None,
        world: WorldState | None = None,
        *,
        dock: tuple[float, float] = (0.0, 0.0),
        base_pose: Pose | None = None,
        reach_m: float = 1.2,
        gripper_max_effort: float = 40.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(transport=transport, world=world, dock=dock, **kwargs)
        # A fixed arm's world pose is its base; reuse the mirror unless overridden.
        self.base_pose = base_pose or Pose(
            self._world.robot.x, self._world.robot.y, self._world.robot.yaw
        )
        self.reach_m = float(reach_m)
        self.gripper_max_effort = float(gripper_max_effort)
        self._joints: tuple[float, ...] = tuple(self.home_joints)

    # ------------------------------------------------------- joint-space API
    @property
    def joints(self) -> tuple[float, ...]:
        return self._joints

    def snapshot(self) -> dict[str, Any]:
        data = super().snapshot()
        data["joints_rad"] = list(self._joints)
        return data

    def restore(self, snapshot: dict[str, Any]) -> None:
        super().restore(snapshot)
        joints = snapshot.get("joints_rad") if isinstance(snapshot, dict) else None
        if joints:
            self._joints = tuple(float(value) for value in joints)

    def _unsupported(self, skill: str, capability: str) -> SkillResult:
        return adapter_failure(
            skill,
            FailureCode.UNSUPPORTED,
            f"{self.display_name} is a fixed base and has no {capability}; "
            "this skill needs a mobile body (the task graph is unchanged, only the adapter layer differs)",
            backend=self.device_name,
            capability=capability,
            supported=False,
            fixed_base=True,
        )

    def move_joints(
        self,
        positions: Sequence[float],
        timeout_s: float = 10.0,
        *,
        velocities: Sequence[float] | None = None,
        time_from_start_s: float = 2.0,
    ) -> SkillResult:
        """Command all six joints in radians (the arm's native capability)."""
        try:
            targets = [float(value) for value in positions]
        except (TypeError, ValueError):
            return adapter_failure(
                "move_joints", FailureCode.PRECONDITION_FAILED, "joint positions must be numbers (radians)"
            )
        if len(targets) != len(self.joint_names):
            return adapter_failure(
                "move_joints",
                FailureCode.PRECONDITION_FAILED,
                f"expected {len(self.joint_names)} joint positions, got {len(targets)}",
            )
        if any(not math.isfinite(value) for value in targets):
            return adapter_failure(
                "move_joints", FailureCode.PRECONDITION_FAILED, "joint positions must be finite radians"
            )
        velocity_list = [0.0] * len(targets)
        if velocities is not None:
            velocity_list = [float(value) for value in velocities]
            if len(velocity_list) != len(targets):
                return adapter_failure(
                    "move_joints",
                    FailureCode.PRECONDITION_FAILED,
                    "velocities must have the same length as positions",
                )
        goal = {
            "joint_names": list(self.joint_names),
            "joint_space": True,
            "units": "rad",
            "points": [
                {
                    "positions": [round(value, 6) for value in targets],
                    "velocities": [round(value, 6) for value in velocity_list],
                    "time_from_start_s": float(time_from_start_s),
                }
            ],
        }
        result = self._dispatch_action("move_joints", self.joint_action, goal, float(timeout_s))
        if result.ok:
            self._joints = tuple(targets)
        return result

    def goto_home(self, timeout_s: float = 10.0) -> SkillResult:
        return self.move_joints(self.home_joints, timeout_s)

    # -------------------------------------------------------- manipulation
    def world_to_base(self, x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
        """World metres -> arm base frame metres (2D rotation around the base)."""
        base_x, base_y = self._world_to_base(x, y, self.base_pose)
        return base_x, base_y, float(z)

    def _reach_check(self, skill: str, base_x: float, base_y: float) -> SkillResult | None:
        distance = math.hypot(base_x, base_y)
        if distance > self.reach_m:
            return adapter_failure(
                skill,
                FailureCode.OUT_OF_BOUNDS,
                f"target is {distance:.2f}m from the arm base, beyond the {self.reach_m:.2f}m reach",
                base_target_m=[round(base_x, 4), round(base_y, 4)],
                reach_m=self.reach_m,
            )
        return None

    def _gripper(self, position: float, timeout_s: float, skill: str) -> SkillResult:
        return self._dispatch_action(
            skill,
            self.gripper_action,
            {
                "command": "close" if position <= 0.0 else "open",
                "position": float(position),
                "max_effort": self.gripper_max_effort,
            },
            timeout_s,
        )

    def _pick(self, object_id: str, timeout_s: float) -> SkillResult:
        obj = self._world.objects.get(object_id)
        if obj is None:
            return adapter_failure("pick", FailureCode.NOT_FOUND, f"object {object_id!r} not in world model")
        base_x, base_y, base_z = self.world_to_base(obj.x, obj.y)
        out_of_reach = self._reach_check("pick", base_x, base_y)
        if out_of_reach is not None:
            return out_of_reach
        move = self.move_joints(self.pick_joints, timeout_s=timeout_s)
        if not move.ok:
            return adapter_failure(
                "pick", move.code, f"joint-space move failed: {move.detail}", **dict(move.data)
            )
        grip = self._gripper(0.0, min(10.0, timeout_s), "pick")
        if not grip.ok:
            return adapter_failure(
                "pick", grip.code, f"gripper close failed: {grip.detail}", **dict(grip.data)
            )
        self._world.remove_object(object_id)
        self._world.holding = object_id
        self._world.held_meta = obj.to_dict()
        self._world.last_detection = None
        return adapter_success(
            "pick",
            f"grasped {object_id} at arm base frame ({base_x:.2f}, {base_y:.2f})m",
            object_id=object_id,
            color=obj.color,
            category=obj.category,
            base_target_m=[round(base_x, 4), round(base_y, 4), round(base_z, 4)],
            joints_rad=list(self._joints),
            backend=self.device_name,
        )

    def _place(self, x: float, y: float, zone: str, timeout_s: float) -> SkillResult:
        if not self._world.holding:
            return adapter_failure("place", FailureCode.NOT_HOLDING, "no object in gripper")
        meta = dict(self._world.held_meta or {})
        base_x, base_y, base_z = self.world_to_base(x, y)
        out_of_reach = self._reach_check("place", base_x, base_y)
        if out_of_reach is not None:
            return out_of_reach
        move = self.move_joints(self.place_joints, timeout_s=timeout_s)
        if not move.ok:
            return adapter_failure(
                "place", move.code, f"joint-space move failed: {move.detail}", **dict(move.data)
            )
        grip = self._gripper(0.04, min(10.0, timeout_s), "place")
        if not grip.ok:
            return adapter_failure(
                "place", grip.code, f"gripper open failed: {grip.detail}", **dict(grip.data)
            )
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
            base_target_m=[round(base_x, 4), round(base_y, 4), round(base_z, 4)],
            joints_rad=list(self._joints),
            zone=zone,
            backend=self.device_name,
        )
