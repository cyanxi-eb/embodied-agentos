"""ROS2 (rclpy) skill backend - the real-robot adapter.

Import-guarded on purpose: the development host is Windows without ROS2, so
``ROS2_AVAILABLE`` is ``False`` here and importing this module must never fail.
On Ubuntu 24.04 + ROS2 Jazzy it is the backend used instead of
``SimSkillBackend``; the agent runtime and skill layer do not change.

Not executed on the development host.  See ``docs/ros2_integration.md``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agentos.backends.base import RobotBackend
from agentos.types import FailureCode, Pose, SkillResult, WorldState

ROS2_AVAILABLE = False
ROS2_IMPORT_ERROR = ""

try:  # pragma: no cover - exercised only on a ROS2 host
    import rclpy
    from geometry_msgs.msg import Twist
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float32, String

    try:
        from agentos_interfaces.action import ExecuteSkill
        from agentos_interfaces.srv import ResetEstop

        AGENTOS_INTERFACES_AVAILABLE = True
    except Exception:  # interfaces not built yet
        ExecuteSkill = None  # type: ignore[assignment]
        ResetEstop = None  # type: ignore[assignment]
        AGENTOS_INTERFACES_AVAILABLE = False

    ROS2_AVAILABLE = True
except Exception as error:  # pragma: no cover - Windows/macOS development host
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment,misc]
    ActionClient = object  # type: ignore[assignment]
    NavigateToPose = None  # type: ignore[assignment]
    Twist = None  # type: ignore[assignment]
    Float32 = None  # type: ignore[assignment]
    String = None  # type: ignore[assignment]
    QoSProfile = object  # type: ignore[assignment]
    ExecuteSkill = None  # type: ignore[assignment]
    ResetEstop = None  # type: ignore[assignment]
    AGENTOS_INTERFACES_AVAILABLE = False
    ROS2_IMPORT_ERROR = f"{type(error).__name__}: {error}"


@dataclass
class Ros2Config:
    node_name: str = "agentos_skill_backend"
    nav_action: str = "navigate_to_pose"  # nav2_msgs/action/NavigateToPose
    skill_action: str = "agentos/execute_skill"  # agentos_interfaces/action/ExecuteSkill
    estop_service: str = "agentos/reset_estop"  # agentos_interfaces/srv/ResetEstop
    cmd_vel_topic: str = "cmd_vel"  # only used for stop(); the agent never streams it
    battery_topic: str = "battery_state"  # sensor_msgs/BatteryState in the real deploy
    odom_topic: str = "odom"
    action_server_timeout_s: float = 5.0
    default_timeout_s: float = 45.0
    max_linear: float = 0.5
    max_angular: float = 1.0
    qos_depth: int = 10
    qos_reliability: str = "reliable"
    qos_durability: str = "volatile"


class Ros2SkillBackend(RobotBackend):
    """Adapter from the AgentOS skill interface to Nav2 + robot skill actions.

    Only one goal may be active at a time: ``cancel_active_goal`` cancels the
    pending goal handle and clears the internal dictionary so a cancelled skill
    can never leak a stale goal into the next dispatch (the exact failure mode
    described in the resume: "技能取消时统一清理 Action goal 与黑板状态").
    """

    def __init__(self, node: Any = None, config: Ros2Config | None = None) -> None:
        if not ROS2_AVAILABLE:
            raise RuntimeError(
                "rclpy is not available on this host; the ROS2 backend requires Ubuntu 24.04 + ROS2 Jazzy. "
                f"Import error: {ROS2_IMPORT_ERROR}"
            )
        self.config = config or Ros2Config()
        self._owns_node = node is None
        self.node = node or Node(self.config.node_name)
        self._world = WorldState(
            battery=100.0,
            zones={},
            dock=(0.0, 0.0),
        )
        self._active_goal: Any = None
        self._active_kind: str | None = None
        self._last_feedback: dict[str, Any] = {}
        self._skill_client = None
        self._nav_client = None
        self._estop_client = None
        self._cmd_vel_publisher = None
        self._estop_engaged = False
        self._setup_qos()
        self._setup_clients()

    # ---------------------------------------------------------------- setup
    def _qos(self) -> Any:
        reliability = (
            ReliabilityPolicy.RELIABLE
            if self.config.qos_reliability == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        )
        durability = (
            DurabilityPolicy.TRANSIENT_LOCAL
            if self.config.qos_durability == "transient_local"
            else DurabilityPolicy.VOLATILE
        )
        return QoSProfile(
            depth=self.config.qos_depth,
            history=HistoryPolicy.KEEP_LAST,
            reliability=reliability,
            durability=durability,
        )

    def _setup_qos(self) -> None:
        self._qos_profile = self._qos()

    def _setup_clients(self) -> None:
        self._nav_client = ActionClient(self.node, NavigateToPose, self.config.nav_action)
        if AGENTOS_INTERFACES_AVAILABLE:
            self._skill_client = ActionClient(self.node, ExecuteSkill, self.config.skill_action)
            self._estop_client = self.node.create_client(ResetEstop, self.config.estop_service)
        else:
            self.node.get_logger().warn(
                "agentos_interfaces not found: build ros2_ws/src/agentos_interfaces first; "
                "only navigation will work"
            )
        self._cmd_vel_publisher = self.node.create_publisher(Twist, self.config.cmd_vel_topic, self._qos_profile)

    # ---------------------------------------------------------------- world
    @property
    def world(self) -> WorldState:
        return self._world

    @property
    def sim_time(self) -> float:
        return time.monotonic()

    @property
    def active_goal(self) -> Any:
        return self._active_goal

    def snapshot(self) -> dict[str, Any]:
        return {"backend": "ros2", "world": self._world.to_dict(), "estop": self._estop_engaged}

    def restore(self, snapshot: dict[str, Any]) -> None:
        """ROS2 world state lives on the robot; restoring is not supported (documented)."""
        raise NotImplementedError("Ros2SkillBackend cannot restore world state from a checkpoint")

    # ------------------------------------------------------------- internals
    def _spin_until(self, future: Any, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return future.done()

    def _cancel_handle(self) -> dict[str, Any]:
        cancelled = False
        if self._active_goal is not None:
            try:
                self._active_goal.cancel_goal_async()
                cancelled = True
            except Exception as error:  # pragma: no cover - hardware dependent
                self.node.get_logger().error(f"cancel failed: {error}")
        self._active_goal = None
        self._active_kind = None
        self._last_feedback = {}
        return {"cancelled": cancelled, "handle_cleared": True}

    def cancel_active_goal(self) -> dict[str, Any]:
        return self._cancel_handle()

    # ------------------------------------------------------------ navigation
    def navigate_to(self, x: float, y: float, yaw: float | None = None, timeout_s: float = 45.0) -> SkillResult:
        if self._estop_engaged:
            return SkillResult.failure("navigate_to", FailureCode.ESTOP, "e-stop engaged")
        if not self._nav_client.wait_for_server(timeout_sec=self.config.action_server_timeout_s):
            return SkillResult.failure("navigate_to", FailureCode.BACKEND_ERROR, "Nav2 action server unavailable")
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        yaw = 0.0 if yaw is None else float(yaw)
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        send_future = self._nav_client.send_goal_async(goal, feedback_callback=self._nav_feedback)
        if not self._spin_until(send_future, self.config.action_server_timeout_s):
            return SkillResult.failure("navigate_to", FailureCode.BACKEND_ERROR, "send_goal timed out")
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return SkillResult.failure("navigate_to", FailureCode.BLOCKED, "Nav2 rejected the goal")
        self._active_goal = goal_handle
        self._active_kind = "navigate_to"
        result_future = goal_handle.get_result_async()
        started = time.monotonic()
        try:
            while not result_future.done():
                if time.monotonic() - started > timeout_s:
                    self._cancel_handle()
                    return SkillResult.failure("navigate_to", FailureCode.TIMEOUT, f"Nav2 timeout after {timeout_s}s")
                rclpy.spin_once(self.node, timeout_sec=0.05)
            wrapped = result_future.result()
            status = wrapped.status
            if status == 4:  # GoalStatus.STATUS_SUCCEEDED
                return SkillResult.success(
                    "navigate_to", "Nav2 reached goal", time.monotonic() - started, final_pose={"x": x, "y": y, "yaw": yaw}
                )
            if status == 5:  # STATUS_CANCELED
                return SkillResult.failure("navigate_to", FailureCode.CANCELLED, "Nav2 goal cancelled")
            return SkillResult.failure("navigate_to", FailureCode.BLOCKED, f"Nav2 failed with status {status}")
        finally:
            self._active_goal = None
            self._active_kind = None

    def _nav_feedback(self, feedback: Any) -> None:
        try:
            pose = feedback.feedback.current_pose.pose
            self._last_feedback = {"x": pose.position.x, "y": pose.position.y}
            self._world.robot = Pose(pose.position.x, pose.position.y, 0.0)
        except Exception:  # pragma: no cover
            pass

    # -------------------------------------------------------- generic skills
    def _execute_skill_action(
        self, skill: str, params: dict[str, Any], timeout_s: float
    ) -> SkillResult:
        if ExecuteSkill is None:  # pragma: no cover - only when interfaces are missing
            return SkillResult.failure(skill, FailureCode.BACKEND_ERROR, "agentos_interfaces not built")
        if not self._skill_client.wait_for_server(timeout_sec=self.config.action_server_timeout_s):
            return SkillResult.failure(skill, FailureCode.BACKEND_ERROR, "skill action server unavailable")
        goal = ExecuteSkill.Goal()
        goal.skill_name = skill
        goal.params_json = json.dumps(params, ensure_ascii=False)
        goal.timeout_s = float(timeout_s)
        send_future = self._skill_client.send_goal_async(goal)
        if not self._spin_until(send_future, self.config.action_server_timeout_s):
            return SkillResult.failure(skill, FailureCode.BACKEND_ERROR, "send_goal timed out")
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return SkillResult.failure(skill, FailureCode.PRECONDITION_FAILED, "skill server rejected the goal")
        self._active_goal = goal_handle
        self._active_kind = skill
        result_future = goal_handle.get_result_async()
        started = time.monotonic()
        try:
            while not result_future.done():
                if time.monotonic() - started > timeout_s:
                    self._cancel_handle()
                    return SkillResult.failure(skill, FailureCode.TIMEOUT, f"skill timeout after {timeout_s}s")
                rclpy.spin_once(self.node, timeout_sec=0.05)
            wrapped = result_future.result()
            payload = json.loads(wrapped.result.result_json or "{}")
            code = FailureCode(payload.get("code", "OK")) if payload.get("code") else FailureCode.OK
            return SkillResult(
                skill=skill,
                ok=bool(payload.get("ok", wrapped.status == 4)),
                code=code if code != FailureCode.OK else FailureCode.OK,
                detail=str(payload.get("detail", "")),
                elapsed_s=time.monotonic() - started,
                data=dict(payload.get("data") or {}),
            )
        finally:
            self._active_goal = None
            self._active_kind = None

    # -------------------------------------------------------------- interface
    def approach_object(self, object_id: str, standoff_m: float = 0.6, timeout_s: float = 45.0) -> SkillResult:
        return self._execute_skill_action(
            "approach_object", {"object_id": object_id, "standoff": standoff_m}, timeout_s
        )

    def pick(self, object_id: str, timeout_s: float = 20.0) -> SkillResult:
        return self._execute_skill_action("pick", {"object_id": object_id}, timeout_s)

    def place(self, x: float, y: float, zone: str = "", timeout_s: float = 20.0) -> SkillResult:
        return self._execute_skill_action("place", {"target": {"x": x, "y": y}, "zone": zone}, timeout_s)

    def speak(self, text: str) -> SkillResult:
        return self._execute_skill_action("speak", {"text": text}, 10.0)

    def dock(self, timeout_s: float = 60.0) -> SkillResult:
        return self._execute_skill_action("dock", {}, timeout_s)

    def retreat(self, distance_m: float = 0.35) -> SkillResult:
        return self._execute_skill_action("retreat", {"distance_m": distance_m}, 10.0)

    def stop(self) -> SkillResult:
        """Zero velocity once (the e-stop lives in the safety gateway, not here)."""
        self._cancel_handle()
        message = Twist()
        self._cmd_vel_publisher.publish(message)
        return SkillResult.success("stop", "published zero velocity and cancelled active goal")

    # -------------------------------------------------------------- operator
    def reset_estop(self, operator_id: str, code: str) -> dict[str, Any]:
        if not AGENTOS_INTERFACES_AVAILABLE or self._estop_client is None:
            return {"ok": False, "reason": "agentos_interfaces not built; use the hardware e-stop panel"}
        if not self._estop_client.wait_for_service(timeout_sec=3.0):
            return {"ok": False, "reason": f"service {self.config.estop_service} unavailable"}
        request = ResetEstop.Request()
        request.operator_id = operator_id
        request.code = code
        future = self._estop_client.call_async(request)
        if not self._spin_until(future, 5.0):
            return {"ok": False, "reason": "reset service timed out"}
        self._estop_engaged = False
        return {"ok": bool(future.result().ok), "reason": str(future.result().message)}

    def destroy(self) -> None:
        self._cancel_handle()
        if self._owns_node:
            self.node.destroy_node()
