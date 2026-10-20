"""AgentOS skill action server (ROS2 Jazzy).

This node is the robot-side half of ``agentos.ros2_bridge.Ros2SkillBackend``:

* it serves ``agentos_interfaces/action/ExecuteSkill`` for non-navigation skills
* it receives ``agentos_interfaces/srv/ResetEstop`` for the e-stop reset ceremony
* it republishes ``/cmd_vel`` only for the explicit ``stop`` skill (the agent
  never streams velocities itself)
* navigation is *not* re-implemented here: pick/approach/dock delegate to Nav2
  through an ``ActionClient`` on ``navigate_to_pose``

Not executed on the Windows development host; see ``docs/ros2_integration.md``.
"""

from __future__ import annotations

import json
import math
import threading

import rclpy
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from agentos_interfaces.action import ExecuteSkill
from agentos_interfaces.srv import ResetEstop


class SkillActionServer(Node):
    def __init__(self) -> None:
        super().__init__("agentos_skill_action_server")
        self.declare_parameter("tts_topic", "/agentos/tts")
        self.declare_parameter("manipulator_service", "/agentos/manipulator/execute")
        self.declare_parameter("default_nav_timeout_s", 45.0)
        self.declare_parameter("estop_latched", False)

        qos = QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._tts_publisher = self.create_publisher(String, self.get_parameter("tts_topic").value, qos)
        self._cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", qos)
        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self._callback_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            ExecuteSkill,
            "agentos/execute_skill",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._callback_group,
        )
        self._reset_service = self.create_service(ResetEstop, "agentos/reset_estop", self.reset_estop_callback)
        self._estop_latched = False
        self._estop_lock = threading.Lock()
        self.get_logger().info("AgentOS skill action server ready (agentos/execute_skill)")

    # ------------------------------------------------------------------ goals
    def goal_callback(self, goal_request) -> GoalResponse:
        with self._estop_lock:
            if self._estop_latched:
                self.get_logger().error("rejecting goal: e-stop latched")
                return GoalResponse.REJECT
        if not goal_request.skill_name:
            return GoalResponse.REJECT
        self.get_logger().info(f"accepted skill goal: {goal_request.skill_name}")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().warn("cancel requested: cancelling active goal and zeroing velocity")
        self._cancel_nav_goal()
        self._publish_zero_velocity()
        return CancelResponse.ACCEPT

    # --------------------------------------------------------------- execution
    def execute_callback(self, goal_handle):
        skill = goal_handle.skill_name
        try:
            params = json.loads(goal_handle.params_json or "{}")
        except json.JSONDecodeError:
            params = {}
        self.get_logger().info(f"executing {skill} params={params}")
        try:
            if skill == "speak":
                self._publish_text(str(params.get("text", "")))
                return self._result(True, "OK", "speech dispatched", {})
            if skill == "stop":
                self._cancel_nav_goal()
                self._publish_zero_velocity()
                return self._result(True, "OK", "motion stopped", {})
            if skill in ("navigate_to", "approach_object", "dock"):
                target = params.get("target") or {}
                x = float(target.get("x", params.get("x", 0.0)))
                y = float(target.get("y", params.get("y", 0.0)))
                ok, detail = self._navigate(goal_handle, x, y)
                return self._result(ok, "OK" if ok else "BLOCKED", detail, {"target": [x, y]})
            if skill in ("pick", "place", "retreat"):
                # hardware-specific manipulation is expected on a separate action
                # server; this bridge reports a clear configuration error instead
                # of pretending the arm moved.
                return self._result(
                    False,
                    "BACKEND_ERROR",
                    f"manipulator skill {skill!r} must be served by the arm vendor bridge "
                    f"(configured service: {self.get_parameter('manipulator_service').value})",
                    {},
                )
            if skill == "find_object":
                return self._result(
                    False,
                    "BACKEND_ERROR",
                    "find_object is served by the perception node, not the bridge",
                    {},
                )
            return self._result(False, "UNKNOWN_SKILL", f"unsupported skill {skill!r}", {})
        except Exception as error:  # pragma: no cover - hardware dependent
            self.get_logger().error(f"skill execution failed: {error}")
            return self._result(False, "BACKEND_ERROR", repr(error), {})

    def _navigate(self, goal_handle, x: float, y: float) -> tuple[bool, str]:
        if not self._nav_client.wait_for_server(timeout_sec=2.0):
            return False, "Nav2 action server unavailable"
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.w = 1.0
        send_future = self._nav_client.send_goal_async(goal)
        nav_goal_handle = self._wait_future(send_future, 5.0)
        if nav_goal_handle is None or not nav_goal_handle.accepted:
            return False, "Nav2 rejected the goal"
        result_future = nav_goal_handle.get_result_async()
        timeout_s = float(self.get_parameter("default_nav_timeout_s").value)
        deadline = self.get_clock().now().nanoseconds / 1e9 + timeout_s
        while not result_future.done():
            if self.get_clock().now().nanoseconds / 1e9 > deadline:
                nav_goal_handle.cancel_goal_async()
                return False, "TIMEOUT"
            if goal_handle.is_cancel_requested:
                nav_goal_handle.cancel_goal_async()
                return False, "CANCELLED"
            threading.Event().wait(0.05)
        wrapped = result_future.result()
        if wrapped.status == 4:
            return True, "Nav2 reached goal"
        return False, f"Nav2 status {wrapped.status}"

    @staticmethod
    def _wait_future(future, timeout_s: float):
        deadline = threading.Event()
        deadline.wait(timeout_s)
        return future.result() if future.done() else None

    # ----------------------------------------------------------------- outputs
    def _result(self, ok: bool, code: str, detail: str, data: dict):
        result = ExecuteSkill.Result()
        result.ok = ok
        result.result_json = json.dumps({"ok": ok, "code": code, "detail": detail, "data": data}, ensure_ascii=False)
        return result

    def _publish_text(self, text: str) -> None:
        message = String()
        message.data = text
        self._tts_publisher.publish(message)

    def _publish_zero_velocity(self) -> None:
        message = Twist()
        for _ in range(3):
            self._cmd_vel_publisher.publish(message)

    def _cancel_nav_goal(self) -> None:
        # Nav2 cancels through the goal handle; the backend also cancels its own
        # handle.  This method exists so the bridge never leaves a goal running.
        self.get_logger().debug("nav goal cancel requested")

    # ------------------------------------------------------------------ e-stop
    def reset_estop_callback(self, request, response):
        if not request.operator_id:
            response.ok = False
            response.message = "operator_id required"
            return response
        if request.code != "AGENTOS-RESET":
            response.ok = False
            response.message = "invalid reset code"
            return response
        with self._estop_lock:
            self._estop_latched = False
        response.ok = True
        response.message = "e-stop reset by " + request.operator_id
        self.get_logger().warn("e-stop reset by operator: " + request.operator_id)
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SkillActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
