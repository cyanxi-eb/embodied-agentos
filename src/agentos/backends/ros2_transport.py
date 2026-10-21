"""ROS2 transport seam shared by the three body adapters (X1-like humanoid, AMR +
arm, fixed-base 6-axis arm).

The adapters never import ``rclpy`` themselves.  They translate one skill call
into a *(interface, JSON payload)* pair and hand it to a transport:

* :class:`FakeRos2Transport` records every call and returns scripted results.
  It is what ``tests/test_adapters.py`` uses, so the wire goals can be asserted
  exactly (units, frames, reverse flags) without a robot.
* :class:`RclpyTransport` is the real client.  ``rclpy`` is import-guarded, and
  the class raises a readable error when ROS2 is missing.  It has **not** been
  executed on this offline workstation; see ``docs/adapters.md``.  Vendor
  action/service message types are supplied by the deployment through
  :class:`InterfaceSpec`, because those interfaces live in the robot workspace
  and are deliberately not vendored here.

Payloads stay plain JSON-native dicts on purpose: the adapter owns unit/frame
conversion and the transport owns ROS2 message plumbing.  That is also why a
fake transport is a faithful stand-in for the adapter tests - it sees exactly
what a real action server would see.
"""

from __future__ import annotations

import importlib
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable

from agentos.types import FailureCode

ROS2_AVAILABLE = False
ROS2_IMPORT_ERROR = ""

try:  # pragma: no cover - only true on a ROS2 host
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node

    ROS2_AVAILABLE = True
except Exception as error:  # pragma: no cover - the offline workstation
    rclpy = None  # type: ignore[assignment]
    ActionClient = None  # type: ignore[assignment]
    Node = object  # type: ignore[assignment,misc]
    ROS2_IMPORT_ERROR = f"{type(error).__name__}: {error}"


# ---------------------------------------------------------------------------
# Result type crossing the transport boundary
# ---------------------------------------------------------------------------
@dataclass
class TransportResult:
    """Outcome of one ROS2 action/service call, before skill-level mapping."""

    ok: bool
    code: FailureCode = FailureCode.OK
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, detail: str = "", **data: Any) -> "TransportResult":
        return cls(ok=True, code=FailureCode.OK, detail=detail, data=dict(data))

    @classmethod
    def failure(cls, code: FailureCode, detail: str = "", **data: Any) -> "TransportResult":
        return cls(ok=False, code=code, detail=detail, data=dict(data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "code": self.code.value,
            "detail": self.detail,
            "data": dict(self.data),
        }


# ---------------------------------------------------------------------------
# Transport protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class Ros2Transport(Protocol):
    """What an adapter needs from ROS2; deliberately four methods wide."""

    def send_action(self, name: str, goal: dict[str, Any], timeout_s: float = 5.0) -> TransportResult: ...

    def call_service(self, name: str, request: dict[str, Any], timeout_s: float = 5.0) -> TransportResult: ...

    def publish(self, topic: str, message: dict[str, Any]) -> None: ...

    def cancel(self, name: str) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Fake transport (tests / dry runs)
# ---------------------------------------------------------------------------
class FakeRos2Transport:
    """Deterministic recorder used by the adapter tests and dry-run demos.

    ``responses`` maps an interface name to a :class:`TransportResult`, a list of
    results (consumed in order), or a callable receiving the payload.  An
    unconfigured interface succeeds, which keeps the "happy path" tests short.
    """

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses: dict[str, Any] = dict(responses or {})
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ record
    def _record(self, kind: str, name: str, payload: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        entry = {"kind": kind, "name": name, "payload": dict(payload), "at": len(self.calls)}
        if timeout_s is not None:
            entry["timeout_s"] = float(timeout_s)
        with self._lock:
            self.calls.append(entry)
        return entry

    def _resolve(self, name: str, payload: dict[str, Any]) -> TransportResult:
        response = self._responses.get(name)
        if response is None:
            return TransportResult.success(f"{name} accepted")
        if isinstance(response, list):
            response = response.pop(0) if response else TransportResult.failure(
                FailureCode.BACKEND_ERROR, f"no scripted response left for {name}"
            )
        if callable(response):
            response = response(payload)
        if isinstance(response, TransportResult):
            return response
        if isinstance(response, dict):
            data = dict(response.get("data") or {})
            return TransportResult(
                ok=bool(response.get("ok", True)),
                code=FailureCode(str(response.get("code", FailureCode.OK.value))),
                detail=str(response.get("detail", "")),
                data=data,
            )
        raise TypeError(f"unsupported scripted response for {name!r}: {type(response).__name__}")

    # ------------------------------------------------------------------- calls
    def send_action(self, name: str, goal: dict[str, Any], timeout_s: float = 5.0) -> TransportResult:
        call = self._record("action", name, goal, timeout_s)
        result = self._resolve(name, goal)
        call["result_code"] = result.code.value
        return result

    def call_service(self, name: str, request: dict[str, Any], timeout_s: float = 5.0) -> TransportResult:
        call = self._record("service", name, request, timeout_s)
        result = self._resolve(name, request)
        call["result_code"] = result.code.value
        return result

    def publish(self, topic: str, message: dict[str, Any]) -> None:
        self._record("publish", topic, message)

    def cancel(self, name: str) -> dict[str, Any]:
        self._record("cancel", name, {})
        return {"cancelled": True, "handle_cleared": True}

    # ---------------------------------------------------------------- helpers
    def set_response(self, name: str, response: Any) -> None:
        self._responses[name] = response

    def calls_to(self, name: str) -> list[dict[str, Any]]:
        with self._lock:
            return [call for call in self.calls if call["name"] == name]

    def last_call(self, name: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            candidates = [call for call in self.calls if name is None or call["name"] == name]
        return candidates[-1] if candidates else None

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()


# ---------------------------------------------------------------------------
# Real rclpy transport
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InterfaceSpec:
    """One logical ROS2 interface used by an adapter.

    ``type_path`` is ``"<python.module>.<Class>"`` (e.g.
    ``"control_msgs.action.FollowJointTrajectory"``).  It is empty when the
    vendor action/service type is only available inside the robot workspace: the
    deployment then passes a spec with the resolved type instead of silently
    guessing a message layout.
    """

    kind: str  # "action" | "service"
    name: str
    type_path: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("action", "service"):
            raise ValueError(f"interface kind must be 'action' or 'service', got {self.kind!r}")
        if not self.name:
            raise ValueError("interface name must not be empty")


def _message_to_dict(message: Any) -> Any:
    """Best-effort ROS2 message -> JSON-native dict (for result/response data)."""
    if message is None:
        return None
    fields = getattr(type(message), "get_fields_and_field_types", None)
    if fields is None:
        return message
    result: dict[str, Any] = {}
    for name in fields():
        value = getattr(message, name, None)
        if hasattr(value, "get_fields_and_field_types"):
            result[name] = _message_to_dict(value)
        elif isinstance(value, (list, tuple)):
            result[name] = [
                _message_to_dict(item) if hasattr(item, "get_fields_and_field_types") else item
                for item in value
            ]
        else:
            result[name] = value
    return result


class RclpyTransport:
    """Real ROS2 client.  Not executed on the offline workstation.

    The class is intentionally conservative: it only knows how to build goals for
    the standard interfaces used here (``NavigateToPose``,
    ``FollowJointTrajectory``, ``GripperCommand``) and refuses anything else with
    a readable error instead of sending a guessed message.
    """

    def __init__(
        self,
        node: Any = None,
        interfaces: Iterable[InterfaceSpec] = (),
        *,
        spin_timeout_s: float = 0.05,
        server_wait_s: float = 5.0,
    ) -> None:
        if not ROS2_AVAILABLE:
            raise RuntimeError(
                "rclpy is not available on this host; the real ROS2 transport requires "
                "Ubuntu 24.04 + ROS2 Jazzy (or an rclpy-enabled Python environment). "
                f"Import error: {ROS2_IMPORT_ERROR or 'rclpy not importable'}"
            )
        self.node = node if node is not None else Node("agentos_ros2_transport")
        self._owns_node = node is None
        self._specs: dict[str, InterfaceSpec] = {spec.name: spec for spec in interfaces}
        self._spin_timeout_s = float(spin_timeout_s)
        self._server_wait_s = float(server_wait_s)
        self._action_clients: dict[str, Any] = {}
        self._service_clients: dict[str, Any] = {}
        self._publishers: dict[str, Any] = {}
        self._active: dict[str, Any] = {}

    # ---------------------------------------------------------------- internal
    def _spec(self, name: str, kind: str) -> InterfaceSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise RuntimeError(
                f"no ROS2 interface configured for {kind} {name!r}; "
                "pass InterfaceSpec entries to RclpyTransport(interfaces=...)"
            )
        if spec.kind != kind:
            raise RuntimeError(f"interface {name!r} is a {spec.kind}, not a {kind}")
        if not spec.type_path:
            raise RuntimeError(
                f"interface {name!r} has no ROS2 message type; provide "
                f"InterfaceSpec(kind={kind!r}, name={name!r}, type_path='<pkg>.<module>.<Class>')"
            )
        return spec

    @staticmethod
    def _resolve_type(type_path: str) -> Any:
        module_name, _, class_name = type_path.rpartition(".")
        if not module_name:
            raise RuntimeError(f"invalid ROS2 type path {type_path!r}")
        module = importlib.import_module(module_name)
        try:
            return getattr(module, class_name)
        except AttributeError as error:  # pragma: no cover - depends on the robot workspace
            raise RuntimeError(f"ROS2 type {type_path!r} not found: {error}") from error

    def _spin_until(self, future: Any, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=self._spin_timeout_s)
        return bool(future.done())

    def _action_client(self, spec: InterfaceSpec) -> Any:
        client = self._action_clients.get(spec.name)
        if client is None:
            message_type = self._resolve_type(spec.type_path)
            client = ActionClient(self.node, message_type, spec.name)
            self._action_clients[spec.name] = client
        return client

    @staticmethod
    def _build_goal(type_path: str, payload: dict[str, Any]) -> Any:
        message_type = RclpyTransport._resolve_type(type_path)
        leaf = type_path.rsplit(".", 1)[-1]
        goal = message_type.Goal()
        if leaf == "NavigateToPose":
            from geometry_msgs.msg import PoseStamped

            yaw = float(payload.get("yaw_rad") or 0.0)
            pose = PoseStamped()
            pose.header.frame_id = str(payload.get("frame_id") or "map")
            pose.pose.position.x = float(payload.get("x_m", 0.0))
            pose.pose.position.y = float(payload.get("y_m", 0.0))
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            goal.pose = pose
            return goal
        if leaf == "FollowJointTrajectory":
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

            trajectory = JointTrajectory()
            trajectory.joint_names = [str(name) for name in payload.get("joint_names", [])]
            for raw in payload.get("points", []):
                point = JointTrajectoryPoint()
                point.positions = [float(value) for value in raw.get("positions", [])]
                velocities = raw.get("velocities")
                if velocities is not None:
                    point.velocities = [float(value) for value in velocities]
                seconds = float(raw.get("time_from_start_s", 1.0))
                point.time_from_start.sec = int(seconds)
                point.time_from_start.nanosec = int(round((seconds - int(seconds)) * 1e9))
                trajectory.points.append(point)
            goal.trajectory = trajectory
            return goal
        if leaf == "GripperCommand":
            command = payload.get("command")
            if not isinstance(command, dict):
                command = payload
            goal.command.position = float(command.get("position", 0.0))
            goal.command.max_effort = float(command.get("max_effort", 0.0))
            return goal
        raise RuntimeError(
            f"no goal codec for {type_path!r}; add one in RclpyTransport._build_goal "
            "or expose a JSON bridge action in the robot workspace"
        )

    def _publisher(self, topic: str, message_type: Any) -> Any:
        publisher = self._publishers.get(topic)
        if publisher is None:
            publisher = self.node.create_publisher(message_type, topic, 10)
            self._publishers[topic] = publisher
        return publisher

    # ------------------------------------------------------------------ public
    def send_action(self, name: str, goal: dict[str, Any], timeout_s: float = 5.0) -> TransportResult:
        try:
            spec = self._spec(name, "action")
            client = self._action_client(spec)
            if not client.wait_for_server(timeout_sec=self._server_wait_s):
                return TransportResult.failure(
                    FailureCode.BACKEND_ERROR, f"action server {spec.name!r} unavailable"
                )
            goal_message = self._build_goal(spec.type_path, goal)
            send_future = client.send_goal_async(goal_message)
            if not self._spin_until(send_future, self._server_wait_s):
                return TransportResult.failure(FailureCode.BACKEND_ERROR, f"send_goal to {spec.name!r} timed out")
            handle = send_future.result()
            if handle is None or not handle.accepted:
                return TransportResult.failure(FailureCode.BLOCKED, f"action {spec.name!r} rejected the goal")
            self._active[spec.name] = handle
            result_future = handle.get_result_async()
            started = time.monotonic()
            while not result_future.done():
                if time.monotonic() - started > timeout_s:
                    handle.cancel_goal_async()
                    return TransportResult.failure(
                        FailureCode.TIMEOUT, f"action {spec.name!r} exceeded {timeout_s:.1f}s"
                    )
                rclpy.spin_once(self.node, timeout_sec=self._spin_timeout_s)
            wrapped = result_future.result()
            status = int(getattr(wrapped, "status", 0))
            if status == 4:  # GoalStatus.STATUS_SUCCEEDED
                return TransportResult.success(
                    f"{spec.name} succeeded",
                    status=status,
                    result=_message_to_dict(getattr(wrapped, "result", None)),
                )
            if status == 5:  # GoalStatus.STATUS_CANCELED
                return TransportResult.failure(FailureCode.CANCELLED, f"action {spec.name!r} cancelled")
            return TransportResult.failure(
                FailureCode.BLOCKED, f"action {spec.name!r} failed with status {status}"
            )
        except Exception as error:  # pragma: no cover - hardware/workspace dependent
            return TransportResult.failure(
                FailureCode.BACKEND_ERROR, f"{type(error).__name__}: {error}"
            )
        finally:
            self._active.pop(name, None)

    def call_service(self, name: str, request: dict[str, Any], timeout_s: float = 5.0) -> TransportResult:
        try:
            spec = self._spec(name, "service")
            message_type = self._resolve_type(spec.type_path)
            client = self._service_clients.get(spec.name)
            if client is None:
                client = self.node.create_client(message_type, spec.name)
                self._service_clients[spec.name] = client
            if not client.wait_for_service(timeout_sec=self._server_wait_s):
                return TransportResult.failure(FailureCode.BACKEND_ERROR, f"service {spec.name!r} unavailable")
            request_message = message_type.Request()
            self._fill_message(request_message, request)
            future = client.call_async(request_message)
            if not self._spin_until(future, timeout_s):
                return TransportResult.failure(FailureCode.TIMEOUT, f"service {spec.name!r} timed out")
            response = _message_to_dict(future.result())
            data = {"response": response}
            success = True
            detail = ""
            if isinstance(response, dict):
                success = bool(response.get("success", True))
                detail = str(response.get("message", "") or "")
            if not success:
                return TransportResult.failure(FailureCode.BACKEND_ERROR, detail or "service reported failure", **data)
            return TransportResult.success(detail or f"{spec.name} succeeded", **data)
        except Exception as error:  # pragma: no cover - hardware/workspace dependent
            return TransportResult.failure(FailureCode.BACKEND_ERROR, f"{type(error).__name__}: {error}")

    @classmethod
    def _fill_message(cls, message: Any, values: dict[str, Any]) -> None:
        fields = getattr(type(message), "get_fields_and_field_types", lambda: {})()
        for key, value in values.items():
            if key not in fields:
                continue
            current = getattr(message, key)
            if isinstance(value, dict) and hasattr(current, "get_fields_and_field_types"):
                cls._fill_message(current, value)
            else:
                setattr(message, key, list(value) if isinstance(value, tuple) else value)

    def publish(self, topic: str, message: dict[str, Any]) -> None:
        if topic.endswith("cmd_vel"):
            from geometry_msgs.msg import Twist

            twist = Twist()
            linear = message.get("linear") or {}
            angular = message.get("angular") or {}
            twist.linear.x = float(linear.get("x", 0.0))
            twist.linear.y = float(linear.get("y", 0.0))
            twist.linear.z = float(linear.get("z", 0.0))
            twist.angular.x = float(angular.get("x", 0.0))
            twist.angular.y = float(angular.get("y", 0.0))
            twist.angular.z = float(angular.get("z", 0.0))
            self._publisher(topic, Twist).publish(twist)
            return
        from std_msgs.msg import String

        text = String()
        text.data = str(message.get("data", ""))
        self._publisher(topic, String).publish(text)

    def cancel(self, name: str) -> dict[str, Any]:
        handle = self._active.pop(name, None)
        cancelled = False
        if handle is not None:
            try:  # pragma: no cover - hardware dependent
                handle.cancel_goal_async()
                cancelled = True
            except Exception:
                cancelled = False
        return {"cancelled": cancelled, "handle_cleared": True}

    def destroy(self) -> None:
        self._active.clear()
        if self._owns_node:
            self.node.destroy_node()
