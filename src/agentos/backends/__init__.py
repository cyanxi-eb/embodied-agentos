"""Robot backends: the only layer allowed to command a robot.

``SimSkillBackend`` (2D simulator) and ``Ros2SkillBackend`` (generic Nav2 body)
plus the three body adapters behind :func:`agentos.backends.registry.make_backend`.
Every module here is import-safe on a host without ROS2: the rclpy import is
guarded inside :mod:`agentos.backends.ros2_transport`.
"""

from agentos.backends.arm6 import Arm6Backend
from agentos.backends.base import RobotBackend
from agentos.backends.mobile_manipulator import MobileManipulatorBackend
from agentos.backends.registry import (
    UnknownBackendError,
    available_backends,
    describe_backends,
    make_backend,
    register_backend,
)
from agentos.backends.ros2_adapter import Ros2AdapterBackend
from agentos.backends.ros2_transport import (
    FakeRos2Transport,
    InterfaceSpec,
    RclpyTransport,
    Ros2Transport,
    TransportResult,
)
from agentos.backends.wheeled_humanoid import WheeledHumanoidBackend

__all__ = [
    "Arm6Backend",
    "FakeRos2Transport",
    "InterfaceSpec",
    "MobileManipulatorBackend",
    "RclpyTransport",
    "RobotBackend",
    "Ros2AdapterBackend",
    "Ros2Transport",
    "TransportResult",
    "UnknownBackendError",
    "WheeledHumanoidBackend",
    "available_backends",
    "describe_backends",
    "make_backend",
    "register_backend",
]
