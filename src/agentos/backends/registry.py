"""Device registry: configuration name -> adapter factory.

This is the switch the runtime/evaluation harness uses:

.. code-block:: python

    backend = make_backend("mobile_manipulator", transport=FakeRos2Transport())
    backend = make_backend("wheeled_humanoid")          # real ROS2 transport

The task graph, planner, skills, behavior tree and safety gateway are *not*
involved: swapping the body is a one-line configuration change, which is what
``tests/test_adapters.py::test_same_task_graph_across_backends`` checks.

Third-party bodies can be added without editing this file:

.. code-block:: python

    register_backend("my_amr", MyAmrBackend)
    backend = make_backend("my_amr", transport=transport)
"""

from __future__ import annotations

from typing import Any, Callable

from agentos.backends.arm6 import Arm6Backend
from agentos.backends.base import RobotBackend
from agentos.backends.mobile_manipulator import MobileManipulatorBackend
from agentos.backends.ros2_adapter import Ros2AdapterBackend
from agentos.backends.wheeled_humanoid import WheeledHumanoidBackend

BackendFactory = Callable[..., RobotBackend]


class UnknownBackendError(ValueError):
    """Raised when a configuration name has no registered adapter factory."""


_DEFAULT_FACTORIES: dict[str, BackendFactory] = {
    WheeledHumanoidBackend.device_name: WheeledHumanoidBackend,
    MobileManipulatorBackend.device_name: MobileManipulatorBackend,
    Arm6Backend.device_name: Arm6Backend,
}
_FACTORIES: dict[str, BackendFactory] = dict(_DEFAULT_FACTORIES)


def available_backends() -> tuple[str, ...]:
    """Registered device names, sorted for stable CLI/API output."""
    return tuple(sorted(_FACTORIES))


def register_backend(name: str, factory: BackendFactory, *, replace: bool = False) -> None:
    """Register a new body factory (e.g. another brand's adapter)."""
    key = str(name).strip()
    if not key:
        raise ValueError("backend name must be a non-empty string")
    if not callable(factory):
        raise TypeError("backend factory must be callable")
    if key in _FACTORIES and not replace:
        raise ValueError(f"backend {key!r} is already registered (pass replace=True to override)")
    _FACTORIES[key] = factory


def make_backend(name: str, **kwargs: Any) -> RobotBackend:
    """Instantiate the adapter registered under ``name``.

    ``transport`` and ``world`` are forwarded to the adapter constructor:
    tests pass a :class:`~agentos.backends.ros2_transport.FakeRos2Transport`,
    a deployment simply omits ``transport`` to get the real rclpy one.
    """
    key = str(name).strip()
    factory = _FACTORIES.get(key)
    if factory is None:
        raise UnknownBackendError(
            f"unknown backend {name!r}; available backends: {', '.join(available_backends())}. "
            "Register a new body with register_backend(name, factory)."
        )
    return factory(**kwargs)


def describe_backends() -> dict[str, dict[str, Any]]:
    """Small capability matrix for the API/docs (no instantiation, no ROS2)."""
    descriptions: dict[str, dict[str, Any]] = {}
    for name, factory in sorted(_FACTORIES.items()):
        candidate = factory
        if isinstance(candidate, type) and issubclass(candidate, Ros2AdapterBackend):
            descriptions[name] = {
                "display_name": candidate.display_name,
                "navigation": bool(candidate.supports_navigation),
                "manipulation": bool(candidate.supports_manipulation),
                "interfaces": [
                    {"kind": spec.kind, "name": spec.name, "type_path": spec.type_path}
                    for spec in candidate.interfaces
                ],
            }
        else:
            descriptions[name] = {"display_name": name, "navigation": None, "manipulation": None, "interfaces": []}
    return descriptions
