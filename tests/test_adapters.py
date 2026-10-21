"""Cross-body adapter tests.

These tests are the code behind "换设备只换适配层，上层技能图不动":

* the same instruction builds the *same* task graph for all three bodies;
* the graph really runs (plan -> validate -> dispatch -> BT execute -> verify),
  so the claim is not just "three classes exist";
* the wire commands differ exactly where the three-layer heterogeneity table
  says they do (interface, action semantics, units/frames);
* a body without a capability degrades to a readable ``UNSUPPORTED`` failure and
  the trace reports it instead of crashing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pytest

from agentos.backends import (
    Arm6Backend,
    FakeRos2Transport,
    MobileManipulatorBackend,
    RobotBackend,
    UnknownBackendError,
    WheeledHumanoidBackend,
    available_backends,
    make_backend,
    register_backend,
)
from agentos.backends.ros2_transport import TransportResult
from agentos.config import RuntimeConfig
from agentos.graph.nodes import AgentNodes, route
from agentos.graph.parser import parse_instruction
from agentos.graph.planner import PlannerContext, RuleBasedPlanner
from agentos.graph.runtime import ControlRegistry
from agentos.perception.sim_perception import SimPerception
from agentos.perception.sim_vlm import SimVLM
from agentos.safety.gateway import SafetyGateway
from agentos.sim.scenes import build_scene, sample_objects
from agentos.skills.registry import SkillRegistry
from agentos.trace.recorder import TraceRecorder
from agentos.types import FailureCode, GameObject, Pose, WorldState

FETCH = "把A区的红色盒子拿到工作站台"
BACKEND_NAMES = ("wheeled_humanoid", "mobile_manipulator", "arm6")
EXPECTED_SKILLS = ["navigate_to", "find_object", "approach_object", "pick", "navigate_to", "place"]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _scene():
    return build_scene("warehouse")


def _world(scene, seed: int = 3) -> WorldState:
    """A world mirror with real scene objects, used by all adapter backends."""
    objects = sample_objects(scene, scene.planning_grid(), seed)
    return WorldState(
        robot=Pose(scene.start[0], scene.start[1], scene.start_yaw),
        objects=objects,
        dock=scene.dock,
        zones={name: [x, y] for name, (x, y) in scene.zones.items()},
        battery=100.0,
    )


def _make(name: str, scene=None, **kwargs: Any):
    scene = scene or _scene()
    transport = FakeRos2Transport()
    backend = make_backend(name, transport=transport, world=_world(scene), **kwargs)
    return backend, transport


def _build_plan(backend: RobotBackend, scene):
    """Use the exact planner entry point the runtime uses."""
    registry = SkillRegistry()
    planner = RuleBasedPlanner(registry)
    parsed = parse_instruction(FETCH)
    context = PlannerContext(instruction=FETCH, scene=scene, world=backend.world, parsed=parsed)
    return planner.plan(context)


def _graph_shape(plan_dict: dict[str, Any]) -> tuple[int, int]:
    nodes = list(plan_dict.get("nodes") or [])
    edges = sum(len(node.get("depends_on") or []) for node in nodes)
    return len(nodes), edges


# ---------------------------------------------------------------------------
# Minimal runtime shim: exactly the surface AgentNodes touches
# ---------------------------------------------------------------------------
@dataclass
class _Context:
    session_id: str
    scene: Any
    backend: RobotBackend
    safety: SafetyGateway
    perception: Any
    vlm: Any
    memory: Any
    store: Any
    recorder: TraceRecorder
    config: RuntimeConfig


@dataclass
class _GraphRun:
    status: str
    failure: dict[str, Any] | None
    plan: dict[str, Any]
    results: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


class _AdapterRuntime:
    """Runs the real LangGraph node functions against an adapter backend.

    ``AgentRuntime`` owns an SQLite checkpointer and hard-wires
    ``SimSkillBackend`` in its context factory, so the tests reuse the node
    objects themselves (the same code that runs in production) with an in-memory
    context.  No graph node logic is duplicated here.
    """

    def __init__(self, backend: RobotBackend, scene, registry: SkillRegistry) -> None:
        self.backend = backend
        self.registry = registry
        self.control = ControlRegistry()
        self.config = RuntimeConfig(
            scene=scene.name,
            bt_max_retries=0,
            max_replans=0,
            memory_enabled=False,
            persist_events=False,
        )
        self.planner = RuleBasedPlanner(registry)
        self.recorder = TraceRecorder("adapter-test")
        safety = SafetyGateway(registry_names=registry.names())
        perception = SimPerception(scene.collision_grid())
        self.context = _Context(
            session_id="adapter-test",
            scene=scene,
            backend=backend,
            safety=safety,
            perception=perception,
            vlm=SimVLM(perception),
            memory=None,
            store=None,
            recorder=self.recorder,
            config=self.config,
        )

    def ensure_context(self, state: dict[str, Any]) -> _Context:
        return self.context

    def cleanup_session(self, session_id: str, ctx: _Context | None = None) -> dict[str, Any]:
        return {"session_id": session_id, "backend_goal": self.backend.cancel_active_goal()}


def _run_graph(backend: RobotBackend, scene) -> _GraphRun:
    registry = SkillRegistry()
    runtime = _AdapterRuntime(backend, scene, registry)
    nodes = AgentNodes(runtime)
    state: dict[str, Any] = {
        "session_id": "adapter-test",
        "instruction": FETCH,
        "scene": scene.name,
        "seed": 3,
        "faults": [],
        "status": "pending",
        "plan_index": 0,
        "results": [],
        "replans": 0,
        "searched_zones": [],
        "skill_attempts": {},
        "memory_hits": [],
        "step_delay_ms": 0,
        "verify_enabled": True,
        "replan_enabled": True,
        "plan_latency_ms": [],
        "decision_latency_ms": [],
        "execute_latency_ms": [],
        "verify_latency_ms": [],
    }
    state.update(nodes.plan(state))
    node_functions = {
        "plan": nodes.plan,
        "validate_plan": nodes.validate_plan,
        "dispatch_skill": nodes.dispatch_skill,
        "execute_skill": nodes.execute_skill,
        "verify_postcondition": nodes.verify_postcondition,
        "finalize": nodes.finalize,
    }
    # Follow the real conditional-edge router (``nodes.route``) exactly like the
    # compiled StateGraph does, instead of hand-rolling the transition order.
    for _ in range(40):
        node_name = route(state)
        state.update(node_functions[node_name](state))
        if node_name == "finalize":
            break
    else:  # pragma: no cover - defensive
        raise AssertionError("task graph did not reach finalize within 40 steps")
    return _GraphRun(
        status=str(state.get("status", "failed")),
        failure=state.get("failure"),
        plan=dict(state.get("plan") or {}),
        results=list(state.get("results") or []),
        events=runtime.recorder.events,
    )


# ---------------------------------------------------------------------------
# 1. Same graph, three bodies
# ---------------------------------------------------------------------------
def test_same_task_graph_across_backends():
    """One instruction -> identical task graph; only the adapter changes."""
    scene = _scene()
    baseline_plan: dict[str, Any] | None = None
    baseline_shape: tuple[int, int] | None = None
    runs: dict[str, _GraphRun] = {}

    for name in BACKEND_NAMES:
        backend, _transport = _make(name, scene)
        plan = _build_plan(backend, scene)
        plan_dict = plan.to_dict()
        shape = _graph_shape(plan_dict)
        if baseline_shape is None:
            baseline_shape = shape
            baseline_plan = plan_dict
        assert shape == baseline_shape, name
        assert plan_dict == baseline_plan, f"{name} changed the task graph"
        assert shape == (6, 5)  # 6 skill nodes, 5 sequential dependency edges
        assert [node["skill"] for node in plan_dict["nodes"]] == EXPECTED_SKILLS
        assert {node["node_id"] for node in plan_dict["nodes"]} == {f"n{n:02d}" for n in range(6)}

        run = _run_graph(backend, scene)
        runs[name] = run
        # Execution must not reshape the graph either.
        assert run.plan == baseline_plan, f"{name} rewrote the graph while executing"
        assert _graph_shape(run.plan) == baseline_shape

    # The two mobile bodies can execute the plan; the fixed arm degrades at the
    # first navigation node instead of pretending to drive.
    assert runs["wheeled_humanoid"].status == "finished"
    assert runs["mobile_manipulator"].status == "finished"
    for name in ("wheeled_humanoid", "mobile_manipulator"):
        assert [item["skill"] for item in runs[name].results] == EXPECTED_SKILLS
        assert all(item["ok"] for item in runs[name].results), name
    assert runs["arm6"].status == "failed"
    assert runs["arm6"].failure is not None
    assert runs["arm6"].failure["code"] == FailureCode.UNSUPPORTED.value


# ---------------------------------------------------------------------------
# 2. Unsupported capability degrades, not crashes
# ---------------------------------------------------------------------------
def test_unsupported_skill_degrades():
    scene = _scene()
    backend, transport = _make("arm6", scene)

    for call in (
        lambda: backend.navigate_to(4.5, 5.0),
        lambda: backend.dock(),
        lambda: backend.retreat(0.35),
    ):
        result = call()
        assert result.ok is False
        assert result.code is FailureCode.UNSUPPORTED
        assert "fixed base" in result.detail.lower()
        assert result.data["capability"]
    assert transport.calls == [], "an unsupported skill must not emit a ROS2 command"

    run = _run_graph(backend, scene)
    assert run.status == "failed"
    assert run.failure is not None
    assert run.failure["code"] == FailureCode.UNSUPPORTED.value
    assert run.failure["skill"] == "navigate_to"
    assert "fixed base" in run.failure["detail"].lower()
    assert [entry["skill"] for entry in run.results] == ["navigate_to"]

    skill_results = [event for event in run.events if event["type"] == "skill_result"]
    assert skill_results, "the graph trace must record the skill call"
    assert skill_results[0]["skill"] == "navigate_to"
    assert skill_results[0]["code"] == FailureCode.UNSUPPORTED.value
    assert skill_results[0]["ok"] is False
    failures = [event for event in run.events if event["type"] == "skill_failed"]
    assert failures and failures[0]["code"] == FailureCode.UNSUPPORTED.value
    finishes = [event for event in run.events if event["type"] == "finish"]
    assert finishes and finishes[-1]["status"] == "failed"


# ---------------------------------------------------------------------------
# 3. Units and frames
# ---------------------------------------------------------------------------
def test_mobile_manipulator_millimetre_boundary():
    backend, transport = _make("mobile_manipulator")
    assert isinstance(backend, MobileManipulatorBackend)
    backend.world.robot = Pose(2.0, 3.0, 0.0)

    result = backend.navigate_to(2.5, 3.5)
    assert result.ok
    goal = transport.last_call("/amr/navigate")["payload"]
    assert transport.last_call("/amr/navigate")["kind"] == "action"
    # skill/world layer speaks metres; the AMR consumes millimetres
    assert goal["x_mm"] == pytest.approx(2500.0)
    assert goal["y_mm"] == pytest.approx(3500.0)
    assert goal["reverse"] is False

    # robot -> adapter: the reached pose comes back in millimetres and must be
    # converted back to metres before it enters the world mirror.
    transport.set_response(
        "/amr/navigate",
        TransportResult.success(
            "reached",
            reached_pose_mm={"x_mm": 2600.0, "y_mm": 3400.0, "yaw_rad": 0.25},
        ),
    )
    result = backend.navigate_to(2.6, 3.4, yaw=0.25)
    assert result.ok
    assert backend.world.robot.x == pytest.approx(2.6)
    assert backend.world.robot.y == pytest.approx(3.4)
    assert backend.world.robot.yaw == pytest.approx(0.25)

    # arm commands go out in the AMR's base_link frame, still in millimetres
    obj = next(iter(backend.world.objects.values()))
    backend.world.robot = Pose(obj.x - 0.3, obj.y, 0.0)
    picked = backend.pick(obj.object_id)
    assert picked.ok
    request = transport.last_call("/arm/move")
    assert request["kind"] == "service"
    assert request["payload"]["x_mm"] == pytest.approx(300.0)
    assert request["payload"]["frame_id"] == "base_link"
    assert backend.world.holding == obj.object_id


def test_mobile_manipulator_backward_requires_reverse():
    backend, transport = _make("mobile_manipulator")
    backend.world.robot = Pose(5.0, 5.0, 0.0)

    rejected = backend.navigate_to(4.0, 5.0, allow_reverse=False)
    assert rejected.ok is False
    assert rejected.code is FailureCode.PRECONDITION_FAILED
    assert "behind" in rejected.detail.lower()
    assert transport.calls_to("/amr/navigate") == []

    accepted = backend.navigate_to(4.0, 5.0)
    assert accepted.ok
    goal = transport.last_call("/amr/navigate")["payload"]
    assert goal["reverse"] is True
    assert goal["x_mm"] == pytest.approx(4000.0)
    assert goal["align_first"] is False


def test_wheeled_humanoid_relative_body_goal():
    backend, transport = _make("wheeled_humanoid")
    assert isinstance(backend, WheeledHumanoidBackend)

    # Facing +y (yaw=pi/2): world +y is body +x.
    backend.world.robot = Pose(2.0, 3.0, math.pi / 2)
    assert backend.navigate_to(2.0, 5.0).ok
    goal = transport.last_call("/humanoid/nav_to")["payload"]
    assert goal["dx_m"] == pytest.approx(2.0)
    assert goal["dy_m"] == pytest.approx(0.0, abs=1e-9)
    assert goal["dyaw_rad"] == pytest.approx(0.0, abs=1e-9)
    assert goal["holonomic"] is True

    # World +x is body -y (a strafe the non-holonomic AMR cannot express).
    backend.world.robot = Pose(2.0, 3.0, math.pi / 2)
    assert backend.navigate_to(3.0, 3.0).ok
    goal = transport.last_call("/humanoid/nav_to")["payload"]
    assert goal["dx_m"] == pytest.approx(0.0, abs=1e-9)
    assert goal["dy_m"] == pytest.approx(-1.0)

    # Rotation in place is legal only for the omnidirectional body.
    backend.world.robot = Pose(2.0, 3.0, math.pi / 2)
    assert backend.navigate_to(2.0, 3.0, yaw=math.pi / 2 + 0.5).ok
    goal = transport.last_call("/humanoid/nav_to")["payload"]
    assert goal["rotate_in_place"] is True
    assert goal["dx_m"] == pytest.approx(0.0, abs=1e-9)
    assert goal["dy_m"] == pytest.approx(0.0, abs=1e-9)
    assert goal["dyaw_rad"] == pytest.approx(0.5)


def test_arm6_base_frame_and_joint_radians():
    scene = _scene()
    world = _world(scene)
    world.objects = {
        "obj_red_box": GameObject(
            object_id="obj_red_box", color="red", category="box", x=0.9, y=0.6, zone="zone_a"
        )
    }
    backend = make_backend(
        "arm6",
        transport=FakeRos2Transport(),
        world=world,
        base_pose=Pose(0.0, 0.0, 0.0),
        reach_m=1.2,
    )
    assert isinstance(backend, Arm6Backend)
    assert backend.supports_navigation is False

    # Base-frame conversion: world (0.9, 0.6) with a base at the origin.
    assert backend.world_to_base(0.9, 0.6)[:2] == pytest.approx((0.9, 0.6))

    # Joint-space action: six radians, correct joint names, no metres.
    joints = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    moved = backend.move_joints(joints)
    assert moved.ok
    goal = backend.transport.last_call("/arm/follow_joint_trajectory")["payload"]
    assert goal["units"] == "rad"
    assert goal["joint_names"] == list(Arm6Backend.joint_names)
    assert goal["points"][0]["positions"] == pytest.approx(joints)
    assert backend.joints == pytest.approx(tuple(joints))

    # Wrong arity / non-finite values fail loudly instead of sending a goal.
    calls_before = len(backend.transport.calls)
    assert backend.move_joints([0.0, 0.1]).code is FailureCode.PRECONDITION_FAILED
    assert backend.move_joints([float("nan")] * 6).code is FailureCode.PRECONDITION_FAILED
    assert len(backend.transport.calls) == calls_before

    # Pick validates the object in the base frame and records it in the result.
    picked = backend.pick("obj_red_box")
    assert picked.ok
    assert picked.data["base_target_m"][:2] == pytest.approx([0.9, 0.6])
    assert backend.world.holding == "obj_red_box"

    # Out-of-reach target is rejected before any motion command.
    backend.world.add_object(
        GameObject(object_id="far", color="blue", category="box", x=5.0, y=0.0, zone="zone_c")
    )
    calls_before = len(backend.transport.calls)
    rejected = backend.pick("far")
    assert rejected.ok is False
    assert rejected.code is FailureCode.OUT_OF_BOUNDS
    assert len(backend.transport.calls) == calls_before


# ---------------------------------------------------------------------------
# 4. Registry
# ---------------------------------------------------------------------------
def test_registry_and_unknown_name():
    assert available_backends() == ("arm6", "mobile_manipulator", "wheeled_humanoid")
    expected = {
        "wheeled_humanoid": WheeledHumanoidBackend,
        "mobile_manipulator": MobileManipulatorBackend,
        "arm6": Arm6Backend,
    }
    for name, cls in expected.items():
        backend = make_backend(name, transport=FakeRos2Transport())
        assert isinstance(backend, cls)
        assert isinstance(backend, RobotBackend)
        assert backend.device_name == name

    with pytest.raises(UnknownBackendError) as error:
        make_backend("quadruped_x")
    message = str(error.value)
    assert "unknown backend" in message
    assert "wheeled_humanoid" in message and "mobile_manipulator" in message and "arm6" in message

    with pytest.raises(UnknownBackendError):
        make_backend("")
    with pytest.raises(ValueError):
        register_backend("arm6", Arm6Backend)  # duplicate without replace=True


# ---------------------------------------------------------------------------
# 5. Cancellation clears the goal handle
# ---------------------------------------------------------------------------
def test_backend_cancel_clears_goal():
    backend, transport = _make("wheeled_humanoid")
    observed: list[dict[str, Any]] = []

    def cancel_while_running(goal: dict[str, Any]) -> TransportResult:
        # Simulates a cancel arriving from the API thread while the goal is
        # in flight; the adapter must clear its handle and ask ROS2 to cancel.
        observed.append(backend.cancel_active_goal())
        return TransportResult.failure(FailureCode.CANCELLED, "goal cancelled in flight")

    transport.set_response("/humanoid/nav_to", cancel_while_running)
    result = backend.navigate_to(4.5, 5.0)

    assert result.ok is False
    assert result.code is FailureCode.CANCELLED
    assert backend.active_goal is None
    assert observed and observed[0]["handle_cleared"] is True
    assert observed[0]["interface"] == "/humanoid/nav_to"
    cancels = [call for call in transport.calls if call["kind"] == "cancel"]
    assert len(cancels) == 1
    assert cancels[0]["name"] == "/humanoid/nav_to"
