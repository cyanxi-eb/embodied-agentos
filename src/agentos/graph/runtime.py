"""Agent runtime: LangGraph checkpointer, session contexts, pause/resume/cancel.

The runtime is the single owner of cross-session resources (SQLite checkpointer,
Qdrant memory, session store) and of the *in-process* live contexts (simulator +
behavior tree).  Contexts can be rebuilt from a checkpointed world snapshot, which
is what makes ``Command(resume=...)`` work after a process restart.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agentos.config import RuntimeConfig
from agentos.graph.nodes import AgentNodes, route
from agentos.graph.planner import PlannerBackend, RuleBasedPlanner, build_planner
from agentos.graph.state import AgentState
from agentos.memory.session_store import SessionStore
from agentos.memory.vector_store import MemoryStore, build_memory_store
from agentos.perception.sim_vlm import SimVLM
from agentos.safety.gateway import SafetyConfig, SafetyGateway
from agentos.sim.backend import SimSkillBackend
from agentos.sim.faults import FaultSpec
from agentos.sim.scenes import Scene, build_scene
from agentos.skills.registry import SkillRegistry, default_registry
from agentos.trace.recorder import TraceRecorder
from agentos.types import WorldState


@dataclass
class SessionContext:
    session_id: str
    scene: Scene
    backend: SimSkillBackend
    safety: SafetyGateway
    perception: Any
    vlm: SimVLM
    memory: MemoryStore | None
    store: SessionStore
    recorder: TraceRecorder
    config: RuntimeConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "scene": self.scene.name,
            "sim_time": round(self.backend.sim_time, 2),
            "robot": self.backend.world.robot.to_dict(),
            "battery": round(self.backend.world.battery, 1),
            "holding": self.backend.world.holding,
            "estop": self.safety.estop_engaged,
        }


@dataclass
class RunResult:
    session_id: str
    status: str
    state: dict[str, Any]
    interrupt: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "interrupt": self.interrupt,
            "state": _trim_state(self.state),
        }


def _trim_state(state: dict[str, Any]) -> dict[str, Any]:
    """Drop the non-JSON interrupt marker before returning state to callers."""
    return {key: value for key, value in state.items() if not str(key).startswith("__")}


class ControlRegistry:
    """Thread-safe pause / cancel flags shared between API threads and graph nodes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._flags: dict[str, dict[str, bool]] = {}

    def _ensure(self, session_id: str) -> dict[str, bool]:
        return self._flags.setdefault(session_id, {"pause": False, "cancel": False})

    def flags(self, session_id: str) -> dict[str, bool]:
        with self._lock:
            return dict(self._ensure(session_id))

    def request_pause(self, session_id: str) -> dict[str, bool]:
        with self._lock:
            self._ensure(session_id)["pause"] = True
            return dict(self._flags[session_id])

    def clear_pause(self, session_id: str) -> None:
        with self._lock:
            self._ensure(session_id)["pause"] = False

    def request_cancel(self, session_id: str) -> dict[str, bool]:
        with self._lock:
            self._ensure(session_id)["cancel"] = True
            return dict(self._flags[session_id])

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._flags.pop(session_id, None)


class AgentRuntime:
    def __init__(
        self,
        config: RuntimeConfig | None = None,
        registry: SkillRegistry | None = None,
        planner: PlannerBackend | None = None,
        memory: MemoryStore | None = None,
        store: SessionStore | None = None,
        checkpointer: SqliteSaver | None = None,
        frame_hook: Callable[[WorldState, dict[str, Any]], None] | None = None,
        frame_every: int = 3,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.paths = self.config.ensure_dirs()
        self.registry = registry or default_registry()
        self.control = ControlRegistry()
        self.store = store or SessionStore(self.paths["sessions"])
        self.memory = memory
        if self.memory is None and self.config.memory_enabled:
            self.memory = build_memory_store(self.paths["qdrant"], backend=self.config.memory_backend)
        self.checkpointer = checkpointer or _build_checkpointer(self.paths["checkpoints"])
        self.planner = planner or build_planner(self.config.planner, self.registry)
        self.frame_hook = frame_hook
        self.frame_every = frame_every
        self._contexts: dict[str, SessionContext] = {}
        self._recorders: dict[str, TraceRecorder] = {}
        self._final_worlds: dict[str, WorldState] = {}
        self._lock = threading.RLock()
        self.nodes = AgentNodes(self)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------- graph
    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("plan", self.nodes.plan)
        builder.add_node("validate_plan", self.nodes.validate_plan)
        builder.add_node("dispatch_skill", self.nodes.dispatch_skill)
        builder.add_node("execute_skill", self.nodes.execute_skill)
        builder.add_node("verify_postcondition", self.nodes.verify_postcondition)
        builder.add_node("finalize", self.nodes.finalize)
        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan", route, {"validate_plan": "validate_plan", "finalize": "finalize"}
        )
        builder.add_conditional_edges(
            "validate_plan",
            route,
            {"validate_plan": "validate_plan", "dispatch_skill": "dispatch_skill", "finalize": "finalize"},
        )
        builder.add_conditional_edges(
            "dispatch_skill",
            route,
            {"execute_skill": "execute_skill", "finalize": "finalize", "plan": "plan"},
        )
        builder.add_conditional_edges(
            "execute_skill", route, {"verify_postcondition": "verify_postcondition", "finalize": "finalize"}
        )
        builder.add_conditional_edges(
            "verify_postcondition",
            route,
            {"dispatch_skill": "dispatch_skill", "plan": "plan", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self.checkpointer)

    # ----------------------------------------------------------------- context
    def _safety_config(self) -> SafetyConfig:
        sim = self.config.sim
        return SafetyConfig(
            max_linear=sim.max_linear,
            max_angular=sim.max_angular,
            map_bounds=(0.0, 0.0, 12.0, 12.0),
        )

    def _create_context(
        self,
        session_id: str,
        scene_name: str,
        seed: int,
        faults: list[dict[str, Any]] | None,
        recorder: TraceRecorder,
        world_snapshot: dict[str, Any] | None = None,
    ) -> SessionContext:
        scene = build_scene(scene_name)
        fault_specs = [FaultSpec.from_dict(fault) for fault in (faults or [])]
        backend = SimSkillBackend(
            scene,
            config=self.config.sim,
            safety_config=self._safety_config(),
            seed=seed,
            faults=fault_specs,
            frame_hook=self.frame_hook,
            frame_every=self.frame_every,
        )
        backend.safety.register_skills(self.registry.names())
        if world_snapshot:
            backend.restore(world_snapshot)
        return SessionContext(
            session_id=session_id,
            scene=scene,
            backend=backend,
            safety=backend.safety,
            perception=backend.perception,
            vlm=SimVLM(backend.perception),
            memory=self.memory,
            store=self.store,
            recorder=recorder,
            config=self.config,
        )

    def _new_recorder(self, session_id: str) -> TraceRecorder:
        recorder = TraceRecorder(
            session_id,
            store=self.store if self.config.persist_events else None,
            trace_dir=self.paths["traces"],
        )
        self._recorders[session_id] = recorder
        return recorder

    def get_recorder(self, session_id: str) -> TraceRecorder | None:
        return self._recorders.get(session_id)

    def ensure_context(self, state: dict[str, Any]) -> SessionContext:
        session_id = str(state["session_id"])
        with self._lock:
            context = self._contexts.get(session_id)
            if context is not None:
                return context
            recorder = self._recorders.get(session_id) or self._new_recorder(session_id)
            context = self._create_context(
                session_id,
                str(state.get("scene", self.config.scene)),
                int(state.get("seed", 0)),
                list(state.get("faults") or []),
                recorder,
                world_snapshot=state.get("world_snapshot"),
            )
            self._contexts[session_id] = context
            return context

    # -------------------------------------------------------------------- runs
    def start(
        self,
        instruction: str,
        scene: str | None = None,
        seed: int | None = None,
        faults: list[dict[str, Any] | FaultSpec] | None = None,
        session_id: str | None = None,
        verify_enabled: bool | None = None,
        replan_enabled: bool | None = None,
        step_delay_ms: int | None = None,
        recorder: TraceRecorder | None = None,
    ) -> RunResult:
        session_id = session_id or f"s-{uuid.uuid4().hex[:12]}"
        scene_name = scene or self.config.scene
        seed = int(seed if seed is not None else self.config.seed)
        fault_dicts = [fault.to_dict() if isinstance(fault, FaultSpec) else dict(fault) for fault in (faults or [])]
        recorder = recorder or self._new_recorder(session_id)
        self.store.create_session(
            session_id,
            instruction,
            scene_name,
            "running",
            meta={"seed": seed, "faults": fault_dicts, "config": self.config.to_dict()},
        )
        with self._lock:
            self._contexts[session_id] = self._create_context(
                session_id, scene_name, seed, fault_dicts, recorder
            )
        initial: AgentState = {
            "session_id": session_id,
            "instruction": instruction,
            "scene": scene_name,
            "seed": seed,
            "faults": fault_dicts,
            "status": "pending",
            "plan_index": 0,
            "results": [],
            "replans": 0,
            "searched_zones": [],
            "skill_attempts": {},
            "memory_hits": [],
            "step_delay_ms": int(step_delay_ms if step_delay_ms is not None else self.config.step_delay_ms),
            "verify_enabled": bool(
                verify_enabled if verify_enabled is not None else self.config.verify_enabled
            ),
            "replan_enabled": bool(
                replan_enabled if replan_enabled is not None else self.config.replan_enabled
            ),
            "plan_latency_ms": [],
            "decision_latency_ms": [],
            "execute_latency_ms": [],
            "verify_latency_ms": [],
        }
        recorder.emit(
            "session_start",
            instruction=instruction,
            scene=scene_name,
            seed=seed,
            faults=fault_dicts,
            verify_enabled=initial["verify_enabled"],
            replan_enabled=initial["replan_enabled"],
        )
        started = time.perf_counter()
        result = self.graph.invoke(initial, config=self._thread_config(session_id), recursion_limit=120)
        return self._handle_result(session_id, result, started)

    def resume(self, session_id: str, decision: str = "resume") -> RunResult:
        if decision == "cancel":
            self.control.request_cancel(session_id)
        recorder = self._recorders.get(session_id)
        if recorder is not None:
            recorder.emit("resume_requested", decision=decision)
            recorder.emit("resumed", decision=decision)
        started = time.perf_counter()
        result = self.graph.invoke(
            Command(resume=decision),
            config=self._thread_config(session_id),
            recursion_limit=120,
        )
        return self._handle_result(session_id, result, started)

    def _thread_config(self, session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}, "recursion_limit": 120}

    def _handle_result(self, session_id: str, result: dict[str, Any], started: float) -> RunResult:
        interrupt = None
        raw_interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if raw_interrupts:
            interrupt = {
                "payloads": [
                    {"value": getattr(item, "value", None), "id": getattr(item, "id", None)}
                    for item in raw_interrupts
                ]
            }
            status = "paused"
        else:
            status = str(result.get("status", "failed"))
        state = _trim_state(dict(result))
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        self.store.update_session(
            session_id,
            status,
            {
                "status": status,
                "goal_verified": bool(result.get("goal_verified", False)),
                "replans": int(result.get("replans", 0)),
                "latency_ms": elapsed_ms,
                "failure": result.get("failure"),
            },
        )
        recorder = self._recorders.get(session_id)
        if recorder is not None:
            recorder.emit("session_status", status=status, latency_ms=elapsed_ms, goal_verified=bool(result.get("goal_verified", False)))
        if status in ("finished", "failed", "cancelled"):
            context = self._contexts.get(session_id)
            if context is not None:
                self._final_worlds[session_id] = context.backend.world.clone()
            if recorder is not None:
                recorder.close()
            self.control.clear(session_id)
        return RunResult(session_id=session_id, status=status, state=state, interrupt=interrupt)

    # ------------------------------------------------------------ control plane
    def pause(self, session_id: str) -> dict[str, bool]:
        flags = self.control.request_pause(session_id)
        recorder = self._recorders.get(session_id)
        if recorder is not None:
            recorder.emit("pause_requested", source="api")
        return flags

    def cancel(self, session_id: str) -> dict[str, Any]:
        flags = self.control.request_cancel(session_id)
        state = self.get_state(session_id) or {}
        status = state.get("status")
        if status == "paused":
            result = self.resume(session_id, "cancel")
            return {"cancelled": True, "flags": flags, "status": result.status}
        return {"cancelled": True, "flags": flags, "status": status}

    def engage_estop(self, reason: str = "api emergency stop", session_id: str | None = None) -> dict[str, Any]:
        events = []
        targets = [session_id] if session_id else list(self._contexts)
        for sid in targets:
            context = self._contexts.get(sid)
            if context is not None:
                events.append(context.safety.engage_estop(reason, source="api"))
                context.recorder.emit("estop_engaged", reason=reason)
        return {"engaged": True, "reason": reason, "sessions": events}

    def reset_estop(self, operator_id: str, code: str, session_id: str | None = None) -> dict[str, Any]:
        results = []
        targets = [session_id] if session_id else list(self._contexts)
        for sid in targets:
            context = self._contexts.get(sid)
            if context is not None:
                result = context.safety.reset_estop(operator_id, code)
                result["session_id"] = sid
                results.append(result)
        return {"results": results}

    # -------------------------------------------------------------------- reads
    def get_state(self, session_id: str) -> dict[str, Any] | None:
        try:
            snapshot = self.graph.get_state(self._thread_config(session_id))
        except Exception:
            return None
        return _trim_state(dict(snapshot.values or {})) if snapshot else None

    def get_run_status(self, session_id: str) -> tuple[str, dict[str, Any] | None]:
        state = self.get_state(session_id)
        if not state:
            return "unknown", None
        status = str(state.get("status", "unknown"))
        try:
            snapshot = self.graph.get_state(self._thread_config(session_id))
            if snapshot is not None and snapshot.next:
                status = "paused" if state.get("status") != "cancelled" else status
        except Exception:
            pass
        return status, state

    def get_context(self, session_id: str) -> SessionContext | None:
        return self._contexts.get(session_id)

    def world(self, session_id: str) -> WorldState | None:
        context = self._contexts.get(session_id)
        if context is not None:
            return context.backend.world
        return self._final_worlds.get(session_id)

    def trace_events(self, session_id: str) -> list[dict[str, Any]]:
        recorder = self._recorders.get(session_id)
        if recorder is not None:
            return recorder.events
        return self.store.get_events(session_id)

    # --------------------------------------------------------------- teardown
    def cleanup_session(self, session_id: str, ctx: SessionContext | None = None) -> dict[str, Any]:
        context = ctx or self._contexts.get(session_id)
        cleanup: dict[str, Any] = {"session_id": session_id}
        if context is not None:
            try:
                cleanup["backend_goal"] = context.backend.cancel_active_goal()
            except Exception as error:  # pragma: no cover
                cleanup["backend_goal"] = {"error": repr(error)}
            cleanup["blackboard_keys"] = []  # the behavior tree clears its own namespace
        self.control.clear(session_id)
        return cleanup

    def close(self) -> None:
        for session_id in list(self._recorders):
            try:
                self._recorders[session_id].close()
            except Exception:  # pragma: no cover
                pass
        try:
            if self.memory is not None:
                self.memory.close()
        finally:
            try:
                self.store.close()
            except Exception:  # pragma: no cover
                pass


def _build_checkpointer(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=5000")
    saver = SqliteSaver(connection)
    saver.setup()
    return saver
