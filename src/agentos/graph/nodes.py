"""Graph nodes: plan -> validate_plan -> dispatch_skill -> execute_skill ->
verify_postcondition -> (next_skill | replan | finish | failed).

Design notes
------------
* ``dispatch_skill`` is the *only* place where a pause can take effect, so a
  pause never interrupts a half-finished motion primitive; the checkpoint has a
  consistent world snapshot at that boundary.
* ``verify_postcondition`` is the final authority.  It re-evaluates the skill's
  postconditions against world state even if the behavior tree already verified
  them, so "the model said it worked" and "the robot actually did it" stay
  separated (this is the core honesty mechanism of the project).
* ``interrupt()`` stores the pause payload in the SQLite checkpoint; resuming uses
  ``Command(resume=...)`` and works across processes.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.types import interrupt

from agentos.graph.parser import ParsedIntent, parse_instruction
from agentos.graph.planner import PlannerContext, UnplannableError
from agentos.graph.state import AgentState
from agentos.graph.task_graph import TaskNode, TaskPlan, validate_plan
from agentos.skills.base import SkillContext
from agentos.types import FailureCode, RECOVERABLE_CODES


class AgentNodes:
    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    # ------------------------------------------------------------------ helpers
    def _ctx(self, state: AgentState) -> Any:
        return self.runtime.ensure_context(state)

    def _ctl(self, state: AgentState) -> dict[str, bool]:
        return self.runtime.control.flags(state["session_id"])

    def _trace(self, state: AgentState, event_type: str, **payload: Any) -> None:
        ctx = self.runtime.ensure_context(state)
        ctx.recorder.emit(event_type, **payload)

    def _cancel(self, state: AgentState, ctx: Any, stage: str) -> dict[str, Any]:
        cleanup = self.runtime.cleanup_session(state["session_id"], ctx=ctx)
        ctx.recorder.emit("cancelled", stage=stage, cleanup=cleanup)
        return {
            "status": "cancelled",
            "next_action": "finish",
            "cancel_cleanup": cleanup,
            "failure": {"code": FailureCode.CANCELLED.value, "detail": f"cancelled at {stage}", "stage": stage},
        }

    def _append(self, values: list[float] | None, value: float) -> list[float]:
        result = list(values or [])
        result.append(round(value, 3))
        return result

    @staticmethod
    def _skill_context(ctx: Any) -> SkillContext:
        return SkillContext(
            backend=ctx.backend,
            safety=ctx.safety,
            perception=ctx.perception,
            vlm=ctx.vlm,
            memory=ctx.memory,
            session_id=ctx.session_id,
        )

    # --------------------------------------------------------------------- plan
    def plan(self, state: AgentState) -> dict[str, Any]:
        ctx = self._ctx(state)
        if self._ctl(state)["cancel"]:
            return self._cancel(state, ctx, "plan")
        started = time.perf_counter()
        parsed = (
            ParsedIntent.from_dict(state["parse"])
            if state.get("parse")
            else parse_instruction(state["instruction"])
        )
        previous_plan = TaskPlan.from_dict(state["plan"]) if state.get("plan") else None
        failure = state.get("failure") or None
        planner_ctx = PlannerContext(
            instruction=state["instruction"],
            scene=ctx.scene,
            world=ctx.backend.world,
            parsed=parsed,
            memory=ctx.memory if self.runtime.config.memory_enabled else None,
            searched_zones=list(state.get("searched_zones") or []),
            replan_feedback=failure,
            previous_plan=previous_plan,
            failed_index=int((failure or {}).get("node_index", 0)),
            skill_attempts=dict(state.get("skill_attempts") or {}),
        )
        try:
            plan = self.runtime.planner.plan(planner_ctx)
        except UnplannableError as error:
            ctx.recorder.emit("unplannable", detail=str(error), instruction=state["instruction"])
            return {
                "status": "failed",
                "next_action": "finish",
                "parse": parsed.to_dict(),
                "failure": {
                    "code": FailureCode.UNPLANNABLE.value,
                    "detail": str(error),
                    "stage": "plan",
                    "node_index": state.get("plan_index", 0),
                },
            }
        latency = (time.perf_counter() - started) * 1000.0
        memory_hit = planner_ctx.memory_hit.to_dict() if planner_ctx.memory_hit else None
        hints: list[dict[str, Any]] = list(state.get("memory_hits") or [])
        if memory_hit:
            hints.append(memory_hit)
        ctx.recorder.emit(
            "plan",
            instruction=state["instruction"],
            goal=plan.goal,
            nodes=[node.to_dict() for node in plan.nodes],
            replanned=plan.replanned,
            notes=plan.notes,
            latency_ms=round(latency, 3),
            memory_hit=memory_hit,
        )
        return {
            "parse": parsed.to_dict(),
            "plan": plan.to_dict(),
            "plan_index": int(plan.resume_index),
            "status": "planning",
            "next_action": "validate",
            "memory_hits": hints,
            "plan_latency_ms": self._append(state.get("plan_latency_ms"), latency),
        }

    # ------------------------------------------------------------- validate_plan
    def validate_plan(self, state: AgentState) -> dict[str, Any]:
        ctx = self._ctx(state)
        plan = TaskPlan.from_dict(state["plan"])  # type: ignore[arg-type]
        validation = validate_plan(plan, self.runtime.registry, ctx.backend.world)
        ctx.recorder.emit("plan_validated", **validation.to_dict(), nodes=plan.node_ids())
        if not validation.ok:
            return {
                "status": "failed",
                "next_action": "finish",
                "failure": {
                    "code": FailureCode.PLAN_INVALID.value,
                    "detail": "; ".join(validation.errors[:5]),
                    "stage": "validate_plan",
                    "node_index": 0,
                    "validation": validation.to_dict(),
                },
            }
        return {"status": "executing", "next_action": "dispatch"}

    # ------------------------------------------------------------ dispatch_skill
    def dispatch_skill(self, state: AgentState) -> dict[str, Any]:
        ctx = self._ctx(state)
        started = time.perf_counter()
        plan = TaskPlan.from_dict(state["plan"])  # type: ignore[arg-type]
        index = int(state.get("plan_index", 0))
        if index >= len(plan.nodes):
            return {"status": "finished", "next_action": "finish"}
        node = plan.nodes[index]
        ctl = self._ctl(state)
        if ctl["cancel"]:
            return self._cancel(state, ctx, "dispatch")
        if ctl["pause"]:
            # clear the request *before* interrupting so the resumed re-run does
            # not interrupt itself again (LangGraph re-executes the node on resume)
            self.runtime.control.clear_pause(state["session_id"])
            ctx.recorder.emit(
                "pause_requested",
                next_skill=node.skill,
                plan_index=index,
                instruction=state["instruction"],
            )
            decision = interrupt(
                {
                    "reason": "pause_requested",
                    "session_id": state["session_id"],
                    "next_skill": node.to_dict(),
                    "plan_index": index,
                    "results_so_far": len(state.get("results") or []),
                }
            )
            if decision == "cancel":
                self.runtime.control.request_cancel(state["session_id"])
                return self._cancel(state, ctx, "paused")
            ctx.recorder.emit("resumed", decision=decision, next_skill=node.skill)
            if self._ctl(state)["cancel"]:
                return self._cancel(state, ctx, "resumed")

        skill = self.runtime.registry.require(node.skill)
        verdict = ctx.safety.authorize(
            node.skill, node.params, ctx.backend.world, timeout_s=skill.timeout_s
        )
        decision_latency = (time.perf_counter() - started) * 1000.0
        if not verdict.allowed:
            ctx.recorder.emit(
                "safety_rejected",
                skill=node.skill,
                params=node.params,
                code=verdict.code.value,
                reason=verdict.reason,
            )
            return {
                "status": "failed",
                "next_action": "finish",
                "current": None,
                "failure": {
                    "code": verdict.code.value,
                    "detail": verdict.reason,
                    "stage": "dispatch",
                    "skill": node.skill,
                    "node_index": index,
                },
                "decision_latency_ms": self._append(state.get("decision_latency_ms"), decision_latency),
            }
        params = dict(verdict.sanitized)
        params.pop("__clamped__", None)
        node = TaskNode(
            node_id=node.node_id,
            skill=node.skill,
            params=params,
            depends_on=list(node.depends_on),
            purpose=node.purpose,
        )
        world_before = ctx.backend.world.snapshot(f"{node.node_id}:before")
        ctx.recorder.emit(
            "skill_dispatch",
            node_id=node.node_id,
            skill=node.skill,
            params=params,
            purpose=node.purpose,
            plan_index=index,
            world=world_before,
        )
        return {
            "current": node.to_dict(),
            "world_before": world_before,
            "status": "executing",
            "next_action": "execute",
            "decision_latency_ms": self._append(state.get("decision_latency_ms"), decision_latency),
        }

    # ------------------------------------------------------------- execute_skill
    def execute_skill(self, state: AgentState) -> dict[str, Any]:
        from agentos.bt.executor import SkillExecutionTree

        ctx = self._ctx(state)
        started = time.perf_counter()
        node = TaskNode.from_dict(state["current"])  # type: ignore[arg-type]
        skill = self.runtime.registry.require(node.skill)
        delay_ms = int(state.get("step_delay_ms") or 0)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        if self._ctl(state)["cancel"]:
            ctx.backend.cancel_active_goal()
            return self._cancel(state, ctx, "execute")
        tree = SkillExecutionTree(
            self._skill_context(ctx),
            self.runtime.registry,
            max_retries=self.runtime.config.bt_max_retries,
            allow_backtrack=True,
            verify=bool(state.get("verify_enabled", True)),
            on_event=lambda event: ctx.recorder.emit("bt_event", **event),
        )
        bt_outcome = tree.run(node.skill, node.params, cancel_check=lambda: self._ctl(state)["cancel"])
        latency = (time.perf_counter() - started) * 1000.0
        ctx.recorder.emit(
            "skill_result",
            node_id=node.node_id,
            skill=node.skill,
            ok=bt_outcome.ok,
            outcome=bt_outcome.outcome,
            code=bt_outcome.code.value,
            detail=bt_outcome.detail,
            attempts=bt_outcome.attempts,
            retried=bt_outcome.retried,
            backtracked=bt_outcome.backtracked,
            latency_ms=round(latency, 3),
            world=ctx.backend.world.snapshot(f"{node.node_id}:after"),
        )
        # --- memory: remember where a successfully found object was -----------
        if bt_outcome.ok and node.skill == "find_object" and ctx.memory is not None:
            detection = ctx.backend.world.last_detection or {}
            if detection.get("x") is not None:
                ctx.memory.remember_object(
                    ctx.scene.name,
                    node.params.get("query", {}),
                    str(detection.get("object_id", "")),
                    float(detection["x"]),
                    float(detection["y"]),
                    str(node.params.get("zone", "")),
                    session_id=ctx.session_id,
                )
        return {
            "bt": bt_outcome.to_dict(),
            "results": list(state.get("results") or [])
            + [
                {
                    "node_id": node.node_id,
                    "skill": node.skill,
                    "ok": bt_outcome.ok,
                    "outcome": bt_outcome.outcome,
                    "code": bt_outcome.code.value,
                    "attempts": bt_outcome.attempts,
                    "retried": bt_outcome.retried,
                    "backtracked": bt_outcome.backtracked,
                }
            ],
            "world_snapshot": ctx.backend.snapshot(),
            "status": "executing",
            "next_action": "verify",
            "execute_latency_ms": self._append(state.get("execute_latency_ms"), latency),
        }

    # ------------------------------------------------------ verify_postcondition
    def verify_postcondition(self, state: AgentState) -> dict[str, Any]:
        ctx = self._ctx(state)
        started = time.perf_counter()
        bt = state.get("bt") or {}
        if state.get("status") == "cancelled":
            return {"status": "cancelled", "next_action": "finish"}
        node = TaskNode.from_dict(state["current"])  # type: ignore[arg-type]
        skill = self.runtime.registry.require(node.skill)
        if bt.get("outcome") == "cancelled":
            return self._cancel(state, ctx, "execute")
        if not bt:
            # defensive: execution did not produce a behavior-tree outcome
            return {
                "status": "failed",
                "next_action": "finish",
                "failure": {
                    "code": FailureCode.BACKEND_ERROR.value,
                    "detail": "execute_skill produced no behavior-tree outcome",
                    "stage": "verify_postcondition",
                    "skill": node.skill,
                    "node_index": int(state.get("plan_index", 0)),
                },
            }
        verify_enabled = bool(state.get("verify_enabled", True))

        verified = True
        report: dict[str, Any] = {"ok": True, "skipped": not verify_enabled, "failures": []}
        if bt.get("outcome") != "ok":
            verified = False
            report = {"ok": False, "skipped": False, "failures": [bt.get("detail", "")]}
        elif verify_enabled:
            condition_report = skill.check_postconditions(ctx.backend.world, node.params)
            verified = condition_report.ok
            report = condition_report.to_dict()
        latency = (time.perf_counter() - started) * 1000.0
        ctx.recorder.emit(
            "postcondition",
            node_id=node.node_id,
            skill=node.skill,
            verified=verified,
            verify_enabled=verify_enabled,
            report=report,
            world=ctx.backend.world.snapshot(f"{node.node_id}:verified"),
        )

        if verified:
            attempts = dict(state.get("skill_attempts") or {})
            attempts.pop(node.skill, None)
            new_index = int(state.get("plan_index", 0)) + 1
            plan = TaskPlan.from_dict(state["plan"])  # type: ignore[arg-type]
            finished = new_index >= len(plan.nodes)
            return {
                "plan_index": new_index,
                "skill_attempts": attempts,
                "status": "finished" if finished else "executing",
                "next_action": "finish" if finished else "dispatch",
                "verify_latency_ms": self._append(state.get("verify_latency_ms"), latency),
                "world_snapshot": ctx.backend.snapshot(),
            }

        # ---------------------------------------------------------- failure path
        code_value = str(bt.get("code") or FailureCode.POSTCONDITION_FAILED.value)
        try:
            code = FailureCode(code_value)
        except ValueError:
            code = FailureCode.POSTCONDITION_FAILED
        attempts_map = dict(state.get("skill_attempts") or {})
        attempts = int(attempts_map.get(node.skill, 0)) + 1
        attempts_map[node.skill] = attempts
        searched_zones = list(state.get("searched_zones") or [])
        if node.skill == "find_object":
            zone = str(node.params.get("zone") or "")
            if zone and zone not in searched_zones:
                searched_zones.append(zone)
        failure = {
            "code": code.value,
            "detail": str(bt.get("detail", "") or report.get("failures")),
            "stage": "verify_postcondition",
            "skill": node.skill,
            "node_index": int(state.get("plan_index", 0)),
            "attempts": attempts,
            "recoverable": code in RECOVERABLE_CODES,
            "bt": {k: bt.get(k) for k in ("outcome", "code", "detail", "attempts", "retried", "backtracked")},
            "report": report,
        }
        replans = int(state.get("replans", 0))
        can_replan = (
            bool(state.get("replan_enabled", True))
            and code in RECOVERABLE_CODES
            and replans < self.runtime.config.max_replans
        )
        ctx.recorder.emit(
            "skill_failed",
            node_id=node.node_id,
            skill=node.skill,
            code=code.value,
            attempts=attempts,
            replan=can_replan,
            replans=replans,
            searched_zones=searched_zones,
        )
        if can_replan:
            ctx.recorder.emit("replan_scheduled", failure=failure, next_replans=replans + 1)
            return {
                "status": "replanning",
                "next_action": "plan",
                "failure": failure,
                "replans": replans + 1,
                "plan_index": int(state.get("plan_index", 0)),
                "skill_attempts": attempts_map,
                "searched_zones": searched_zones,
                "verify_latency_ms": self._append(state.get("verify_latency_ms"), latency),
            }
        failure["detail"] = str(failure["detail"]) + " (replan budget exhausted or non-recoverable)"
        return {
            "status": "failed",
            "next_action": "finish",
            "failure": failure,
            "replans": replans,
            "skill_attempts": attempts_map,
            "searched_zones": searched_zones,
            "verify_latency_ms": self._append(state.get("verify_latency_ms"), latency),
        }

    # ----------------------------------------------------------------- finalize
    def finalize(self, state: AgentState) -> dict[str, Any]:
        ctx = self._ctx(state)
        status = str(state.get("status", "failed"))
        plan = TaskPlan.from_dict(state["plan"]) if state.get("plan") else None
        index = int(state.get("plan_index", 0))
        goal_verified = status == "finished" and plan is not None and index >= len(plan.nodes)
        cleanup = None
        if status == "cancelled":
            cleanup = self.runtime.cleanup_session(state["session_id"], ctx=ctx)
        ctx.recorder.emit(
            "finish",
            status=status,
            goal_verified=goal_verified,
            plan_index=index,
            plan_nodes=len(plan.nodes) if plan else 0,
            replans=int(state.get("replans", 0)),
            failure=state.get("failure"),
            cleanup=cleanup,
        )
        if ctx.memory is not None and self.runtime.config.memory_enabled:
            ctx.memory.remember(
                "task",
                state["instruction"],
                scene=ctx.scene.name,
                data={
                    "status": status,
                    "goal_verified": goal_verified,
                    "replans": int(state.get("replans", 0)),
                    "session_id": state["session_id"],
                },
            )
        return {"status": status, "goal_verified": goal_verified, "cancel_cleanup": cleanup}


ROUTE_MAP = {
    "validate": "validate_plan",
    "dispatch": "dispatch_skill",
    "execute": "execute_skill",
    "verify": "verify_postcondition",
    "plan": "plan",
    "finish": "finalize",
}


def route(state: AgentState) -> str:
    return ROUTE_MAP.get(str(state.get("next_action", "finish")), "finalize")
