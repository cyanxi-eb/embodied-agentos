"""Behavior-tree skill execution with recovery branches.

The tree is deliberately small and readable:

.. code-block:: text

    Selector("skill_execution")
    ├── SyncRetry( Sequence(PreconditionCheck -> Execute -> VerifyPostcondition) )
    ├── Sequence(GateRecoverable -> Backtrack -> Sequence(Execute -> Verify))
    └── RequestReplan

* ``SyncRetry`` retries the full attempt (execute + verify) a bounded number of
  times; this covers transient failures such as ``SLIP``.
* ``Backtrack`` physically backs the robot off before a final retry; this covers
  ``BLOCKED`` situations where re-executing from the same pose cannot help.
* ``RequestReplan`` never claims success: it marks the outcome as ``replan`` so
  the LangGraph layer can rebuild the task graph (a node-level concern the BT
  must not own).

The py_trees blackboard is shared by all behaviours (context, result, failure
code) and is explicitly cleared on completion/cancellation so no residual state
leaks into the next skill attempt.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import py_trees
from py_trees.blackboard import Blackboard
from py_trees.common import Status

from agentos.skills.base import Skill, SkillContext
from agentos.skills.registry import SkillRegistry
from agentos.types import FailureCode


@dataclass
class BTOutcome:
    ok: bool
    outcome: str  # ok | replan | cancelled | failed
    code: FailureCode = FailureCode.OK
    detail: str = ""
    attempts: int = 0
    retried: bool = False
    backtracked: bool = False
    result: dict[str, Any] | None = None
    pre_report: dict[str, Any] = field(default_factory=dict)
    post_report: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "outcome": self.outcome,
            "code": self.code.value,
            "detail": self.detail,
            "attempts": self.attempts,
            "retried": self.retried,
            "backtracked": self.backtracked,
            "result": self.result,
            "pre_report": self.pre_report,
            "post_report": self.post_report,
            "events": list(self.events),
        }


class _BB:
    """Namespaced access to the py_trees blackboard for one execution."""

    def __init__(self, namespace: str) -> None:
        self.ns = namespace
        self._keys: set[str] = set()

    def key(self, name: str) -> str:
        key = f"{self.ns}/{name}"
        self._keys.add(key)
        return key

    def set(self, name: str, value: Any) -> None:
        Blackboard.set(self.key(name), value)

    def get(self, name: str, default: Any = None) -> Any:
        key = self.key(name)
        return Blackboard.get(key) if Blackboard.exists(key) else default

    def cleanup(self) -> list[str]:
        removed = []
        for key in sorted(self._keys):
            if Blackboard.exists(key):
                Blackboard.unset(key)
                removed.append(key)
        self._keys.clear()
        return removed


class _SkillBehaviour(py_trees.behaviour.Behaviour):
    """Base class holding the context/skill/blackboard handles."""

    def __init__(self, name: str, ctx: SkillContext, bb: _BB) -> None:
        # attributes must exist before Behaviour.__init__ (it creates the tick iterator)
        self.ctx = ctx
        self.bb = bb
        super().__init__(name=name)

    @property
    def skill(self) -> Skill:
        return self.bb.get("skill")

    @property
    def params(self) -> dict[str, Any]:
        return self.bb.get("params", {})

    def _fail(self, code: FailureCode, detail: str) -> Status:
        self.bb.set("code", code.value)
        self.bb.set("detail", detail)
        return Status.FAILURE

    def _record_event(self, kind: str, **payload: Any) -> None:
        events = self.bb.get("events", [])
        events.append({"kind": kind, **payload})
        self.bb.set("events", events)


class _PreconditionCheck(_SkillBehaviour):
    def update(self) -> Status:
        report = self.skill.check_preconditions(self.ctx.world, self.params)
        self.bb.set("pre_report", report.to_dict())
        if not report.ok:
            return self._fail(FailureCode.PRECONDITION_FAILED, "; ".join(report.failures))
        return Status.SUCCESS


class _Execute(_SkillBehaviour):
    def __init__(self, name: str, ctx: SkillContext, bb: _BB, cancel_check: Callable[[], bool] | None) -> None:
        super().__init__(name, ctx, bb)
        self.cancel_check = cancel_check

    def update(self) -> Status:
        attempts = int(self.bb.get("attempts", 0)) + 1
        self.bb.set("attempts", attempts)
        if attempts > 1:
            self.bb.set("retried", True)
        if self.cancel_check is not None and self.cancel_check():
            self._record_event("cancelled", attempt=attempts)
            return self._fail(FailureCode.CANCELLED, "cancellation requested before execution")
        try:
            result = self.skill.execute(self.ctx, self.params)
        except Exception as error:  # defensive: backend bugs must not kill the agent loop
            self._record_event("exception", attempt=attempts, error=repr(error))
            return self._fail(FailureCode.BACKEND_ERROR, f"skill raised {type(error).__name__}: {error}")
        self.bb.set("result", result.to_dict())
        self._record_event("execute", attempt=attempts, ok=result.ok, code=result.code.value, detail=result.detail)
        if not result.ok:
            return self._fail(result.code, result.detail)
        self.bb.set("code", FailureCode.OK.value)
        self.bb.set("detail", result.detail)
        return Status.SUCCESS


class _VerifyPostcondition(_SkillBehaviour):
    def __init__(self, name: str, ctx: SkillContext, bb: _BB, verify: bool = True) -> None:
        self.verify = verify
        super().__init__(name, ctx, bb)

    def update(self) -> Status:
        if not self.verify:
            # ablation mode: trust the skill result, do not check world state
            result = self.bb.get("result") or {}
            if result.get("ok"):
                self.bb.set("outcome", "ok")
                self.bb.set("post_report", {"ok": True, "skipped": True, "failures": []})
                self._record_event("verify_skipped")
                return Status.SUCCESS
            return self._fail(
                FailureCode(str(result.get("code", FailureCode.POSTCONDITION_FAILED.value))),
                "skill reported failure",
            )
        report = self.skill.check_postconditions(self.ctx.world, self.params)
        self.bb.set("post_report", report.to_dict())
        if report.ok:
            self.bb.set("outcome", "ok")
            self._record_event("verify_ok")
            return Status.SUCCESS
        self._record_event("verify_failed", failures=report.failures)
        return self._fail(FailureCode.POSTCONDITION_FAILED, "; ".join(report.failures))


class _GateRecoverable(py_trees.behaviour.Behaviour):
    """Only let the backtrack branch run for recoverable failure codes."""

    def __init__(self, name: str, bb: _BB) -> None:
        super().__init__(name=name)
        self.bb = bb

    def update(self) -> Status:
        code = self.bb.get("code")
        skill = self.bb.get("skill")
        try:
            failure = FailureCode(code)
        except (ValueError, TypeError):
            return Status.FAILURE
        if skill is not None and failure in skill.recoverable_codes:
            return Status.SUCCESS
        return Status.FAILURE


class _Backtrack(_SkillBehaviour):
    def update(self) -> Status:
        self.bb.set("backtracked", True)
        result = self.ctx.backend.retreat(0.35)
        self._record_event("backtrack", ok=result.ok, detail=result.detail)
        return Status.SUCCESS if result.ok else Status.FAILURE


class _RequestReplan(_SkillBehaviour):
    def update(self) -> Status:
        code = str(self.bb.get("code", FailureCode.POSTCONDITION_FAILED.value))
        self.bb.set("outcome", "cancelled" if code == FailureCode.CANCELLED.value else "replan")
        self._record_event("request_replan", code=code)
        return Status.SUCCESS


class _SyncRetry(py_trees.behaviour.Behaviour):
    """Retry the child synchronously within one tick (all leaves are instantaneous)."""

    def __init__(self, name: str, child: py_trees.behaviour.Behaviour, num_retries: int) -> None:
        self.child = child
        self.num_retries = max(0, int(num_retries))
        super().__init__(name=name)

    def update(self) -> Status:
        status = Status.FAILURE
        for _attempt in range(self.num_retries + 1):
            self.child.tick_once()
            status = self.child.status
            if status != Status.FAILURE:
                return status
        return status


class SkillExecutionTree:
    """Builds and runs the behavior tree for one skill call."""

    def __init__(
        self,
        ctx: SkillContext,
        registry: SkillRegistry,
        max_retries: int = 1,
        allow_backtrack: bool = True,
        verify: bool = True,
        namespace: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.ctx = ctx
        self.registry = registry
        self.max_retries = max(0, int(max_retries))
        self.allow_backtrack = allow_backtrack
        self.verify = verify
        self.namespace = namespace or f"bt_{uuid.uuid4().hex[:10]}"
        self.bb = _BB(self.namespace)
        self.on_event = on_event
        self._tree: py_trees.trees.BehaviourTree | None = None
        self.last_cleanup: list[str] = []

    # ------------------------------------------------------------------ build
    def _build(self, cancel_check: Callable[[], bool] | None) -> None:
        attempt = py_trees.composites.Sequence("attempt", memory=False)
        attempt.add_children(
            [
                _PreconditionCheck("precondition", self.ctx, self.bb),
                _Execute("execute", self.ctx, self.bb, cancel_check),
                _VerifyPostcondition("verify_postcondition", self.ctx, self.bb, self.verify),
            ]
        )
        retry_attempt = _SyncRetry("retry_attempt", attempt, self.max_retries)

        second_attempt = py_trees.composites.Sequence("attempt_after_backtrack", memory=False)
        second_attempt.add_children(
            [
                _Execute("execute_after_backtrack", self.ctx, self.bb, cancel_check),
                _VerifyPostcondition("verify_after_backtrack", self.ctx, self.bb, self.verify),
            ]
        )
        backtrack_branch = py_trees.composites.Sequence("backtrack_branch", memory=False)
        backtrack_branch.add_children(
            [
                _GateRecoverable("gate_recoverable", self.bb),
                _Backtrack("backtrack", self.ctx, self.bb),
                second_attempt,
            ]
        )

        root = py_trees.composites.Selector("skill_execution", memory=False)
        children: list[py_trees.behaviour.Behaviour] = [retry_attempt]
        if self.allow_backtrack:
            children.append(backtrack_branch)
        children.append(_RequestReplan("request_replan", self.ctx, self.bb))
        root.add_children(children)
        self._tree = py_trees.trees.BehaviourTree(root)
        self._root = root

    # -------------------------------------------------------------------- run
    def run(
        self,
        skill_name: str,
        params: dict[str, Any],
        cancel_check: Callable[[], bool] | None = None,
    ) -> BTOutcome:
        skill = self.registry.get(skill_name)
        if skill is None:
            return BTOutcome(
                ok=False,
                outcome="failed",
                code=FailureCode.UNKNOWN_SKILL,
                detail=f"unknown skill {skill_name!r}",
            )
        self.bb.set("skill", skill)
        self.bb.set("params", dict(params or {}))
        self.bb.set("attempts", 0)
        self.bb.set("retried", False)
        self.bb.set("backtracked", False)
        self.bb.set("events", [])
        self._build(cancel_check)
        assert self._tree is not None
        self._tree.tick()
        outcome_name = str(self.bb.get("outcome", "failed"))
        code_value = str(self.bb.get("code", FailureCode.OK.value))
        try:
            code = FailureCode(code_value)
        except ValueError:
            code = FailureCode.BACKEND_ERROR
        result = self.bb.get("result")
        bt_outcome = BTOutcome(
            ok=outcome_name == "ok",
            outcome=outcome_name,
            code=code,
            detail=str(self.bb.get("detail", "")),
            attempts=int(self.bb.get("attempts", 0) or 0),
            retried=bool(self.bb.get("retried", False)),
            backtracked=bool(self.bb.get("backtracked", False)),
            result=result,
            pre_report=self.bb.get("pre_report", {}) or {},
            post_report=self.bb.get("post_report", {}) or {},
            events=list(self.bb.get("events", []) or []),
        )
        if self.on_event:
            for event in bt_outcome.events:
                self.on_event(event)
        self._cleanup()
        return bt_outcome

    # ---------------------------------------------------------------- cleanup
    def _cleanup(self) -> None:
        if self._tree is not None:
            self._tree.root.stop(Status.INVALID)
        self.bb.set("skill", None)
        self.bb.set("params", {})
        self.last_cleanup = self.bb.cleanup()
        self._tree = None

    def cancel(self) -> dict[str, Any]:
        """Explicit cancellation: clear blackboard + backend goal handle."""
        cleanup = self._cleanup()
        goal = self.ctx.backend.cancel_active_goal()
        return {"blackboard_keys_cleared": cleanup, "backend_goal": goal}
