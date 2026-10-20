"""Behavior tree: retry / backtrack / replan branches and blackboard cleanup."""

from __future__ import annotations

import pytest
from py_trees.blackboard import Blackboard

from agentos.bt.executor import SkillExecutionTree
from agentos.perception.sim_vlm import SimVLM
from agentos.sim.backend import SimSkillBackend
from agentos.sim.scenes import build_scene
from agentos.skills.base import Condition, Skill, SkillContext
from agentos.skills.registry import SkillRegistry
from agentos.types import FailureCode, SkillCategory, SkillResult, WorldState


class FlakySkill(Skill):
    """Fails a configurable number of times with a configurable code."""

    name = "flaky"
    category = SkillCategory.MANIPULATION
    description = "test skill that fails on purpose"
    params_schema = {"type": "object", "properties": {"fail_times": {"type": "integer"}}, "additionalProperties": False}
    preconditions = (Condition("always", "always true", lambda world, params: True),)
    postconditions = (Condition("always", "always true", lambda world, params: True),)
    timeout_s = 5.0
    failure_codes = (FailureCode.SLIP, FailureCode.BLOCKED, FailureCode.NOT_FOUND)
    recoverable_codes = frozenset({FailureCode.SLIP, FailureCode.BLOCKED, FailureCode.NOT_FOUND})
    idempotent = True
    fail_code = FailureCode.SLIP

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, ctx: SkillContext, params: dict) -> SkillResult:
        self.calls += 1
        if self.calls <= int(params.get("fail_times", 0)):
            return SkillResult.failure(self.name, self.fail_code, "injected failure")
        return SkillResult.success(self.name, "ok")


class NeverSucceeds(FlakySkill):
    name = "never"

    def execute(self, ctx: SkillContext, params: dict) -> SkillResult:
        self.calls += 1
        return SkillResult.failure(self.name, FailureCode.NOT_FOUND, "object missing")


class PostconditionLiar(FlakySkill):
    name = "liar"
    postconditions = (
        Condition("never_true", "world never satisfies", lambda world, params: False),
    )


@pytest.fixture()
def env():
    registry = SkillRegistry()
    backend = SimSkillBackend(build_scene("warehouse"), seed=3)
    backend.safety.register_skills(list(registry.names()) + ["flaky", "never", "liar"])
    ctx = SkillContext(backend=backend, safety=backend.safety, perception=backend.perception, vlm=SimVLM(backend.perception))
    return registry, backend, ctx


def test_success_first_try(env, registry_and_backend=None):
    registry, backend, ctx = env
    registry.register(FlakySkill())
    tree = SkillExecutionTree(ctx, registry, max_retries=1)
    outcome = tree.run("flaky", {"fail_times": 0})
    assert outcome.ok and outcome.outcome == "ok"
    assert outcome.attempts == 1


def test_retry_branch_recovers_transient_failure(env):
    registry, backend, ctx = env
    registry.register(FlakySkill())
    tree = SkillExecutionTree(ctx, registry, max_retries=2)
    outcome = tree.run("flaky", {"fail_times": 1})
    assert outcome.ok
    assert outcome.retried
    assert outcome.attempts >= 2


def test_backtrack_branch_is_used_when_retry_cannot_help(env):
    registry, backend, ctx = env
    skill = FlakySkill()
    skill.fail_code = FailureCode.BLOCKED
    registry.register(skill)
    tree = SkillExecutionTree(ctx, registry, max_retries=0)
    start_pose = (backend.world.robot.x, backend.world.robot.y)
    outcome = tree.run("flaky", {"fail_times": 1})
    assert outcome.ok
    assert outcome.backtracked
    # backtrack physically moved the robot
    assert (backend.world.robot.x, backend.world.robot.y) != start_pose


def test_replan_branch_requests_replan(env):
    registry, backend, ctx = env
    registry.register(NeverSucceeds())
    tree = SkillExecutionTree(ctx, registry, max_retries=1)
    outcome = tree.run("never", {})
    assert not outcome.ok
    assert outcome.outcome == "replan"
    assert outcome.code == FailureCode.NOT_FOUND


def test_postcondition_failure_triggers_replan(env):
    registry, backend, ctx = env
    registry.register(PostconditionLiar())
    tree = SkillExecutionTree(ctx, registry, max_retries=0)
    outcome = tree.run("liar", {"fail_times": 0})
    assert not outcome.ok
    assert outcome.outcome == "replan"
    assert outcome.code == FailureCode.POSTCONDITION_FAILED


def test_cancel_short_circuits_and_cleans_up(env):
    registry, backend, ctx = env
    registry.register(FlakySkill())
    tree = SkillExecutionTree(ctx, registry, max_retries=1)
    outcome = tree.run("flaky", {"fail_times": 0}, cancel_check=lambda: True)
    assert outcome.outcome == "cancelled"
    assert outcome.code == FailureCode.CANCELLED
    keys = [key for key in Blackboard.keys() if tree.namespace in key and Blackboard.exists(key)]
    assert keys == []
    assert backend.active_goal is None


def test_unknown_skill(env):
    registry, backend, ctx = env
    tree = SkillExecutionTree(ctx, registry)
    outcome = tree.run("rm_rf", {})
    assert not outcome.ok
    assert outcome.code == FailureCode.UNKNOWN_SKILL


def test_precondition_failure_is_reported(env):
    registry, backend, ctx = env
    pick = registry.require("pick")
    tree = SkillExecutionTree(ctx, registry)
    outcome = tree.run("pick", {"query": {"color": "red", "category": "box"}})
    assert not outcome.ok
    assert outcome.code == FailureCode.PRECONDITION_FAILED
    assert outcome.pre_report["ok"] is False


def test_blackboard_is_cleared_after_every_run(env):
    registry, backend, ctx = env
    registry.register(FlakySkill())
    tree = SkillExecutionTree(ctx, registry)
    tree.run("flaky", {"fail_times": 0})
    assert [key for key in Blackboard.keys() if tree.namespace in key and Blackboard.exists(key)] == []
