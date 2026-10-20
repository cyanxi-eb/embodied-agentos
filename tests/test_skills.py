"""Skill contracts: schemas, pre/post conditions, failure codes, verification lies."""

from __future__ import annotations

import pytest

from agentos.perception.sim_vlm import SimVLM
from agentos.sim.backend import SimSkillBackend
from agentos.sim.faults import FaultSpec
from agentos.sim.scenes import build_scene
from agentos.skills.base import SkillContext
from agentos.skills.registry import SkillRegistry
from agentos.types import FailureCode

SKILL_NAMES = {
    "navigate_to",
    "find_object",
    "approach_object",
    "pick",
    "place",
    "speak",
    "stop",
    "dock",
}


@pytest.fixture()
def skill_env():
    registry = SkillRegistry()
    backend = SimSkillBackend(build_scene("warehouse"), seed=3)
    backend.safety.register_skills(registry.names())
    ctx = SkillContext(
        backend=backend,
        safety=backend.safety,
        perception=backend.perception,
        vlm=SimVLM(backend.perception),
    )
    return registry, backend, ctx


def test_registry_exposes_exactly_eight_skills(registry):
    assert set(registry.names()) == SKILL_NAMES
    for name in SKILL_NAMES:
        description = registry.describe(name)
        assert description["description"]
        assert description["params_schema"]["type"] == "object"
        assert description["failure_codes"]
        assert description["cancel_semantics"]
        assert description["timeout_s"] > 0


def test_parameter_validation_rejects_bad_values(registry):
    assert registry.validate_call("navigate_to", {})
    assert registry.validate_call("navigate_to", {"target": {"x": 1.0}})
    errors = registry.validate_call("navigate_to", {"zone": "not_a_zone"})
    assert errors
    errors = registry.validate_call("speak", {"text": ""})
    assert errors
    errors = registry.validate_call("pick", {"unknown": 1})
    assert errors
    assert registry.validate_call("navigate_to", {"zone": "zone_a"}) == []
    assert registry.validate_call("find_object", {"query": {"color": "red"}}) == []


def test_find_object_success_sets_detection(skill_env):
    registry, backend, ctx = skill_env
    backend.navigate_to(*build_scene("warehouse").zones["zone_a"], timeout_s=45.0)
    skill = registry.require("find_object")
    params = {"query": {"color": "red", "category": "box"}, "zone": "zone_a"}
    assert skill.check_preconditions(backend.world, params).ok
    result = skill.execute(ctx, params)
    assert result.ok
    assert backend.world.last_detection["object_id"]
    assert skill.check_postconditions(backend.world, params).ok


def test_find_object_failure_code(skill_env):
    registry, backend, ctx = skill_env
    skill = registry.require("find_object")
    params = {"query": {"color": "green", "category": "ball"}, "zone": "zone_a"}
    result = skill.execute(ctx, params)
    assert not result.ok
    assert result.code == FailureCode.NOT_FOUND
    assert not skill.check_postconditions(backend.world, params).ok


def test_pick_precondition_requires_approach(skill_env):
    registry, backend, ctx = skill_env
    skill = registry.require("pick")
    report = skill.check_preconditions(backend.world, {"query": {"color": "red", "category": "box"}})
    assert not report.ok
    assert any("object_in_reach" in failure for failure in report.failures)


def test_full_manipulation_chain(skill_env):
    registry, backend, ctx = skill_env
    scene = backend.scene
    steps = [
        ("navigate_to", {"zone": "zone_a"}),
        ("find_object", {"query": {"color": "red", "category": "box"}, "zone": "zone_a"}),
        ("approach_object", {"query": {"color": "red", "category": "box"}}),
        ("pick", {"query": {"color": "red", "category": "box"}}),
        ("navigate_to", {"zone": "table"}),
        ("place", {"zone": "table"}),
    ]
    for name, params in steps:
        skill = registry.require(name)
        assert skill.check_preconditions(backend.world, params).ok, name
        result = skill.execute(ctx, params)
        assert result.ok, f"{name}: {result.code} {result.detail}"
        assert skill.check_postconditions(backend.world, params).ok, name
    assert backend.world.holding is None
    assert backend.world.last_placed["zone"] == "table"


def test_place_without_holding_fails(skill_env):
    registry, backend, ctx = skill_env
    skill = registry.require("place")
    assert not skill.check_preconditions(backend.world, {"zone": "table"}).ok
    result = skill.execute(ctx, {"zone": "table"})
    assert result.code == FailureCode.NOT_HOLDING


def test_dock_precondition_while_holding(skill_env):
    registry, backend, ctx = skill_env
    backend.world.holding = "obj_red_box_0"
    assert not registry.require("dock").check_preconditions(backend.world, {}).ok


def test_postcondition_is_not_trusting_the_result(skill_env):
    """A skill result may claim success; world-state verification must still fail."""
    registry, backend, ctx = skill_env
    backend.navigate_to(*build_scene("warehouse").zones["zone_a"], timeout_s=45.0)
    find = registry.require("find_object")
    find.execute(ctx, {"query": {"color": "red", "category": "box"}, "zone": "zone_a"})
    approach = registry.require("approach_object")
    approach.execute(ctx, {"query": {"color": "red", "category": "box"}})
    pick = registry.require("pick")
    params = {"query": {"color": "red", "category": "box"}}
    result = pick.execute(ctx, params)
    assert result.ok
    # simulate a lying/accidental state loss: the world says nothing is held
    backend.world.holding = None
    backend.world.held_meta = None
    assert not pick.check_postconditions(backend.world, params).ok


def test_speak_and_stop(skill_env):
    registry, backend, ctx = skill_env
    speak = registry.require("speak")
    assert speak.execute(ctx, {"text": "任务完成"}).ok
    assert speak.check_postconditions(backend.world, {"text": "任务完成"}).ok
    stop = registry.require("stop")
    backend.world.v = 0.3
    assert stop.check_postconditions(backend.world, {}).ok is False
    assert stop.execute(ctx, {}).ok
    assert stop.check_postconditions(backend.world, {}).ok


def test_navigate_out_of_bounds_precondition(skill_env):
    registry, backend, ctx = skill_env
    skill = registry.require("navigate_to")
    result = skill.execute(ctx, {"target": {"x": 40.0, "y": 40.0}})
    assert not result.ok
    assert not skill.check_postconditions(backend.world, {"target": {"x": 40.0, "y": 40.0}}).ok


def test_slip_retry_is_possible_via_tree(registry, backend, ctx):
    """The pick skill returns SLIP once; a second call succeeds (fault consumed)."""
    backend = SimSkillBackend(build_scene("warehouse"), seed=3, faults=[FaultSpec(kind="pick_slip", target="red:box")])
    backend.safety.register_skills(registry.names())
    ctx = SkillContext(backend=backend, safety=backend.safety, perception=backend.perception, vlm=SimVLM(backend.perception))
    backend.navigate_to(*build_scene("warehouse").zones["zone_a"], timeout_s=45.0)
    find = registry.require("find_object")
    find.execute(ctx, {"query": {"color": "red", "category": "box"}, "zone": "zone_a"})
    registry.require("approach_object").execute(ctx, {"query": {"color": "red", "category": "box"}})
    pick = registry.require("pick")
    first = pick.execute(ctx, {"query": {"color": "red", "category": "box"}})
    second = pick.execute(ctx, {"query": {"color": "red", "category": "box"}})
    assert first.code == FailureCode.SLIP
    assert second.ok
