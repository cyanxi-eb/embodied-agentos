"""Task graph validation + instruction parser + planner/replan strategies."""

from __future__ import annotations

import pytest

from agentos.graph.parser import parse_instruction
from agentos.graph.planner import PlannerContext, RuleBasedPlanner, UnplannableError
from agentos.graph.task_graph import (
    PlanError,
    TaskNode,
    TaskPlan,
    chain_plan,
    topological_order,
    validate_plan,
)
from agentos.memory.vector_store import InMemoryMemoryStore
from agentos.sim.scenes import build_scene
from agentos.skills.registry import SkillRegistry
from agentos.types import WorldState


@pytest.fixture()
def registry():
    return SkillRegistry()


@pytest.fixture()
def scene():
    return build_scene("warehouse")


@pytest.fixture()
def planner(registry):
    return RuleBasedPlanner(registry)


def _ctx(planner, scene, instruction, memory=None, **kwargs):
    parsed = parse_instruction(instruction)
    return PlannerContext(
        instruction=instruction,
        scene=scene,
        world=WorldState(zones={k: list(v) for k, v in scene.zones.items()}, dock=scene.dock),
        parsed=parsed,
        memory=memory,
        **kwargs,
    )


# ---------------------------------------------------------------- task graph
def test_chain_plan_topological_order():
    plan = chain_plan("goal", [("navigate_to", {"zone": "zone_a"}, "a"), ("find_object", {"query": {"color": "red"}}, "b")])
    assert topological_order(plan) == ["n00", "n01"]
    assert plan.nodes[1].depends_on == ["n00"]


def test_cycle_detection():
    plan = TaskPlan(
        goal="cycle",
        nodes=[
            TaskNode("a", "speak", {"text": "x"}, ["b"]),
            TaskNode("b", "speak", {"text": "y"}, ["a"]),
        ],
    )
    with pytest.raises(PlanError):
        topological_order(plan)
    report = validate_plan(plan, SkillRegistry())
    assert not report.ok
    assert "cycle" in report.errors[0]


def test_unknown_dependency():
    plan = TaskPlan(goal="x", nodes=[TaskNode("a", "speak", {"text": "x"}, ["missing"])])
    report = validate_plan(plan, SkillRegistry())
    assert not report.ok


def test_validate_rejects_bad_skill_and_params(registry):
    plan = TaskPlan(
        goal="bad",
        nodes=[
            TaskNode("a", "does_not_exist", {}, []),
            TaskNode("b", "speak", {"text": ""}, ["a"]),
        ],
    )
    report = validate_plan(plan, registry)
    assert not report.ok
    assert len(report.errors) >= 2


def test_validate_enforces_static_ordering(registry):
    plan = TaskPlan(goal="bad order", nodes=[TaskNode("a", "pick", {"query": {"color": "red"}}, [])])
    report = validate_plan(plan, registry)
    assert not report.ok
    assert "needs one of" in report.errors[0]


def test_valid_full_plan(registry, planner, scene):
    ctx = _ctx(planner, scene, "把A区的红色盒子拿到工作站台")
    plan = planner.plan(ctx)
    report = validate_plan(plan, registry)
    assert report.ok, report.errors
    assert [node.skill for node in plan.nodes] == [
        "navigate_to",
        "find_object",
        "approach_object",
        "pick",
        "navigate_to",
        "place",
    ]


# -------------------------------------------------------------------- parser
@pytest.mark.parametrize(
    "text,intent",
    [
        ("把A区的红色盒子拿到工作站台", "fetch"),
        ("帮我找到蓝色杯子并放到充电桩旁边", "fetch"),
        ("把A区的红色盒子和B区的蓝色杯子都拿到工作站台", "multi_fetch"),
        ("回充电桩", "dock"),
        ("播报“任务完成”", "speak"),
        ("立刻停止", "stop"),
        ("依次检查A区和B区", "patrol"),
        ("找到C区的绿色球后播报“找到了”", "find_report"),
        ("先去A区", "navigate"),
        ("打开空调", "unknown"),
    ],
)
def test_parser_intents(text, intent):
    parsed = parse_instruction(text)
    assert parsed.intent == intent


def test_parser_extracts_object_and_destination():
    parsed = parse_instruction("把B区的蓝色杯子拿到充电桩")
    assert parsed.objects == [{"color": "blue", "category": "cup"}]
    assert parsed.destination == "dock"
    assert parsed.zones == ["zone_b"]


def test_parser_extracts_two_objects():
    parsed = parse_instruction("把A区的红色盒子和B区的蓝色杯子都拿到工作站台")
    assert len(parsed.objects) == 2


# ------------------------------------------------------------------- planner
def test_fetch_plan_with_dock_extras(planner, scene):
    ctx = _ctx(planner, scene, "把A区的红色盒子放到工作站台，然后回充")
    plan = planner.plan(ctx)
    assert plan.nodes[-1].skill == "dock"


def test_fetch_plan_with_speak(planner, scene):
    ctx = _ctx(planner, scene, "把A区的红色盒子拿到工作站台，然后播报“已送达”")
    plan = planner.plan(ctx)
    assert plan.nodes[-1].skill == "speak"
    assert plan.nodes[-1].params["text"] == "已送达"


def test_multi_fetch_plan_has_two_pick_place_cycles(planner, scene):
    ctx = _ctx(planner, scene, "把A区的红色盒子和B区的蓝色杯子都拿到工作站台")
    plan = planner.plan(ctx)
    skills = [node.skill for node in plan.nodes]
    assert skills.count("pick") == 2
    assert skills.count("place") == 2


def test_patrol_plan(planner, scene):
    ctx = _ctx(planner, scene, "依次检查A区和B区")
    plan = planner.plan(ctx)
    assert [node.skill for node in plan.nodes].count("navigate_to") == 2
    assert plan.nodes[-1].skill == "speak"


def test_memory_influences_initial_plan(registry, scene):
    memory = InMemoryMemoryStore()
    memory.remember_object("warehouse", {"color": "red", "category": "box"}, "obj_red_box_0", 4.4, 4.9, "zone_a")
    planner = RuleBasedPlanner(registry)
    ctx = _ctx(planner, scene, "把A区的红色盒子拿到工作站台", memory=memory)
    plan = planner.plan(ctx)
    assert plan.nodes[0].skill == "navigate_to"
    assert "target" in plan.nodes[0].params
    assert "记忆" in plan.nodes[0].purpose
    assert any("memory_reuse=true" == note for note in plan.notes)


def test_memory_does_not_match_a_different_object(registry, scene):
    memory = InMemoryMemoryStore()
    memory.remember_object("warehouse", {"color": "blue", "category": "cup"}, "obj_blue_cup_1", 5.9, 8.1, "zone_b")
    planner = RuleBasedPlanner(registry)
    ctx = _ctx(planner, scene, "把A区的红色盒子拿到工作站台", memory=memory)
    plan = planner.plan(ctx)
    assert "target" not in plan.nodes[0].params, "memory of another object must not be reused"


def test_replan_not_found_patches_plan_in_place(planner, registry, scene):
    ctx = _ctx(planner, scene, "把B区的蓝色杯子拿到工作站台")
    plan = planner.plan(ctx)
    validation = validate_plan(plan, registry)
    assert validation.ok
    failed_index = 1  # find_object
    feedback = {"code": "NOT_FOUND", "skill": "find_object", "attempts": 1, "node_index": failed_index}
    replan_ctx = _ctx(
        planner,
        scene,
        "把B区的蓝色杯子拿到工作站台",
        searched_zones=["zone_b"],
        replan_feedback=feedback,
        previous_plan=plan,
        failed_index=failed_index,
    )
    new_plan = planner.plan(replan_ctx)
    assert new_plan.replanned
    assert new_plan.resume_index == failed_index
    # prefix (navigate) is preserved, the failed search is replaced by another zone
    assert new_plan.nodes[0].skill == "navigate_to"  # already executed prefix
    assert new_plan.nodes[1].skill == "navigate_to"  # replan: next search zone
    assert new_plan.nodes[2].skill == "find_object"
    assert new_plan.nodes[2].params["zone"] in {"zone_a", "zone_c"}
    skills = [node.skill for node in new_plan.nodes]
    assert skills.count("pick") == 1, "replan must not duplicate the fetch tail"
    assert len(new_plan.nodes) == 7  # executed nav + replan nav/find + remaining approach/pick/nav/place
    assert validate_plan(new_plan, registry).ok


def test_replan_navigation_retry(planner, registry, scene):
    ctx = _ctx(planner, scene, "把A区的红色盒子拿到工作站台")
    plan = planner.plan(ctx)
    feedback = {"code": "BLOCKED", "skill": "navigate_to", "attempts": 1, "node_index": 0}
    replan_ctx = _ctx(
        planner,
        scene,
        "把A区的红色盒子拿到工作站台",
        replan_feedback=feedback,
        previous_plan=plan,
        failed_index=0,
    )
    new_plan = planner.plan(replan_ctx)
    assert new_plan.replanned
    assert new_plan.resume_index == 0
    assert new_plan.nodes[0].skill == "navigate_to"
    assert "retry" in new_plan.nodes[0].purpose
    assert validate_plan(new_plan, registry).ok


def test_replan_gives_up_after_budget(planner, scene):
    ctx = _ctx(planner, scene, "把A区的红色盒子拿到工作站台")
    plan = planner.plan(ctx)
    feedback = {"code": "BLOCKED", "skill": "navigate_to", "attempts": 2, "node_index": 0}
    replan_ctx = _ctx(
        planner,
        scene,
        "把A区的红色盒子拿到工作站台",
        replan_feedback=feedback,
        previous_plan=plan,
        failed_index=0,
    )
    with pytest.raises(UnplannableError):
        planner.plan(replan_ctx)


def test_replan_not_found_exhausted_zones(planner, scene):
    ctx = _ctx(planner, scene, "帮我找到红色盒子并送到工作站台", searched_zones=["zone_a", "zone_b", "zone_c"])
    feedback = {"code": "NOT_FOUND", "skill": "find_object", "attempts": 4, "node_index": 1}
    previous = planner._initial_after_failure  # noqa: F841 - documentation aid
    plan = planner.plan(_ctx(planner, scene, "帮我找到红色盒子并送到工作站台"))
    replan_ctx = _ctx(
        planner,
        scene,
        "帮我找到红色盒子并送到工作站台",
        searched_zones=["zone_a", "zone_b", "zone_c"],
        replan_feedback=feedback,
        previous_plan=plan,
        failed_index=1,
    )
    with pytest.raises(UnplannableError):
        planner.plan(replan_ctx)


def test_llm_planner_falls_back_without_credentials(registry, scene):
    from agentos.graph.planner import LLMPlanner

    planner = LLMPlanner(registry)
    planner.base_url = ""
    planner.model = ""
    ctx = _ctx(planner, scene, "回充电桩")
    plan = planner.plan(ctx)
    assert plan.nodes[-1].skill == "dock"
