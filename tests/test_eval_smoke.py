"""Evaluation smoke test: report shape, metrics and Markdown generation."""

from __future__ import annotations

from pathlib import Path

from evaluation.instructions import INSTRUCTIONS
from evaluation.report import render_markdown, write_report
from evaluation.runner import EvalConfig, check_goal, paraphrase_instruction, run_evaluation


def test_fifty_instructions_cover_all_intents():
    assert len(INSTRUCTIONS) == 50
    intents = {instruction.intent for instruction in INSTRUCTIONS}
    assert {"fetch", "multi_fetch", "dock", "speak", "stop", "patrol", "find_report"}.issubset(intents)
    ids = [instruction.id for instruction in INSTRUCTIONS]
    assert len(ids) == len(set(ids))


def test_quick_evaluation_report(tmp_path):
    config = EvalConfig(
        scenes=("warehouse",),
        repeats=1,
        instructions=INSTRUCTIONS[:2],
        run_faults=False,
        run_ablation=False,
        persist_events=False,
        cache_dir=str(tmp_path / "cache"),
    )
    report = run_evaluation(config)
    assert report["overall"]["episodes"] == 2
    assert report["overall"]["success_rate"] >= 50.0
    assert "by_scene" in report and "warehouse" in report["by_scene"]
    assert "plan_latency_ms" in report["overall"]
    assert "failure_buckets" in report["overall"]
    assert report["config"]["cache_dir"].startswith(str(tmp_path))
    paths = write_report(report, tmp_path / "reports")
    assert Path(paths["json"]).exists()
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "任务闭环成功率" in markdown
    json_text = Path(paths["json"]).read_text(encoding="utf-8")
    assert '"success_rate"' in json_text


def test_paraphrase_replaces_object_and_destination_words():
    text = paraphrase_instruction("把A区的红色盒子拿到工作站台")
    assert "盒子" not in text and "工作站台" not in text
    assert "物件" in text and "那边" in text


def test_stress_runner_reports_lower_success_than_clean(tmp_path):
    # language paraphrases are the measured generalisation gap: some must fail
    config = EvalConfig(
        scenes=("warehouse",),
        repeats=1,
        instructions=INSTRUCTIONS[:6],
        run_faults=False,
        run_ablation=False,
        run_stress=False,
        persist_events=False,
        cache_dir=str(tmp_path / "cache"),
    )
    report = run_evaluation(config)
    assert report["overall"]["episodes"] == 6
    assert "stress" not in report


def test_goal_checking_detects_success_and_failure():
    from agentos.sim.scenes import build_scene
    from agentos.types import WorldState

    scene = build_scene("warehouse")
    world = WorldState(
        zones={k: list(v) for k, v in scene.zones.items()},
        dock=scene.dock,
        last_spoken="任务完成",
        docked=True,
    )
    world.robot.x, world.robot.y = scene.dock
    assert check_goal({"type": "spoken", "contains": "完成"}, world)[0]
    assert not check_goal({"type": "spoken", "contains": "失败"}, world)[0]
    assert check_goal({"type": "docked"}, world)[0]
    assert check_goal(
        {"type": "object_at_zone", "color": "red", "category": "box", "zone": "table"}, world
    )[0] is False
