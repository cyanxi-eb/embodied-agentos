"""Evaluation runner: 50 instructions x 3 scenes x 3 repeats + fault injection + ablation."""

from __future__ import annotations

import json
import math
import statistics
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from agentos.config import RuntimeConfig, SimConfig
from agentos.graph.runtime import AgentRuntime
from agentos.sim.faults import FaultSpec, fault_suite
from agentos.sim.scenes import SCENE_NAMES
from agentos.types import FailureCode, WorldState
from evaluation.instructions import INSTRUCTIONS, Instruction, stress_instructions


# ---------------------------------------------------------------------------
# Goal verification (evaluator ground truth, independent from the runtime)
# ---------------------------------------------------------------------------
def _zone_center(world: WorldState, zone: str) -> tuple[float, float] | None:
    if zone == "dock":
        return world.dock
    if zone in world.zones:
        return (float(world.zones[zone][0]), float(world.zones[zone][1]))
    return None


def _object_at_zone(world: WorldState, color: str, category: str, zone: str, tolerance: float = 0.95) -> bool:
    center = _zone_center(world, zone)
    if center is None:
        return False
    for obj in world.free_objects():
        if obj.color == color and obj.category == category:
            if math.hypot(obj.x - center[0], obj.y - center[1]) <= tolerance:
                return True
    return False


def check_goal(goal: dict[str, Any], world: WorldState) -> tuple[bool, str]:
    """Return ``(satisfied, detail)`` for the evaluator's ground-truth goal."""
    goal_type = goal.get("type")
    if goal_type == "object_at_zone":
        ok = _object_at_zone(world, goal["color"], goal["category"], goal["zone"])
        return ok, f"object {goal['color']}/{goal['category']} at {goal['zone']}"
    if goal_type == "objects_at_zone":
        details = []
        ok = True
        for obj in goal["objects"]:
            item_ok = _object_at_zone(world, obj["color"], obj["category"], goal["zone"])
            ok = ok and item_ok
            details.append(f"{obj['color']}/{obj['category']}={item_ok}")
        return ok, ", ".join(details)
    if goal_type == "composite":
        details = []
        ok = True
        for sub_goal in goal.get("goals", []):
            sub_ok, detail = check_goal(sub_goal, world)
            ok = ok and sub_ok
            details.append(f"{sub_goal.get('type')}={sub_ok}({detail})")
        return ok, "; ".join(details)
    if goal_type == "docked":
        near = math.hypot(world.robot.x - world.dock[0], world.robot.y - world.dock[1]) <= 0.45
        return bool(world.docked and near), f"docked={world.docked} near_dock={near}"
    if goal_type == "spoken":
        text = str(goal.get("contains", ""))
        ok = bool(world.last_spoken and text in world.last_spoken)
        return ok, f"last_spoken={world.last_spoken!r}"
    if goal_type == "stopped":
        ok = abs(world.v) < 1e-6 and abs(world.w) < 1e-6
        return ok, f"v={world.v:.3f} w={world.w:.3f}"
    if goal_type == "patrol":
        zones = list(goal.get("zones", []))
        ok = all(zone in world.visited_zones for zone in zones)
        return ok, f"visited={world.visited_zones}"
    if goal_type == "find_report":
        detection = world.last_detection or {}
        detection_ok = (
            detection.get("color") == goal.get("color") and detection.get("category") == goal.get("category")
        )
        spoken = str(goal.get("contains", ""))
        spoken_ok = bool(world.last_spoken and spoken in world.last_spoken)
        return bool(detection_ok and spoken_ok), f"detection={detection.get('object_id')} spoken={world.last_spoken!r}"
    return False, f"unknown goal type {goal_type!r}"


# ---------------------------------------------------------------------------
# Episode execution
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    instruction_id: str
    instruction: str
    intent: str
    scene: str
    repeat: int
    seed: int
    status: str
    goal_verified: bool
    evaluator_ok: bool
    goal_detail: str
    replans: int
    memory_hit: bool
    sim_time_s: float
    wall_time_s: float
    plan_latencies_ms: list[float] = field(default_factory=list)
    decision_latencies_ms: list[float] = field(default_factory=list)
    execute_latencies_ms: list[float] = field(default_factory=list)
    failure_code: str | None = None
    failure_detail: str | None = None
    fault: dict[str, Any] | None = None
    skills: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.status == "finished" and self.goal_verified and self.evaluator_ok

    @property
    def runtime_success(self) -> bool:
        return self.status == "finished" and self.goal_verified

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "instruction": self.instruction,
            "intent": self.intent,
            "scene": self.scene,
            "repeat": self.repeat,
            "seed": self.seed,
            "status": self.status,
            "goal_verified": self.goal_verified,
            "evaluator_ok": self.evaluator_ok,
            "goal_detail": self.goal_detail,
            "success": self.success,
            "replans": self.replans,
            "memory_hit": self.memory_hit,
            "sim_time_s": round(self.sim_time_s, 3),
            "wall_time_s": round(self.wall_time_s, 4),
            "plan_latencies_ms": list(self.plan_latencies_ms),
            "decision_latencies_ms": list(self.decision_latencies_ms),
            "execute_latencies_ms": list(self.execute_latencies_ms),
            "failure_code": self.failure_code,
            "failure_detail": self.failure_detail,
            "fault": self.fault,
            "skills": list(self.skills),
        }


def episode_seed(instruction_id: str, scene: str, repeat: int) -> int:
    payload = f"{instruction_id}|{scene}|{repeat}".encode("utf-8")
    return zlib.crc32(payload) & 0xFFFF


def run_episode(
    runtime: AgentRuntime,
    instruction: Instruction,
    scene: str,
    repeat: int,
    fault: FaultSpec | None = None,
    verify_enabled: bool | None = None,
    replan_enabled: bool | None = None,
) -> EpisodeResult:
    seed = episode_seed(instruction.id, scene, repeat)
    started = time.perf_counter()
    result = runtime.start(
        instruction.text,
        scene=scene,
        seed=seed,
        faults=[fault.to_dict()] if fault is not None else [],
        verify_enabled=verify_enabled,
        replan_enabled=replan_enabled,
    )
    wall = time.perf_counter() - started
    state = result.state or {}
    world = runtime.world(result.session_id)
    if world is None:  # pragma: no cover - defensive
        world = WorldState()
    evaluator_ok, detail = check_goal(instruction.goal, world)
    failure = state.get("failure") or {}
    skills = [
        {"skill": item.get("skill"), "ok": item.get("ok"), "code": item.get("code")}
        for item in state.get("results") or []
    ]
    episode = EpisodeResult(
        instruction_id=instruction.id,
        instruction=instruction.text,
        intent=instruction.intent,
        scene=scene,
        repeat=repeat,
        seed=seed,
        status=result.status,
        goal_verified=bool(state.get("goal_verified", False)),
        evaluator_ok=evaluator_ok,
        goal_detail=detail,
        replans=int(state.get("replans", 0) or 0),
        memory_hit=bool(state.get("memory_hits")),
        sim_time_s=float(world.sim_time),
        wall_time_s=wall,
        plan_latencies_ms=[float(value) for value in (state.get("plan_latency_ms") or [])],
        decision_latencies_ms=[float(value) for value in (state.get("decision_latency_ms") or [])],
        execute_latencies_ms=[float(value) for value in (state.get("execute_latency_ms") or [])],
        failure_code=(failure.get("code") if failure else None),
        failure_detail=(failure.get("detail") if failure else None),
        fault=fault.to_dict() if fault else None,
        skills=skills,
    )
    runtime.cleanup_session(result.session_id)
    return episode


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def percentile(values: Iterable[float], q: float) -> float:
    data = sorted(float(value) for value in values)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return data[lower]
    weight = position - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "p50": round(percentile(values, 0.5), 3),
        "p95": round(percentile(values, 0.95), 3),
        "mean": round(statistics.fmean(values), 3) if values else 0.0,
        "max": round(max(values), 3) if values else 0.0,
    }


def summarize(episodes: list[EpisodeResult]) -> dict[str, Any]:
    total = len(episodes)
    success = sum(1 for episode in episodes if episode.success)
    runtime_success = sum(1 for episode in episodes if episode.runtime_success)
    with_replan = [episode for episode in episodes if episode.replans > 0]
    recovered = [episode for episode in with_replan if episode.success]
    memory_episodes = [episode for episode in episodes if episode.memory_hit]
    memory_success = [episode for episode in memory_episodes if episode.success]
    no_memory_episodes = [episode for episode in episodes if not episode.memory_hit]
    buckets: dict[str, int] = {}
    for episode in episodes:
        if not episode.success:
            code = episode.failure_code or ("EVALUATOR_MISMATCH" if episode.runtime_success else "UNKNOWN")
            buckets[code] = buckets.get(code, 0) + 1
    plan_latencies = [value for episode in episodes for value in episode.plan_latencies_ms]
    decision_latencies = [value for episode in episodes for value in episode.decision_latencies_ms]
    return {
        "episodes": total,
        "success": success,
        "success_rate": round(100.0 * success / total, 2) if total else 0.0,
        "runtime_declared_success": runtime_success,
        "runtime_declared_success_rate": round(100.0 * runtime_success / total, 2) if total else 0.0,
        "runtime_evaluator_mismatch": sum(1 for episode in episodes if episode.runtime_success != episode.success),
        "replan_episodes": len(with_replan),
        "replan_rate": round(100.0 * len(with_replan) / total, 2) if total else 0.0,
        "replan_recovered": len(recovered),
        "replan_recovery_rate": round(100.0 * len(recovered) / len(with_replan), 2) if with_replan else 0.0,
        "memory_hit_episodes": len(memory_episodes),
        "memory_hit_rate": round(100.0 * len(memory_episodes) / total, 2) if total else 0.0,
        "memory_success_rate": round(100.0 * len(memory_success) / len(memory_episodes), 2) if memory_episodes else 0.0,
        "no_memory_success_rate": round(
            100.0 * sum(1 for episode in no_memory_episodes if episode.success) / len(no_memory_episodes), 2
        )
        if no_memory_episodes
        else 0.0,
        "sim_time_s": _latency_summary([episode.sim_time_s for episode in episodes]),
        "wall_time_per_episode_s": _latency_summary([episode.wall_time_s for episode in episodes]),
        "plan_latency_ms": _latency_summary(plan_latencies),
        "decision_latency_ms": _latency_summary(decision_latencies),
        "execute_latency_ms": _latency_summary([value for episode in episodes for value in episode.execute_latencies_ms]),
        "failure_buckets": dict(sorted(buckets.items(), key=lambda item: (-item[1], item[0]))),
    }


def group_metrics(episodes: list[EpisodeResult], key: str) -> dict[str, Any]:
    groups: dict[str, list[EpisodeResult]] = {}
    for episode in episodes:
        groups.setdefault(str(getattr(episode, key)), []).append(episode)
    return {name: summarize(group) for name, group in sorted(groups.items())}


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    scenes: tuple[str, ...] = SCENE_NAMES
    repeats: int = 3
    instructions: list[Instruction] | None = None
    run_faults: bool = True
    run_ablation: bool = True
    run_stress: bool = True
    persist_events: bool = False
    cache_dir: str | None = None
    output_dir: str = "reports"
    quick: bool = False

    def instruction_list(self) -> list[Instruction]:
        if self.instructions is not None:
            return self.instructions
        return INSTRUCTIONS


def run_evaluation(config: EvalConfig, progress: Any = None) -> dict[str, Any]:
    started = time.perf_counter()
    instructions = config.instruction_list()
    if config.quick:
        instructions = instructions[:10]
        scenes = config.scenes[:1]
        repeats = 1
        run_faults = True
        fault_limit = 4
        ablation_instructions = instructions
        ablation_scene = scenes[0]
    else:
        scenes = config.scenes
        repeats = config.repeats
        run_faults = config.run_faults
        fault_limit = 20
        ablation_instructions = instructions[:25]
        ablation_scene = "warehouse"

    # A fresh cache dir keeps runs reproducible: memory must not leak between
    # evaluation runs (it does accumulate *within* a run, which is intended).
    # It lives under .cache/ rather than tempfile.gettempdir(): some sandboxes
    # point TEMP at the repository root and pollute the working tree.
    cache_dir = config.cache_dir or _fresh_cache_dir("eval")
    runtime = AgentRuntime(
        RuntimeConfig(scene="warehouse", cache_dir=cache_dir, persist_events=config.persist_events)
    )
    episodes: list[EpisodeResult] = []

    def report_progress(index: int, total: int, episode: EpisodeResult) -> None:
        if progress is not None:
            progress(index, total, episode)

    total_main = len(instructions) * len(scenes) * repeats
    index = 0
    for instruction in instructions:
        for scene in scenes:
            for repeat in range(repeats):
                episode = run_episode(runtime, instruction, scene, repeat)
                episodes.append(episode)
                index += 1
                report_progress(index, total_main, episode)

    report: dict[str, Any] = {
        "schema": "agentos-eval/v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "host": {
            "python": _python_version(),
            "cpu_count": _cpu_count(),
            "platform": "windows-10 (development host, no GPU)",
        },
        "config": {
            "cache_dir": str(cache_dir),
            "instructions": len(instructions),
            "scenes": list(scenes),
            "repeats": repeats,
            "episodes": total_main,
            "quick": config.quick,
            "persist_events": config.persist_events,
        },
        "overall": summarize(episodes),
        "by_scene": group_metrics(episodes, "scene"),
        "by_intent": group_metrics(episodes, "intent"),
        "failures": [episode.to_dict() for episode in episodes if not episode.success][:30],
        "episodes": [episode.to_dict() for episode in episodes],
    }

    # ---- fault injection: 4 fault kinds x 5 targets = 20 -------------------
    if run_faults:
        fault_episodes: list[EpisodeResult] = []
        fault_cases = fault_suite()[:fault_limit]
        for fault_case in fault_cases:
            instruction = _fault_instruction(fault_case)
            episode = run_episode(runtime, instruction, "warehouse", 0, fault=fault_case)
            episode.fault = fault_case.to_dict()
            fault_episodes.append(episode)
            report_progress(index, total_main + len(fault_cases), episode)
        report["fault_injection"] = {
            "cases": len(fault_episodes),
            "summary": summary_with_faults(fault_episodes),
            "episodes": [episode.to_dict() for episode in fault_episodes],
        }

    # ---- ablation: trust the skill result, never replan --------------------
    # A separate runtime + cache isolates the ablation variable: memory from the
    # main run must not leak in, otherwise the comparison would be confounded.
    ablation_runtime: AgentRuntime | None = None
    if config.run_ablation:
        ablation_runtime = AgentRuntime(
            RuntimeConfig(
                scene="warehouse",
                cache_dir=_fresh_cache_dir("ablation"),
                persist_events=config.persist_events,
            )
        )
        ablation_episodes: list[EpisodeResult] = []
        for instruction in ablation_instructions:
            episode = run_episode(
                ablation_runtime,
                instruction,
                ablation_scene,
                0,
                verify_enabled=False,
                replan_enabled=False,
            )
            episode.intent = f"ablation:{episode.intent}"
            ablation_episodes.append(episode)
        fault_ablation: list[EpisodeResult] = []
        for fault_case in (fault_suite()[:8] if not config.quick else fault_suite()[:2]):
            instruction = _fault_instruction(fault_case)
            episode = run_episode(
                ablation_runtime, instruction, "warehouse", 0, fault=fault_case, verify_enabled=False, replan_enabled=False
            )
            episode.intent = f"ablation_fault:{episode.intent}"
            fault_ablation.append(episode)
        report["ablation"] = {
            "description": "verify_enabled=false + replan_enabled=false (trust the skill result, never replan)",
            "clean_scene": ablation_scene,
            "clean": summarize(ablation_episodes),
            "fault_recovery": summary_with_faults(fault_ablation),
            "fault_episodes": [episode.to_dict() for episode in fault_ablation],
        }

    # ---- stress / generalisation subset ----------------------------------
    if config.run_stress:
        report["stress"] = run_stress_evaluation(config, progress=progress)

    report["wall_time_s"] = round(time.perf_counter() - started, 2)
    report["episodes_total"] = (
        len(episodes)
        + len(report.get("fault_injection", {}).get("episodes", []))
        + (len(ablation_episodes) if config.run_ablation else 0)
        + (len(fault_ablation) if config.run_ablation else 0)
        + len(report.get("stress", {}).get("episodes", []))
    )
    runtime.close()
    if ablation_runtime is not None:
        ablation_runtime.close()
    return report


STRESS_VARIANTS: list[dict[str, Any]] = [
    {
        "name": "perturbed_start",
        "description": "初始位姿扰动：随机平移 0.3-1.0m + 朝向 ±0.6rad",
        "sim": {"start_jitter_m": 1.0, "start_jitter_yaw": 0.6},
        "fault_kind": None,
    },
    {
        "name": "moved_target",
        "description": "目标物体被挪到下一个搜索区域（object_missing 故障）",
        "sim": {},
        "fault_kind": "object_missing",
    },
    {
        "name": "noisy_perception_blocked_path",
        "description": "15% 检测漏检 + 0.12m 定位噪声 + 路径中途临时阻断",
        "sim": {"perception_dropout_p": 0.15, "perception_noise_m": 0.12},
        "fault_kind": "path_blocked",
    },
    {
        "name": "paraphrased_language",
        "description": "改写/省略措辞：物体类别与目的地用同义/模糊说法（如“物件”“那边”），检验语言泛化",
        "sim": {},
        "fault_kind": None,
        "language": True,
    },
]


_AMBIGUOUS_CATEGORY_WORDS = ("盒子", "杯子", "球", "包裹", "工具", "瓶子")
_AMBIGUOUS_DEST_WORDS = ("工作站台", "工作台", "充电桩", "餐桌", "样品交接台", "交接台")


def paraphrase_instruction(text: str) -> str:
    """Rewrite an instruction with vague references: the rule parser will mis-plan
    some of them - that is exactly the language-generalisation gap being measured."""
    result = text
    for word in _AMBIGUOUS_CATEGORY_WORDS:
        result = result.replace(word, "物件")
    for word in _AMBIGUOUS_DEST_WORDS:
        result = result.replace(word, "那边")
    return result


def _goal_objects(goal: dict[str, Any]) -> list[dict[str, Any]]:
    goal_type = goal.get("type")
    if goal_type == "object_at_zone":
        return [{"color": goal["color"], "category": goal["category"]}]
    if goal_type == "objects_at_zone":
        return list(goal.get("objects", []))
    if goal_type == "composite":
        objects: list[dict[str, Any]] = []
        for sub_goal in goal.get("goals", []):
            objects.extend(_goal_objects(sub_goal))
        return objects
    return []


def run_stress_evaluation(config: EvalConfig, progress: Any = None) -> dict[str, Any]:
    """Stress / generalisation subset: 30 instructions x 3 perturbed scenarios x 1 repeat.

    The main 450-episode suite is a clean deterministic world; this subset adds
    initial-pose perturbation, relocated targets, perception noise/dropout and a
    temporary path blockage so the reported success rate is not artificially 100%.
    """
    instructions = stress_instructions(30)
    variants = STRESS_VARIANTS[:2] if config.quick else STRESS_VARIANTS
    episodes: list[EpisodeResult] = []
    variant_reports: list[dict[str, Any]] = []
    total = len(instructions) * len(variants)
    index = 0
    for variant in variants:
        sim_defaults = {field: getattr(SimConfig(), field) for field in SimConfig.__dataclass_fields__}
        sim_defaults.update(variant["sim"])
        runtime = AgentRuntime(
            RuntimeConfig(
                scene="warehouse",
                sim=SimConfig(**sim_defaults),
                cache_dir=_fresh_cache_dir(f"stress-{variant['name']}"),
                persist_events=config.persist_events,
            )
        )
        variant_episodes: list[EpisodeResult] = []
        for instruction in instructions:
            episode_instruction = instruction
            if variant.get("language"):
                episode_instruction = Instruction(
                    id=f"{instruction.id}-para",
                    text=paraphrase_instruction(instruction.text),
                    intent=instruction.intent,
                    goal=instruction.goal,
                    object_zone=instruction.object_zone,
                )
            fault = None
            if variant["fault_kind"]:
                objects = _goal_objects(instruction.goal)
                if objects:
                    target = f"{objects[0]['color']}:{objects[0]['category']}"
                    fault_kind = variant["fault_kind"]
                else:
                    target = ""
                    fault_kind = "path_blocked"
                fault = FaultSpec(kind=fault_kind, target=target)
            episode = run_episode(runtime, episode_instruction, "warehouse", 0, fault=fault)
            episode.instruction_id = instruction.id
            episode.intent = f"stress:{variant['name']}:{episode.intent}"
            episode.fault = fault.to_dict() if fault else None
            variant_episodes.append(episode)
            episodes.append(episode)
            index += 1
            if progress is not None:
                progress(index, total, episode)
        runtime.close()
        summary = summarize(variant_episodes)
        summary.update({"name": variant["name"], "description": variant["description"], "sim_config": variant["sim"]})
        variant_reports.append(summary)
    return {
        "instructions": len(instructions),
        "variants": variant_reports,
        "overall": summarize(episodes),
        "episodes": [episode.to_dict() for episode in episodes],
    }


def summary_with_faults(episodes: list[EpisodeResult]) -> dict[str, Any]:
    summary = summarize(episodes)
    recovered = sum(1 for episode in episodes if episode.success)
    summary["recovery_rate"] = round(100.0 * recovered / len(episodes), 2) if episodes else 0.0
    return summary


def _fresh_cache_dir(prefix: str) -> str:
    """Fresh cache dir under the repo's git-ignored .cache/ directory."""
    import uuid
    from pathlib import Path

    base = Path(__file__).resolve().parents[1] / ".cache" / "eval-tmp" / f"{prefix}-{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def _fault_instruction(fault_case: FaultSpec) -> Instruction:
    """Instruction whose first search zone is the faulted object's canonical zone."""
    from evaluation.instructions import CATEGORY_ZH, COLOR_ZH, ZONE_ZH

    target = fault_case.target.split(":") if ":" in fault_case.target else ["red", "box"]
    color, category = target[0], target[1]
    zone = _canonical_zone(color, category)
    return Instruction(
        id=f"F-{fault_case.kind}-{color}-{category}",
        text=f"把{ZONE_ZH[zone]}的{COLOR_ZH[color]}{CATEGORY_ZH[category]}拿到工作站台",
        intent="fetch",
        goal={"type": "object_at_zone", "color": color, "category": category, "zone": "table"},
        object_zone=zone,
    )


def _canonical_zone(color: str, category: str) -> str:
    mapping = {
        ("red", "box"): "zone_a",
        ("blue", "cup"): "zone_b",
        ("green", "ball"): "zone_c",
        ("yellow", "package"): "zone_b",
        ("black", "tool"): "zone_a",
        ("white", "bottle"): "zone_c",
        ("red", "cup"): "zone_b",
        ("blue", "box"): "zone_a",
    }
    return mapping.get((color, category), "zone_a")


def _python_version() -> str:
    import sys

    return sys.version.split()[0]


def _cpu_count() -> int:
    import os

    return os.cpu_count() or 1
