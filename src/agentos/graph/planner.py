"""Planner layer: instruction -> task graph.

``RuleBasedPlanner`` is the deterministic baseline used by tests and the
evaluation.  ``LLMPlanner`` is the OpenAI-compatible structured-output adapter
for a real deployment (Qwen2.5 / DeepSeek / any chat-completions endpoint); it is
not exercised without credentials and is documented as optional.

Replanning is planner-owned: the graph only says *which node failed and how*, and
the planner decides whether a new search zone, a retry or "give up" is correct.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agentos.graph.parser import ParsedIntent, parse_instruction
from agentos.graph.task_graph import TaskNode, TaskPlan, chain_plan
from agentos.memory.vector_store import MemoryHit, MemoryStore
from agentos.sim.scenes import Scene
from agentos.skills.registry import SkillRegistry
from agentos.types import FailureCode, WorldState
from agentos.vocab import zone_label


class UnplannableError(RuntimeError):
    """The planner has no remaining strategy (object really absent, retries spent...)."""


@dataclass
class PlannerContext:
    instruction: str
    scene: Scene
    world: WorldState
    parsed: ParsedIntent
    memory: MemoryStore | None = None
    memory_hit: MemoryHit | None = None
    searched_zones: list[str] = field(default_factory=list)
    replan_feedback: dict[str, Any] | None = None
    previous_plan: TaskPlan | None = None
    failed_index: int | None = None
    skill_attempts: dict[str, int] = field(default_factory=dict)


class PlannerBackend(ABC):
    name: str = "base"

    @abstractmethod
    def plan(self, ctx: PlannerContext) -> TaskPlan: ...


# ---------------------------------------------------------------------------
# Deterministic planner
# ---------------------------------------------------------------------------
class RuleBasedPlanner(PlannerBackend):
    name = "rule-based"

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry

    # ------------------------------------------------------------------ zones
    def _zone_order(self, ctx: PlannerContext) -> list[str]:
        order: list[str] = []
        for zone in list(ctx.parsed.zones) + list(ctx.scene.search_order):
            if zone in ctx.scene.zones and zone not in order:
                order.append(zone)
        return order

    def _destination_zone(self, ctx: PlannerContext) -> str:
        destination = ctx.parsed.destination or "table"
        if destination not in ctx.scene.zones and destination != "dock":
            return "table"
        return destination

    def _steps_for_object(
        self,
        ctx: PlannerContext,
        obj: dict[str, Any],
        search_zone: str | None,
        memory_hit: MemoryHit | None,
        include_tail: bool = True,
        destination: str | None = None,
        include_manipulation: bool = True,
    ) -> list[tuple[str, dict[str, Any], str]]:
        query = {"color": obj["color"], "category": obj["category"]}
        steps: list[tuple[str, dict[str, Any], str]] = []
        if memory_hit is not None:
            steps.append(
                (
                    "navigate_to",
                    {"target": {"x": float(memory_hit.data["x"]), "y": float(memory_hit.data["y"])}},
                    f"按记忆前往{obj['color']} {obj['category']}上次所在位置",
                )
            )
            steps.append(("find_object", {"query": query, "zone": "memory"}, "确认记忆中物体仍然存在"))
        else:
            zone = search_zone or ctx.scene.search_order[0]
            steps.append(("navigate_to", {"zone": zone}, f"前往{zone_label(zone)}搜索目标物体"))
            steps.append(("find_object", {"query": query, "zone": zone}, "在搜索区域内定位物体"))
        if include_manipulation:
            steps.append(("approach_object", {"query": query, "standoff": 0.6}, "导航到抓取位姿"))
            steps.append(("pick", {"query": query}, "抓取目标物体"))
        if include_tail:
            destination = destination or self._destination_zone(ctx)
            steps.append(("navigate_to", {"zone": destination}, f"把物体送到{zone_label(destination)}"))
            steps.append(("place", {"zone": destination}, "放置物体并校验落点"))
        return steps

    def _extras(self, ctx: PlannerContext) -> list[tuple[str, dict[str, Any], str]]:
        steps: list[tuple[str, dict[str, Any], str]] = []
        if ctx.parsed.speak_text:
            steps.append(("speak", {"text": ctx.parsed.speak_text}, "播报结果"))
        if ctx.parsed.wants_dock:
            steps.append(("dock", {}, "回充电桩"))
        if ctx.parsed.wants_stop:
            steps.append(("stop", {}, "停止运动"))
        return steps

    # ---------------------------------------------------------------- initial
    def plan(self, ctx: PlannerContext) -> TaskPlan:
        if ctx.replan_feedback is not None:
            return self._replan(ctx)
        intent = ctx.parsed.intent
        if intent == "unknown":
            raise UnplannableError(f"无法从指令中解析出可执行意图: {ctx.instruction!r}")
        if intent in ("fetch", "find_report"):
            object_query = ctx.parsed.objects[0]
            memory_hit = self._memory_hit(ctx, object_query)
            search_zone = self._first_zone(ctx)
            steps = self._steps_for_object(
                ctx,
                object_query,
                search_zone,
                memory_hit,
                include_tail=(intent == "fetch"),
                include_manipulation=(intent == "fetch"),
            )
            if intent == "find_report":
                text = ctx.parsed.speak_text or "已经找到目标物体"
                steps.append(("speak", {"text": text}, "播报搜索结果"))
            elif intent == "fetch":
                steps.extend(self._extras(ctx))
            return self._to_plan(ctx, steps, f"fetch:{object_query}")
        if intent == "multi_fetch":
            steps: list[tuple[str, dict[str, Any], str]] = []
            for obj in ctx.parsed.objects[:2]:
                memory_hit = self._memory_hit(ctx, obj)
                steps.extend(
                    self._steps_for_object(
                        ctx, obj, self._first_zone(ctx), memory_hit, include_tail=True
                    )
                )
            steps.extend(self._extras(ctx))
            return self._to_plan(ctx, steps, "multi_fetch")
        if intent == "patrol":
            zones = list(ctx.parsed.zones) or self._zone_order(ctx)[:2]
            steps = []
            for zone in zones:
                steps.append(("navigate_to", {"zone": zone}, f"巡检{zone_label(zone)}"))
                steps.append(("speak", {"text": f"已完成{zone_label(zone)}巡检"}, "播报巡检结果"))
            steps.extend(self._extras(ctx))
            return self._to_plan(ctx, steps, "patrol")
        if intent == "speak":
            text = ctx.parsed.speak_text or "收到指令"
            return chain_plan("speak", [("speak", {"text": text}, "播报文本")])
        if intent == "dock":
            return chain_plan("dock", [("dock", {}, "回充电桩")])
        if intent == "stop":
            steps = []
            if ctx.parsed.zones:
                steps.append(("navigate_to", {"zone": ctx.parsed.zones[0]}, "先移动到指定区域"))
            steps.append(("stop", {}, "停止运动"))
            return self._to_plan(ctx, steps, "stop")
        if intent == "navigate":
            steps = [("navigate_to", {"zone": zone}, f"移动到{zone_label(zone)}") for zone in ctx.parsed.zones]
            return self._to_plan(ctx, steps, "navigate")
        raise UnplannableError(f"unsupported intent {intent!r}")

    def _to_plan(
        self, ctx: PlannerContext, steps: list[tuple[str, dict[str, Any], str]], goal: str
    ) -> TaskPlan:
        if not steps:
            raise UnplannableError("planner produced no steps")
        plan = chain_plan(goal, steps)
        plan.notes.append(f"planner={self.name}")
        if ctx.memory_hit is not None and any("memory" in node.purpose or "记忆" in node.purpose for node in plan.nodes):
            plan.notes.append("memory_reuse=true")
        return plan

    def _first_zone(self, ctx: PlannerContext) -> str:
        zones = self._zone_order(ctx)
        for zone in zones:
            if zone not in ctx.searched_zones:
                return zone
        return zones[0] if zones else ctx.scene.search_order[0]

    def _memory_hit(self, ctx: PlannerContext, obj: dict[str, Any]) -> MemoryHit | None:
        if ctx.memory is None:
            return None
        if ctx.memory_hit is not None:
            return ctx.memory_hit
        if "memory" in " ".join(ctx.searched_zones):
            return None
        hit = ctx.memory.last_object_location(ctx.scene.name, obj)
        if hit is not None:
            ctx.memory_hit = hit
        return hit

    # ----------------------------------------------------------------- replan
    def _rebuild(
        self,
        previous: TaskPlan,
        prefix: list[TaskNode],
        patch: list[TaskNode],
        rest: list[TaskNode],
        note: str,
    ) -> TaskPlan:
        steps = [(node.skill, node.params, node.purpose) for node in prefix + patch + rest]
        plan = chain_plan(previous.goal, steps)
        plan.replanned = True
        plan.resume_index = len(prefix)
        plan.notes.append(note)
        return plan

    def _replan(self, ctx: PlannerContext) -> TaskPlan:
        feedback = ctx.replan_feedback or {}
        code = str(feedback.get("code", ""))
        skill = str(feedback.get("skill", ""))
        attempts = int(feedback.get("attempts", 0) or 0)
        previous = ctx.previous_plan
        failed_index = ctx.failed_index if ctx.failed_index is not None else 0
        if previous is None or not previous.nodes:
            return self._initial_after_failure(ctx, code)
        failed_index = max(0, min(failed_index, len(previous.nodes) - 1))
        failed_node = previous.nodes[failed_index]
        prefix = previous.nodes[:failed_index]
        rest = previous.nodes[failed_index + 1 :]

        if code in {
            FailureCode.ESTOP.value,
            FailureCode.SAFETY_REJECTED.value,
            FailureCode.OUT_OF_BOUNDS.value,
            FailureCode.CANCELLED.value,
            FailureCode.UNKNOWN_SKILL.value,
            FailureCode.PLAN_INVALID.value,
        }:
            raise UnplannableError(f"failure {code} is not replannable")
        recoverable = {
            FailureCode.NOT_FOUND.value,
            FailureCode.BLOCKED.value,
            FailureCode.TIMEOUT.value,
            FailureCode.NO_PATH.value,
            FailureCode.SLIP.value,
            FailureCode.PRECONDITION_FAILED.value,
            FailureCode.POSTCONDITION_FAILED.value,
        }
        if code not in recoverable:
            raise UnplannableError(f"unknown failure code {code!r}")

        # 1) object not found -> replace the failed search with the next zone
        if code == FailureCode.NOT_FOUND.value and failed_node.skill == "find_object":
            if attempts >= 4:
                raise UnplannableError("target object not found after exhausting search zones and replans")
            next_zone = self._next_unsearched_zone(ctx)
            if next_zone is None:
                raise UnplannableError("all search zones exhausted and object still not found")
            query = failed_node.params.get("query")
            if not query and ctx.parsed.objects:
                obj = ctx.parsed.objects[0]
                query = {"color": obj["color"], "category": obj["category"]}
            patch = [
                TaskNode("patch0", "navigate_to", {"zone": next_zone}, [], f"replan: 前往{zone_label(next_zone)}搜索"),
                TaskNode("patch1", "find_object", {"query": query or {}, "zone": next_zone}, ["patch0"], "replan: 下一搜索区域"),
            ]
            return self._rebuild(previous, prefix, patch, rest, f"replan=search_next_zone:{next_zone}")

        # 2) navigation blocked / timed out -> retry the same node (A* sees the new obstacle)
        if failed_node.skill == "navigate_to" or code in (
            FailureCode.BLOCKED.value,
            FailureCode.TIMEOUT.value,
            FailureCode.NO_PATH.value,
        ):
            if attempts >= 2:
                raise UnplannableError(f"navigation kept failing ({code}) after {attempts} attempts")
            retry = TaskNode(
                "patch0",
                failed_node.skill,
                dict(failed_node.params),
                [],
                failed_node.purpose + " (retry)",
            )
            return self._rebuild(previous, prefix, [retry], rest, f"replan=retry_navigation:{code}")

        # 3) manipulation failure -> re-approach and retry
        if failed_node.skill in ("pick", "approach_object", "place"):
            if attempts >= 2:
                raise UnplannableError(f"manipulation kept failing ({code}) after {attempts} attempts")
            query: dict[str, Any] = {}
            if ctx.parsed.objects:
                obj = ctx.parsed.objects[0]
                query = {"color": obj["color"], "category": obj["category"]}
            if failed_node.skill == "pick" and query:
                patch = [
                    TaskNode("patch0", "approach_object", {"query": query, "standoff": 0.6}, [], "replan: 重新接近"),
                    TaskNode("patch1", "pick", {"query": query}, ["patch0"], "replan: 重新抓取"),
                ]
            elif failed_node.skill == "approach_object" and query:
                patch = [
                    TaskNode("patch0", "approach_object", {"query": query, "standoff": 0.6}, [], "replan: 重新接近")
                ]
            else:
                patch = [
                    TaskNode(
                        "patch0",
                        failed_node.skill,
                        dict(failed_node.params),
                        [],
                        failed_node.purpose + " (retry)",
                    )
                ]
            return self._rebuild(previous, prefix, patch, rest, f"replan=retry_manipulation:{code}")

        raise UnplannableError(f"no replan strategy for skill={skill!r} code={code!r}")

    def _initial_after_failure(self, ctx: PlannerContext, code: str) -> TaskPlan:
        ctx.replan_feedback = None
        plan = self.plan(ctx)
        plan.replanned = True
        plan.notes.append(f"replan=from_scratch:{code}")
        return plan

    def _next_unsearched_zone(self, ctx: PlannerContext) -> str | None:
        zones = self._zone_order(ctx)
        searched = set(ctx.searched_zones)
        for zone in zones:
            if zone not in searched:
                return zone
        return None


# ---------------------------------------------------------------------------
# Optional LLM planner (OpenAI-compatible endpoints)
# ---------------------------------------------------------------------------
class LLMPlanner(PlannerBackend):
    """Structured-output planner over any OpenAI-compatible chat endpoint.

    Configuration: ``AGENTOS_PLANNER_BASE_URL``, ``AGENTOS_PLANNER_API_KEY``,
    ``AGENTOS_PLANNER_MODEL``.  Not used by tests; falls back to the rule planner
    if the model returns an invalid plan.
    """

    name = "llm"

    def __init__(self, registry: SkillRegistry, fallback: RuleBasedPlanner | None = None) -> None:
        self.registry = registry
        self.fallback = fallback or RuleBasedPlanner(registry)
        self.base_url = os.environ.get("AGENTOS_PLANNER_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("AGENTOS_PLANNER_API_KEY", "")
        self.model = os.environ.get("AGENTOS_PLANNER_MODEL", "")
        self.timeout_s = float(os.environ.get("AGENTOS_PLANNER_TIMEOUT_S", "20"))
        self.last_error = ""

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.model)

    def plan(self, ctx: PlannerContext) -> TaskPlan:
        if not self.available:
            return self.fallback.plan(ctx)
        try:
            return self._llm_plan(ctx)
        except Exception as error:  # graceful degradation, logged for the trace
            self.last_error = f"{type(error).__name__}: {error}"
            return self.fallback.plan(ctx)

    def _llm_plan(self, ctx: PlannerContext) -> TaskPlan:
        import httpx

        skills = [
            {
                "name": skill.name,
                "description": skill.description,
                "params_schema": skill.params_schema,
            }
            for skill in self.registry.all()
        ]
        system = (
            "你是机器人的任务规划器。只能使用给定的技能与 JSON Schema 参数，"
            "输出任务图 JSON：{\"goal\": str, \"nodes\": [{\"node_id\": str, \"skill\": str, "
            "\"params\": object, \"depends_on\": [str], \"purpose\": str}]}。"
            "禁止输出速度、电机、关节等底层控制参数。"
        )
        user = json.dumps(
            {
                "instruction": ctx.instruction,
                "scene": {"name": ctx.scene.name, "zones": ctx.scene.zones, "dock": ctx.scene.dock},
                "world": ctx.world.snapshot(),
                "memory": ctx.memory_hit.to_dict() if ctx.memory_hit else None,
                "skills": skills,
            },
            ensure_ascii=False,
        )
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        start, end = content.find("{"), content.rfind("}")
        data = json.loads(content[start : end + 1])
        nodes = [
            TaskNode(
                node_id=str(node["node_id"]),
                skill=str(node["skill"]),
                params=dict(node.get("params") or {}),
                depends_on=list(node.get("depends_on") or []),
                purpose=str(node.get("purpose", "llm")),
            )
            for node in data.get("nodes", [])
        ]
        plan = TaskPlan(goal=str(data.get("goal", ctx.instruction)), nodes=nodes, notes=["planner=llm"])
        from agentos.graph.task_graph import validate_plan

        validation = validate_plan(plan, self.registry)
        if not validation.ok:
            raise ValueError(f"LLM plan failed validation: {validation.errors[:3]}")
        return plan


def build_planner(backend: str, registry: SkillRegistry) -> PlannerBackend:
    if backend == "llm":
        return LLMPlanner(registry)
    return RuleBasedPlanner(registry)
