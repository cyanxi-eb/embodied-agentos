"""Task graph: plan nodes, dependencies, topological scheduling and validation.

The plan is a DAG (in practice a chain, but the validator enforces the general
case) so "按技能前置条件调度" is a graph property, not an ad-hoc list index.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

from agentos.skills.registry import SkillRegistry
from agentos.types import WorldState


class PlanError(ValueError):
    pass


@dataclass
class TaskNode:
    node_id: str
    skill: str
    params: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    purpose: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "skill": self.skill,
            "params": self.params,
            "depends_on": list(self.depends_on),
            "purpose": self.purpose,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskNode":
        return cls(
            node_id=str(data["node_id"]),
            skill=str(data["skill"]),
            params=dict(data.get("params") or {}),
            depends_on=list(data.get("depends_on") or []),
            purpose=str(data.get("purpose", "")),
        )


@dataclass
class TaskPlan:
    goal: str
    nodes: list[TaskNode] = field(default_factory=list)
    replanned: bool = False
    #: index of the first node that still has to run (used by replanning: the
    #: prefix of the plan has already been verified and must not re-execute)
    resume_index: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "nodes": [node.to_dict() for node in self.nodes],
            "replanned": self.replanned,
            "resume_index": int(self.resume_index),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskPlan":
        return cls(
            goal=data.get("goal", ""),
            nodes=[TaskNode.from_dict(node) for node in data.get("nodes", [])],
            replanned=bool(data.get("replanned", False)),
            resume_index=int(data.get("resume_index", 0) or 0),
            notes=list(data.get("notes", [])),
        )

    def node_ids(self) -> list[str]:
        return [node.node_id for node in self.nodes]


@dataclass
class PlanValidation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "order": list(self.order),
            "warnings": list(self.warnings),
        }


def topological_order(plan: TaskPlan) -> list[str]:
    """Kahn topological sort; raises ``PlanError`` on unknown deps or cycles."""
    ids = plan.node_ids()
    if len(ids) != len(set(ids)):
        raise PlanError("duplicate node ids in plan")
    known = set(ids)
    indegree = {node_id: 0 for node_id in ids}
    children: dict[str, list[str]] = {node_id: [] for node_id in ids}
    for node in plan.nodes:
        for dependency in node.depends_on:
            if dependency not in known:
                raise PlanError(f"node {node.node_id!r} depends on unknown node {dependency!r}")
            indegree[node.node_id] += 1
            children[dependency].append(node.node_id)
    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        current = heapq.heappop(ready)
        order.append(current)
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(ids):
        raise PlanError("plan contains a dependency cycle")
    return order


def ancestors(plan: TaskPlan) -> dict[str, set[str]]:
    mapping = {node.node_id: node for node in plan.nodes}
    memo: dict[str, set[str]] = {}

    def visit(node_id: str) -> set[str]:
        if node_id in memo:
            return memo[node_id]
        result: set[str] = set()
        for dependency in mapping[node_id].depends_on:
            result.add(dependency)
            result |= visit(dependency)
        memo[node_id] = result
        return result

    return {node_id: visit(node_id) for node_id in mapping}


def validate_plan(
    plan: TaskPlan,
    registry: SkillRegistry,
    world: WorldState | None = None,
) -> PlanValidation:
    """Static validation: skills, schemas, dependencies, ordering prerequisites."""
    errors: list[str] = []
    warnings: list[str] = []
    if not plan.nodes:
        errors.append("plan has no nodes")
        return PlanValidation(ok=False, errors=errors)
    try:
        order = topological_order(plan)
    except PlanError as error:
        return PlanValidation(ok=False, errors=[str(error)])
    mapping = {node.node_id: node for node in plan.nodes}
    skill_by_id: dict[str, str] = {node.node_id: node.skill for node in plan.nodes}
    ancestor_map = ancestors(plan)

    for node in plan.nodes:
        skill = registry.get(node.skill)
        if skill is None:
            errors.append(f"node {node.node_id}: unknown skill {node.skill!r}")
            continue
        param_errors = skill.validate(node.params)
        for error in param_errors:
            errors.append(f"node {node.node_id} ({node.skill}): {error}")
        if skill.requires_prior:
            ancestors_skills = {skill_by_id[other] for other in ancestor_map[node.node_id]}
            if not ancestors_skills & set(skill.requires_prior):
                errors.append(
                    f"node {node.node_id} ({node.skill}) needs one of {sorted(skill.requires_prior)} "
                    f"earlier in the task graph, found {sorted(ancestors_skills)}"
                )
        if not node.purpose:
            warnings.append(f"node {node.node_id} has no purpose annotation")
    if world is not None and world.estop:
        errors.append("world e-stop is latched; plan cannot be executed")
    return PlanValidation(ok=not errors, errors=errors, order=order, warnings=warnings)


def chain_plan(goal: str, steps: list[tuple[str, dict[str, Any], str]]) -> TaskPlan:
    """Convenience builder: a linear chain with sequential dependencies."""
    nodes: list[TaskNode] = []
    for index, (skill, params, purpose) in enumerate(steps):
        node_id = f"n{index:02d}"
        depends_on = [f"n{index - 1:02d}"] if index > 0 else []
        nodes.append(TaskNode(node_id=node_id, skill=skill, params=params, depends_on=depends_on, purpose=purpose))
    return TaskPlan(goal=goal, nodes=nodes)
