"""Skill abstraction: parameter schema + pre/post conditions + failure codes.

A skill has exactly one job: turn validated parameters into a backend command and
return a structured result.  It never decides *when* it is called and it never
verifies itself: the behavior tree and the agent loop verify against world state.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from agentos.perception.base import PerceptionBackend, VLMBackend
from agentos.safety.gateway import SafetyGateway
from agentos.skills.schema import validate_params
from agentos.types import FailureCode, SkillCategory, WorldState

Predicate = Callable[[WorldState, dict[str, Any]], bool]


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    predicate: Predicate

    def holds(self, world: WorldState, params: dict[str, Any]) -> bool:
        try:
            return bool(self.predicate(world, params))
        except Exception:  # a malformed world must not crash the verification path
            return False


@dataclass
class ConditionReport:
    ok: bool
    failures: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "failures": list(self.failures), "checked": list(self.checked)}


@dataclass
class SkillContext:
    """Everything a skill may touch.  Built per session by the agent runtime."""

    backend: Any
    safety: SafetyGateway
    perception: PerceptionBackend
    vlm: VLMBackend
    memory: Any = None
    session_id: str = ""

    @property
    def world(self) -> WorldState:
        return self.backend.world


class Skill(ABC):
    name: str = ""
    category: SkillCategory = SkillCategory.NAVIGATION
    description: str = ""
    params_schema: dict[str, Any] = {}
    preconditions: tuple[Condition, ...] = ()
    postconditions: tuple[Condition, ...] = ()
    timeout_s: float = 45.0
    failure_codes: tuple[FailureCode, ...] = ()
    recoverable_codes: frozenset[FailureCode] = frozenset()
    idempotent: bool = True
    #: Static plan-ordering requirement: at least one of these skills must appear earlier.
    requires_prior: tuple[str, ...] = ()
    #: Human-readable cancellation semantics (shown by MCP / the debug console).
    cancel_semantics: str = "cancel active goal and clear its handle"

    # -------------------------------------------------------------- conditions
    def check_preconditions(self, world: WorldState, params: dict[str, Any]) -> ConditionReport:
        failures: list[str] = []
        checked: list[str] = []
        for condition in self.preconditions:
            checked.append(condition.name)
            if not condition.holds(world, params):
                failures.append(f"{condition.name}: {condition.description}")
        return ConditionReport(ok=not failures, failures=failures, checked=checked)

    def check_postconditions(self, world: WorldState, params: dict[str, Any]) -> ConditionReport:
        failures: list[str] = []
        checked: list[str] = []
        for condition in self.postconditions:
            checked.append(condition.name)
            if not condition.holds(world, params):
                failures.append(f"{condition.name}: {condition.description}")
        return ConditionReport(ok=not failures, failures=failures, checked=checked)

    # ------------------------------------------------------------------ calls
    def validate(self, params: dict[str, Any]) -> list[str]:
        return validate_params(self.params_schema, params)

    @abstractmethod
    def execute(self, ctx: SkillContext, params: dict[str, Any]) -> Any:
        """Execute and return a ``SkillResult`` (typed loosely to avoid cycles)."""

    # --------------------------------------------------------------- metadata
    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "params_schema": self.params_schema,
            "preconditions": [{"name": c.name, "description": c.description} for c in self.preconditions],
            "postconditions": [{"name": c.name, "description": c.description} for c in self.postconditions],
            "timeout_s": self.timeout_s,
            "failure_codes": [code.value for code in self.failure_codes],
            "recoverable_codes": sorted(code.value for code in self.recoverable_codes),
            "idempotent": self.idempotent,
            "requires_prior": list(self.requires_prior),
            "cancel_semantics": self.cancel_semantics,
        }


# ---------------------------------------------------------------------------
# Condition helpers shared by the 8 skills
# ---------------------------------------------------------------------------
def battery_ok(world: WorldState, params: dict[str, Any] | None = None) -> bool:
    return world.battery > 2.0


def not_estop(world: WorldState, params: dict[str, Any] | None = None) -> bool:
    return not world.estop


def dist(world: WorldState, x: float, y: float) -> float:
    return math.hypot(world.robot.x - x, world.robot.y - y)


def resolve_zone_xy(world: WorldState, zone: str) -> tuple[float, float] | None:
    if zone == "dock":
        return world.dock
    if zone in world.zones:
        return (float(world.zones[zone][0]), float(world.zones[zone][1]))
    return None


def resolve_query(params: dict[str, Any]) -> dict[str, str]:
    """Accept either ``{"query": {"color","category"}}`` or flat keys."""
    query: dict[str, str] = {}
    nested = params.get("query")
    if isinstance(nested, dict):
        if nested.get("color"):
            query["color"] = str(nested["color"])
        if nested.get("category"):
            query["category"] = str(nested["category"])
    if params.get("color"):
        query["color"] = str(params["color"])
    if params.get("category"):
        query["category"] = str(params["category"])
    return query


def resolve_target_xy(params: dict[str, Any], world: WorldState) -> tuple[float, float] | None:
    target = params.get("target")
    if isinstance(target, dict) and "x" in target and "y" in target:
        return (float(target["x"]), float(target["y"]))
    if "x" in params and "y" in params:
        return (float(params["x"]), float(params["y"]))
    zone = params.get("zone") or params.get("destination")
    if isinstance(zone, str) and zone:
        return resolve_zone_xy(world, zone)
    return None


def detection_matches(detection: dict[str, Any] | None, query: dict[str, str]) -> bool:
    if not detection:
        return False
    if query.get("color") and str(detection.get("color", "")).lower() != query["color"].lower():
        return False
    if query.get("category") and str(detection.get("category", "")).lower() != query["category"].lower():
        return False
    return True


def known_object_id(world: WorldState, params: dict[str, Any]) -> str | None:
    """Resolve which object a pick/approach refers to, from world knowledge."""
    object_id = params.get("object_id")
    if isinstance(object_id, str) and object_id:
        return object_id
    query = resolve_query(params)
    if not query:
        if world.last_detection:
            return str(world.last_detection.get("object_id"))
        if world.last_approached:
            return str(world.last_approached.get("object_id"))
        return None
    # 1) last detection / last approach
    for candidate in (world.last_detection, world.last_approached):
        if candidate and detection_matches(candidate, query):
            return str(candidate.get("object_id"))
    # 2) nearest known object matching the query
    candidates = [
        obj for obj in world.free_objects() if obj.matches(query)
    ]
    if candidates:
        return min(
            candidates,
            key=lambda obj: math.hypot(obj.x - world.robot.x, obj.y - world.robot.y),
        ).object_id
    return None
