"""LangGraph state schema (must stay JSON/SQLite-serializable)."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # identity
    session_id: str
    instruction: str
    scene: str
    seed: int
    faults: list[dict[str, Any]]
    step_delay_ms: int
    verify_enabled: bool
    replan_enabled: bool
    # planning
    parse: dict[str, Any]
    plan: dict[str, Any] | None
    plan_index: int
    searched_zones: list[str]
    memory_hits: list[dict[str, Any]]
    # execution
    current: dict[str, Any] | None
    world_before: dict[str, Any] | None
    world_snapshot: dict[str, Any] | None
    bt: dict[str, Any] | None
    results: list[dict[str, Any]]
    replans: int
    skill_attempts: dict[str, int]
    # control / outcome
    status: str
    next_action: str
    failure: dict[str, Any] | None
    goal_verified: bool
    cancel_cleanup: dict[str, Any] | None
    # metrics (wall-clock milliseconds per phase)
    plan_latency_ms: list[float]
    decision_latency_ms: list[float]
    execute_latency_ms: list[float]
    verify_latency_ms: list[float]
    total_latency_ms: list[float]
