"""LangGraph runtime: end-to-end tasks, faults, pause/resume, cancel, e-stop, SQLite resume."""

from __future__ import annotations

import pytest

from agentos.config import RuntimeConfig
from agentos.graph.runtime import AgentRuntime
from agentos.sim.faults import FaultSpec

FETCH = "把A区的红色盒子拿到工作站台"


def _status(runtime: AgentRuntime, session_id: str) -> dict:
    return runtime.get_state(session_id) or {}


def test_end_to_end_fetch_succeeds(runtime):
    result = runtime.start(FETCH, scene="warehouse", seed=3)
    assert result.status == "finished"
    assert result.state.get("goal_verified") is True
    assert result.state.get("replans") == 0
    assert [item["skill"] for item in result.state["results"]] == [
        "navigate_to",
        "find_object",
        "approach_object",
        "pick",
        "navigate_to",
        "place",
    ]
    world = runtime.world(result.session_id)
    assert world is not None
    assert world.holding is None
    assert world.last_placed["zone"] == "table"
    assert runtime.store.get_session(result.session_id)["status"] == "finished"


def test_object_missing_fault_is_recovered_by_replan(runtime):
    result = runtime.start(
        FETCH,
        scene="warehouse",
        seed=3,
        faults=[FaultSpec(kind="object_missing", target="red:box")],
    )
    assert result.status == "finished"
    assert result.state.get("goal_verified") is True
    assert result.state.get("replans", 0) >= 1
    trace = runtime.trace_events(result.session_id)
    assert any(event["type"] == "replan_scheduled" for event in trace)


def test_pause_before_start_and_resume(runtime):
    session_id = "s-pause-test"
    runtime.control.request_pause(session_id)
    result = runtime.start(FETCH, scene="warehouse", seed=3, session_id=session_id)
    assert result.status == "paused"
    assert result.interrupt is not None
    payload = result.interrupt["payloads"][0]["value"]
    assert payload["reason"] == "pause_requested"
    resumed = runtime.resume(session_id, "resume")
    assert resumed.status == "finished"
    assert resumed.state.get("goal_verified") is True


def test_cancel_from_pause(runtime):
    session_id = "s-cancel-test"
    runtime.control.request_pause(session_id)
    result = runtime.start(FETCH, scene="warehouse", seed=3, session_id=session_id)
    assert result.status == "paused"
    resumed = runtime.resume(session_id, "cancel")
    assert resumed.status == "cancelled"
    cleanup = resumed.state.get("cancel_cleanup") or {}
    assert cleanup.get("backend_goal", {}).get("handle_cleared") is True
    assert runtime.get_context(session_id).safety.estop_engaged is False


def test_estop_blocks_dispatch_until_operator_reset(runtime):
    session_id = "s-estop-test"
    runtime.control.request_pause(session_id)
    runtime.start(FETCH, scene="warehouse", seed=3, session_id=session_id)
    runtime.engage_estop("test estop", session_id=session_id)
    failed = runtime.resume(session_id, "resume")
    assert failed.status == "failed"
    assert failed.state["failure"]["code"] == "ESTOP"
    assert runtime.reset_estop("operator-1", "WRONG", session_id)["results"][0]["ok"] is False
    assert runtime.reset_estop("operator-1", "AGENTOS-RESET", session_id)["results"][0]["ok"] is True


def test_verify_disabled_still_succeeds_on_clean_task(runtime):
    result = runtime.start(FETCH, scene="warehouse", seed=5, verify_enabled=False)
    assert result.status == "finished"


def test_replan_disabled_fails_on_fault(runtime):
    result = runtime.start(
        FETCH,
        scene="warehouse",
        seed=3,
        faults=[FaultSpec(kind="object_missing", target="red:box")],
        replan_enabled=False,
    )
    assert result.status == "failed"
    assert result.state["failure"]["code"] == "NOT_FOUND"


def test_resume_after_process_restart_via_sqlite_checkpoint(tmp_path):
    cache = tmp_path / "shared-cache"
    config = RuntimeConfig(cache_dir=str(cache), persist_events=False, memory_enabled=False)
    runtime_a = AgentRuntime(config)
    session_id = "s-cross-process"
    runtime_a.control.request_pause(session_id)
    paused = runtime_a.start(FETCH, scene="warehouse", seed=7, session_id=session_id)
    assert paused.status == "paused"
    # A brand new runtime (new simulation context) resumes from the SQLite checkpoint.
    runtime_b = AgentRuntime(RuntimeConfig(cache_dir=str(cache), persist_events=False, memory_enabled=False))
    resumed = runtime_b.resume(session_id, "resume")
    assert resumed.status == "finished", resumed.state.get("failure")
    assert resumed.state.get("goal_verified") is True
    runtime_a.close()
    runtime_b.close()


def test_unknown_instruction_fails_cleanly(runtime):
    result = runtime.start("打开空调", scene="warehouse", seed=1)
    assert result.status == "failed"
    assert result.state["failure"]["code"] in {"UNPLANNABLE", "PLAN_INVALID"}
    trace = runtime.trace_events(result.session_id)
    assert any(event["type"] == "unplannable" for event in trace)


def test_trace_contains_plan_and_verification_events(runtime):
    result = runtime.start(FETCH, scene="warehouse", seed=3)
    trace = runtime.trace_events(result.session_id)
    types = [event["type"] for event in trace]
    for expected in ("session_start", "plan", "plan_validated", "skill_dispatch", "skill_result", "postcondition", "finish"):
        assert expected in types, expected
    # every skill result is followed by an independent world-state verification
    assert types.count("skill_result") == types.count("postcondition")
