"""FastAPI debug console: task lifecycle, pause/resume, SSE, safety endpoints."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from agentos.api.server import create_app
from agentos.config import RuntimeConfig
from agentos.graph.runtime import AgentRuntime


@pytest.fixture()
def client(tmp_path):
    config = RuntimeConfig(
        scene="warehouse",
        cache_dir=str(tmp_path / "cache"),
        persist_events=False,
        memory_enabled=False,
    )
    runtime = AgentRuntime(config)
    app = create_app(runtime)
    with TestClient(app) as test_client:
        yield test_client
    runtime.close()


def _wait_for_status(client: TestClient, task_id: str, statuses: set[str], timeout_s: float = 30.0) -> dict:
    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        response = client.get(f"/tasks/{task_id}")
        assert response.status_code == 200
        last = response.json()
        if last["status"] in statuses:
            return last
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} never reached {statuses}; last={last}")


def test_index_and_health(client):
    index = client.get("/")
    assert index.status_code == 200
    assert "Embodied AgentOS" in index.text
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert health["skills"] == 8
    assert health["sim"] == "2d-deterministic"


def test_skills_endpoint(client):
    payload = client.get("/skills").json()
    names = {skill["name"] for skill in payload["skills"]}
    assert len(names) == 8
    assert "navigate_to" in names


def test_create_task_and_list(client):
    response = client.post(
        "/tasks",
        json={"instruction": "把A区的红色盒子拿到工作站台", "scene": "warehouse", "seed": 3},
    )
    assert response.status_code == 200
    task = response.json()
    assert task["status"] == "running"
    final = _wait_for_status(client, task["id"], {"finished", "failed"})
    assert final["status"] == "finished"
    assert final["goal_verified"] is True
    tasks = client.get("/tasks").json()["tasks"]
    assert any(item["id"] == task["id"] for item in tasks)


def test_pause_resume_via_api(client):
    response = client.post(
        "/tasks",
        json={
            "instruction": "把A区的红色盒子拿到工作站台",
            "scene": "warehouse",
            "seed": 3,
            "step_delay_ms": 80,
        },
    )
    task = response.json()
    assert client.post(f"/tasks/{task['id']}/pause").status_code == 200
    paused = _wait_for_status(client, task["id"], {"paused", "finished", "failed"})
    assert paused["status"] == "paused", "step delay should keep the task alive long enough to pause"
    resume = client.post(f"/tasks/{task['id']}/resume")
    assert resume.status_code == 200
    final = _wait_for_status(client, task["id"], {"finished", "failed", "cancelled"})
    assert final["status"] == "finished"


def test_cancel_task(client):
    response = client.post(
        "/tasks",
        json={
            "instruction": "把A区的红色盒子拿到工作站台",
            "scene": "warehouse",
            "seed": 3,
            "step_delay_ms": 120,
        },
    )
    task = response.json()
    time.sleep(0.15)
    cancel = client.post(f"/tasks/{task['id']}/cancel")
    assert cancel.status_code == 200
    final = _wait_for_status(client, task["id"], {"cancelled", "failed"})
    assert final["status"] in {"cancelled", "failed"}


def test_sse_stream_emits_plan_event(client):
    response = client.post(
        "/tasks",
        json={"instruction": "回充电桩", "scene": "warehouse", "seed": 1},
    )
    task = response.json()
    seen = []
    with client.stream("GET", f"/events/{task['id']}") as stream:
        for line in stream.iter_lines():
            if line.startswith("data: "):
                seen.append(line)
            if len(seen) >= 3:
                break
    assert seen, "SSE stream returned no events"
    assert "session_start" in seen[0] or "plan" in seen[0]


def test_estop_and_reset_endpoints(client):
    estop = client.post("/safety/estop", json={"reason": "test"}).json()
    assert estop["engaged"] is True
    wrong = client.post("/safety/reset", json={"operator_id": "op", "code": "NOPE"}).json()
    assert wrong["results"] == [] or wrong["results"][0]["ok"] is False
    right = client.post("/safety/reset", json={"operator_id": "op", "code": "AGENTOS-RESET"}).json()
    assert right["results"] == [] or right["results"][0]["ok"] is True


def test_unknown_task_returns_404(client):
    assert client.get("/tasks/nope").status_code == 404
    assert client.post("/tasks/nope/pause").status_code == 404


def test_invalid_scene_rejected(client):
    response = client.post("/tasks", json={"instruction": "回充电桩", "scene": "moon"})
    assert response.status_code == 400
