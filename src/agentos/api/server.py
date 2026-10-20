"""FastAPI debug console with SSE event streaming.

Endpoints
---------
POST   /tasks                  create a task (instruction + scene + seed)
GET    /tasks                  list tasks
GET    /tasks/{id}             task detail
POST   /tasks/{id}/pause       request pause (takes effect at the next skill boundary)
POST   /tasks/{id}/resume      resume a paused task
POST   /tasks/{id}/cancel      cancel
GET    /events/{id}            SSE stream of trace events
POST   /safety/estop           latched e-stop
POST   /safety/reset           operator reset (operator_id + code)
GET    /skills                 skill catalogue (same data as MCP)
GET    /                       single-page debug console
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agentos.config import RuntimeConfig
from agentos.graph.runtime import AgentRuntime
from agentos.sim.faults import FaultSpec
from agentos.sim.scenes import SCENE_NAMES
from agentos.trace.recorder import TraceRecorder

TERMINAL_STATUSES = {"finished", "failed", "cancelled"}


class TaskRequest(BaseModel):
    instruction: str = Field(..., min_length=1, max_length=300)
    scene: str = Field(default="warehouse")
    seed: int = 0
    faults: list[dict[str, Any]] = Field(default_factory=list)
    step_delay_ms: int = Field(default=0, ge=0, le=2000, description="debug throttling for pause/resume demos")


class EstopRequest(BaseModel):
    reason: str = "api emergency stop"
    session_id: str | None = None


class ResetRequest(BaseModel):
    operator_id: str
    code: str
    session_id: str | None = None


class TaskRecord:
    def __init__(self, session_id: str, instruction: str, scene: str, seed: int) -> None:
        self.session_id = session_id
        self.instruction = instruction
        self.scene = scene
        self.seed = seed
        self.status = "created"
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.error: str | None = None
        self.result: dict[str, Any] | None = None
        self.thread: threading.Thread | None = None
        self.recorder: TraceRecorder | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.session_id,
            "instruction": self.instruction,
            "scene": self.scene,
            "seed": self.seed,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "result": self.result,
        }


class TaskManager:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime
        self.tasks: dict[str, TaskRecord] = {}
        self.lock = threading.RLock()

    # --------------------------------------------------------------- lifecycle
    def create(self, request: TaskRequest) -> TaskRecord:
        if request.scene not in SCENE_NAMES:
            raise HTTPException(status_code=400, detail=f"unknown scene {request.scene!r}")
        session_id = f"s-{uuid.uuid4().hex[:12]}"
        record = TaskRecord(session_id, request.instruction, request.scene, request.seed)
        recorder = TraceRecorder(
            session_id,
            store=self.runtime.store if self.runtime.config.persist_events else None,
            trace_dir=self.runtime.paths["traces"],
        )
        record.recorder = recorder
        with self.lock:
            self.tasks[session_id] = record
        record.status = "running"
        record.thread = threading.Thread(
            target=self._run,
            args=(record, request),
            name=f"agentos-{session_id}",
            daemon=True,
        )
        record.thread.start()
        return record

    def _run(self, record: TaskRecord, request: TaskRequest) -> None:
        try:
            result = self.runtime.start(
                request.instruction,
                scene=request.scene,
                seed=request.seed,
                faults=[FaultSpec.from_dict(fault).to_dict() for fault in request.faults],
                session_id=record.session_id,
                step_delay_ms=request.step_delay_ms,
                recorder=record.recorder,
            )
            self._update(record, result.status, result.to_dict())
        except Exception as error:  # pragma: no cover - defensive for the console
            self._update(record, "failed", None, error=f"{type(error).__name__}: {error}")

    def _update(self, record: TaskRecord, status: str, result: dict[str, Any] | None, error: str | None = None) -> None:
        with self.lock:
            record.status = status
            record.updated_at = time.time()
            record.result = result
            record.error = error

    def get(self, session_id: str) -> TaskRecord:
        with self.lock:
            record = self.tasks.get(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown task {session_id!r}")
        return record

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            records = sorted(self.tasks.values(), key=lambda item: item.created_at, reverse=True)
        return [record.to_dict() for record in records]

    # -------------------------------------------------------------- control
    def pause(self, session_id: str) -> dict[str, Any]:
        record = self.get(session_id)
        flags = self.runtime.pause(session_id)
        return {"id": session_id, "status": record.status, "flags": flags}

    def resume(self, session_id: str, decision: str = "resume") -> dict[str, Any]:
        record = self.get(session_id)
        if record.status != "paused":
            raise HTTPException(status_code=409, detail=f"task is {record.status}, not paused")
        record.status = "running"
        record.thread = threading.Thread(
            target=self._resume_worker,
            args=(record, decision),
            name=f"agentos-resume-{session_id}",
            daemon=True,
        )
        record.thread.start()
        return {"id": session_id, "status": "running"}

    def _resume_worker(self, record: TaskRecord, decision: str) -> None:
        try:
            result = self.runtime.resume(record.session_id, decision)
            self._update(record, result.status, result.to_dict())
        except Exception as error:  # pragma: no cover
            self._update(record, "failed", None, error=f"{type(error).__name__}: {error}")

    def cancel(self, session_id: str) -> dict[str, Any]:
        record = self.get(session_id)
        result = self.runtime.cancel(session_id)
        if record.status == "paused":
            decoded = json.loads(json.dumps(result, ensure_ascii=False))
            self._update(record, str(decoded.get("status", "cancelled")), None)
        return {"id": session_id, "cancel": result}

    def estop(self, request: EstopRequest) -> dict[str, Any]:
        return self.runtime.engage_estop(request.reason, session_id=request.session_id)

    def reset_estop(self, request: ResetRequest) -> dict[str, Any]:
        return self.runtime.reset_estop(request.operator_id, request.code, session_id=request.session_id)


def create_app(runtime: AgentRuntime | None = None) -> FastAPI:
    runtime = runtime or AgentRuntime(RuntimeConfig())
    manager = TaskManager(runtime)
    app = FastAPI(title="Embodied AgentOS Debug Console", version="0.1.0")
    app.state.runtime = runtime
    app.state.manager = manager

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "memory_backend": getattr(runtime.memory, "backend_name", "disabled"),
            "memory_items": runtime.memory.count() if runtime.memory else 0,
            "skills": len(runtime.registry.names()),
            "planner": runtime.planner.name,
            "sim": "2d-deterministic",
        }

    @app.get("/skills")
    def skills() -> dict[str, Any]:
        return {"skills": runtime.registry.list_descriptions()}

    @app.post("/tasks")
    def create_task(request: TaskRequest) -> dict[str, Any]:
        record = manager.create(request)
        return record.to_dict()

    @app.get("/tasks")
    def list_tasks() -> dict[str, Any]:
        return {"tasks": manager.list()}

    @app.get("/tasks/{session_id}")
    def get_task(session_id: str) -> dict[str, Any]:
        record = manager.get(session_id)
        state = runtime.get_state(session_id) or {}
        return {
            **record.to_dict(),
            "plan": state.get("plan"),
            "plan_index": state.get("plan_index"),
            "replans": state.get("replans", 0),
            "failure": state.get("failure"),
            "goal_verified": state.get("goal_verified"),
        }

    @app.post("/tasks/{session_id}/pause")
    def pause_task(session_id: str) -> dict[str, Any]:
        return manager.pause(session_id)

    @app.post("/tasks/{session_id}/resume")
    def resume_task(session_id: str, decision: str = "resume") -> dict[str, Any]:
        return manager.resume(session_id, decision)

    @app.post("/tasks/{session_id}/cancel")
    def cancel_task(session_id: str) -> dict[str, Any]:
        return manager.cancel(session_id)

    @app.post("/safety/estop")
    def estop(request: EstopRequest) -> dict[str, Any]:
        return manager.estop(request)

    @app.post("/safety/reset")
    def reset_estop(request: ResetRequest) -> dict[str, Any]:
        return manager.reset_estop(request)

    @app.get("/events/{session_id}")
    async def events(session_id: str, max_events: int = 5000) -> StreamingResponse:
        record = manager.get(session_id)
        recorder = record.recorder

        async def stream():
            index = 0
            sent = 0
            idle_since = time.time()
            yield "retry: 1000\n\n"
            while True:
                if recorder is not None:
                    batch = recorder.events
                else:
                    batch = []
                if index < len(batch):
                    for event in batch[index:]:
                        if sent >= max_events:
                            return
                        yield f"event: {event.get('type', 'message')}\n"
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                        sent += 1
                    index = len(batch)
                    idle_since = time.time()
                status = record.status
                if status in TERMINAL_STATUSES and index >= len(batch):
                    return
                if time.time() - idle_since > 300:
                    return
                await asyncio.sleep(0.05)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    return app


_INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


app = create_app() if __name__ == "__main__" else None
