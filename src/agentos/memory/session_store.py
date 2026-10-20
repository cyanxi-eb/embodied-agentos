"""SQLite session and trace storage.

Sessions outlive a process: the agent loop checkpoints into SQLite (LangGraph
SqliteSaver) and the debug console reads the session/event history from this
small store as well.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    instruction TEXT NOT NULL,
    scene TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    result_json TEXT
);
CREATE TABLE IF NOT EXISTS events (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts REAL NOT NULL,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, seq);
"""


class SessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        # WAL + NORMAL keeps per-event commits cheap on local disks (fsync is the
        # bottleneck for a 450-episode evaluation); events are diagnostic data,
        # not a durability-critical ledger.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --------------------------------------------------------------- sessions
    def create_session(
        self,
        session_id: str,
        instruction: str,
        scene: str,
        status: str = "pending",
        meta: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(session_id, instruction, scene, status, created_at, updated_at, result_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    instruction,
                    scene,
                    status,
                    now,
                    now,
                    json.dumps(meta or {}, ensure_ascii=False),
                ),
            )
            self._conn.commit()

    def update_session(self, session_id: str, status: str, result: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ?, result_json = COALESCE(?, result_json) "
                "WHERE session_id = ?",
                (
                    status,
                    time.time(),
                    json.dumps(result, ensure_ascii=False) if result is not None else None,
                    session_id,
                ),
            )
            self._conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT session_id, instruction, scene, status, created_at, updated_at, result_json "
                "FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0],
            "instruction": row[1],
            "scene": row[2],
            "status": row[3],
            "created_at": row[4],
            "updated_at": row[5],
            "result": json.loads(row[6]) if row[6] else None,
        }

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, instruction, scene, status, created_at, updated_at "
                "FROM sessions ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            {
                "session_id": row[0],
                "instruction": row[1],
                "scene": row[2],
                "status": row[3],
                "created_at": row[4],
                "updated_at": row[5],
            }
            for row in rows
        ]

    # ----------------------------------------------------------------- events
    def append_event(self, session_id: str, seq: int, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO events (session_id, seq, ts, type, payload_json) VALUES (?, ?, ?, ?, ?)",
                (session_id, int(seq), time.time(), event_type, json.dumps(payload, ensure_ascii=False)),
            )
            self._conn.commit()

    def get_events(self, session_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, ts, type, payload_json FROM events "
                "WHERE session_id = ? AND seq > ? ORDER BY seq ASC",
                (session_id, int(after_seq)),
            ).fetchall()
        return [
            {"seq": row[0], "ts": row[1], "type": row[2], "payload": json.loads(row[3])}
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
