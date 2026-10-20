"""Structured trace recorder: the "what the agent said vs what the robot did" log.

Each event is written to SQLite (for the debug console) and, on close, to a JSONL
file (for offline replay with ``scripts/replay_trace.py``).  Observers can
subscribe to live events (used by the SSE endpoint and the demo recorder).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from agentos.memory.session_store import SessionStore

Observer = Callable[[dict[str, Any]], None]


class TraceRecorder:
    def __init__(
        self,
        session_id: str,
        store: SessionStore | None = None,
        trace_dir: str | Path | None = None,
        echo: bool = False,
    ) -> None:
        self.session_id = session_id
        self.store = store
        self.echo = echo
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self._events: list[dict[str, Any]] = []
        self._observers: list[Observer] = []
        self._lock = threading.RLock()
        self._seq = 0

    # ---------------------------------------------------------------- observe
    def subscribe(self, observer: Observer) -> Callable[[], None]:
        with self._lock:
            self._observers.append(observer)

        def unsubscribe() -> None:
            with self._lock:
                if observer in self._observers:
                    self._observers.remove(observer)

        return unsubscribe

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    # ------------------------------------------------------------------ emit
    def emit(self, event_type: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            event = {
                "seq": self._seq,
                "ts": round(time.time(), 4),
                "type": event_type,
                "session_id": self.session_id,
                **payload,
            }
            self._events.append(event)
            observers = list(self._observers)
        if self.store is not None:
            try:
                self.store.append_event(self.session_id, self._seq, event_type, event)
            except Exception:  # pragma: no cover - persistence must not break the loop
                pass
        if self.echo:
            print(json.dumps(event, ensure_ascii=False))
        for observer in observers:
            try:
                observer(event)
            except Exception:  # pragma: no cover
                pass
        return event

    # ----------------------------------------------------------------- close
    def write_jsonl(self) -> str | None:
        if self.trace_dir is None:
            return None
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{self.session_id}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return str(path)

    def close(self) -> str | None:
        return self.write_jsonl()
