"""Configuration objects and repo-relative paths.

The repository never writes runtime state into the source tree: Qdrant local
mode, LangGraph checkpoints and trace JSONL all live under ``.cache/`` which is
git-ignored (see ``.gitignore``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Repository root (``.../embodied-agentos``): ``src/agentos/config.py`` -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]


def default_cache_dir() -> Path:
    return Path(os.environ.get("AGENTOS_CACHE_DIR", REPO_ROOT / ".cache"))


@dataclass
class SimConfig:
    """Deterministic simulator knobs."""

    dt: float = 0.1
    max_linear: float = 0.5
    max_angular: float = 1.0
    lookahead_m: float = 0.65
    goal_tolerance_m: float = 0.3
    robot_radius_m: float = 0.24
    grid_resolution_m: float = 0.15
    inflation_m: float = 0.30
    perception_radius_m: float = 3.2
    nav_timeout_s: float = 45.0
    max_steps_per_skill: int = 900
    battery_drain_per_s: float = 0.05
    #: stress evaluation knobs (all deterministic, seeded by the episode seed)
    start_jitter_m: float = 0.0
    start_jitter_yaw: float = 0.0
    perception_dropout_p: float = 0.0
    perception_noise_m: float = 0.0


@dataclass
class RuntimeConfig:
    """Everything the agent runtime needs; roughly one "session template"."""

    scene: str = "warehouse"
    seed: int = 0
    max_replans: int = 3
    bt_max_retries: int = 1
    verify_enabled: bool = True
    replan_enabled: bool = True
    memory_enabled: bool = True
    memory_backend: str = "qdrant"  # qdrant | memory
    planner: str = "rule"  # rule | llm
    memory_limit: int = 3
    sim: SimConfig = field(default_factory=SimConfig)
    step_delay_ms: int = 0
    #: write every trace event into SQLite (debug console); evaluation disables it for speed
    persist_events: bool = True
    cache_dir: str | None = None
    checkpoint_path: str | None = None
    trace_dir: str | None = None
    qdrant_path: str | None = None
    session_db_path: str | None = None

    def resolved_paths(self) -> dict[str, Path]:
        cache = Path(self.cache_dir) if self.cache_dir else default_cache_dir()
        return {
            "cache": cache,
            "checkpoints": Path(self.checkpoint_path) if self.checkpoint_path else cache / "checkpoints.sqlite",
            "traces": Path(self.trace_dir) if self.trace_dir else cache / "traces",
            "qdrant": Path(self.qdrant_path) if self.qdrant_path else cache / "qdrant",
            "sessions": Path(self.session_db_path) if self.session_db_path else cache / "sessions.sqlite",
        }

    def ensure_dirs(self) -> dict[str, Path]:
        paths = self.resolved_paths()
        for key in ("cache", "traces"):
            paths[key].mkdir(parents=True, exist_ok=True)
        paths["qdrant"].mkdir(parents=True, exist_ok=True)
        paths["checkpoints"].parent.mkdir(parents=True, exist_ok=True)
        paths["sessions"].parent.mkdir(parents=True, exist_ok=True)
        return paths

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        sim_data = data.get("sim") or {}
        sim = SimConfig(**{k: v for k, v in sim_data.items() if k in SimConfig.__dataclass_fields__})
        kwargs = {
            k: v
            for k, v in data.items()
            if k in cls.__dataclass_fields__ and k not in {"sim"}
        }
        cfg = cls(sim=sim, **kwargs)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            data[key] = value.to_dict() if hasattr(value, "to_dict") else value
        # SimConfig has no to_dict; serialize manually for traceability
        data["sim"] = {k: getattr(self.sim, k) for k in SimConfig.__dataclass_fields__}
        return data

