"""Shared pytest fixtures.

The repository is used as ``src`` layout without an editable install in some
environments, so the tests put ``src`` (and the repo root for ``evaluation``) on
``sys.path`` explicitly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agentos.config import RuntimeConfig  # noqa: E402
from agentos.graph.runtime import AgentRuntime  # noqa: E402
from agentos.sim.backend import SimSkillBackend  # noqa: E402
from agentos.sim.scenes import build_scene  # noqa: E402
from agentos.skills.base import SkillContext  # noqa: E402
from agentos.skills.registry import SkillRegistry  # noqa: E402


@pytest.fixture()
def registry() -> SkillRegistry:
    return SkillRegistry()


@pytest.fixture()
def scene():
    return build_scene("warehouse")


@pytest.fixture()
def backend(scene):
    backend = SimSkillBackend(scene, seed=3)
    return backend


@pytest.fixture()
def registry_and_backend():
    registry = SkillRegistry()
    backend = SimSkillBackend(build_scene("warehouse"), seed=3)
    backend.safety.register_skills(registry.names())
    return registry, backend


@pytest.fixture()
def ctx(registry_and_backend):
    from agentos.perception.sim_vlm import SimVLM

    registry, backend = registry_and_backend
    return SkillContext(
        backend=backend,
        safety=backend.safety,
        perception=backend.perception,
        vlm=SimVLM(backend.perception),
    )


@pytest.fixture()
def runtime(tmp_path) -> AgentRuntime:
    config = RuntimeConfig(
        scene="warehouse",
        seed=3,
        cache_dir=str(tmp_path / "cache"),
        persist_events=False,
        memory_enabled=False,
    )
    runtime = AgentRuntime(config)
    yield runtime
    runtime.close()
