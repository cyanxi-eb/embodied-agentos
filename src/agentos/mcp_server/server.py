"""MCP stdio server exposing the 8-skill library to an LLM client.

Run: ``python -m agentos.mcp_server.server``

Tools:
* ``list_skills`` – catalogue (name, category, description, failure codes)
* ``get_skill``   – full contract: JSON schema + pre/postconditions + timeout
* ``call_skill``  – execute one skill against the deterministic simulator,
  returning the result *and* the independently verified postcondition report

The LLM can only choose a skill and fill its parameters; raw velocity commands,
deny-listed parameters and unknown skills are rejected by the safety gateway.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from agentos.perception.sim_vlm import SimVLM
from agentos.safety.gateway import SafetyConfig
from agentos.sim.backend import SimSkillBackend
from agentos.sim.scenes import SCENE_NAMES, build_scene
from agentos.skills.base import SkillContext
from agentos.skills.registry import SkillRegistry

mcp = FastMCP(
    "embodied-agentos",
    instructions=(
        "具身机器人技能库。只能通过 list_skills/get_skill 了解接口，用 call_skill 选择技能并填参数；"
        "禁止直接下发速度/电机指令。每次 call_skill 都是无状态仿真执行，返回结果与独立后置条件校验。"
    ),
)

_REGISTRY = SkillRegistry()


def _registry() -> SkillRegistry:
    return _REGISTRY


@mcp.tool()
def list_skills() -> list[dict[str, Any]]:
    """列出全部机器人技能（名称、类别、描述、超时、失败码）。"""
    return _registry().summary()


@mcp.tool()
def get_skill(name: str) -> dict[str, Any]:
    """获取单个技能的完整契约：参数 JSON Schema、前置/后置条件、失败码、取消语义。"""
    skill = _registry().get(name)
    if skill is None:
        return {"error": f"unknown skill {name!r}", "available": _registry().names()}
    return skill.describe()


@mcp.tool()
def call_skill(name: str, params: dict[str, Any] | None = None, scene: str = "warehouse", seed: int = 3) -> dict[str, Any]:
    """在确定性仿真中执行一个技能（无状态），返回结果与后置条件校验。

    Args:
        name: 技能名，必须是 list_skills 返回的 8 个之一
        params: 技能参数，需符合 get_skill 的 JSON Schema
        scene: warehouse | lab | home
        seed: 物体摆放种子，保持可复现
    """
    params = params or {}
    if scene not in SCENE_NAMES:
        return {"error": f"unknown scene {scene!r}", "available": list(SCENE_NAMES)}
    registry = _registry()
    errors = registry.validate_call(name, params)
    if errors:
        return {"ok": False, "error": "parameter validation failed", "details": errors}
    scene_obj = build_scene(scene)
    backend = SimSkillBackend(scene_obj, safety_config=SafetyConfig(), seed=seed)
    backend.safety.register_skills(registry.names())
    ctx = SkillContext(
        backend=backend,
        safety=backend.safety,
        perception=backend.perception,
        vlm=SimVLM(backend.perception),
    )
    skill = registry.require(name)
    verdict = backend.safety.authorize(name, params, backend.world, timeout_s=skill.timeout_s)
    if not verdict.allowed:
        return {
            "ok": False,
            "code": verdict.code.value,
            "rejected_by": "safety_gateway",
            "reason": verdict.reason,
        }
    pre = skill.check_preconditions(backend.world, params)
    if not pre.ok:
        return {"ok": False, "code": "PRECONDITION_FAILED", "preconditions": pre.to_dict()}
    result = skill.execute(ctx, params)
    post = skill.check_postconditions(backend.world, params)
    return {
        "ok": bool(result.ok and post.ok),
        "result": result.to_dict(),
        "postcondition": post.to_dict(),
        "verified_by": "world_state (not by the skill result)",
        "world": backend.world.snapshot(),
        "sim_time": round(backend.sim_time, 2),
        "safety": backend.safety.status(),
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
