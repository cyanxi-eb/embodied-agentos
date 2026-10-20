"""MCP server: real stdio subprocess, tool listing, catalogue and skill call."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"


def _server_params():
    from mcp import StdioServerParameters

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentos.mcp_server.server"],
        env=env,
    )


def _payload(result) -> object:
    """Decode an MCP tool result into JSON (structured or content-based)."""
    structured = getattr(result, "structuredContent", None)
    if structured:
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured
    texts = [item.text for item in getattr(result, "content", []) if getattr(item, "type", "") == "text"]
    if not texts:
        return None
    if len(texts) == 1:
        return json.loads(texts[0])
    decoded = []
    for text in texts:
        try:
            decoded.append(json.loads(text))
        except json.JSONDecodeError:
            decoded.append(text)
    return decoded


async def _with_session(coro_fn):
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await coro_fn(session)


def test_mcp_lists_tools_and_skill_catalogue():
    async def scenario(session):
        tools = await session.list_tools()
        names = {tool.name for tool in tools.tools}
        assert {"list_skills", "get_skill", "call_skill"} <= names
        payload = _payload(await session.call_tool("list_skills", {}))
        assert isinstance(payload, list) and len(payload) == 8, payload
        skill_names = {item["name"] for item in payload}
        assert "navigate_to" in skill_names and "dock" in skill_names
        contract = _payload(await session.call_tool("get_skill", {"name": "navigate_to"}))
        assert contract["params_schema"]["type"] == "object"
        assert contract["preconditions"] and contract["postconditions"]
        assert "TIMEOUT" in contract["failure_codes"]
        missing = _payload(await session.call_tool("get_skill", {"name": "does_not_exist"}))
        assert "error" in missing
        return True

    assert asyncio.run(_with_session(scenario)) is True


def test_mcp_call_skill_executes_and_verifies():
    async def scenario(session):
        payload = _payload(
            await session.call_tool(
                "call_skill",
                {"name": "speak", "params": {"text": "MCP 技能调用测试"}, "scene": "warehouse", "seed": 1},
            )
        )
        assert payload["ok"] is True
        assert payload["result"]["code"] == "OK"
        assert payload["postcondition"]["ok"] is True
        assert payload["verified_by"] == "world_state (not by the skill result)"

        rejected = _payload(
            await session.call_tool("call_skill", {"name": "cmd_vel", "params": {"v": 1.0}, "scene": "warehouse"})
        )
        assert rejected["ok"] is False

        invalid = _payload(await session.call_tool("call_skill", {"name": "speak", "params": {}}))
        assert invalid["ok"] is False
        return True

    assert asyncio.run(_with_session(scenario)) is True
