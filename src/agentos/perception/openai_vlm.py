"""OpenAI-compatible VLM adapter (Qwen2.5-VL / vLLM / DashScope compatible mode).

This is the "real backend" seam.  It is configured purely through environment
variables and is **never** used by the test suite:

    AGENTOS_VLM_BASE_URL   e.g. http://127.0.0.1:8000/v1
    AGENTOS_VLM_API_KEY    bearer token (may be a dummy for local vLLM)
    AGENTOS_VLM_MODEL      e.g. Qwen2.5-VL-7B-Instruct
    AGENTOS_VLM_TIMEOUT_S  default 15

On the development host (Windows, no GPU, no model server) the adapter reports
``available == False`` and raises a clear error instead of pretending to work.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

from agentos.perception.base import VLMBackend
from agentos.types import Detection, WorldState


class OpenAICompatVLM(VLMBackend):
    name = "openai-compat-vlm"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("AGENTOS_VLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("AGENTOS_VLM_API_KEY", "")
        self.model = model or os.environ.get("AGENTOS_VLM_MODEL", "")
        self.timeout_s = float(timeout_s or os.environ.get("AGENTOS_VLM_TIMEOUT_S", "15"))

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.model)

    # ------------------------------------------------------------------ http
    def _chat(self, prompt: str, image: Any = None) -> str:
        if not self.available:
            raise RuntimeError(
                "OpenAICompatVLM is not configured; set AGENTOS_VLM_BASE_URL and AGENTOS_VLM_MODEL "
                "to a Qwen2.5-VL compatible endpoint"
            )
        import httpx

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            b64 = _encode_image(image)
            if b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    # ------------------------------------------------------------------- api
    def locate(self, query: dict[str, Any], world: WorldState, image: Any = None) -> list[Detection]:
        prompt = (
            "你是机器人视觉定位模块。根据图像与任务描述，输出 JSON 数组，"
            "每个元素包含 object_id, color, category, x, y, confidence。"
            "x/y 是物体在机器人局部坐标系中的米制坐标（前方为 +x）。"
            f"目标：{json.dumps(query, ensure_ascii=False)}。只输出 JSON。"
        )
        raw = self._chat(prompt, image=image)
        detections: list[Detection] = []
        for item in _parse_json_list(raw):
            try:
                detections.append(
                    Detection(
                        object_id=str(item.get("object_id", "vlm")),
                        color=str(item.get("color", "")),
                        category=str(item.get("category", "")),
                        x=float(item.get("x", 0.0)),
                        y=float(item.get("y", 0.0)),
                        confidence=float(item.get("confidence", 0.5)),
                        source=self.name,
                    )
                )
            except (TypeError, ValueError):
                continue
        return detections

    def answer(self, question: str, world: WorldState, image: Any = None) -> str:
        prompt = (
            "你是家庭/仓储机器人的视觉问答模块。结合图像和下面的机器人状态，用中文简短回答一个问题。\n"
            f"机器人状态：{json.dumps(world.snapshot(), ensure_ascii=False)}\n问题：{question}"
        )
        return self._chat(prompt, image=image).strip()

    def describe_scene(self, world: WorldState, image: Any = None) -> dict[str, Any]:
        prompt = (
            "用 JSON 描述图像中机器人视野内可抓取物体，字段为 "
            '{"objects": [{"object_id","color","category","x","y","confidence"}], "summary": "一句话"}。'
            "只输出 JSON。"
        )
        raw = self._chat(prompt, image=image)
        payload = _parse_json_object(raw)
        payload.setdefault("vlm", self.name)
        return payload


def _encode_image(image: Any) -> str | None:
    if image is None:
        return None
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("ascii")
    if isinstance(image, str) and os.path.exists(image):
        with open(image, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")
    return None


def _parse_json_list(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
