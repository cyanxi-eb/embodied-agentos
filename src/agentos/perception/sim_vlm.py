"""Offline "sim-VLM": deterministic scene QA over simulator ground truth.

This is explicitly *not* a neural network.  It exists so the perception/VLM
seam is exercisable on a CPU-only Windows host: the same interface is what a
Qwen2.5-VL adapter implements in a real deployment.
"""

from __future__ import annotations

from typing import Any

from agentos.perception.base import VLMBackend
from agentos.perception.sim_perception import SimPerception
from agentos.types import Detection, WorldState
from agentos.vocab import CATEGORY_ZH, COLOR_ZH, zone_label


class SimVLM(VLMBackend):
    name = "sim-vlm"

    def __init__(self, perception: SimPerception) -> None:
        self.perception = perception

    # ----------------------------------------------------------------- locate
    def locate(self, query: dict[str, Any], world: WorldState, image: Any = None) -> list[Detection]:
        detections = self.perception.locate(query, world)
        return [
            Detection(
                object_id=d.object_id,
                color=d.color,
                category=d.category,
                x=d.x,
                y=d.y,
                confidence=d.confidence,
                source=self.name,
                distance=d.distance,
            )
            for d in detections
        ]

    # ----------------------------------------------------------------- answer
    def answer(self, question: str, world: WorldState, image: Any = None) -> str:
        question = (question or "").strip()
        visible = self.perception.detect_all(world)
        if not question:
            return "请给出需要查询的场景问题。"
        if "电量" in question or "battery" in question.lower():
            return f"当前电量 {world.battery:.0f}%，机器人位于 ({world.robot.x:.1f}, {world.robot.y:.1f})。"
        if "有什么" in question or "看到" in question or "场景" in question:
            if not visible:
                return "当前视野内没有可识别的物体。"
            items = "、".join(f"{_color_zh(d.color)}{_category_zh(d.category)}" for d in visible[:5])
            return f"当前视野内看到 {len(visible)} 个物体：{items}。"
        query = _query_from_text(question)
        if query:
            matches = [d for d in visible if _matches_query(d, query)]
            if matches:
                nearest = matches[0]
                return (
                    f"{_color_zh(nearest.color)}{_category_zh(nearest.category)}"
                    f"在机器人{(nearest.distance or 0):.1f}米处，坐标 ({nearest.x:.1f}, {nearest.y:.1f})。"
                )
            return f"视野内没有找到{_describe_query(query)}，可以先移动到搜索区域再找。"
        return (
            f"机器人位于 ({world.robot.x:.1f}, {world.robot.y:.1f})，"
            f"电量 {world.battery:.0f}%，视野内 {len(visible)} 个物体。"
        )

    # ---------------------------------------------------------------- describe
    def describe_scene(self, world: WorldState, image: Any = None) -> dict[str, Any]:
        perception_description = self.perception.describe(world)
        perception_description["vlm"] = self.name
        perception_description["summary"] = self.answer("场景里有什么", world)
        perception_description["zone_labels"] = {
            "zone_a": zone_label("zone_a"),
            "zone_b": zone_label("zone_b"),
            "zone_c": zone_label("zone_c"),
        }
        return perception_description


def _color_zh(color: str) -> str:
    for zh, token in COLOR_ZH.items():
        if token == color:
            return zh if zh.endswith("色") else zh + "色"
    return color


def _category_zh(category: str) -> str:
    for zh, token in CATEGORY_ZH.items():
        if token == category:
            return zh
    return category


def _query_from_text(text: str) -> dict[str, str]:
    query: dict[str, str] = {}
    for zh, token in COLOR_ZH.items():
        if zh in text:
            query["color"] = token
            break
    for zh, token in CATEGORY_ZH.items():
        if zh in text:
            query["category"] = token
            break
    return query


def _describe_query(query: dict[str, str]) -> str:
    parts = []
    if "color" in query:
        parts.append(_color_zh(query["color"]))
    if "category" in query:
        parts.append(_category_zh(query["category"]))
    return "".join(parts) or "目标物体"


def _matches_query(detection: Detection, query: dict[str, str]) -> bool:
    if "color" in query and detection.color != query["color"]:
        return False
    if "category" in query and detection.category != query["category"]:
        return False
    return True
