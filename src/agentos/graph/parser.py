"""Deterministic Chinese instruction parser (offline planner baseline).

This is intentionally a *rule* parser, not an LLM: the evaluation must be
reproducible on a CPU-only machine without an API key.  The same ``ParsedIntent``
structure is what an LLM planner would fill via structured output (see
``agentos.graph.planner.LLMPlanner``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agentos.vocab import CATEGORY_ZH, COLOR_ZH, DEST_ZH, ZONE_ZH


@dataclass
class ParsedIntent:
    raw: str
    intent: str = "unknown"
    objects: list[dict[str, str]] = field(default_factory=list)
    destination: str | None = None
    zones: list[str] = field(default_factory=list)
    speak_text: str | None = None
    wants_dock: bool = False
    wants_stop: bool = False
    patrol: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "intent": self.intent,
            "objects": list(self.objects),
            "destination": self.destination,
            "zones": list(self.zones),
            "speak_text": self.speak_text,
            "wants_dock": self.wants_dock,
            "wants_stop": self.wants_stop,
            "patrol": self.patrol,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedIntent":
        return cls(
            raw=data.get("raw", ""),
            intent=data.get("intent", "unknown"),
            objects=[dict(o) for o in data.get("objects", [])],
            destination=data.get("destination"),
            zones=list(data.get("zones", [])),
            speak_text=data.get("speak_text"),
            wants_dock=bool(data.get("wants_dock", False)),
            wants_stop=bool(data.get("wants_stop", False)),
            patrol=bool(data.get("patrol", False)),
            notes=list(data.get("notes", [])),
        )


_QUOTED = re.compile(r"[\"“'‘]([^\"”'’]{1,80})[\"”'’]")
_COLOR_WORDS = sorted(COLOR_ZH, key=len, reverse=True)
_CATEGORY_WORDS = sorted(CATEGORY_ZH, key=len, reverse=True)


def _find_objects(text: str) -> list[dict[str, str]]:
    """Extract (color, category) mentions in text order."""
    matches: list[tuple[int, dict[str, str]]] = []
    occupied: list[tuple[int, int]] = []
    for category_word in _CATEGORY_WORDS:
        start = 0
        while True:
            index = text.find(category_word, start)
            if index < 0:
                break
            start = index + len(category_word)
            if any(lo <= index < hi for lo, hi in occupied):
                continue
            color = _color_before(text, index)
            if color is None:
                continue
            occupied.append((index, index + len(category_word)))
            matches.append((index, {"color": COLOR_ZH[color], "category": CATEGORY_ZH[category_word]}))
    matches.sort(key=lambda item: item[0])
    objects: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, obj in matches:
        key = (obj["color"], obj["category"])
        if key in seen:
            continue
        seen.add(key)
        objects.append(obj)
    return objects


def _color_before(text: str, index: int) -> str | None:
    window = text[max(0, index - 4) : index]
    for color_word in _COLOR_WORDS:
        if window.endswith(color_word):
            return color_word
    return None


def _find_zones(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for word, zone in sorted(ZONE_ZH.items(), key=lambda item: len(item[0]), reverse=True):
        index = text.find(word)
        if index >= 0:
            found.append((index, zone))
    found.sort()
    zones: list[str] = []
    for _, zone in found:
        if zone not in zones:
            zones.append(zone)
    return zones


def _find_destination(text: str) -> str | None:
    found: list[tuple[int, str]] = []
    for word, zone in sorted(DEST_ZH.items(), key=lambda item: len(item[0]), reverse=True):
        index = text.find(word)
        if index >= 0:
            found.append((index, zone))
    if not found:
        return None
    found.sort()
    return found[0][1]


def _find_speak_text(text: str) -> str | None:
    quoted = _QUOTED.search(text)
    if quoted:
        return quoted.group(1).strip()
    for marker in ("播报", "说一句", "说", "通知"):
        index = text.find(marker)
        if index >= 0:
            tail = text[index + len(marker) :].strip(" ：:，,。.!！")
            if tail:
                return tail[:80]
    return None


def parse_instruction(text: str) -> ParsedIntent:
    raw = (text or "").strip()
    intent = ParsedIntent(raw=raw)
    if not raw:
        intent.notes.append("empty instruction")
        return intent

    objects = _find_objects(raw)
    zones = _find_zones(raw)
    destination = _find_destination(raw)
    speak_text = _find_speak_text(raw) if any(m in raw for m in ("播报", "说", "通知")) else None
    wants_dock = any(word in raw for word in ("回充", "充电", "回桩"))
    wants_stop = any(word in raw for word in ("停止", "停下", "急停", "停一下", "站住"))
    patrol = any(word in raw for word in ("依次", "巡检", "逐个检查", "巡查"))

    intent.objects = objects
    intent.zones = zones
    intent.destination = destination
    intent.speak_text = speak_text
    intent.wants_stop = wants_stop
    intent.patrol = patrol
    # dock as a *destination* is handled as a destination ("拿到充电桩"); dock as
    # an action is detected by the verbs below.
    intent.wants_dock = wants_dock and not (objects and destination == "dock" and "回充" not in raw)

    if patrol:
        intent.intent = "patrol"
    elif objects:
        if len(objects) >= 2 and destination:
            intent.intent = "multi_fetch"
        elif destination:
            intent.intent = "fetch"
        elif speak_text:
            intent.intent = "find_report"
        else:
            intent.intent = "fetch"
            intent.destination = "table"
            intent.notes.append("no destination specified, defaulting to table")
    elif speak_text:
        intent.intent = "speak"
    elif wants_dock:
        intent.intent = "dock"
    elif wants_stop:
        intent.intent = "stop"
    elif zones:
        intent.intent = "navigate"
    else:
        intent.intent = "unknown"
        intent.notes.append("no object / action keyword recognised")
    return intent
