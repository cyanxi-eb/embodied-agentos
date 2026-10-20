"""The 50-instruction evaluation set.

Chinese natural-language instructions covering all 8 skills and their
combinations.  Every instruction is valid in all three scenes (the object
catalogue and zone keys are shared), which is what makes "50 instructions x 3
scenes x 3 repeats = 450 episodes" meaningful.

The structured ``goal`` dict is used **only by the evaluator** to check the final
world state; the agent receives the ``text`` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

COLOR_ZH = {
    "red": "红色",
    "blue": "蓝色",
    "green": "绿色",
    "yellow": "黄色",
    "black": "黑色",
    "white": "白色",
}
CATEGORY_ZH = {
    "box": "盒子",
    "cup": "杯子",
    "ball": "球",
    "package": "包裹",
    "tool": "工具",
    "bottle": "瓶子",
}
ZONE_ZH = {"zone_a": "A区", "zone_b": "B区", "zone_c": "C区"}
DEST_ZH = {"table": "工作站台", "dock": "充电桩"}


@dataclass(frozen=True)
class Instruction:
    id: str
    text: str
    intent: str
    goal: dict[str, Any]
    #: canonical object zone (for fault targeting / diagnostics)
    object_zone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "intent": self.intent,
            "goal": self.goal,
            "object_zone": self.object_zone,
        }


def _obj(color: str, category: str) -> str:
    return f"{COLOR_ZH[color]}{CATEGORY_ZH[category]}"


def _fetch_goal(color: str, category: str, destination: str) -> dict[str, Any]:
    return {"type": "object_at_zone", "color": color, "category": category, "zone": destination}


def build_instructions() -> list[Instruction]:
    data: list[dict[str, Any]] = []

    def add(text: str, intent: str, goal: dict[str, Any], zone: str | None = None) -> None:
        data.append({"text": text, "intent": intent, "goal": goal, "object_zone": zone})

    # ---- fetch to table, with zone hint (14) -------------------------------
    for color, category, zone in (
        ("red", "box", "zone_a"),
        ("blue", "cup", "zone_b"),
        ("green", "ball", "zone_c"),
        ("yellow", "package", "zone_b"),
        ("black", "tool", "zone_a"),
        ("white", "bottle", "zone_c"),
        ("red", "cup", "zone_b"),
        ("blue", "box", "zone_a"),
    ):
        add(
            f"把{ZONE_ZH[zone]}的{_obj(color, category)}拿到{DEST_ZH['table']}",
            "fetch",
            _fetch_goal(color, category, "table"),
            zone,
        )
    add(
        f"去A区把黑色工具取过来放到工作站台",
        "fetch",
        _fetch_goal("black", "tool", "table"),
        "zone_a",
    )
    add(
        f"帮我把B区的红色杯子放到工作站台",
        "fetch",
        _fetch_goal("red", "cup", "table"),
        "zone_b",
    )
    add(
        f"把C区的绿色球移到工作站台上",
        "fetch",
        _fetch_goal("green", "ball", "table"),
        "zone_c",
    )
    add(
        f"从B区取黄色包裹送到工作站台",
        "fetch",
        _fetch_goal("yellow", "package", "table"),
        "zone_b",
    )

    # ---- fetch to dock (6) ------------------------------------------------
    for color, category, zone in (
        ("red", "box", "zone_a"),
        ("blue", "cup", "zone_b"),
        ("black", "tool", "zone_a"),
    ):
        add(
            f"把{ZONE_ZH[zone]}的{_obj(color, category)}放到{DEST_ZH['dock']}旁边",
            "fetch",
            _fetch_goal(color, category, "dock"),
            zone,
        )
    add(
        "把B区的蓝色杯子和C区的绿色球都拿到工作站台",
        "multi_fetch",
        {
            "type": "objects_at_zone",
            "objects": [{"color": "blue", "category": "cup"}, {"color": "green", "category": "ball"}],
            "zone": "table",
        },
        "zone_b",
    )
    add(
        "把A区的红色盒子和A区的蓝色盒子一起送到工作站台",
        "multi_fetch",
        {
            "type": "objects_at_zone",
            "objects": [{"color": "red", "category": "box"}, {"color": "blue", "category": "box"}],
            "zone": "table",
        },
        "zone_a",
    )
    add(
        "先把C区的白色瓶子送到工作台，再回充电桩",
        "fetch_dock",
        {"type": "composite", "goals": [_fetch_goal("white", "bottle", "table"), {"type": "docked"}]},
        "zone_c",
    )

    # ---- fetch + speak (6) ------------------------------------------------
    for color, category, zone, text in (
        ("red", "box", "zone_a", "红色盒子已送达"),
        ("blue", "cup", "zone_b", "蓝色杯子已送达"),
        ("yellow", "package", "zone_b", "包裹已送到工作站"),
        ("white", "bottle", "zone_c", "瓶子已送达"),
        ("black", "tool", "zone_a", "工具已取到"),
        ("green", "ball", "zone_c", "小球已送达"),
    ):
        add(
            f"把{ZONE_ZH[zone]}的{_obj(color, category)}拿到工作站台，然后播报“{text}”",
            "fetch_speak",
            {
                "type": "composite",
                "goals": [_fetch_goal(color, category, "table"), {"type": "spoken", "contains": text}],
            },
            zone,
        )

    # ---- fetch then dock (5) ---------------------------------------------
    for color, category, zone in (
        ("red", "box", "zone_a"),
        ("green", "ball", "zone_c"),
        ("blue", "cup", "zone_b"),
        ("black", "tool", "zone_a"),
        ("yellow", "package", "zone_b"),
    ):
        add(
            f"把{ZONE_ZH[zone]}的{_obj(color, category)}放到工作站台，然后回充",
            "fetch_dock",
            {"type": "composite", "goals": [_fetch_goal(color, category, "table"), {"type": "docked"}]},
            zone,
        )

    # ---- dock only (4) ---------------------------------------------------
    add("电量低了，回充电桩充电", "dock", {"type": "docked"})
    add("回到充电桩", "dock", {"type": "docked"})
    add("任务结束，回充", "dock", {"type": "docked"})
    add("请回到充电桩待命", "dock", {"type": "docked"})

    # ---- speak only (4) --------------------------------------------------
    for text in ("全部任务已完成", "机器人已就绪", "检测到异常请人工确认", "开始执行搬运任务"):
        add(f"播报“{text}”", "speak", {"type": "spoken", "contains": text})

    # ---- stop (3) --------------------------------------------------------
    add("立刻停止", "stop", {"type": "stopped"})
    add("先去A区，然后停下", "stop", {"type": "stopped"}, "zone_a")
    add("停下并保持静止", "stop", {"type": "stopped"})

    # ---- find + report (2) ----------------------------------------------
    add(
        "找到C区的绿色球后播报“目标已找到”",
        "find_report",
        {"type": "find_report", "color": "green", "category": "ball", "contains": "目标已找到"},
        "zone_c",
    )
    add(
        "找到A区的黑色工具后播报“工具在这里”",
        "find_report",
        {"type": "find_report", "color": "black", "category": "tool", "contains": "工具在这里"},
        "zone_a",
    )

    # ---- patrol (2) ------------------------------------------------------
    add("依次检查A区和B区", "patrol", {"type": "patrol", "zones": ["zone_a", "zone_b"]})
    add("巡检C区和B区并播报", "patrol", {"type": "patrol", "zones": ["zone_c", "zone_b"]})

    # ---- no zone hint: force a real search/replan (3) --------------------
    add(
        "帮我找到红色盒子并送到工作站台",
        "fetch",
        _fetch_goal("red", "box", "table"),
        "zone_a",
    )
    add(
        "把蓝色杯子拿过来放到工作站台",
        "fetch",
        _fetch_goal("blue", "cup", "table"),
        "zone_b",
    )
    add(
        "找到绿色球并放到充电桩旁边",
        "fetch",
        _fetch_goal("green", "ball", "dock"),
        "zone_c",
    )

    # ---- final 3: extra combinations (50 total) ---------------------------
    add(
        "把A区的蓝色盒子拿到工作站台",
        "fetch",
        _fetch_goal("blue", "box", "table"),
        "zone_a",
    )
    add(
        "把C区的白色瓶子拿到充电桩旁边，然后播报“瓶子已放到充电桩”",
        "fetch_speak",
        {
            "type": "composite",
            "goals": [
                _fetch_goal("white", "bottle", "dock"),
                {"type": "spoken", "contains": "瓶子已放到充电桩"},
            ],
        },
        "zone_c",
    )
    add(
        "去A区巡检并播报“A区正常”",
        "patrol",
        {"type": "patrol", "zones": ["zone_a"]},
        "zone_a",
    )

    return [
        Instruction(id=f"L{index + 1:02d}", text=item["text"], intent=item["intent"], goal=item["goal"], object_zone=item["object_zone"])
        for index, item in enumerate(data)
    ]


INSTRUCTIONS: list[Instruction] = build_instructions()
INSTRUCTION_BY_ID = {instruction.id: instruction for instruction in INSTRUCTIONS}


def stress_instructions(limit: int = 30) -> list[Instruction]:
    """Balanced subset across intents for the stress/generalisation evaluation."""
    by_intent: dict[str, list[Instruction]] = {}
    for instruction in INSTRUCTIONS:
        by_intent.setdefault(instruction.intent, []).append(instruction)
    selected: list[Instruction] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for intent in sorted(by_intent):
            items = by_intent[intent]
            if depth < len(items) and len(selected) < limit:
                selected.append(items[depth])
                added = True
        if not added:
            break
        depth += 1
    return selected
