"""Shared bilingual vocabulary (Chinese instruction surface -> internal tokens).

The vocabulary is deliberately small and explicit: the offline planner is a
deterministic baseline, so its language coverage must be inspectable instead of
hidden in a prompt.
"""

from __future__ import annotations

#: Chinese colour words -> internal colour token
COLOR_ZH: dict[str, str] = {
    "红": "red",
    "红色": "red",
    "橙": "orange",
    "橙色": "orange",
    "黄": "yellow",
    "黄色": "yellow",
    "绿": "green",
    "绿色": "green",
    "蓝": "blue",
    "蓝色": "blue",
    "白": "white",
    "白色": "white",
    "黑": "black",
    "黑色": "black",
}

#: Chinese object words -> internal category token
CATEGORY_ZH: dict[str, str] = {
    "盒子": "box",
    "箱子": "box",
    "纸盒": "box",
    "杯子": "cup",
    "水杯": "cup",
    "茶杯": "cup",
    "球": "ball",
    "小球": "ball",
    "包裹": "package",
    "快递": "package",
    "件": "package",
    "工具": "tool",
    "扳手": "tool",
    "瓶子": "bottle",
    "水瓶": "bottle",
}

#: Chinese search-zone words -> internal zone key
ZONE_ZH: dict[str, str] = {
    "货架区A": "zone_a",
    "货架A区": "zone_a",
    "货架区a": "zone_a",
    "A区": "zone_a",
    "货架区B": "zone_b",
    "货架B区": "zone_b",
    "货架区b": "zone_b",
    "B区": "zone_b",
    "货架区C": "zone_c",
    "货架C区": "zone_c",
    "右侧开放区": "zone_c",
    "C区": "zone_c",
    "实验台A区": "zone_a",
    "实验台B区": "zone_b",
    "设备区": "zone_c",
    "客厅A区": "zone_a",
    "卧室区": "zone_b",
    "走廊区": "zone_c",
}

#: Chinese destination words -> internal zone key
DEST_ZH: dict[str, str] = {
    "充电桩": "dock",
    "充电站": "dock",
    "充电座": "dock",
    "充点桩": "dock",
    "工作台": "table",
    "工作站台": "table",
    "桌子": "table",
    "餐桌": "table",
    "样品交接台": "table",
    "交接台": "table",
    "桌面": "table",
}

#: internal -> human label
ZONE_LABEL: dict[str, str] = {
    "zone_a": "货架区A",
    "zone_b": "货架区B",
    "zone_c": "右侧开放区",
    "table": "工作台",
    "dock": "充电桩",
    "start": "起点",
}


def zone_label(zone: str) -> str:
    return ZONE_LABEL.get(zone, zone)
