"""Three hand-authored 12x12 m scenes: warehouse, lab, home.

Every scene shares the same object catalogue (so one natural-language
instruction is valid in all three scenes) but has its own geometry, search
zones, dock and start pose.  That is what the "3 scenes" evaluation dimension
measures: the same instruction under different spatial layouts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from agentos.sim.world import OccupancyGrid, Rect
from agentos.types import GameObject

MAP_W = 12.0
MAP_H = 12.0
GRID_RES = 0.15
PLANNING_INFLATION = 0.42
COLLISION_INFLATION = 0.22


@dataclass(frozen=True)
class ObjectSpec:
    color: str
    category: str
    zone: str


#: Same 8 objects in every scene (colours/categories are the instruction vocabulary).
OBJECT_SPECS: tuple[ObjectSpec, ...] = (
    ObjectSpec("red", "box", "zone_a"),
    ObjectSpec("blue", "cup", "zone_b"),
    ObjectSpec("green", "ball", "zone_c"),
    ObjectSpec("yellow", "package", "zone_b"),
    ObjectSpec("black", "tool", "zone_a"),
    ObjectSpec("white", "bottle", "zone_c"),
    ObjectSpec("red", "cup", "zone_b"),
    ObjectSpec("blue", "box", "zone_a"),
)


@dataclass
class Scene:
    name: str
    description: str
    obstacles: tuple[Rect, ...]
    zones: dict[str, tuple[float, float]]
    dock: tuple[float, float]
    start: tuple[float, float]
    start_yaw: float = 0.0
    width: float = MAP_W
    height: float = MAP_H
    zone_labels: dict[str, str] = field(default_factory=dict)
    #: Preferred search order for "find a missing object" replanning.
    search_order: tuple[str, ...] = ("zone_a", "zone_b", "zone_c")

    def zone_center(self, zone: str) -> tuple[float, float]:
        if zone in self.zones:
            return self.zones[zone]
        raise KeyError(f"unknown zone {zone!r} in scene {self.name!r}")

    def planning_grid(self) -> OccupancyGrid:
        return OccupancyGrid(self.width, self.height, GRID_RES, self.obstacles, inflation=PLANNING_INFLATION)

    def collision_grid(self) -> OccupancyGrid:
        return OccupancyGrid(self.width, self.height, GRID_RES, self.obstacles, inflation=COLLISION_INFLATION)


def build_warehouse() -> Scene:
    obstacles = (
        Rect(2.0, 2.6, 7.0, 3.4),   # shelf row 1
        Rect(2.0, 6.6, 7.0, 7.4),   # shelf row 2
        Rect(2.0, 9.0, 7.0, 9.8),   # shelf row 3
        Rect(9.0, 3.0, 10.2, 4.0),  # crate stack
        Rect(9.2, 8.0, 10.4, 9.2),  # crate stack
    )
    return Scene(
        name="warehouse",
        description="12x12 仓储场景：三排货架 + 右侧开放区 + 右上角充电桩",
        obstacles=obstacles,
        zones={
            "zone_a": (4.5, 5.0),
            "zone_b": (5.5, 8.2),
            "zone_c": (9.9, 6.0),
            "table": (8.0, 11.0),
        },
        dock=(10.8, 10.8),
        start=(1.0, 1.0),
        start_yaw=0.0,
        zone_labels={
            "zone_a": "货架区A",
            "zone_b": "货架区B",
            "zone_c": "右侧开放区",
            "table": "工作站台",
            "dock": "充电桩",
        },
        search_order=("zone_a", "zone_b", "zone_c"),
    )


def build_lab() -> Scene:
    obstacles = (
        Rect(2.0, 3.0, 4.0, 3.8),   # bench 1
        Rect(6.5, 3.0, 8.5, 3.8),   # bench 2
        Rect(2.0, 7.5, 4.5, 8.3),   # bench 3
        Rect(8.0, 7.0, 10.0, 8.0),  # equipment cell
    )
    return Scene(
        name="lab",
        description="12x12 实验室场景：四组实验台/设备区 + 左下角充电桩",
        obstacles=obstacles,
        zones={
            "zone_a": (3.0, 5.5),
            "zone_b": (6.0, 5.5),
            "zone_c": (9.0, 5.0),
            "table": (6.0, 10.5),
        },
        dock=(1.2, 1.0),
        start=(1.0, 10.5),
        start_yaw=-1.5708,
        zone_labels={
            "zone_a": "实验台A区",
            "zone_b": "实验台B区",
            "zone_c": "设备区",
            "table": "样品交接台",
            "dock": "充电桩",
        },
        search_order=("zone_b", "zone_c", "zone_a"),
    )


def build_home() -> Scene:
    obstacles = (
        Rect(1.5, 2.0, 4.5, 3.0),   # sofa
        Rect(5.0, 6.0, 7.0, 7.0),   # coffee table
        Rect(8.0, 1.5, 11.0, 3.5),  # bed
        Rect(1.5, 8.0, 3.0, 9.0),   # cabinet
        Rect(9.5, 9.0, 11.0, 10.0), # tv stand
    )
    return Scene(
        name="home",
        description="12x12 家庭场景：沙发/床/茶几/柜子 + 中部充电桩",
        obstacles=obstacles,
        zones={
            "zone_a": (3.5, 6.0),
            "zone_b": (8.5, 6.0),
            "zone_c": (6.0, 9.0),
            "table": (6.5, 4.5),
        },
        dock=(2.0, 5.5),
        start=(6.0, 11.2),
        start_yaw=-1.5708,
        zone_labels={
            "zone_a": "客厅A区",
            "zone_b": "卧室区",
            "zone_c": "走廊区",
            "table": "餐桌",
            "dock": "充电桩",
        },
        search_order=("zone_a", "zone_c", "zone_b"),
    )


SCENE_BUILDERS = {
    "warehouse": build_warehouse,
    "lab": build_lab,
    "home": build_home,
}

SCENE_NAMES = tuple(SCENE_BUILDERS)


def build_scene(name: str) -> Scene:
    if name not in SCENE_BUILDERS:
        raise KeyError(f"unknown scene {name!r}; choose from {sorted(SCENE_BUILDERS)}")
    return SCENE_BUILDERS[name]()


# ---------------------------------------------------------------------------
# Deterministic object placement
# ---------------------------------------------------------------------------
def _place_near_zone(
    scene: Scene,
    planning: OccupancyGrid,
    rng: random.Random,
    center: tuple[float, float],
    taken: list[tuple[float, float]],
    radius: float = 0.9,
    min_gap: float = 0.65,
) -> tuple[float, float]:
    cx, cy = center
    for _ in range(400):
        angle = rng.uniform(0.0, 2.0 * 3.141592653589793)
        distance = radius * (rng.random() ** 0.5)
        x = cx + distance * cos(angle)
        y = cy + distance * sin(angle)
        if not planning.is_free(x, y):
            continue
        if any((x - tx) ** 2 + (ty - y) ** 2 < min_gap**2 for tx, ty in taken):
            continue
        return (x, y)
    # Deterministic fallback ring search around the zone centre
    for ring in (0.2, 0.35, 0.5, 0.7, 0.9, 1.1):
        for k in range(8):
            angle = 2.0 * 3.141592653589793 * k / 8
            x = cx + ring * cos(angle)
            y = cy + ring * sin(angle)
            if planning.is_free(x, y) and all(
                (x - tx) ** 2 + (ty - y) ** 2 >= (min_gap * 0.8) ** 2 for tx, ty in taken
            ):
                return (x, y)
    return (cx, cy)


def cos(angle: float) -> float:
    """Local cos to avoid importing math at module import time (kept explicit)."""
    import math

    return math.cos(angle)


def sin(angle: float) -> float:
    import math

    return math.sin(angle)


def _stable_seed(*parts: object) -> int:
    """Deterministic across processes (``hash()`` is salted per process)."""
    import zlib

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


def sample_objects(scene: Scene, planning: OccupancyGrid, seed: int) -> dict[str, GameObject]:
    """Place the scene object catalogue deterministically for a given seed."""
    rng = random.Random(_stable_seed(scene.name, seed))
    objects: dict[str, GameObject] = {}
    taken: list[tuple[float, float]] = []
    for index, spec in enumerate(OBJECT_SPECS):
        zone_xy = scene.zone_center(spec.zone)
        x, y = _place_near_zone(scene, planning, rng, zone_xy, taken)
        taken.append((x, y))
        object_id = f"obj_{spec.color}_{spec.category}_{index}"
        objects[object_id] = GameObject(
            object_id=object_id,
            color=spec.color,
            category=spec.category,
            x=round(x, 3),
            y=round(y, 3),
            zone=spec.zone,
        )
    return objects
