"""Core data types shared by every layer.

Everything that crosses a checkpoint / API / trace boundary is either a plain
dataclass with ``to_dict`` or a JSON-native value.  LangGraph checkpoints the
agent state into SQLite, so keeping the state serializable is a hard rule.
"""

from __future__ import annotations

import copy
import enum
import math
from dataclasses import dataclass, field
from typing import Any


class FailureCode(str, enum.Enum):
    """Failure taxonomy for skills and for the agent loop.

    The string value is what shows up in traces / evaluation reports, so the
    enum is deliberately explicit instead of a bare string.
    """

    OK = "OK"
    TIMEOUT = "TIMEOUT"
    NOT_FOUND = "NOT_FOUND"
    BLOCKED = "BLOCKED"
    SLIP = "SLIP"
    ESTOP = "ESTOP"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    POSTCONDITION_FAILED = "POSTCONDITION_FAILED"
    SAFETY_REJECTED = "SAFETY_REJECTED"
    NO_PATH = "NO_PATH"
    LOW_BATTERY = "LOW_BATTERY"
    CANCELLED = "CANCELLED"
    UNKNOWN_SKILL = "UNKNOWN_SKILL"
    PLAN_INVALID = "PLAN_INVALID"
    UNPLANNABLE = "UNPLANNABLE"
    BACKEND_ERROR = "BACKEND_ERROR"
    NOT_HOLDING = "NOT_HOLDING"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Codes a replan can plausibly fix.  Everything else needs a human / operator.
RECOVERABLE_CODES: frozenset[FailureCode] = frozenset(
    {
        FailureCode.NOT_FOUND,
        FailureCode.BLOCKED,
        FailureCode.TIMEOUT,
        FailureCode.SLIP,
        FailureCode.NO_PATH,
        FailureCode.POSTCONDITION_FAILED,
        FailureCode.PRECONDITION_FAILED,
    }
)

#: Codes that must never be "retried around": they need an operator or end the task.
FATAL_CODES: frozenset[FailureCode] = frozenset(
    {
        FailureCode.ESTOP,
        FailureCode.SAFETY_REJECTED,
        FailureCode.OUT_OF_BOUNDS,
        FailureCode.CANCELLED,
        FailureCode.UNKNOWN_SKILL,
        FailureCode.LOW_BATTERY,
        FailureCode.PLAN_INVALID,
    }
)


class SkillCategory(str, enum.Enum):
    NAVIGATION = "navigation"
    PERCEPTION = "perception"
    MANIPULATION = "manipulation"
    SPEECH = "speech"
    SAFETY = "safety"
    POWER = "power"


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"x": round(self.x, 4), "y": round(self.y, 4), "yaw": round(self.yaw, 4)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pose":
        return cls(float(data.get("x", 0.0)), float(data.get("y", 0.0)), float(data.get("yaw", 0.0)))

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)

    def angle_to(self, x: float, y: float) -> float:
        return normalize_angle(math.atan2(y - self.y, x - self.x) - self.yaw)


def normalize_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


@dataclass
class GameObject:
    object_id: str
    color: str
    category: str
    x: float
    y: float
    graspable: bool = True
    zone: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "color": self.color,
            "category": self.category,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "graspable": self.graspable,
            "zone": self.zone,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameObject":
        return cls(
            object_id=data["object_id"],
            color=data["color"],
            category=data["category"],
            x=float(data["x"]),
            y=float(data["y"]),
            graspable=bool(data.get("graspable", True)),
            zone=data.get("zone", ""),
        )

    def matches(self, query: dict[str, Any]) -> bool:
        """Does this object satisfy a ``{color?, category?}`` query?"""
        color = query.get("color")
        category = query.get("category")
        if color and str(color).lower() != self.color.lower():
            return False
        if category and str(category).lower() != self.category.lower():
            return False
        return True


@dataclass
class WorldState:
    """Dynamic world state.

    Static geometry lives in ``agentos.sim.world.Scene``; this object is what the
    simulation mutates and what checkpoints / postcondition checks read.
    """

    sim_time: float = 0.0
    robot: Pose = field(default_factory=Pose)
    v: float = 0.0
    w: float = 0.0
    battery: float = 100.0
    holding: str | None = None
    held_meta: dict[str, Any] | None = None
    objects: dict[str, GameObject] = field(default_factory=dict)
    dock: tuple[float, float] = (0.0, 0.0)
    #: static zone name -> [x, y]; copied from the scene so that skills can
    #: resolve symbolic destinations ("充电桩", "工作台") without owning geometry.
    zones: dict[str, list[float]] = field(default_factory=dict)
    docked: bool = False
    estop: bool = False
    last_error: str | None = None
    # ---- memory of the running session (what the robot "knows") ----
    known_objects: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_detection: dict[str, Any] | None = None
    last_approached: dict[str, Any] | None = None
    last_placed: dict[str, Any] | None = None
    last_spoken: str | None = None
    visited_zones: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ utils
    def clone(self) -> "WorldState":
        return copy.deepcopy(self)

    def free_objects(self) -> list[GameObject]:
        """Objects that are still on the floor (not currently held)."""
        return [o for o in self.objects.values() if o.object_id != self.holding]

    def object_by_id(self, object_id: str) -> GameObject | None:
        return self.objects.get(object_id)

    def nearest_object(self, query: dict[str, Any], x: float | None = None, y: float | None = None) -> GameObject | None:
        rx = self.robot.x if x is None else x
        ry = self.robot.y if y is None else y
        candidates = [o for o in self.free_objects() if o.matches(query)]
        if not candidates:
            return None
        return min(candidates, key=lambda o: math.hypot(o.x - rx, o.y - ry))

    def add_object(self, obj: GameObject) -> None:
        self.objects[obj.object_id] = obj

    def remove_object(self, object_id: str) -> GameObject | None:
        return self.objects.pop(object_id, None)

    def mark_detection(self, detection: dict[str, Any]) -> None:
        self.last_detection = dict(detection)
        self.known_objects[detection["object_id"]] = dict(detection)

    def zone_visited(self, zone: str) -> bool:
        return zone in self.visited_zones

    # ------------------------------------------------------------- serialize
    def to_dict(self) -> dict[str, Any]:
        return {
            "sim_time": round(self.sim_time, 4),
            "robot": self.robot.to_dict(),
            "v": round(self.v, 4),
            "w": round(self.w, 4),
            "battery": round(self.battery, 3),
            "holding": self.holding,
            "held_meta": dict(self.held_meta) if self.held_meta else None,
            "objects": {oid: obj.to_dict() for oid, obj in self.objects.items()},
            "dock": [round(self.dock[0], 4), round(self.dock[1], 4)],
            "zones": {k: [round(float(v[0]), 4), round(float(v[1]), 4)] for k, v in self.zones.items()},
            "docked": self.docked,
            "estop": self.estop,
            "last_error": self.last_error,
            "known_objects": {oid: dict(v) for oid, v in self.known_objects.items()},
            "last_detection": dict(self.last_detection) if self.last_detection else None,
            "last_approached": dict(self.last_approached) if self.last_approached else None,
            "last_placed": dict(self.last_placed) if self.last_placed else None,
            "last_spoken": self.last_spoken,
            "visited_zones": list(self.visited_zones),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldState":
        world = cls(
            sim_time=float(data.get("sim_time", 0.0)),
            robot=Pose.from_dict(data.get("robot", {})),
            v=float(data.get("v", 0.0)),
            w=float(data.get("w", 0.0)),
            battery=float(data.get("battery", 100.0)),
            holding=data.get("holding"),
            held_meta=dict(data["held_meta"]) if data.get("held_meta") else None,
            dock=tuple(data.get("dock", (0.0, 0.0))),  # type: ignore[arg-type]
            zones={k: [float(v[0]), float(v[1])] for k, v in (data.get("zones") or {}).items()},
            docked=bool(data.get("docked", False)),
            estop=bool(data.get("estop", False)),
            last_error=data.get("last_error"),
            known_objects={k: dict(v) for k, v in (data.get("known_objects") or {}).items()},
            last_detection=dict(data["last_detection"]) if data.get("last_detection") else None,
            last_approached=dict(data["last_approached"]) if data.get("last_approached") else None,
            last_placed=dict(data["last_placed"]) if data.get("last_placed") else None,
            last_spoken=data.get("last_spoken"),
            visited_zones=list(data.get("visited_zones") or []),
        )
        world.objects = {oid: GameObject.from_dict(o) for oid, o in (data.get("objects") or {}).items()}
        return world

    def snapshot(self, label: str = "") -> dict[str, Any]:
        """Compact snapshot used by traces and by the graph's before/after diff."""
        return {
            "label": label,
            "sim_time": round(self.sim_time, 3),
            "robot": self.robot.to_dict(),
            "v": round(self.v, 3),
            "w": round(self.w, 3),
            "battery": round(self.battery, 2),
            "holding": self.holding,
            "docked": self.docked,
            "estop": self.estop,
            "free_objects": len(self.free_objects()),
        }


@dataclass
class Detection:
    """A single perception / VLM detection."""

    object_id: str
    color: str
    category: str
    x: float
    y: float
    confidence: float = 1.0
    source: str = "perception"
    distance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "object_id": self.object_id,
            "color": self.color,
            "category": self.category,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }
        if self.distance is not None:
            data["distance"] = round(self.distance, 3)
        return data


@dataclass
class SkillResult:
    """What a skill backend reports back.  Note: never trusted for verification."""

    skill: str
    ok: bool
    code: FailureCode = FailureCode.OK
    detail: str = ""
    elapsed_s: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill,
            "ok": self.ok,
            "code": self.code.value,
            "detail": self.detail,
            "elapsed_s": round(self.elapsed_s, 3),
            "data": _jsonable(self.data),
        }

    @classmethod
    def success(cls, skill: str, detail: str = "", elapsed_s: float = 0.0, **data: Any) -> "SkillResult":
        return cls(skill=skill, ok=True, code=FailureCode.OK, detail=detail, elapsed_s=elapsed_s, data=dict(data))

    @classmethod
    def failure(
        cls, skill: str, code: FailureCode, detail: str = "", elapsed_s: float = 0.0, **data: Any
    ) -> "SkillResult":
        return cls(skill=skill, ok=False, code=code, detail=detail, elapsed_s=elapsed_s, data=dict(data))


def _jsonable(value: Any) -> Any:
    """Best-effort conversion so trace/API payloads stay JSON serializable."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Pose):
        return value.to_dict()
    return str(value)
