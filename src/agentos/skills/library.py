"""The 8 robot skills.

Each skill declares a JSON-schema parameter contract, preconditions and
postconditions evaluated against *world state*, a timeout, an explicit failure
code set and cancellation semantics.  The LLM/MCP layer can only choose one of
these skills and fill its parameters; nothing here accepts raw velocities.
"""

from __future__ import annotations

import math
from typing import Any

from agentos.skills.base import (
    Condition,
    Skill,
    SkillContext,
    battery_ok,
    detection_matches,
    dist,
    known_object_id,
    not_estop,
    resolve_query,
    resolve_target_xy,
    resolve_zone_xy,
)
from agentos.types import FailureCode, SkillCategory, SkillResult, WorldState

_QUERY_SCHEMA = {
    "type": "object",
    "description": "物体查询条件：颜色 / 类别，至少给一个",
    "properties": {
        "color": {"type": "string", "description": "red/blue/green/yellow/black/white"},
        "category": {"type": "string", "description": "box/cup/ball/package/tool/bottle"},
    },
    "additionalProperties": False,
}

_TARGET_SCHEMA = {
    "type": "object",
    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
    "required": ["x", "y"],
    "additionalProperties": False,
}


class NavigateToSkill(Skill):
    name = "navigate_to"
    category = SkillCategory.NAVIGATION
    description = "规划全局路径并移动到目标坐标或命名区域；模型只给目标，不给速度指令"
    params_schema = {
        "type": "object",
        "properties": {
            "target": _TARGET_SCHEMA,
            "x": {"type": "number"},
            "y": {"type": "number"},
            "zone": {"type": "string", "enum": ["zone_a", "zone_b", "zone_c", "table", "dock"]},
            "yaw": {"type": "number", "minimum": -3.15, "maximum": 3.15},
            "speed": {"type": "number", "minimum": 0.0, "maximum": 0.5, "description": "建议速度，网关会再裁剪"},
        },
        "additionalProperties": False,
        "anyOf": [{"required": ["target"]}, {"required": ["x", "y"]}, {"required": ["zone"]}],
    }
    preconditions = (
        Condition("estop_clear", "急停未触发", not_estop),
        Condition("battery_ok", "电量高于 2%", battery_ok),
    )
    postconditions = (
        Condition(
            "reached_target",
            "机器人距目标点不超过 0.5m",
            lambda world, params: _post_nav_distance(world, params) <= 0.5,
        ),
    )
    timeout_s = 45.0
    failure_codes = (
        FailureCode.TIMEOUT,
        FailureCode.BLOCKED,
        FailureCode.NO_PATH,
        FailureCode.OUT_OF_BOUNDS,
        FailureCode.ESTOP,
        FailureCode.CANCELLED,
    )
    recoverable_codes = frozenset({FailureCode.TIMEOUT, FailureCode.BLOCKED, FailureCode.NO_PATH})
    idempotent = True
    cancel_semantics = "cancel Nav2 NavigateToPose goal, clear goal handle, keep pose"

    def execute(self, ctx: SkillContext, params: dict[str, Any]) -> SkillResult:
        target = resolve_target_xy(params, ctx.world)
        if target is None:
            return SkillResult.failure(self.name, FailureCode.PRECONDITION_FAILED, "no resolvable navigation target")
        result = ctx.backend.navigate_to(
            target[0], target[1], yaw=params.get("yaw"), timeout_s=self.timeout_s
        )
        zone = params.get("zone")
        if result.ok and zone and zone not in ctx.world.visited_zones:
            ctx.world.visited_zones.append(zone)
        return result


def _post_nav_distance(world: WorldState, params: dict[str, Any]) -> float:
    target = resolve_target_xy(params, world)
    if target is None:
        return math.inf
    return dist(world, target[0], target[1])


class FindObjectSkill(Skill):
    name = "find_object"
    category = SkillCategory.PERCEPTION
    description = "在当前位置用相机/VLM 搜索目标物体并记录其世界坐标，不移动底盘"
    params_schema = {
        "type": "object",
        "properties": {
            "query": _QUERY_SCHEMA,
            "color": {"type": "string"},
            "category": {"type": "string"},
            "zone": {"type": "string", "description": "当前搜索区域（仅用于失败诊断与记忆）"},
        },
        "additionalProperties": False,
        "anyOf": [{"required": ["query"]}, {"required": ["color"]}, {"required": ["category"]}],
    }
    preconditions = (
        Condition("estop_clear", "急停未触发", not_estop),
        Condition("battery_ok", "电量高于 2%", battery_ok),
        Condition("gripper_free", "机械臂未占用", lambda world, params: world.holding is None),
    )
    postconditions = (
        Condition(
            "target_detected",
            "最新检测结果与查询条件一致",
            lambda world, params: detection_matches(world.last_detection, resolve_query(params)),
        ),
    )
    timeout_s = 10.0
    failure_codes = (
        FailureCode.NOT_FOUND,
        FailureCode.TIMEOUT,
        FailureCode.ESTOP,
        FailureCode.CANCELLED,
    )
    recoverable_codes = frozenset({FailureCode.NOT_FOUND, FailureCode.TIMEOUT})
    idempotent = True
    cancel_semantics = "clear last_detection for this attempt; no hardware goal to cancel"

    def execute(self, ctx: SkillContext, params: dict[str, Any]) -> SkillResult:
        query = resolve_query(params)
        if not query:
            return SkillResult.failure(self.name, FailureCode.PRECONDITION_FAILED, "empty object query")
        world = ctx.world
        zone = params.get("zone", "")
        detections = ctx.perception.locate(query, world)
        source = "camera"
        if not detections and ctx.vlm is not None:
            detections = ctx.vlm.locate(query, world)
            source = "vlm"
        if not detections:
            where = f"在{zone}" if zone else "在当前视野"
            return SkillResult.failure(
                self.name,
                FailureCode.NOT_FOUND,
                f"{where}未检测到匹配 {query} 的物体",
                data={"query": query, "zone": zone, "visible": len(ctx.perception.detect_all(world))},
            )
        best = detections[0]
        world.mark_detection(best.to_dict())
        return SkillResult.success(
            self.name,
            f"detected {best.color} {best.category} at ({best.x:.2f},{best.y:.2f})",
            data={"detection": best.to_dict(), "visible": len(detections), "source": source, "zone": zone},
        )


class ApproachObjectSkill(Skill):
    name = "approach_object"
    category = SkillCategory.MANIPULATION
    description = "导航到目标物体的抓取位姿（保持安全间距并朝向物体）"
    params_schema = {
        "type": "object",
        "properties": {
            "query": _QUERY_SCHEMA,
            "color": {"type": "string"},
            "category": {"type": "string"},
            "object_id": {"type": "string"},
            "standoff": {"type": "number", "minimum": 0.4, "maximum": 1.2},
        },
        "additionalProperties": False,
    }
    preconditions = (
        Condition("estop_clear", "急停未触发", not_estop),
        Condition("battery_ok", "电量高于 2%", battery_ok),
        Condition("gripper_free", "机械臂未占用", lambda world, params: world.holding is None),
        Condition(
            "object_known",
            "已知目标物体（先执行 find_object）",
            lambda world, params: known_object_id(world, params) is not None,
        ),
    )
    postconditions = (
        Condition(
            "at_standoff",
            "机器人与物体距离在 standoff+0.3m 内",
            lambda world, params: _approach_post_ok(world, params),
        ),
    )
    timeout_s = 45.0
    failure_codes = (
        FailureCode.NOT_FOUND,
        FailureCode.BLOCKED,
        FailureCode.NO_PATH,
        FailureCode.TIMEOUT,
        FailureCode.ESTOP,
        FailureCode.CANCELLED,
    )
    recoverable_codes = frozenset(
        {FailureCode.NOT_FOUND, FailureCode.BLOCKED, FailureCode.NO_PATH, FailureCode.TIMEOUT}
    )
    idempotent = True
    requires_prior = ("find_object",)
    cancel_semantics = "cancel approach navigation goal and clear goal handle"

    def execute(self, ctx: SkillContext, params: dict[str, Any]) -> SkillResult:
        object_id = known_object_id(ctx.world, params)
        if object_id is None:
            return SkillResult.failure(self.name, FailureCode.NOT_FOUND, "no known object to approach")
        standoff = float(params.get("standoff", 0.6))
        return ctx.backend.approach_object(object_id, standoff_m=standoff, timeout_s=self.timeout_s)


def _approach_post_ok(world: WorldState, params: dict[str, Any]) -> bool:
    approached = world.last_approached
    if not approached:
        return False
    query = resolve_query(params)
    if query and not detection_matches(approached, query):
        return False
    standoff = float(params.get("standoff", 0.6))
    return float(approached.get("distance", 99.0)) <= standoff + 0.4


class PickSkill(Skill):
    name = "pick"
    category = SkillCategory.MANIPULATION
    description = "闭合夹爪抓取已靠近的目标物体；失败返回 SLIP 等失败码"
    params_schema = {
        "type": "object",
        "properties": {
            "query": _QUERY_SCHEMA,
            "color": {"type": "string"},
            "category": {"type": "string"},
            "object_id": {"type": "string"},
        },
        "additionalProperties": False,
    }
    preconditions = (
        Condition("estop_clear", "急停未触发", not_estop),
        Condition("battery_ok", "电量高于 2%", battery_ok),
        Condition("gripper_free", "机械臂未占用", lambda world, params: world.holding is None),
        Condition(
            "object_in_reach",
            "目标物体在 0.95m 抓取范围内",
            lambda world, params: _object_in_reach(world, params),
        ),
    )
    postconditions = (
        Condition(
            "holding_target",
            "夹爪持有匹配查询的物体",
            lambda world, params: world.holding is not None
            and detection_matches(world.held_meta, resolve_query(params)),
        ),
    )
    timeout_s = 20.0
    failure_codes = (
        FailureCode.SLIP,
        FailureCode.NOT_FOUND,
        FailureCode.PRECONDITION_FAILED,
        FailureCode.TIMEOUT,
        FailureCode.ESTOP,
        FailureCode.CANCELLED,
    )
    recoverable_codes = frozenset({FailureCode.SLIP, FailureCode.NOT_FOUND, FailureCode.TIMEOUT})
    idempotent = False
    requires_prior = ("approach_object",)
    cancel_semantics = "open gripper, clear held object metadata, no residual grasp goal"

    def execute(self, ctx: SkillContext, params: dict[str, Any]) -> SkillResult:
        object_id = known_object_id(ctx.world, params)
        if object_id is None:
            return SkillResult.failure(self.name, FailureCode.NOT_FOUND, "no known object to pick")
        return ctx.backend.pick(object_id, timeout_s=self.timeout_s)


def _object_in_reach(world: WorldState, params: dict[str, Any]) -> bool:
    approached = world.last_approached
    if not approached:
        return False
    query = resolve_query(params)
    if query and not detection_matches(approached, query):
        return False
    if float(approached.get("distance", 99.0)) <= 1.0:
        return True
    # also accept a geometric check when world knowledge is richer
    object_id = known_object_id(world, params)
    obj = world.object_by_id(object_id) if object_id else None
    if obj is None:
        return False
    return dist(world, obj.x, obj.y) <= 1.0


class PlaceSkill(Skill):
    name = "place"
    category = SkillCategory.MANIPULATION
    description = "在目标区域放置夹爪中的物体，并校验物体落点与区域中心误差"
    params_schema = {
        "type": "object",
        "properties": {
            "zone": {"type": "string", "enum": ["zone_a", "zone_b", "zone_c", "table", "dock"]},
            "target": _TARGET_SCHEMA,
        },
        "additionalProperties": False,
        "anyOf": [{"required": ["zone"]}, {"required": ["target"]}],
    }
    preconditions = (
        Condition("estop_clear", "急停未触发", not_estop),
        Condition("battery_ok", "电量高于 2%", battery_ok),
        Condition("holding_object", "夹爪中已有物体", lambda world, params: world.holding is not None),
    )
    postconditions = (
        Condition(
            "object_near_zone",
            "物体落点在目标区域 0.9m 内且夹爪已释放",
            lambda world, params: _place_post_ok(world, params),
        ),
    )
    timeout_s = 20.0
    failure_codes = (
        FailureCode.NOT_HOLDING,
        FailureCode.PRECONDITION_FAILED,
        FailureCode.TIMEOUT,
        FailureCode.ESTOP,
        FailureCode.CANCELLED,
    )
    recoverable_codes = frozenset({FailureCode.TIMEOUT, FailureCode.PRECONDITION_FAILED})
    idempotent = False
    requires_prior = ("pick",)
    cancel_semantics = "release object at current pose, clear held metadata"

    def execute(self, ctx: SkillContext, params: dict[str, Any]) -> SkillResult:
        zone = str(params.get("zone", ""))
        target = resolve_target_xy(params, ctx.world)
        if target is None:
            return SkillResult.failure(self.name, FailureCode.PRECONDITION_FAILED, "no place target")
        return ctx.backend.place(target[0], target[1], zone=zone, timeout_s=self.timeout_s)


def _place_post_ok(world: WorldState, params: dict[str, Any]) -> bool:
    placed = world.last_placed
    if not placed or world.holding is not None:
        return False
    zone = str(params.get("zone", ""))
    if zone:
        target = resolve_zone_xy(world, zone)
    else:
        target = resolve_target_xy(params, world)
    if target is None:
        return False
    return math.hypot(float(placed.get("x", 0.0)) - target[0], float(placed.get("y", 0.0)) - target[1]) <= 0.9


class SpeakSkill(Skill):
    name = "speak"
    category = SkillCategory.SPEECH
    description = "通过 TTS 播报文本（长度受限，模型不能借此下发动作）"
    params_schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "minLength": 1, "maxLength": 200}},
        "required": ["text"],
        "additionalProperties": False,
    }
    preconditions = (Condition("estop_clear", "急停未触发", not_estop),)
    postconditions = (
        Condition(
            "spoken_text",
            "最近播报内容与参数一致",
            lambda world, params: world.last_spoken == str(params.get("text", "")),
        ),
    )
    timeout_s = 10.0
    failure_codes = (FailureCode.ESTOP, FailureCode.CANCELLED, FailureCode.SAFETY_REJECTED)
    recoverable_codes = frozenset()
    idempotent = True
    cancel_semantics = "stop TTS playback; no motion state involved"

    def execute(self, ctx: SkillContext, params: dict[str, Any]) -> SkillResult:
        return ctx.backend.speak(str(params["text"]))


class StopSkill(Skill):
    name = "stop"
    category = SkillCategory.SAFETY
    description = "立即停止运动（非闩锁急停；闩锁急停只能由操作员复位）"
    params_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    preconditions = ()
    postconditions = (
        Condition(
            "velocity_zero",
            "线速度与角速度均为 0",
            lambda world, params: abs(world.v) < 1e-6 and abs(world.w) < 1e-6,
        ),
    )
    timeout_s = 5.0
    failure_codes = (FailureCode.CANCELLED, FailureCode.BACKEND_ERROR)
    recoverable_codes = frozenset()
    idempotent = True
    cancel_semantics = "idempotent; zero velocities and clear any active motion goal"

    def execute(self, ctx: SkillContext, params: dict[str, Any]) -> SkillResult:
        return ctx.backend.stop()


class DockSkill(Skill):
    name = "dock"
    category = SkillCategory.POWER
    description = "导航回充电桩并对接充电，完成后电量恢复"
    params_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    preconditions = (
        Condition("estop_clear", "急停未触发", not_estop),
        Condition("gripper_free", "机械臂未占用", lambda world, params: world.holding is None),
    )
    postconditions = (
        Condition(
            "docked",
            "机器人位于充电桩 0.45m 内并处于充电状态",
            lambda world, params: world.docked and dist(world, world.dock[0], world.dock[1]) <= 0.45,
        ),
    )
    timeout_s = 50.0
    failure_codes = (
        FailureCode.BLOCKED,
        FailureCode.NO_PATH,
        FailureCode.TIMEOUT,
        FailureCode.PRECONDITION_FAILED,
        FailureCode.ESTOP,
        FailureCode.CANCELLED,
    )
    recoverable_codes = frozenset({FailureCode.BLOCKED, FailureCode.NO_PATH, FailureCode.TIMEOUT})
    idempotent = True
    cancel_semantics = "cancel dock navigation goal and clear goal handle"

    def execute(self, ctx: SkillContext, params: dict[str, Any]) -> SkillResult:
        return ctx.backend.dock(timeout_s=self.timeout_s)


ALL_SKILL_CLASSES = (
    NavigateToSkill,
    FindObjectSkill,
    ApproachObjectSkill,
    PickSkill,
    PlaceSkill,
    SpeakSkill,
    StopSkill,
    DockSkill,
)


def build_default_skills() -> list[Skill]:
    return [cls() for cls in ALL_SKILL_CLASSES]
