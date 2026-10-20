"""Deterministic 2D simulator backend implementing ``RobotBackend``.

This is the layer that would be replaced by ``Ros2SkillBackend`` on a real
robot.  It owns:

* the world state (objects, battery, docking, e-stop mirror)
* A* + pure-pursuit navigation with collision detection
* grasp/place/dock primitives
* one-shot fault injection
* frame hooks so demos can record real simulator runs

All velocity commands go through ``SafetyGateway.filter_velocity`` before the
differential-drive model integrates them.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable

from agentos.backends.base import RobotBackend
from agentos.config import SimConfig
from agentos.perception.sim_perception import SimPerception
from agentos.safety.gateway import SafetyConfig, SafetyGateway, SimClock
from agentos.sim.controller import PurePursuit, PurePursuitConfig
from agentos.sim.faults import FaultSpec
from agentos.sim.planner import astar, path_length, simplify_path
from agentos.sim.robot import DiffDriveRobot
from agentos.sim.scenes import Scene, sample_objects
from agentos.sim.world import OccupancyGrid, Rect
from agentos.types import FailureCode, GameObject, Pose, SkillResult, WorldState, normalize_angle


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class SimSkillBackend(RobotBackend):
    def __init__(
        self,
        scene: Scene,
        config: SimConfig | None = None,
        safety_config: SafetyConfig | None = None,
        seed: int = 0,
        faults: Iterable[FaultSpec | dict[str, Any] | str] | None = None,
        frame_hook: Callable[[WorldState, dict[str, Any]], None] | None = None,
        frame_every: int = 3,
    ) -> None:
        self.scene = scene
        self.cfg = config or SimConfig()
        self.seed = int(seed)
        self.clock = SimClock()
        self.safety = SafetyGateway(config=safety_config or SafetyConfig(), clock=self.clock)
        self.planning_grid: OccupancyGrid = scene.planning_grid()
        self.collision_grid: OccupancyGrid = scene.collision_grid()
        self.extra_obstacles: list[Rect] = []
        self.perception = SimPerception(
            self.collision_grid,
            radius_m=self.cfg.perception_radius_m,
            seed=self.seed,
            noise_m=self.cfg.perception_noise_m,
            dropout_p=self.cfg.perception_dropout_p,
        )
        self._rng = random.Random(self.seed * 104729 + 7)
        self._faults: list[FaultSpec] = [FaultSpec.from_dict(f) for f in (faults or [])]
        self._fired_faults: list[dict[str, Any]] = []
        self._frame_hook = frame_hook
        self._frame_every = max(1, int(frame_every))
        self._frame_counter = 0
        self._cancel_requested = False
        self._active_goal: dict[str, Any] | None = None
        self._path: list[tuple[float, float]] | None = None
        self._current_skill: str | None = None
        self._trail: list[tuple[float, float]] = []
        self._event_log: list[dict[str, Any]] = []
        self._nav_attempts = 0
        self._pick_attempts = 0
        self._degraded_nav = False

        # --- build world -----------------------------------------------------
        objects = sample_objects(scene, self.planning_grid, seed)
        start = self.planning_grid.nearest_free(*scene.start, max_radius=0.8) or scene.start
        start, start_yaw = self._jittered_start(start, scene.start_yaw)
        self._world = WorldState(
            robot=Pose(x=start[0], y=start[1], yaw=start_yaw),
            objects=objects,
            dock=scene.dock,
            zones={name: [x, y] for name, (x, y) in scene.zones.items()},
            battery=100.0,
        )
        self._robot = DiffDriveRobot(
            pose=self._world.robot,
            radius_m=self.cfg.robot_radius_m,
            max_linear=self.cfg.max_linear,
            max_angular=self.cfg.max_angular,
        )
        self._tracker_cfg = PurePursuitConfig(
            lookahead_m=self.cfg.lookahead_m,
            max_linear=self.cfg.max_linear * 0.95,
            max_angular=self.cfg.max_angular,
            goal_tolerance_m=self.cfg.goal_tolerance_m,
        )
        self._apply_initial_faults()

    # ------------------------------------------------------------------ basic
    @property
    def world(self) -> WorldState:
        self._sync_world()
        return self._world

    @property
    def sim_time(self) -> float:
        return self._world.sim_time

    @property
    def active_goal(self) -> dict[str, Any] | None:
        return self._active_goal

    @property
    def fired_faults(self) -> list[dict[str, Any]]:
        return list(self._fired_faults)

    @property
    def event_log(self) -> list[dict[str, Any]]:
        return list(self._event_log)

    def _sync_world(self) -> None:
        self._world.robot = self._robot.pose
        self._world.estop = self.safety.estop_engaged
        self._world.docked = self._is_at_dock() or self._world.docked

    def _log(self, event: str, **payload: Any) -> None:
        entry = {"event": event, "t": round(self._world.sim_time, 3), **payload}
        self._event_log.append(entry)

    def _maybe_frame(self) -> None:
        self._frame_counter += 1
        if self._frame_hook and self._frame_counter % self._frame_every == 0:
            self._frame_hook(
                self._world,
                {
                    "path": list(self._path or []),
                    "skill": self._current_skill,
                    "active_goal": self._active_goal,
                    "trail": list(self._trail[-240:]),
                    "obstacles": [r.to_list() for r in self.extra_obstacles],
                },
            )

    def set_frame_hook(
        self, hook: Callable[[WorldState, dict[str, Any]], None] | None, frame_every: int | None = None
    ) -> None:
        self._frame_hook = hook
        if frame_every:
            self._frame_every = max(1, int(frame_every))

    def _jittered_start(self, start: tuple[float, float], yaw: float) -> tuple[tuple[float, float], float]:
        """Stress evaluation: perturb the initial pose but keep it collision-free."""
        if not (self.cfg.start_jitter_m or self.cfg.start_jitter_yaw):
            return start, yaw
        jitter_rng = random.Random(self.seed * 31337 + 11)
        best = start
        for _ in range(80):
            angle = jitter_rng.uniform(0.0, 2.0 * math.pi)
            distance = jitter_rng.uniform(0.3, max(0.3, self.cfg.start_jitter_m))
            x = start[0] + distance * math.cos(angle)
            y = start[1] + distance * math.sin(angle)
            if self.planning_grid.is_free(x, y):
                best = (round(x, 3), round(y, 3))
                break
        yaw_new = yaw + jitter_rng.uniform(-self.cfg.start_jitter_yaw, self.cfg.start_jitter_yaw)
        return best, yaw_new

    # ------------------------------------------------------------- fault setup
    def _apply_initial_faults(self) -> None:
        remaining: list[FaultSpec] = []
        for spec in self._faults:
            if spec.kind == "object_missing":
                self._relocate_target(spec)
            else:
                remaining.append(spec)
        self._faults = remaining

    def _relocate_target(self, spec: FaultSpec) -> None:
        candidates = [o for o in self._world.free_objects() if spec.matches(o)]
        if not candidates:
            self._log("fault_skipped", kind=spec.kind, target=spec.target, reason="no matching object")
            return
        obj = candidates[0]
        order = list(self.scene.search_order)
        try:
            start_index = order.index(obj.zone)
        except ValueError:
            start_index = -1
        target_zone = order[(start_index + 1) % len(order)]
        position = self._free_point_near_zone(target_zone)
        old = (round(obj.x, 2), round(obj.y, 2))
        obj.x, obj.y = position
        obj.zone = target_zone
        self._fired_faults.append(
            {
                "kind": spec.kind,
                "target": obj.object_id,
                "from": old,
                "to": [round(obj.x, 2), round(obj.y, 2)],
                "zone": target_zone,
            }
        )
        self._log("fault_fired", kind=spec.kind, target=obj.object_id, to=list(obj.zone and target_zone))

    def _free_point_near_zone(self, zone: str) -> tuple[float, float]:
        cx, cy = self.scene.zone_center(zone)
        for _ in range(500):
            angle = self._rng.uniform(0.0, 2.0 * math.pi)
            distance = self._rng.uniform(0.0, 1.4)
            x = cx + distance * math.cos(angle)
            y = cy + distance * math.sin(angle)
            if self.planning_grid.is_free(x, y) and self.planning_grid.clearance(x, y, 0.7) > 0.45:
                return (round(x, 3), round(y, 3))
        return (cx, cy)

    def _pending_fault(self, kind: str) -> FaultSpec | None:
        for spec in self._faults:
            if spec.kind == kind:
                return spec
        return None

    def _consume_fault(self, spec: FaultSpec, **payload: Any) -> None:
        if spec in self._faults:
            self._faults.remove(spec)
        self._fired_faults.append({"kind": spec.kind, "target": spec.target, **payload})
        self._log("fault_fired", kind=spec.kind, target=spec.target)

    # ------------------------------------------------------------------ motion
    def _advance(self, dt: float) -> None:
        self.clock.advance(dt)
        self._world.sim_time += dt
        if not self._world.docked:
            self._world.battery = max(0.0, self._world.battery - self.cfg.battery_drain_per_s * dt)
        else:
            self._world.battery = min(100.0, self._world.battery + 0.5 * dt)
        self._sync_world()

    def _step(self, v: float, w: float, dt: float, check_collision: bool = True) -> bool:
        """Integrate one control step.  Returns False on collision (pose rolled back)."""
        v, w = self.safety.filter_velocity(v, w)
        old_pose = Pose(self._robot.pose.x, self._robot.pose.y, self._robot.pose.yaw)
        new_pose = self._robot.step(v, w, dt)
        if check_collision and not self.collision_grid.is_free(new_pose.x, new_pose.y):
            self._robot.pose = old_pose
            self._world.v = 0.0
            self._world.w = 0.0
            return False
        self._world.v, self._world.w = v, w
        self._advance(dt)
        self._trail.append((round(new_pose.x, 3), round(new_pose.y, 3)))
        if len(self._trail) > 600:
            self._trail = self._trail[-600:]
        self._maybe_frame()
        return True

    def _check_interrupts(self, skill: str, token: str) -> SkillResult | None:
        if self._cancel_requested:
            return SkillResult.failure(skill, FailureCode.CANCELLED, "cancel requested; goal cleared")
        if self.safety.estop_engaged:
            return SkillResult.failure(skill, FailureCode.ESTOP, f"e-stop latched: {self.safety.estop_reason}")
        if self.safety.check_timeout(token):
            return SkillResult.failure(skill, FailureCode.TIMEOUT, "skill timeout guard fired")
        return None

    # -------------------------------------------------------------- navigate
    def navigate_to(
        self, x: float, y: float, yaw: float | None = None, timeout_s: float = 45.0, arrive_tolerance: float | None = None
    ) -> SkillResult:
        skill = "navigate_to"
        self._nav_attempts += 1
        token = self.safety.begin_skill(skill, timeout_s)
        started = self._world.sim_time
        self._current_skill = skill
        try:
            if self.safety.estop_engaged:
                return SkillResult.failure(skill, FailureCode.ESTOP, "e-stop latched before navigation")
            if self.planning_grid.nearest_free(x, y, max_radius=0.6) is None:
                return SkillResult.failure(skill, FailureCode.OUT_OF_BOUNDS, f"target ({x:.2f},{y:.2f}) unreachable")
            path = astar(self.planning_grid, (self._world.robot.x, self._world.robot.y), (x, y))
            if not path:
                return SkillResult.failure(skill, FailureCode.NO_PATH, f"no path to ({x:.2f},{y:.2f})")
            path = simplify_path(self.planning_grid, path)
            self._path = path
            self._active_goal = {"type": "navigate_to", "target": [round(x, 3), round(y, 3)], "path_len": round(path_length(path), 2)}
            self._log("nav_start", target=[round(x, 2), round(y, 2)], path_len=round(path_length(path), 2))

            speed_scale = 1.0
            timeout_fault = self._pending_fault("nav_timeout")
            if timeout_fault is not None:
                speed_scale = 0.04
                self._degraded_nav = True
                self._consume_fault(timeout_fault, at=self._world.sim_time)

            tracker_config = self._tracker_cfg
            if arrive_tolerance is not None:
                tracker_config = PurePursuitConfig(
                    lookahead_m=self.cfg.lookahead_m,
                    max_linear=self.cfg.max_linear * 0.95,
                    max_angular=self.cfg.max_angular,
                    goal_tolerance_m=float(arrive_tolerance),
                )
            tracker = PurePursuit(tracker_config)
            tracker.reset()
            dt = self.cfg.dt
            max_steps = min(self.cfg.max_steps_per_skill, int(timeout_s / dt) + 20)
            reached = False
            blocked = False
            travelled = 0.0
            last = (self._world.robot.x, self._world.robot.y)
            for _ in range(max_steps):
                interrupt = self._check_interrupts(skill, token)
                if interrupt is not None:
                    self._log("nav_interrupted", code=interrupt.code.value)
                    return interrupt
                v, w, reached = tracker.command(self._world.robot, path)
                if reached:
                    break
                v *= speed_scale
                if not self._step(v, w, dt):
                    if not self._collision_recovery(tracker, path, dt):
                        blocked = True
                        break
                travelled += math.hypot(self._world.robot.x - last[0], self._world.robot.y - last[1])
                last = (self._world.robot.x, self._world.robot.y)
                self._maybe_inject_path_block(travelled)
                # local re-plan when tracking error grows (analogue of Nav2 recovery)
                if tracker.last_cross_track > 0.9:
                    re_path = astar(
                        self.planning_grid, (self._world.robot.x, self._world.robot.y), (x, y)
                    )
                    if re_path:
                        path = simplify_path(self.planning_grid, re_path)
                        self._path = path
                        tracker.reset()
            else:
                return SkillResult.failure(
                    skill, FailureCode.TIMEOUT, f"navigation exceeded {timeout_s:.1f}s", self._world.sim_time - started
                )
            if blocked:
                self._log("nav_blocked", at=[round(self._world.robot.x, 2), round(self._world.robot.y, 2)])
                return SkillResult.failure(
                    skill,
                    FailureCode.BLOCKED,
                    "collision detected; local escape failed",
                    self._world.sim_time - started,
                )
            if not reached:
                interrupt = self._check_interrupts(skill, token)
                if interrupt is not None:
                    return interrupt
                return SkillResult.failure(
                    skill, FailureCode.TIMEOUT, "navigation loop exhausted", self._world.sim_time - started
                )
            if yaw is not None:
                self._rotate_to(yaw, token)
            distance_to_goal = math.hypot(self._world.robot.x - x, self._world.robot.y - y)
            self._log("nav_done", distance=round(distance_to_goal, 3))
            return SkillResult.success(
                skill,
                f"reached goal with error {distance_to_goal:.2f}m",
                self._world.sim_time - started,
                path_length=round(path_length(path), 2),
                distance_error=round(distance_to_goal, 3),
                final_pose=self._world.robot.to_dict(),
            )
        finally:
            self.safety.end_skill(token)
            self._current_skill = None
            self._active_goal = None
            self._sync_world()

    def _rotate_to(self, yaw: float, token: str) -> bool:
        dt = self.cfg.dt
        for _ in range(120):
            if self.safety.check_timeout(token) or self._cancel_requested:
                return False
            error = normalize_angle(yaw - self._world.robot.yaw)
            if abs(error) < 0.06:
                return True
            w = _clamp(2.2 * error, -self.cfg.max_angular, self.cfg.max_angular)
            if not self._step(0.0, w, dt):
                return False
        return abs(normalize_angle(yaw - self._world.robot.yaw)) < 0.15

    def _collision_recovery(self, tracker: PurePursuit, path: list[tuple[float, float]], dt: float) -> bool:
        """Back off, then rotate toward the carrot before retrying.

        Pure-pursuit on a path that hugs an inflated obstacle can end up with the
        nose against the safety margin.  Rotating in place is not enough (the
        robot does not move), so the recovery must reverse first.
        """
        backed = 0.0
        for _ in range(12):
            if not self._step(-0.22, 0.0, dt):
                break
            backed += 0.22 * dt
        if backed < 0.02:
            return False
        for _ in range(25):
            _v, w, reached = tracker.command(self._world.robot, path)
            if reached:
                return True
            if abs(w) < 0.12:
                break
            if not self._step(0.0, _clamp(w, -self.cfg.max_angular, self.cfg.max_angular), dt):
                break
        return True

    def _maybe_inject_path_block(self, travelled: float) -> None:
        spec = self._pending_fault("path_blocked")
        if spec is None or travelled < 1.0:
            return
        if not self._path or len(self._path) < 3:
            self._consume_fault(spec, at=self._world.sim_time)
            return
        robot = self._world.robot
        # find a point ~1.6 m ahead along the remaining path
        target = None
        for px, py in self._path:
            distance = math.hypot(px - robot.x, py - robot.y)
            if 1.4 <= distance <= 2.2:
                target = (px, py)
                break
        if target is None:
            self._consume_fault(spec, at=self._world.sim_time)
            return
        half = 0.45
        rect = Rect(target[0] - half, target[1] - half, target[0] + half, target[1] + half)
        # never block the robot itself or the final goal region
        goal = self._path[-1]
        if math.hypot(target[0] - goal[0], target[1] - goal[1]) < 1.0:
            self._consume_fault(spec, at=self._world.sim_time)
            return
        self.extra_obstacles.append(rect)
        self._rebuild_grids()
        self._consume_fault(spec, at=self._world.sim_time, rect=rect.to_list())
        self._log("path_blocked", rect=rect.to_list())

    def _rebuild_grids(self) -> None:
        self.planning_grid = OccupancyGrid(
            self.scene.width,
            self.scene.height,
            self.cfg.grid_resolution_m,
            [*self.scene.obstacles, *self.extra_obstacles],
            inflation=self.planning_grid.inflation,
        )
        self.collision_grid = OccupancyGrid(
            self.scene.width,
            self.scene.height,
            self.cfg.grid_resolution_m,
            [*self.scene.obstacles, *self.extra_obstacles],
            inflation=self.collision_grid.inflation,
        )
        self.perception.grid = self.collision_grid

    # -------------------------------------------------------------- approach
    def approach_object(self, object_id: str, standoff_m: float = 0.6, timeout_s: float = 45.0) -> SkillResult:
        skill = "approach_object"
        started = self._world.sim_time
        obj = self._world.objects.get(object_id)
        if obj is None:
            return SkillResult.failure(skill, FailureCode.NOT_FOUND, f"object {object_id!r} not in world model")
        robot = self._world.robot
        dx, dy = robot.x - obj.x, robot.y - obj.y
        norm = math.hypot(dx, dy)
        base_angle = math.atan2(dy, dx) if norm > 1e-6 else 0.0
        chosen: tuple[float, float] | None = None
        for offset in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.4, -2.4, math.pi):
            angle = base_angle + offset
            px = obj.x + standoff_m * math.cos(angle)
            py = obj.y + standoff_m * math.sin(angle)
            if not self.planning_grid.is_free(px, py):
                continue
            if astar(self.planning_grid, (robot.x, robot.y), (px, py)) is None:
                continue
            chosen = (px, py)
            break
        if chosen is None:
            return SkillResult.failure(skill, FailureCode.NO_PATH, "no reachable approach pose around object")
        yaw = math.atan2(obj.y - chosen[1], obj.x - chosen[0])
        nav = self.navigate_to(chosen[0], chosen[1], yaw=yaw, timeout_s=timeout_s, arrive_tolerance=0.2)
        if not nav.ok:
            return SkillResult(
                skill=skill,
                ok=False,
                code=nav.code,
                detail=f"approach navigation failed: {nav.detail}",
                elapsed_s=self._world.sim_time - started,
                data={"nav": nav.to_dict()},
            )
        distance = math.hypot(self._world.robot.x - obj.x, self._world.robot.y - obj.y)
        self._world.last_approached = {
            "object_id": object_id,
            "color": obj.color,
            "category": obj.category,
            "x": round(obj.x, 3),
            "y": round(obj.y, 3),
            "distance": round(distance, 3),
        }
        return SkillResult.success(
            skill,
            f"approached {object_id} at {distance:.2f}m",
            self._world.sim_time - started,
            object_id=object_id,
            distance=round(distance, 3),
            standoff=standoff_m,
        )

    # ------------------------------------------------------------------ pick
    def pick(self, object_id: str, timeout_s: float = 20.0) -> SkillResult:
        skill = "pick"
        started = self._world.sim_time
        self._pick_attempts += 1
        if self._world.holding:
            return SkillResult.failure(
                skill, FailureCode.PRECONDITION_FAILED, f"already holding {self._world.holding}"
            )
        obj = self._world.objects.get(object_id)
        if obj is None:
            return SkillResult.failure(skill, FailureCode.NOT_FOUND, f"object {object_id!r} not in world model")
        distance = math.hypot(self._world.robot.x - obj.x, self._world.robot.y - obj.y)
        if distance > 1.1:
            return SkillResult.failure(
                skill,
                FailureCode.PRECONDITION_FAILED,
                f"object is {distance:.2f}m away; approach first (max 1.1m)",
            )
        if not obj.graspable:
            return SkillResult.failure(skill, FailureCode.PRECONDITION_FAILED, "object is not graspable")
        self._consume_time(0.6)
        slip = self._pending_fault("pick_slip")
        if slip is not None:
            self._consume_fault(slip, object_id=object_id)
            return SkillResult.failure(
                skill,
                FailureCode.SLIP,
                "gripper closed but object slipped out",
                self._world.sim_time - started,
                object_id=object_id,
            )
        self._world.remove_object(object_id)
        self._world.holding = object_id
        self._world.held_meta = obj.to_dict()
        self._world.last_detection = None
        self._log("pick_ok", object_id=object_id)
        return SkillResult.success(
            skill,
            f"grasped {object_id}",
            self._world.sim_time - started,
            object_id=object_id,
            color=obj.color,
            category=obj.category,
        )

    # ----------------------------------------------------------------- place
    def place(self, x: float, y: float, zone: str = "", timeout_s: float = 20.0) -> SkillResult:
        skill = "place"
        started = self._world.sim_time
        if not self._world.holding:
            return SkillResult.failure(skill, FailureCode.NOT_HOLDING, "no object in gripper")
        meta = dict(self._world.held_meta or {})
        self._consume_time(0.6)
        robot = self._world.robot
        candidates = [
            (robot.x + 0.38 * math.cos(robot.yaw), robot.y + 0.38 * math.sin(robot.yaw)),
            (robot.x - 0.38 * math.cos(robot.yaw), robot.y - 0.38 * math.sin(robot.yaw)),
            (robot.x, robot.y),
        ]
        drop = candidates[-1]
        for cx, cy in candidates:
            if self.collision_grid.is_free(cx, cy):
                drop = (cx, cy)
                break
        object_id = str(meta.get("object_id", self._world.holding))
        obj = GameObject(
            object_id=object_id,
            color=str(meta.get("color", "unknown")),
            category=str(meta.get("category", "unknown")),
            x=round(drop[0], 3),
            y=round(drop[1], 3),
            zone=zone,
        )
        self._world.add_object(obj)
        self._world.holding = None
        self._world.held_meta = None
        self._world.last_placed = {
            "object_id": object_id,
            "color": obj.color,
            "category": obj.category,
            "x": obj.x,
            "y": obj.y,
            "zone": zone,
            "target_xy": [round(x, 3), round(y, 3)],
        }
        self._log("place_ok", object_id=object_id, zone=zone)
        return SkillResult.success(
            skill,
            f"placed {object_id} in zone {zone or 'n/a'}",
            self._world.sim_time - started,
            object_id=object_id,
            drop=[obj.x, obj.y],
            zone=zone,
        )

    # --------------------------------------------------------------- various
    def speak(self, text: str) -> SkillResult:
        skill = "speak"
        self._consume_time(0.15)
        self._world.last_spoken = text
        return SkillResult.success(skill, f"spoke {len(text)} chars", data={"text": text})

    def stop(self) -> SkillResult:
        skill = "stop"
        self._world.v = 0.0
        self._world.w = 0.0
        self._world.robot = self._robot.pose
        self._active_goal = None
        self._path = None
        self._log("stop_ok")
        return SkillResult.success(skill, "motion stopped")

    def dock(self, timeout_s: float = 60.0) -> SkillResult:
        skill = "dock"
        started = self._world.sim_time
        if self._world.holding:
            return SkillResult.failure(
                skill, FailureCode.PRECONDITION_FAILED, "cannot dock while holding an object"
            )
        dock_x, dock_y = self.scene.dock
        distance = math.hypot(self._world.robot.x - dock_x, self._world.robot.y - dock_y)
        if distance > self.cfg.goal_tolerance_m:
            nav = self.navigate_to(dock_x, dock_y, timeout_s=timeout_s)
            if not nav.ok:
                return SkillResult(
                    skill=skill,
                    ok=False,
                    code=nav.code,
                    detail=f"dock navigation failed: {nav.detail}",
                    elapsed_s=self._world.sim_time - started,
                    data={"nav": nav.to_dict()},
                )
        self._consume_time(0.8)
        self._world.docked = True
        self._world.battery = 100.0
        self._log("dock_ok")
        return SkillResult.success(
            skill, "docked and charging", self._world.sim_time - started, pose=self._world.robot.to_dict()
        )

    def retreat(self, distance_m: float = 0.35) -> SkillResult:
        skill = "retreat"
        robot = self._world.robot
        target = (
            robot.x - distance_m * math.cos(robot.yaw),
            robot.y - distance_m * math.sin(robot.yaw),
        )
        dt = self.cfg.dt
        steps = max(1, int(abs(distance_m) / (self.cfg.max_linear * 0.6 * dt)))
        moved = 0.0
        for _ in range(steps):
            if self._cancel_requested or self.safety.estop_engaged:
                break
            remaining = math.hypot(target[0] - self._world.robot.x, target[1] - self._world.robot.y)
            if remaining < 0.06:
                break
            v = -min(0.25, remaining / dt * 0.5)
            if not self._step(v, 0.0, dt):
                break
            moved += abs(v) * dt
        return SkillResult.success(skill, f"retreated {moved:.2f}m", data={"moved": round(moved, 3)})

    def cancel_active_goal(self) -> dict[str, Any]:
        was_active = self._active_goal is not None
        goal = self._active_goal
        self._active_goal = None
        self._path = None
        self._cancel_requested = True
        self._log("cancel_goal", was_active=was_active)
        return {"cancelled": True, "was_active": was_active, "goal": goal, "handle_cleared": True}

    def clear_cancel(self) -> None:
        self._cancel_requested = False

    # ------------------------------------------------------------------ misc
    def _is_at_dock(self) -> bool:
        return math.hypot(self._world.robot.x - self.scene.dock[0], self._world.robot.y - self.scene.dock[1]) <= 0.45

    def _consume_time(self, seconds: float) -> None:
        dt = self.cfg.dt / 2.0
        steps = max(1, int(seconds / dt))
        for _ in range(steps):
            self._advance(dt)
            self._maybe_frame()

    def snapshot(self) -> dict[str, Any]:
        return {
            "scene": self.scene.name,
            "seed": self.seed,
            "world": self._world.to_dict(),
            "extra_obstacles": [rect.to_list() for rect in self.extra_obstacles],
            "fired_faults": list(self._fired_faults),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        world_data = snapshot.get("world") or {}
        if world_data:
            self._world = WorldState.from_dict(world_data)
            self._robot.pose = self._world.robot
        extra = snapshot.get("extra_obstacles") or []
        if extra:
            self.extra_obstacles = [Rect(*values) for values in extra]
            self._rebuild_grids()
        self.safety.end_skill("restore")
        self._sync_world()

    @classmethod
    def from_snapshot(
        cls,
        scene: Scene,
        snapshot: dict[str, Any],
        config: SimConfig | None = None,
        safety_config: SafetyConfig | None = None,
        frame_hook: Callable[[WorldState, dict[str, Any]], None] | None = None,
        frame_every: int = 3,
    ) -> "SimSkillBackend":
        backend = cls(
            scene,
            config=config,
            safety_config=safety_config,
            seed=int(snapshot.get("seed", 0)),
            faults=None,
            frame_hook=frame_hook,
            frame_every=frame_every,
        )
        backend.restore(snapshot)
        return backend

    # ------------------------------------------------------------- rendering
    def render_state(self) -> dict[str, Any]:
        self._sync_world()
        return {
            "scene": self.scene,
            "world": self._world,
            "path": list(self._path or []),
            "trail": list(self._trail[-240:]),
            "extra_obstacles": list(self.extra_obstacles),
            "active_goal": self._active_goal,
            "current_skill": self._current_skill,
        }
