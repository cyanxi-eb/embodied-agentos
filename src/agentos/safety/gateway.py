"""Unified safety gateway.

Every skill call passes through ``authorize`` *before* it can touch a robot
backend, and every velocity command passes through ``filter_velocity`` before
it reaches the wheels.  The four guards required by the design are:

1. velocity clamping (linear / angular)
2. latched e-stop with an operator reset ceremony
3. per-skill timeout bookkeeping
4. polygonal geofence + deny-list for dangerous skills / parameters

The gateway deliberately keeps *no* robot state of its own beyond guard state,
so a rejected command can never have side effects.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from agentos.sim.world import point_in_polygon
from agentos.types import FailureCode, WorldState


@dataclass(frozen=True)
class SafetyConfig:
    max_linear: float = 0.5
    min_linear: float = -0.2
    max_angular: float = 1.0
    map_bounds: tuple[float, float, float, float] = (0.0, 0.0, 12.0, 12.0)
    geofence_margin: float = 0.1
    geofence_polygon: tuple[tuple[float, float], ...] | None = None
    max_skill_timeout_s: float = 120.0
    default_skill_timeout_s: float = 45.0
    min_battery_pct: float = 2.0
    max_speak_chars: int = 200
    estop_reset_code: str = "AGENTOS-RESET"
    denied_skills: tuple[str, ...] = (
        "cmd_vel",
        "raw_cmd_vel",
        "direct_control",
        "joint_velocity",
        "motor_pwm",
        "teleop_raw",
    )
    denied_param_keys: tuple[str, ...] = (
        "cmd_vel",
        "raw_cmd_vel",
        "joint_velocity",
        "motor_pwm",
        "pwm",
        "linear_x",
        "angular_z",
        "raw_command",
        "direct_velocity",
    )
    clamp_param_keys_linear: tuple[str, ...] = ("speed", "max_speed", "v", "linear_speed", "linear_velocity")
    clamp_param_keys_angular: tuple[str, ...] = ("omega", "w", "angular_speed", "angular_velocity")
    #: Skills that must stay available even on a nearly empty battery.
    battery_exempt_skills: tuple[str, ...] = ("dock", "stop", "speak")

    def geofence_polygon_resolved(self) -> tuple[tuple[float, float], ...]:
        if self.geofence_polygon:
            return self.geofence_polygon
        x0, y0, x1, y1 = self.map_bounds
        m = self.geofence_margin
        return ((x0 + m, y0 + m), (x1 - m, y0 + m), (x1 - m, y1 - m), (x0 + m, y1 - m))


@dataclass
class SafetyVerdict:
    allowed: bool
    code: FailureCode = FailureCode.OK
    reason: str = ""
    sanitized: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "code": self.code.value,
            "reason": self.reason,
            "sanitized": self.sanitized,
        }


class SafetyClock:
    """Monotonic wall clock by default; the simulator injects a sim clock."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._start


class SimClock(SafetyClock):
    """Clock advanced by the simulator (makes timeout behaviour testable)."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class SafetyGateway:
    def __init__(
        self,
        config: SafetyConfig | None = None,
        clock: SafetyClock | None = None,
        registry_names: Iterable[str] | None = None,
    ) -> None:
        self.config = config or SafetyConfig()
        self.clock = clock or SafetyClock()
        self._registry: set[str] = set(registry_names or ())
        self._estop = False
        self._estop_reason = ""
        self._estop_source = ""
        self._violations: list[dict[str, Any]] = []
        self._clamps: list[dict[str, Any]] = []
        self._tokens: dict[str, dict[str, Any]] = {}
        self._token_seq = 0
        self._estop_events: list[dict[str, Any]] = []

    # ------------------------------------------------------------- registry
    def register_skills(self, names: Iterable[str]) -> None:
        self._registry.update(names)

    @property
    def skill_names(self) -> frozenset[str]:
        return frozenset(self._registry)

    # ---------------------------------------------------------------- e-stop
    @property
    def estop_engaged(self) -> bool:
        return self._estop

    @property
    def estop_reason(self) -> str:
        return self._estop_reason

    def engage_estop(self, reason: str, source: str = "operator") -> dict[str, Any]:
        """Latch the e-stop.  No skill is authorized until ``reset_estop``."""
        self._estop = True
        self._estop_reason = reason
        self._estop_source = source
        event = {
            "event": "estop_engaged",
            "reason": reason,
            "source": source,
            "at": round(self.clock.now(), 4),
        }
        self._estop_events.append(event)
        return event

    def reset_estop(self, operator_id: str, code: str) -> dict[str, Any]:
        """Reset requires an operator identity *and* the reset code (not a UI click)."""
        if not self._estop:
            return {"ok": False, "reason": "e-stop is not engaged"}
        if not operator_id or not str(operator_id).strip():
            return {"ok": False, "reason": "operator_id is required to reset the e-stop"}
        if code != self.config.estop_reset_code:
            return {"ok": False, "reason": "invalid estop reset code"}
        self._estop = False
        reason = self._estop_reason
        self._estop_reason = ""
        event = {
            "event": "estop_reset",
            "operator_id": operator_id,
            "previous_reason": reason,
            "at": round(self.clock.now(), 4),
        }
        self._estop_events.append(event)
        return {"ok": True, **event}

    # --------------------------------------------------------------- timeout
    def begin_skill(self, skill_name: str, timeout_s: float | None = None) -> str:
        self._token_seq += 1
        token = f"skill-{self._token_seq}"
        limit = timeout_s if timeout_s is not None else self.config.default_skill_timeout_s
        self._tokens[token] = {
            "skill": skill_name,
            "start": self.clock.now(),
            "limit": float(limit),
        }
        return token

    def check_timeout(self, token: str) -> bool:
        info = self._tokens.get(token)
        if info is None:
            return False
        if info["limit"] <= 0:
            return False
        return (self.clock.now() - info["start"]) > info["limit"]

    def end_skill(self, token: str) -> None:
        self._tokens.pop(token, None)

    # -------------------------------------------------------------- velocity
    def filter_velocity(self, v: float, w: float) -> tuple[float, float]:
        """Clamp to configured limits; zero the command while the e-stop is latched."""
        if self._estop:
            return 0.0, 0.0
        cv = max(self.config.min_linear, min(self.config.max_linear, float(v)))
        cw = max(-self.config.max_angular, min(self.config.max_angular, float(w)))
        if abs(cv - v) > 1e-9 or abs(cw - w) > 1e-9:
            self._clamps.append(
                {"requested": [round(v, 4), round(w, 4)], "applied": [round(cv, 4), round(cw, 4)]}
            )
        return cv, cw

    # -------------------------------------------------------------- authorize
    def authorize(
        self,
        skill_name: str,
        params: dict[str, Any] | None,
        world: WorldState | None = None,
        timeout_s: float | None = None,
    ) -> SafetyVerdict:
        params = dict(params or {})
        # 1) latched e-stop
        if self._estop:
            return self._reject(
                skill_name, FailureCode.ESTOP, f"e-stop latched: {self._estop_reason or 'operator stop'}"
            )
        # 2) deny-list: raw actuation skills are never callable from the agent
        if skill_name in self.config.denied_skills:
            return self._reject(
                skill_name, FailureCode.SAFETY_REJECTED, f"dangerous skill {skill_name!r} is deny-listed"
            )
        if skill_name not in self._registry:
            return self._reject(skill_name, FailureCode.UNKNOWN_SKILL, f"skill {skill_name!r} is not registered")
        # 3) deny-list parameters (someone trying to bypass the skill abstraction)
        denied = self._find_denied_keys(params)
        if denied:
            return self._reject(
                skill_name,
                FailureCode.SAFETY_REJECTED,
                f"parameters {sorted(denied)} are deny-listed (raw actuation must not come from the model)",
            )
        # 4) per-skill timeout budget
        limit = timeout_s if timeout_s is not None else self.config.default_skill_timeout_s
        if limit > self.config.max_skill_timeout_s:
            return self._reject(
                skill_name,
                FailureCode.SAFETY_REJECTED,
                f"timeout {limit}s exceeds gateway maximum {self.config.max_skill_timeout_s}s",
            )
        # 5) battery
        if world is not None and skill_name not in self.config.battery_exempt_skills:
            if world.battery < self.config.min_battery_pct:
                return self._reject(
                    skill_name,
                    FailureCode.LOW_BATTERY,
                    f"battery {world.battery:.1f}% below {self.config.min_battery_pct}%",
                )
        # 6) speak payload guard
        if skill_name == "speak":
            text = str(params.get("text", ""))
            if not text.strip():
                return self._reject(skill_name, FailureCode.PRECONDITION_FAILED, "speak text is empty")
            if len(text) > self.config.max_speak_chars:
                return self._reject(
                    skill_name,
                    FailureCode.SAFETY_REJECTED,
                    f"speak text longer than {self.config.max_speak_chars} characters",
                )
        # 7) geofence for explicit target coordinates
        polygon = self.config.geofence_polygon_resolved()
        for label, (x, y) in self._extract_targets(params).items():
            if not point_in_polygon(x, y, polygon):
                return self._reject(
                    skill_name,
                    FailureCode.OUT_OF_BOUNDS,
                    f"{label}=({x:.2f},{y:.2f}) is outside the electronic fence",
                )
        # 8) clamp velocity-like parameters (defense in depth; skills rarely carry them)
        sanitized: dict[str, Any] = {}
        clamped = False
        for key, value in params.items():
            if key in self.config.clamp_param_keys_linear and isinstance(value, (int, float)):
                new_value = max(self.config.min_linear, min(self.config.max_linear, float(value)))
                clamped = clamped or abs(new_value - float(value)) > 1e-9
                sanitized[key] = new_value
            elif key in self.config.clamp_param_keys_angular and isinstance(value, (int, float)):
                new_value = max(-self.config.max_angular, min(self.config.max_angular, float(value)))
                clamped = clamped or abs(new_value - float(value)) > 1e-9
                sanitized[key] = new_value
            else:
                sanitized[key] = value
        if clamped:
            self._clamps.append({"skill": skill_name, "requested": params, "applied": sanitized})
            sanitized["__clamped__"] = True
        return SafetyVerdict(allowed=True, code=FailureCode.OK, sanitized=sanitized)

    # ---------------------------------------------------------------- status
    @property
    def violations(self) -> list[dict[str, Any]]:
        return list(self._violations)

    @property
    def clamps(self) -> list[dict[str, Any]]:
        return list(self._clamps)

    @property
    def estop_events(self) -> list[dict[str, Any]]:
        return list(self._estop_events)

    def status(self) -> dict[str, Any]:
        return {
            "estop_engaged": self._estop,
            "estop_reason": self._estop_reason,
            "violations": len(self._violations),
            "clamps": len(self._clamps),
            "active_timeouts": len(self._tokens),
            "max_linear": self.config.max_linear,
            "max_angular": self.config.max_angular,
            "geofence": [list(p) for p in self.config.geofence_polygon_resolved()],
        }

    # --------------------------------------------------------------- helpers
    def _reject(self, skill: str, code: FailureCode, reason: str) -> SafetyVerdict:
        self._violations.append(
            {
                "skill": skill,
                "code": code.value,
                "reason": reason,
                "at": round(self.clock.now(), 4),
            }
        )
        return SafetyVerdict(allowed=False, code=code, reason=reason)

    def _find_denied_keys(self, params: dict[str, Any], prefix: str = "") -> set[str]:
        found: set[str] = set()
        for key, value in params.items():
            if key in self.config.denied_param_keys:
                found.add(f"{prefix}{key}")
            if isinstance(value, dict):
                found |= self._find_denied_keys(value, prefix=f"{prefix}{key}.")
        return found

    @staticmethod
    def _extract_targets(params: dict[str, Any]) -> dict[str, tuple[float, float]]:
        """Collect explicit ``(x, y)`` pairs from common skill parameter shapes."""
        targets: dict[str, tuple[float, float]] = {}
        if "x" in params and "y" in params:
            try:
                targets["target"] = (float(params["x"]), float(params["y"]))
            except (TypeError, ValueError):
                pass
        for key in ("target", "position", "destination", "pose"):
            value = params.get(key)
            if isinstance(value, dict) and "x" in value and "y" in value:
                try:
                    targets[key] = (float(value["x"]), float(value["y"]))
                except (TypeError, ValueError):
                    pass
            elif isinstance(value, (list, tuple)) and len(value) >= 2:
                try:
                    targets[key] = (float(value[0]), float(value[1]))
                except (TypeError, ValueError):
                    pass
        return targets
