"""Safety gateway: clamping, latched e-stop, timeout, geofence, deny-list."""

from __future__ import annotations

import pytest

from agentos.safety.gateway import SafetyConfig, SafetyGateway, SimClock
from agentos.types import WorldState


def make_gateway(**config_overrides) -> tuple[SafetyGateway, SimClock]:
    clock = SimClock()
    config = SafetyConfig(**config_overrides)
    gateway = SafetyGateway(config=config, clock=clock, registry_names=["navigate_to", "find_object", "pick", "place", "speak", "stop", "dock", "approach_object"])
    return gateway, clock


def test_velocity_clamped():
    gateway, _ = make_gateway()
    assert gateway.filter_velocity(5.0, 10.0) == (0.5, 1.0)
    assert gateway.filter_velocity(-3.0, -9.0) == (-0.2, -1.0)
    assert len(gateway.clamps) == 2


def test_estop_latches_until_operator_reset():
    gateway, _ = make_gateway()
    gateway.engage_estop("hardware button pressed", source="test")
    assert gateway.estop_engaged
    assert gateway.filter_velocity(0.4, 0.2) == (0.0, 0.0)
    verdict = gateway.authorize("navigate_to", {"zone": "zone_a"}, WorldState())
    assert not verdict.allowed
    assert verdict.code.value == "ESTOP"

    # reset requires an operator identity and the reset code
    assert gateway.reset_estop("", "AGENTOS-RESET")["ok"] is False
    assert gateway.reset_estop("op-1", "WRONG")["ok"] is False
    assert gateway.estop_engaged
    assert gateway.reset_estop("op-1", "AGENTOS-RESET")["ok"] is True
    assert not gateway.estop_engaged
    assert gateway.authorize("navigate_to", {"zone": "zone_a"}, WorldState()).allowed


def test_estop_stays_latched_after_failed_reset_attempts():
    gateway, _ = make_gateway()
    gateway.engage_estop("watchdog", source="test")
    for _ in range(3):
        gateway.reset_estop("op", "bad-code")
    assert gateway.estop_engaged
    assert len(gateway.estop_events) == 1  # only the engage event is recorded


def test_geofence_rejects_out_of_bounds():
    gateway, _ = make_gateway()
    verdict = gateway.authorize("navigate_to", {"target": {"x": 50.0, "y": 50.0}}, WorldState())
    assert not verdict.allowed
    assert verdict.code.value == "OUT_OF_BOUNDS"
    ok = gateway.authorize("navigate_to", {"target": {"x": 4.0, "y": 4.0}}, WorldState())
    assert ok.allowed


def test_deny_listed_skill_and_parameters():
    gateway, _ = make_gateway()
    verdict = gateway.authorize("cmd_vel", {"v": 0.5}, WorldState())
    assert not verdict.allowed
    assert verdict.code.value == "SAFETY_REJECTED"
    verdict = gateway.authorize("navigate_to", {"target": {"x": 1.0, "y": 1.0}, "cmd_vel": 0.5}, WorldState())
    assert not verdict.allowed
    assert "deny-listed" in verdict.reason


def test_deny_listed_nested_parameter():
    gateway, _ = make_gateway()
    verdict = gateway.authorize(
        "pick", {"query": {"color": "red"}, "extra": {"motor_pwm": 200}}, WorldState()
    )
    assert not verdict.allowed
    assert verdict.code.value == "SAFETY_REJECTED"


def test_unknown_skill_rejected():
    gateway, _ = make_gateway()
    verdict = gateway.authorize("fly_to_moon", {}, WorldState())
    assert not verdict.allowed
    assert verdict.code.value == "UNKNOWN_SKILL"


def test_skill_timeout_guard_uses_backend_clock():
    gateway, clock = make_gateway()
    token = gateway.begin_skill("navigate_to", timeout_s=5.0)
    clock.advance(3.0)
    assert not gateway.check_timeout(token)
    clock.advance(3.0)
    assert gateway.check_timeout(token)
    gateway.end_skill(token)
    assert not gateway.check_timeout(token)


def test_timeout_budget_cannot_exceed_gateway_maximum():
    gateway, _ = make_gateway()
    verdict = gateway.authorize("navigate_to", {"zone": "zone_a"}, WorldState(), timeout_s=999.0)
    assert not verdict.allowed
    assert verdict.code.value == "SAFETY_REJECTED"


def test_low_battery_blocks_motion_but_not_dock():
    gateway, _ = make_gateway()
    world = WorldState(battery=1.0)
    assert not gateway.authorize("navigate_to", {"zone": "zone_a"}, world).allowed
    assert gateway.authorize("navigate_to", {"zone": "zone_a"}, world).code.value == "LOW_BATTERY"
    assert gateway.authorize("dock", {}, world).allowed
    assert gateway.authorize("stop", {}, world).allowed


def test_speak_payload_guards():
    gateway, _ = make_gateway()
    assert not gateway.authorize("speak", {"text": ""}, WorldState()).allowed
    assert not gateway.authorize("speak", {"text": "x" * 500}, WorldState()).allowed
    assert gateway.authorize("speak", {"text": "任务完成"}, WorldState()).allowed


def test_rejection_has_no_world_side_effects():
    world = WorldState()
    snapshot = world.to_dict()
    gateway, _ = make_gateway()
    gateway.authorize("cmd_vel", {"v": 1.0}, world)
    gateway.authorize("navigate_to", {"target": {"x": -100, "y": 0}}, world)
    assert world.to_dict() == snapshot
    assert len(gateway.violations) == 2


def test_velocity_parameter_is_sanitized():
    gateway, _ = make_gateway()
    verdict = gateway.authorize("navigate_to", {"zone": "zone_a", "speed": 9.0}, WorldState())
    assert verdict.allowed
    assert verdict.sanitized["speed"] == pytest.approx(0.5)
    assert verdict.sanitized["__clamped__"] is True


def test_custom_polygon_geofence():
    polygon = ((1.0, 1.0), (5.0, 1.0), (5.0, 5.0), (1.0, 5.0))
    gateway, _ = make_gateway(geofence_polygon=polygon)
    assert gateway.authorize("navigate_to", {"target": {"x": 3.0, "y": 3.0}}, WorldState()).allowed
    assert not gateway.authorize("navigate_to", {"target": {"x": 7.0, "y": 3.0}}, WorldState()).allowed


def test_status_snapshot():
    gateway, _ = make_gateway()
    gateway.engage_estop("test")
    status = gateway.status()
    assert status["estop_engaged"] is True
    assert status["max_linear"] == 0.5
    assert len(status["geofence"]) == 4
