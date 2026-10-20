"""Simulator geometry, A*, pure pursuit and deterministic world state."""

from __future__ import annotations

import math

import pytest

from agentos.sim.backend import SimSkillBackend
from agentos.sim.controller import PurePursuit, PurePursuitConfig
from agentos.sim.faults import FaultSpec
from agentos.sim.planner import astar, path_length, simplify_path
from agentos.sim.robot import DiffDriveRobot
from agentos.sim.scenes import SCENE_NAMES, build_scene
from agentos.sim.world import OccupancyGrid, Rect, point_in_polygon
from agentos.types import Pose


def test_occupancy_grid_marks_walls_and_obstacles():
    grid = OccupancyGrid(12.0, 12.0, 0.15, [Rect(2.0, 2.6, 7.0, 3.4)], inflation=0.3)
    assert not grid.is_free(0.0, 0.0)  # wall
    assert not grid.is_free(3.0, 3.0)  # inflated shelf
    assert grid.is_free(1.0, 1.0)
    assert grid.free_ratio() > 0.7


def test_point_in_polygon():
    square = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert point_in_polygon(0.5, 0.5, square)
    assert not point_in_polygon(1.5, 0.5, square)


@pytest.mark.parametrize("scene_name", SCENE_NAMES)
def test_astar_finds_collision_free_path(scene_name):
    scene = build_scene(scene_name)
    grid = scene.planning_grid()
    path = astar(grid, scene.start, scene.zones["table"])
    assert path is not None
    assert path_length(path) > 3.0
    for x, y in path:
        assert grid.is_free(x, y)
    assert [path[0], path[-1]]


def test_astar_no_path_when_walled_off():
    grid = OccupancyGrid(12.0, 12.0, 0.15, [Rect(4.0, 0.0, 5.0, 12.0)], inflation=0.4)
    path = astar(grid, (1.0, 1.0), (9.0, 9.0))
    assert path is None


def test_simplify_path_preserves_endpoints_and_caps_segments():
    scene = build_scene("warehouse")
    grid = scene.planning_grid()
    path = astar(grid, scene.start, scene.zones["zone_a"])
    simplified = simplify_path(grid, path, max_segment_m=1.6)
    assert simplified[0] == path[0]
    assert simplified[-1] == path[-1]
    assert len(simplified) <= len(path)
    for a, b in zip(simplified, simplified[1:]):
        assert math.hypot(b[0] - a[0], b[1] - a[1]) <= 1.6 + 1e-6


def test_diff_drive_clamps_and_integrates():
    robot = DiffDriveRobot(pose=Pose(0.0, 0.0, 0.0), max_linear=0.5, max_angular=1.0)
    pose = robot.step(5.0, 5.0, 1.0)
    assert pose.x == pytest.approx(0.5)
    assert pose.yaw == pytest.approx(1.0)
    pose = robot.step(0.0, 0.0, 1.0)
    assert pose.x == pytest.approx(0.5)


def test_pure_pursuit_reaches_straight_line():
    scene = build_scene("warehouse")
    backend = SimSkillBackend(scene, seed=1)
    path = [(1.0, 1.0), (1.0, 6.0)]
    tracker = PurePursuit(PurePursuitConfig(goal_tolerance_m=0.25))
    reached = False
    for _ in range(400):
        v, w, reached = tracker.command(backend.world.robot, path)
        if reached:
            break
        assert backend._step(v, w, 0.1)
    assert reached
    assert backend.world.robot.y > 5.5


def test_navigation_succeeds_in_all_scenes():
    for scene_name in SCENE_NAMES:
        scene = build_scene(scene_name)
        backend = SimSkillBackend(scene, seed=2)
        result = backend.navigate_to(*scene.zones["zone_a"], timeout_s=45.0)
        assert result.ok, f"{scene_name}: {result.code} {result.detail}"
        distance = math.hypot(
            backend.world.robot.x - scene.zones["zone_a"][0],
            backend.world.robot.y - scene.zones["zone_a"][1],
        )
        assert distance <= 0.5


def test_objects_are_placed_in_free_space():
    for scene_name in SCENE_NAMES:
        scene = build_scene(scene_name)
        for seed in range(3):
            backend = SimSkillBackend(scene, seed=seed)
            assert len(backend.world.objects) == 8
            for obj in backend.world.objects.values():
                assert scene.planning_grid().is_free(obj.x, obj.y)


def test_object_missing_fault_relocates_object():
    scene = build_scene("warehouse")
    original = SimSkillBackend(scene, seed=4)
    target = original.world.objects["obj_red_box_0"]
    before = (target.x, target.y)
    backend = SimSkillBackend(scene, seed=4, faults=[FaultSpec(kind="object_missing", target="red:box")])
    relocated = backend.world.objects["obj_red_box_0"]
    assert (relocated.x, relocated.y) != before
    assert backend.fired_faults and backend.fired_faults[0]["kind"] == "object_missing"


def test_pick_slip_fault_is_consumed_once():
    scene = build_scene("warehouse")
    backend = SimSkillBackend(scene, seed=3, faults=[FaultSpec(kind="pick_slip", target="red:box")])
    backend.robot_pose_override = None
    backend.navigate_to(*scene.zones["zone_a"], timeout_s=45.0)
    found = backend.perception.locate({"color": "red", "category": "box"}, backend.world)
    assert found, "red box should be visible at zone_a"
    object_id = found[0].object_id
    backend.world.mark_detection(found[0].to_dict())
    obj = backend.world.objects[object_id]
    backend._robot.pose = Pose(obj.x - 0.6, obj.y, 0.0)
    first = backend.pick(object_id)
    assert not first.ok
    assert first.code.value == "SLIP"
    second = backend.pick(object_id)
    assert second.ok
    assert backend.world.holding == object_id


def test_snapshot_restore_roundtrip():
    scene = build_scene("warehouse")
    backend = SimSkillBackend(scene, seed=5)
    backend.navigate_to(*scene.zones["zone_b"], timeout_s=45.0)
    snapshot = backend.snapshot()
    restored = SimSkillBackend.from_snapshot(scene, snapshot)
    assert restored.world.robot.x == pytest.approx(backend.world.robot.x, abs=1e-3)
    assert restored.world.robot.y == pytest.approx(backend.world.robot.y, abs=1e-3)
    assert restored.world.sim_time == pytest.approx(backend.world.sim_time, abs=1e-3)
    assert set(restored.world.objects) == set(backend.world.objects)


def test_cancel_active_goal_marks_navigation_cancelled():
    scene = build_scene("warehouse")
    backend = SimSkillBackend(scene, seed=6)
    backend.cancel_active_goal()
    result = backend.navigate_to(*scene.zones["zone_a"], timeout_s=45.0)
    assert not result.ok
    assert result.code.value == "CANCELLED"
    assert backend.active_goal is None
    backend.clear_cancel()
    result = backend.navigate_to(*scene.zones["zone_a"], timeout_s=45.0)
    assert result.ok


def test_battery_drains_and_dock_recharges():
    scene = build_scene("warehouse")
    backend = SimSkillBackend(scene, seed=7)
    backend.navigate_to(*scene.zones["zone_a"], timeout_s=45.0)
    drained = backend.world.battery
    assert drained < 100.0
    backend._robot.pose = Pose(scene.dock[0], scene.dock[1], 0.0)
    result = backend.dock()
    assert result.ok
    assert backend.world.docked
    assert backend.world.battery == pytest.approx(100.0)
