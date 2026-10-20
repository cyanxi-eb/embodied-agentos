"""ROS2 adapter import guard, ament skeleton presence, py_compile, renderer."""

from __future__ import annotations

import compileall
import io
from pathlib import Path

import pytest

from agentos.ros2_bridge import ROS2_AVAILABLE, ROS2_IMPORT_ERROR, Ros2Config, Ros2SkillBackend

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ros2_import_guard_is_explicit():
    # This environment has no ROS2: the adapter must import cleanly and say why.
    assert ROS2_AVAILABLE is False
    assert ROS2_IMPORT_ERROR
    with pytest.raises(RuntimeError, match="rclpy is not available"):
        Ros2SkillBackend()


def test_ros2_config_defaults_match_topic_contract():
    config = Ros2Config()
    assert config.nav_action == "navigate_to_pose"
    assert config.skill_action == "agentos/execute_skill"
    assert config.estop_service == "agentos/reset_estop"
    assert config.max_linear == 0.5


def test_ament_package_skeleton_is_complete():
    package = REPO_ROOT / "ros2_ws" / "src" / "agentos_bridge"
    expected = [
        package / "package.xml",
        package / "setup.py",
        package / "setup.cfg",
        package / "launch" / "agentos.launch.py",
        package / "config" / "nav2_params.yaml",
        package / "urdf" / "robot.urdf.xacro",
        package / "worlds" / "warehouse.world",
        package / "agentos_bridge" / "skill_action_server.py",
    ]
    for path in expected:
        assert path.exists(), f"missing {path}"
    assert "ament_python" in (package / "package.xml").read_text(encoding="utf-8")
    interfaces = REPO_ROOT / "ros2_ws" / "src" / "agentos_interfaces"
    assert (interfaces / "package.xml").exists()
    assert (interfaces / "CMakeLists.txt").exists()
    assert (interfaces / "action" / "ExecuteSkill.action").exists()
    docs = REPO_ROOT / "docs" / "ros2_integration.md"
    assert docs.exists()
    docs_text = docs.read_text(encoding="utf-8")
    assert "未运行" in docs_text
    assert "warehouse.world" in docs_text and "未在本机验证" in docs_text
    launch_text = (package / "launch" / "agentos.launch.py").read_text(encoding="utf-8")
    assert "warehouse.world" in launch_text
    # the Gazebo world is real SDF 1.9 XML with the same layout as the 2D simulator
    import xml.etree.ElementTree as ElementTree

    world = ElementTree.parse(package / "worlds" / "warehouse.world").getroot()
    assert world.tag == "sdf" and world.get("version") == "1.9"
    world_name = world.find("world")
    assert world_name is not None and world_name.get("name") == "warehouse"
    model_names = {model.get("name") for model in world_name.findall("model")}
    assert {"ground_plane", "shelf_row_1", "shelf_row_2", "shelf_row_3", "charging_dock"} <= model_names
    plugin_names = {plugin.get("filename") for plugin in world_name.findall("plugin")}
    assert "gz-sim-physics-system" in plugin_names
    # setup.py must install the worlds/ directory
    setup_text = (package / "setup.py").read_text(encoding="utf-8")
    assert "worlds/warehouse.world" in setup_text


def test_all_python_sources_compile():
    """py_compile every .py file, including the ROS2-only module (syntax must be valid)."""
    targets = [REPO_ROOT / "src", REPO_ROOT / "evaluation", REPO_ROOT / "scripts", REPO_ROOT / "tests"]
    for target in targets:
        assert compileall.compile_dir(str(target), quiet=2, force=True), f"compile failed under {target}"
    for path in [
        REPO_ROOT / "ros2_ws" / "src" / "agentos_bridge" / "agentos_bridge" / "skill_action_server.py",
        REPO_ROOT / "ros2_ws" / "src" / "agentos_bridge" / "launch" / "agentos.launch.py",
        REPO_ROOT / "ros2_ws" / "src" / "agentos_bridge" / "setup.py",
    ]:
        assert compileall.compile_file(str(path), quiet=2, force=True), f"compile failed: {path}"


def test_scenes_are_consistent_with_eval_instructions():
    from agentos.sim.scenes import SCENE_NAMES, build_scene
    from evaluation.instructions import INSTRUCTIONS

    assert len(INSTRUCTIONS) == 50
    # every scene must contain the zones/objects the instruction set talks about
    for scene_name in SCENE_NAMES:
        scene = build_scene(scene_name)
        for zone in ("zone_a", "zone_b", "zone_c", "table"):
            assert zone in scene.zones
        assert scene.dock
        assert len(scene.planning_grid().blocked) > 0


def test_renderer_produces_gif():
    from PIL import Image

    from agentos.sim.backend import SimSkillBackend
    from agentos.sim.render import SimRenderer, save_gif
    from agentos.sim.scenes import build_scene

    scene = build_scene("warehouse")
    backend = SimSkillBackend(scene, seed=3)
    renderer = SimRenderer(scene)
    frames = []
    try:
        for step in range(3):
            frames.append(
                renderer.frame(
                    backend.world,
                    hud={"skill": "navigate_to", "status": "running", "replans": 0},
                    banner="DEMO" if step == 1 else None,
                )
            )
        assert frames[0].size == (480, 360)
    finally:
        renderer.close()
    output = Path(io.__name__ and "/tmp") / "agentos-test"
    tmp_path = Path(__file__).parent / "_render_tmp.gif"
    try:
        info = save_gif(frames, tmp_path, fps=8)
        assert info["frames"] == 3
        assert info["megabytes"] < 3.5
        assert info["size"] == "480x360"
    finally:
        tmp_path.unlink(missing_ok=True)
