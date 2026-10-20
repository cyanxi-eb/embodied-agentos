"""Bring up Gazebo + robot + Nav2 + the AgentOS skill bridge.

Verified sequence is documented in docs/ros2_integration.md.  This launch file is
part of the (not-yet-run) ROS2 deliverable: no ROS2/Gazebo installation was
available to execute it.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    # Warehouse world shipped with the package (SDF 1.9 XML, .world extension).
    # Override with world:=/path/to/other.world.
    world = LaunchConfiguration(
        "world",
        default=PathJoinSubstitution([FindPackageShare("agentos_bridge"), "worlds", "warehouse.world"]),
    )
    nav2_params = PathJoinSubstitution(
        [FindPackageShare("agentos_bridge"), "config", "nav2_params.yaml"]
    )
    urdf = PathJoinSubstitution([FindPackageShare("agentos_bridge"), "urdf", "robot.urdf.xacro"])
    robot_description = Command(["xacro ", urdf])

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"])
                ),
                launch_arguments={"world": world, "verbose": "true"}.items(),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description, "use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                package="gazebo_ros",
                executable="spawn_entity.py",
                arguments=["-topic", "robot_description", "-entity", "agentos_robot", "-x", "1.0", "-y", "1.0"],
                output="screen",
            ),
            TimerAction(
                period=5.0,
                actions=[
                    IncludeLaunchDescription(
                        PythonLaunchDescriptionSource(
                            PathJoinSubstitution([FindPackageShare("nav2_bringup"), "launch", "navigation_launch.py"])
                        ),
                        launch_arguments={
                            "use_sim_time": use_sim_time,
                            "params_file": nav2_params,
                        }.items(),
                    ),
                ],
            ),
            Node(
                package="agentos_bridge",
                executable="skill_action_server",
                name="agentos_skill_action_server",
                parameters=[{"use_sim_time": use_sim_time, "default_nav_timeout_s": 45.0}],
                output="screen",
            ),
        ]
    )
