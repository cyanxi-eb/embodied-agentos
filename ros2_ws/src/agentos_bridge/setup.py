from setuptools import find_packages, setup

package_name = "agentos_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/agentos.launch.py"]),
        ("share/" + package_name + "/config", ["config/nav2_params.yaml"]),
        ("share/" + package_name + "/urdf", ["urdf/robot.urdf.xacro"]),
        ("share/" + package_name + "/worlds", ["worlds/warehouse.world"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="cyanxi-eb",
    maintainer_email="cyanxi-eb@users.noreply.github.com",
    description="Embodied AgentOS skill action server bridging skills to Nav2 / hardware",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "skill_action_server = agentos_bridge.skill_action_server:main",
            "fake_manipulator = agentos_bridge.fake_manipulator:main",
        ],
    },
)
