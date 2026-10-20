# ROS2 集成说明（Ubuntu 24.04 + ROS2 Jazzy + Gazebo Harmonic + Nav2）

> **说明**：本文档描述的 ROS2 适配层与 ament 包**未运行**过，只做了 `python -m py_compile` 语法验证。
> 仿真评测与 demo 全部来自纯 Python 的 2D 仿真后端，所以 README 里的指标**不代表** Gazebo 或真机链路的性能；
> 它验证的是 Agent 主循环、技能契约、安全网关、行为树与评测方法本身。仓库里没有任何「已真机验证」的说法。

## 1. 目录

```
ros2_ws/src/
├── agentos_interfaces/          # CMake 包：ExecuteSkill.action / ResetEstop.srv
└── agentos_bridge/              # ament_python 包
    ├── agentos_bridge/skill_action_server.py   # 技能 action server（Nav2 ActionClient + TTS + 急停复位服务）
    ├── agentos_bridge/fake_manipulator.py      # 无机械臂时的占位节点（仅 log）
    ├── launch/agentos.launch.py                # Gazebo + robot_state_publisher + spawn + Nav2 + bridge
    ├── config/nav2_params.yaml                 # AMCL/controller/planner/costmap/recovery 参数
    ├── urdf/robot.urdf.xacro                   # 差速底盘 + 2D 雷达 + RGB 相机 + Gazebo 插件
    └── worlds/warehouse.world                  # Gazebo world（SDF 1.9 XML，见 2.2）
```

`src/agentos/ros2_bridge/ros2_skill_backend.py` 是 **agent 侧** 的 `RobotBackend` 实现；
`skill_action_server.py` 是 **机器人侧** 的 action server，两者通过 `agentos_interfaces` 解耦。

## 2. 启动步骤

### 2.1 环境

```bash
sudo apt install ros-jazzy-desktop ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
                 ros-jazzy-gazebo-ros-pkgs ros-jazzy-ros-gz
source /opt/ros/jazzy/setup.bash
```

### 2.2 Gazebo world

仓库已包含 `ros2_ws/src/agentos_bridge/worlds/warehouse.world`（内容是 SDF 1.9 XML，只是用了 Gazebo 同样支持的
`.world` 扩展名）。布局与 2D 仿真 `agentos.sim.scenes.build_warehouse` 对齐：

* 12×12 m 地面、四周 0.15 m 厚、0.8 m 高墙体；
* 三排货架（y = 3.0 / 7.0 / 9.4，x ∈ [2.0, 7.0]，0.8 m 深、1.2 m 高）；
* 两组 crate（约 (9.6, 3.5) 与 (9.8, 8.6)）；
* 充电桩 (10.8, 10.8)；
* 光照（directional sun）与 gz-sim physics / user-commands / scene-broadcaster / sensors / contact 插件。

启动时默认使用包内 world，也可以显式覆盖：

```bash
ros2 launch agentos_bridge agentos.launch.py
ros2 launch agentos_bridge agentos.launch.py world:=/path/to/other.world
```

> **未在本机验证**：这个 world 只做了 XML 结构校验，没在 Gazebo Harmonic 里实际加载过；
> 加载行为和物理表现需要在 Ubuntu 24.04 + Gazebo Harmonic 环境里跑一遍才能确认。

### 2.3 构建与启动

```bash
mkdir -p ~/agentos_ws/src && cp -r ros2_ws/src/* ~/agentos_ws/src/
cd ~/agentos_ws
colcon build --symlink-install
source install/setup.bash

# 1) 启动 Gazebo + Nav2 + AgentOS bridge（另一个终端）
ros2 launch agentos_bridge agentos.launch.py   # 默认使用包内 worlds/warehouse.world

# 2) 启动 agent 侧（仿真后端换成 ROS2 后端）
export AGENTOS_BACKEND=ros2
python scripts/run_api.py --ros2            # 需要 rclpy 的 Python 解释器
```

> `AGENTOS_BACKEND=ros2` 的切换点计划在 `AgentRuntime` 的 context 工厂注入
> `Ros2SkillBackend`；当前仓库只提供适配器与包骨架，未做该分支的自动化验证。

## 3. 话题 / Action / Service 契约

| 方向 | 类型 | 名称 | 说明 |
| --- | --- | --- | --- |
| agent → robot | `nav2_msgs/action/NavigateToPose` | `navigate_to_pose` | navigate_to / approach_object / dock 底层导航 |
| agent → robot | `agentos_interfaces/action/ExecuteSkill` | `agentos/execute_skill` | 非导航技能：find/approach/pick/place/speak/stop/dock |
| agent → robot | `agentos_interfaces/srv/ResetEstop` | `agentos/reset_estop` | 操作员复位闩锁急停（需 operator_id + 复位码） |
| robot → agent | `geometry_msgs/msg/Twist` | `cmd_vel` | 仅 `stop` 技能发布 0 速度；agent 从不直接下发速度 |
| robot → agent | `sensor_msgs/msg/LaserScan` | `scan` | 导航/避障，Nav2 costmap obstacle layer |
| robot → agent | `sensor_msgs/msg/BatteryState` | `battery_state` | 安全网关低电量守卫的数据源 |
| robot → agent | `nav_msgs/msg/Odometry` | `odom` | 位姿反馈 |
| robot → agent | `sensor_msgs/msg/Image` | `camera/image_raw` | Qwen2.5-VL 感知后端输入 |
| robot → agent | `std_msgs/msg/String` | `agentos/tts` | speak 技能送 TTS |

### QoS

* Action / Service：默认 reliable、volatile、depth 10（`agentos/ros2_bridge/ros2_skill_backend.py` 中显式 `QoSProfile`）。
* `/scan`、`/camera/image_raw`：建议 sensor data QoS（best-effort、depth 5）。
* `/map`：transient_local（Nav2 惯例）。

### 取消语义（本项目重点）

`Ros2SkillBackend.cancel_active_goal()`：

1. 取消当前 goal handle（Nav2 或自定义技能 action）；
2. 清空 `_active_goal` / `_active_kind` 与 `_last_feedback`；
3. 行为树在取消时同步清理 py_trees blackboard 命名空间（`agentos/bt/executor.py` 的 `_cleanup`）；
4. 下一个技能不会继承任何旧 goal —— 这就是「技能取消时统一清理 Action goal 与黑板状态」那条设计。

## 4. 与 2D 仿真的对应关系

| 层 | 2D 仿真（仓库内可跑） | ROS2（需 Linux，未验证） |
| --- | --- | --- |
| 技能动作 | `SimSkillBackend.navigate_to/pick/...` | `Ros2SkillBackend` → ActionClient |
| 运动学 | 差速模型 + A* + pure pursuit + 碰撞检测 | Nav2 RegulatedPurePursuit + costmap |
| 安全守卫 | 同一份 `SafetyGateway` | 同一份 `SafetyGateway`（网关在进入行动层前） |
| 感知 | 几何射线可见性 + 可注入噪声的 sim-VLM | 深度相机/YOLO + Qwen2.5-VL |
| 记忆 | Qdrant local（hashing embedding） | 同一接口，可换真实 embedding |
| 评测 | 450 幕仿真 + 20 故障注入 | 未评测（本文档不提供任何 ROS2 指标） |

## 5. 已知未完成项

* `AgentRuntime` 尚未提供 `AGENTOS_BACKEND=ros2` 的正式工厂分支（需要 rclpy 环境才能联调）。
* 机械臂 pick/place 需要接入真实厂商 action server，bridge 目前明确返回 `BACKEND_ERROR` 而不是假装成功。
* `worlds/warehouse.world` 已随包提供，但 Gazebo 里的实际加载与物理表现未验证（见 2.2）。
