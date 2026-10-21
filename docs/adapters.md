# 异构本体适配层（轮式人形 / 复合移动 / 六轴协作臂）

> **状态说明（先看这一段）**：三个适配器和注册表都有测试覆盖，测试走的是
> `FakeRos2Transport`（记录每个 Action/Service 的 payload 并断言单位、坐标系、reverse
> 标志）。真实 `RclpyTransport` 的代码是完整的，但**没有在这台离线工作站上执行过**——
> 这里没有 ROS2 环境，也没有真机。仓库里没有任何“已真机验证”的说法。
>
> 本文对应的测试是 [`tests/test_adapters.py`](../tests/test_adapters.py)，README 里
> “换设备只换适配层，上层技能图不动”这句话由
> `test_same_task_graph_across_backends` 背书。

## 1. 这层解决什么问题

技能层（8 个 skill）只认 `RobotBackend` 接口，参数是 JSON Schema 契约，目标是世界坐标
（米）。不同本体在三个层面上并不一样：

1. **通信层**：到底用 Action 还是 Service、话题名是什么、QoS 怎么配；
2. **动作语义层**：全向底盘可以直接平移/原地转，非全向底盘只能前进+转向、倒车要显式标记，
   固定基座机械臂根本没有导航；
3. **单位/坐标层**：相对机体坐标（m + rad）、世界坐标（mm）、关节空间（rad）混在一起。

适配器的职责就是把这三点收在自己身上。任务图、planner、技能契约、行为树、安全网关、
评估器都只依赖 `RobotBackend`，**不 import 任何具体适配器**。

```text
自然语言 → 任务图（6 节点）→ 8 skills → 行为树/安全网关 → RobotBackend
                                                              ├── SimSkillBackend（2D 仿真）
                                                              ├── Ros2SkillBackend（通用 Nav2 本体）
                                                              ├── WheeledHumanoidBackend
                                                              ├── MobileManipulatorBackend
                                                              └── Arm6Backend
```

## 2. 三层差异表

| 差异层 | 轮式人形（X1 类） | 复合移动协作（AMR + 臂） | 六轴协作臂（固定基座） |
| --- | --- | --- | --- |
| **通信** | Action `/humanoid/nav_to`（导航）+ Action `/humanoid/manipulate`（抓放） | Action `/amr/navigate`（导航）+ **Service** `/arm/move`（抓放） | Action `/arm/follow_joint_trajectory`（关节轨迹）+ Action `/arm/gripper`（夹爪） |
| **动作语义** | 全向：可平移、可原地转向；`navigate(pose)` 接受相对机体位姿 | 非全向：只能前进/倒车+转向；目标在身后时必须显式 `reverse=True`，否则返回 `PRECONDITION_FAILED`；带 `align_first` 提示 | 无导航能力：`navigate_to` / `dock` / `retreat` / `approach_object` 直接返回 `UNSUPPORTED`，不发任何 ROS2 指令 |
| **单位/坐标** | 相对机体坐标，**米 + 弧度**（`dx_m` / `dy_m` / `dyaw_rad`） | 世界坐标，**毫米**（`x_mm` / `y_mm`）；世界↔毫米换算在适配器边界完成 | 关节空间，**弧度**（6 个 joint）；pick/place 的笛卡尔目标用**基座坐标（米）**并做可达性检查 |
| **world 镜像** | 导航成功后更新 `(x, y, yaw)` | 优先用机器人回传的 `reached_pose_mm` 反算，没有则用指令目标 | 基座固定，不更新位姿；只更新 holding / last_placed |

三者的实际 wire payload（测试里断言的形状）：

```jsonc
// wheeled_humanoid -> /humanoid/nav_to (relative body frame)
{"frame_id": "base_link", "dx_m": 2.0, "dy_m": -1.0, "dyaw_rad": 0.5,
 "rotate_in_place": false, "holonomic": true}

// mobile_manipulator -> /amr/navigate (world frame, millimetres)
{"frame_id": "map", "x_mm": 2500.0, "y_mm": 3500.0, "yaw_rad": 0.25,
 "reverse": false, "align_first": true, "holonomic": false}

// arm6 -> /arm/follow_joint_trajectory (joint space, radians)
{"joint_names": ["shoulder_pan_joint", "..."],
 "points": [{"positions": [0.0, -1.15, 0.65, -1.10, 0.0, 0.0],
             "velocities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             "time_from_start_s": 2.0}],
 "units": "rad", "joint_space": true}
```

## 3. 代码结构

```text
src/agentos/backends/
├── base.py                 # 既有：RobotBackend 抽象接口
├── ros2_transport.py       # 新增：Ros2Transport 协议 + FakeRos2Transport + RclpyTransport（import 守卫）
├── ros2_adapter.py         # 新增：三个本体共享的适配器基类（world 镜像 / 取消 / dock / retreat / TTS）
├── wheeled_humanoid.py     # 新增：全向、相对机体坐标
├── mobile_manipulator.py   # 新增：非全向、世界 mm、reverse 标志、Action + Service
├── arm6.py                 # 新增：固定基座、关节弧度、无导航 → UNSUPPORTED
└── registry.py             # 新增：名字 → 适配器工厂
```

关键点：

* **`RobotBackend` 没有被修改**。三个类都完整实现了抽象方法，`tests/test_adapters.py`
  里直接 `isinstance(backend, RobotBackend)` 并实例化，缺一个抽象方法就构造失败。
* **导入安全**：`ros2_transport` 顶部对 `rclpy` 做 `try/except` 守卫，
  `ROS2_AVAILABLE=False` 时三个适配器照常 import、照常配 `FakeRos2Transport` 跑测试。
* **真实传输只在需要时构造**：`make_backend("arm6")` 不传 `transport` 才会去建
  `RclpyTransport`，没有 ROS2 时它抛一条可读的 `RuntimeError`（不是 ImportError）。
* **取消语义**：`_dispatch_action` 在调用期间把 `{"interface", "kind", "goal"}` 挂在
  `_active_goal` 上；`cancel_active_goal()` 调 `transport.cancel(interface)` 并清空句柄。
  测试 `test_backend_cancel_clears_goal` 在 action 执行中途触发取消，断言 handle 清空、
  只发了一次 cancel。
* **未知设备名**：`make_backend("quadruped_x")` 抛
  `UnknownBackendError`，消息里列出当前注册的三个名字；注册新本体用
  `register_backend(name, factory)`。

## 4. 测试证明了什么

| 测试 | 断言 |
| --- | --- |
| `test_same_task_graph_across_backends` | 同一句指令经同一个 `RuleBasedPlanner` 生成的 `TaskPlan.to_dict()` 在三个后端上**完全相等**：6 个节点、5 条依赖边、节点 id 与技能序列一致；三个后端都真的跑了 LangGraph 节点（plan → validate → dispatch → 行为树执行 → 后置条件校验），执行后任务图仍是 6/5。轮式人形与复合移动执行 `finished`，六轴臂在第一个 `navigate_to` 返回 `UNSUPPORTED`。 |
| `test_unsupported_skill_degrades` | 六轴臂的 `navigate_to` / `dock` / `retreat` 返回 `SkillResult(ok=False, code=UNSUPPORTED)`，detail 可读，`transport.calls == []`（没发指令）；任务图执行到该节点时 `failed`，trace 里有 `skill_result.code=UNSUPPORTED`、`skill_failed` 和 `finish(status=failed)`，没有异常。 |
| `test_mobile_manipulator_millimetre_boundary` | 世界米 → `/amr/navigate` 的 `x_mm` / `y_mm` 正确（2.5 m → 2500.0 mm）；机器人回传 `reached_pose_mm` 后 world 镜像变回米；抓取 Service 的 `x_mm` 用当前机体坐标系（0.3 m → 300.0 mm）。 |
| `test_mobile_manipulator_backward_requires_reverse` | 目标在身后且 `allow_reverse=False` 时返回 `PRECONDITION_FAILED` 且不发 Action；允许倒车时 goal 带 `reverse=True`，`x_mm` 仍是世界毫米。 |
| `test_wheeled_humanoid_relative_body_goal` | 朝向 +y 时世界 +y 变成机体 `dx_m=2.0`；世界 +x 变成 `dy_m=-1.0`（横移）；原地转向时 `dx=dy=0, dyaw=0.5, rotate_in_place=true`。 |
| `test_arm6_base_frame_and_joint_radians` | `move_joints` 发 6 个弧度 + 正确 joint_names；关节数不对 / NaN 直接失败且不发指令；pick 返回基座坐标 `(0.9, 0.6)`；超出 `reach_m` 返回 `OUT_OF_BOUNDS` 且不发指令。 |
| `test_registry_and_unknown_name` | `available_backends()` 固定为三个名字；`make_backend` 返回正确类；未知名字 / 空名字抛 `UnknownBackendError`，消息列出可用名字；重复注册抛 `ValueError`。 |
| `test_backend_cancel_clears_goal` | in-flight 取消后 `active_goal` 为空、`handle_cleared=True`、只发一次 `cancel`。 |

## 5. 新增一个品牌要改什么

只写一个适配器，映射 **Action 签名 → 技能 schema**，上层不动：

1. 在 `src/agentos/backends/` 新建 `my_brand.py`，继承 `Ros2AdapterBackend`：

   ```python
   from agentos.backends.ros2_adapter import Ros2AdapterBackend, adapter_success
   from agentos.backends.ros2_transport import InterfaceSpec
   from agentos.types import SkillResult

   class MyBrandBackend(Ros2AdapterBackend):
       device_name = "my_brand_amr"
       display_name = "MyBrand AMR"
       nav_action = "/my_brand/nav"
       interfaces = (InterfaceSpec("action", nav_action),)

       def _move_base(self, x, y, yaw, timeout_s, *, allow_reverse=False) -> SkillResult:
           # 1) 技能层给的是世界坐标（米）；这里换成厂商 Action 的单位/坐标系
           goal = {"x_mm": round(x * 1000, 3), "y_mm": round(y * 1000, 3)}
           # 2) 调传输层；FakeRos2Transport 在测试里记录 goal
           result = self._dispatch_action("navigate_to", self.nav_action, goal, timeout_s)
           # 3) 成功后才更新 world 镜像（后置条件靠它判定）
           if result.ok:
               self._commit_pose(x, y, yaw)
           return result
   ```

2. `registry.register_backend("my_brand_amr", MyBrandBackend)`（或在
   `registry.py` 的 `_DEFAULT_FACTORIES` 里加一行）。
3. 在 `tests/test_adapters.py` 里写一条 `_move_base` 的单位/坐标系断言。
4. 任务图、planner、8 个技能、行为树、安全网关、评测器**都不改**；MCP 工具清单和
   LLM planner 的 prompt 都从同一个 `SkillRegistry.describe()` 生成，也不因本体变化。
   如果新本体缺某个技能，让对应方法返回 `UNSUPPORTED`（参考 `arm6.py`），而不是假装成功。

## 6. 失败码映射

| 场景 | 失败码 | 来源 |
| --- | --- | --- |
| 本体没有该能力（固定臂的导航） | `UNSUPPORTED` | 适配器，新增的字符串枚举值；序列化仍是 `code.value` |
| 非全向底盘的后退目标未显式允许 | `PRECONDITION_FAILED` | `mobile_manipulator._move_base` |
| 目标超出机械臂可达范围 | `OUT_OF_BOUNDS` | `arm6._reach_check` |
| Action server 不在 / 被拒绝 / 超时 / 取消 | `BACKEND_ERROR` / `BLOCKED` / `TIMEOUT` / `CANCELLED` | `RclpyTransport` / `FakeRos2Transport` |
| 传输层自己抛异常 | `BACKEND_ERROR` | `_dispatch_action` / `_dispatch_service` 兜底，不让 agent loop 崩 |

`UNSUPPORTED` 被加进 `FATAL_CODES`（重试/重规划都救不了，应该结束任务或换本体），
没有加进 `RECOVERABLE_CODES`，所以 `verify_postcondition` 不会为它安排 replan。

## 7. 已知限制（不要过度解读）

* **真实 ROS2 传输未执行**：`RclpyTransport` 只做了 `py_compile` 语法验证；没有在
  Ubuntu + ROS2 Jazzy 上跑过，也没有真机。测试全部使用 `FakeRos2Transport`。
* **厂商 Action/Service 类型不在仓库里**：轮式人形的 `/humanoid/nav_to`、
  复合移动的 `/amr/navigate`、`/arm/move` 的消息类型属于机器人 workspace，
  `InterfaceSpec(type_path="")` 表示“由部署方提供”。三个适配器只约定 payload 形状；
  仓库内只给 `control_msgs` 的关节轨迹/夹爪标准类型。
* **没有 IK / 力控 / 视觉伺服**：六轴臂的 pick/place 发送配置好的关节路点，并检查基座
  坐标系下的可达距离；真实部署由厂商 action server / MoveIt2 做 IK。复合移动的
  `/arm/move` Service 同样只发目标位姿，机械臂侧怎么规划不在本仓库范围。
* **world 镜像不是真值**：适配器用指令目标（或机器人回传的 `reached_pose_mm`）更新
  本地 `WorldState`，用于技能后置条件。真实部署应从 `/odom` + TF 重新灌入；重启后的
  `restore()` 只恢复本地镜像。
* **没有真机时序**：`FakeRos2Transport` 是同步的，不模拟 action feedback、QoS 丢包、
  网络抖动和并发取消的真实时序；取消测试是“在 action 调用内部触发取消”的确定性版本。
* 这些适配器的价值是把**异构差异收敛在一层**并用测试固定住，不是证明某台具体机器人
  已经跑通。

## 8. 复现

```powershell
# 只跑适配器测试
.\.venv\Scripts\python.exe -m pytest tests\test_adapters.py -q

# 全量测试（含仿真、安全、技能、行为树、LangGraph、API、MCP）
.\.venv\Scripts\python.exe -m pytest -q

# 不接 ROS2 的 dry-run 示例
.\.venv\Scripts\python.exe -c "from agentos.backends import make_backend, FakeRos2Transport; b = make_backend('mobile_manipulator', transport=FakeRos2Transport()); print(b.navigate_to(2.5, 3.5).to_dict()); print(b.transport.last_call('/amr/navigate'))"
```
