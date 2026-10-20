# 真实踩坑记录（开发过程中实际遇到并修复）

> 下面这些都是开发时真撞上、导致测试挂掉或指标不对的问题，不是编出来的「常见坑」。
> 每条按 现象 → 根因 → 修复 → 验证 来记。

## 1. pure pursuit 让机器人绕目标点公转，永远到不了

* **现象**：导航仿真里机器人到达目标附近后开始原地打转，30s 超时；`sim_time` 很大但位置不变。
* **根因**：lookahead 点在路径长线段上做投影时，`need = lookahead - 到段起点距离` 可能为 0，投影结果落在机器人自身位置，
  `atan2(0,0)` 导致朝向角随机，控制器不断自旋。
* **修复**：改为「沿路径顶点前进 + 当 carrot 偏航过大时自适应缩短 lookahead」；路径简化限制最大段长（1.6m），避免长线段切角。
* **验证**：`tests/test_sim.py::test_navigation_succeeds_in_all_scenes` + 全量评测导航零超时。

## 2. 货架拐角反复碰撞，导航"看起来能走"却一直卡住

* **现象**：仓库场景从起点到 A 区，机器人在货架拐角处 100 步内碰撞 20 次，导航 46s 超时；换 seed 不一定复现。
* **根因**：纯跟踪在拐角切角，进入 `collision inflation` 的障碍格；碰撞恢复只做原地旋转，位置没退，下一轮继续撞。
* **修复**：碰撞恢复改为「先倒车 0.16–0.26m，再转向 carrot」；规划膨胀半径 0.35→0.42；自适应 lookahead 降低切角。
* **验证**：拐角场景碰撞数 20→0，导航 46s→19.3s（`tests/test_sim.py`）。

## 3. 重规划后任务图节点翻倍执行

* **现象**：故障恢复时 replan 计划变成 10 个节点（原 6 个），`pick/place` 出现两次，仿真时长翻倍。
* **根因**：replan 把「失败节点之后的剩余节点」拼接到「重新构造的取物链」后面，而后者本身已包含 approach/pick/place。
* **修复**：改为**原地 patch**：保留已执行前缀 `previous.nodes[:failed_index]`，只替换失败节点区域（如"换搜索区域"= nav+find），
  再拼剩余节点；`TaskPlan.resume_index` 标记从哪个节点继续，图 layer 不再重跑前缀。
* **验证**：`tests/test_task_graph_planner.py::test_replan_not_found_patches_plan_in_place` 断言节点数=7、pick 仅 1 次；
  评测中重规划幕成功率 100%。

## 4. 陈旧记忆被复用成错误计划（最危险的一类）

* **现象**：同一评测进程里，L04「黄色包裹」的 plan 导航到别的位置（红色盒子记忆位置），find 失败后才重规划。
* **根因**：`last_object_location` 只做向量检索，没有校验 payload 里的物体身份；hashing embedding 对 "object yellow package"
  与 "object red box" 有一定相似度，返回了错误记忆。
* **修复**：记忆检索必须**精确匹配颜色+类别**且分数 ≥ 0.5 才复用；规划只把记忆当提示，失败后正常重规划。
* **验证**：`tests/test_memory.py::test_last_object_location_requires_exact_object_match`、
  `tests/test_task_graph_planner.py::test_memory_does_not_match_a_different_object`。

## 5. `find_report` 全部被评测器判失败（假失败）

* **现象**：全量评测 18 幕 `EVALUATOR_MISMATCH`，指令是「找到…后播报…」；运行时 `finished`、goal_verified=True，
  但评测器 detail 显示 `detection=None`、`spoken` 正确。
* **根因**：planner 对 `find_report` 复用了取物步骤（nav→find→**approach→pick**→speak）；`pick` 成功后清空 `last_detection`，
  评测器要求「找到的目标仍在 last_detection」。
* **修复**：`_steps_for_object` 增加 `include_manipulation` 参数，find_report 只生成 nav→find→speak。
* **验证**：重跑全量后该 bucket 归零；`tests/test_eval_smoke.py` 与 450 幕报告。

## 6. 暂停后取消被当成"技能失败"并继续重规划

* **现象**：API 取消测试中任务最终 `finished` 而不是 `cancelled`；trace 里出现 `POSTCONDITION_FAILED` 且 `bt` 为空。
* **根因**：`execute_skill` 在取消时返回 `status=cancelled`（无 bt 结果），但图的固定边仍进入 `verify_postcondition`，
  该节点把空 bt 解释为后置条件失败 → 触发 replan → 继续执行。
* **修复**：`execute_skill` 后加条件边（取消→finalize）；verify 增加防御：空 bt 直接 `BACKEND_ERROR` 而非假失败；状态 cancelled 直接收尾。
* **验证**：`tests/test_api.py::test_cancel_task`、`tests/test_bt.py::test_cancel_short_circuits_and_cleans_up`。

## 7. 未知指令导致 `KeyError: 'plan'`

* **现象**：输入无法解析的指令（"打开空调"）时，图抛 KeyError。
* **根因**：`plan` 节点规划失败时只设置 `status=failed / next_action=finish`，但图里 `plan → validate_plan` 是固定边。
* **修复**：`plan` 后改为条件边，失败直奔 finalize。
* **验证**：`tests/test_graph_runtime.py::test_unknown_instruction_fails_cleanly`。

## 8. 450 幕评测首版要跑 20 分钟以上

* **现象**：首版单幕 4.2s，450 幕 + 故障 + 消融预计 >20 分钟。
* **根因**：① 每条 trace 事件都 SQLite 提交，一次 fsync 就慢一次；② LangGraph 每个节点写 checkpoint 走同一条链路；③ 路径简化对每个候选点做 clearance 栅格扫描。
* **修复**：WAL + `synchronous=NORMAL`；评测关闭 trace 落库（`persist_events=False`，仍保留内存事件）；去掉冗余 clearance 扫描；A* 内部局部变量化；栅格 0.1→0.15m。
* **结果**：单幕均值降到 ~0.4s，全量评测 **~3 分钟**（远低于 20 分钟门槛），报告里记录了实测墙钟。

## 9. 评测临时目录把仓库根目录污染了 151MB

* **现象**：仓库根出现 `pytest-of-*/`、大量 `tmp*/`、`agentos-eval-*/`。
* **根因**：环境把 `TEMP` 指向工作区时，`tempfile.mkdtemp()` 与 pytest basetemp 都会落到仓库根。
* **修复**：pytest `--basetemp=.cache/pytest`；评测临时目录固定到 `.cache/eval-tmp/`；`.gitignore` 兜底 `tmp*/`、`pytest-of-*/`、`agentos-eval-*/`。
* **验证**：清理后仓库根只剩交付目录；`git status` 干净。

## 10. 行为树 `Retry` 装饰器与 py_trees 2.6 的 tick 生成器语义

* **现象**：自定义重试子树在 `Behaviour.__init__` 阶段就报 `AttributeError`（child 未赋值）；`decorators.Retry` 一次 tick 的行为与预期不符。
* **根因**：py_trees 2.6 的 `Behaviour.__init__` 会立即创建 `self.tick()` 生成器；重写 `tick()` 会在 `__init__` 里提前执行 `update()`。
  `decorators.Retry` 在单个 tick 中返回 RUNNING 等待下一次 tick，与「一次 tick 跑完整棵树」的设计冲突。
* **修复**：自定义 `_SyncRetry`，在 `update()` 里用 `child.tick_once()` 同步重试；所有 behaviour 先赋值再 `super().__init__`。
* **验证**：`tests/test_bt.py` 覆盖 retry/backtrack/replan/cancel 四类分支。
