# 评测方法与结果

## 1. 一条命令复现

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py --out reports
```

输出：

* `reports/eval_YYYYMMDD_HHMM.json` —— **唯一数据源**（含逐幕明细、时延数组、失败案例）
* `reports/eval_YYYYMMDD_HHMM.md` —— 人类可读报告（由 JSON 渲染）
* `reports/eval_YYYYMMDD_HHMM.png` —— 分场景成功率 / 失败 bucket 图

评测结束自动打印汇总并给出路径；README 里的指标表和这份 JSON 保持一致。

快速模式（CI 用，约 20 秒）：

```powershell
.\.venv\Scripts\python.exe scripts\run_eval.py --quick --out reports
```

其它开关：`--episodes N`、`--scenes warehouse lab`、`--repeats N`、`--no-faults`、`--no-ablation`、`--persist-events`。

## 2. 评测设计

| 维度 | 设计 |
| --- | --- |
| 指令 | `evaluation/instructions.py` 50 条中文指令（取物 / 取物+播报 / 取物+回充 / 双物体 / 回充 / 播报 / 停止 / 巡检 / 找物播报） |
| 场景 | warehouse / lab / home（12×12m，同一物体目录、不同几何与搜索区） |
| 重复 | 每条 × 每场景 × 3 次；种子 = `crc32(instruction_id|scene|repeat)`，物体位置确定性抖动 |
| 初始条件 | 机器人固定初始位姿（场景定义），电量 100%，无故障（故障另设） |
| 故障注入 | 4 类 × 5 目标 = 20 幕，一次性故障（触发后下一次尝试恢复正常） |
| 消融 | verify+replan 关闭，干净场景 25 幕 + 故障 8 幕，独立 cache（避免记忆串扰） |
| 压力/泛化 | 30 条指令（覆盖全部意图）× 4 类扰动场景 × 1 次 = 120 幕，独立 cache；扰动：① 初始位姿平移≤1m/旋转±0.6rad ② 目标物体被挪走 ③ 15% 感知漏检 + 0.12m 噪声 + 路径阻断 ④ **改写/省略措辞**（物体与目的地换成"物件/那边"等模糊说法，检验规则 planner 的语言泛化） |
| 判定 | 运行时 `finished` + 图节点全部执行 + 后置条件全部通过 + **评测器独立检查世界状态** |
| 失败归因 | 运行时失败码（TIMEOUT/NOT_FOUND/BLOCKED/SLIP/ESTOP/…）+ 失败 bucket 统计 |

## 3. 指标定义

* **成功率** = 成功幕 / 总幕；同时给出「运行时自报成功率」与「运行时/评测器判定不一致」计数，后者应为 0。
* **重规划恢复率** = 发生过 ≥1 次 replan 的幕中最终成功的比例。
* **规划时延**：`plan` 节点 wall-clock（含解析、记忆检索、构图、图校验之外的部分）。
* **决策时延**：`dispatch_skill` 节点 wall-clock（含安全网关裁决）。
* **执行时延**：`execute_skill`（行为树 + 仿真步进）。
* **单幕仿真时长**：仿真的 sim-time（不是墙钟），反映机器人完成任务需要多久。
* **失败 bucket**：最终失败码计数；`EVALUATOR_MISMATCH` 表示运行时自报成功但评测器判定失败（应为 0）。
* **记忆命中率**：plan 节点检索到匹配物体记忆的幕比例；并对比命中/未命中时的成功率。

## 4. 为什么指标是这个量级

* Planner 是**确定性规则 planner**（离线），所以规划/决策时延是个位数毫秒；接入真实 LLM 后会由 API 时延主导。
* 2D 仿真步进是纯 Python、无渲染，单幕墙钟约 0.2–1s；全量评测含 450 幕 + 20 故障 + 33 幕消融，约 3 分钟。
* 成功率与重规划恢复率来自仿真，**不能**外推到 Gazebo 或真机；ROS2 适配层没跑过。

## 5. 最新一次全量结果

见最新 `reports/eval_*.md`（含分场景、分意图、故障注入、消融、失败案例）。README 指标表与该 JSON 完全一致。

## 6. 复现实验的注意事项

* **不要复用 `.cache/`**：评测默认自建独立缓存（`.cache/eval-tmp/`）；手动指定 `--cache-dir` 时确保是空目录，否则记忆会跨运行残留。
* 评测是**确定性**的（固定 seed、固定步进），同机重复运行结果应一致；不同 CPU 只会影响墙钟时延，不影响成功率。
* 若调整仿真/技能参数，请同时跑 `pytest` 与全量评测，并重新生成 README 指标。
