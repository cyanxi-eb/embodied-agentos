# 真实 LLM 规划器（可选）

默认评测用 `RuleBasedPlanner`：确定性、可复现、适合 CI。`--planner llm` 会把主循环换成 `LLMPlanner`——模型只负责输出任务图 JSON，技能执行、后置条件校验、失败重规划仍然由运行时完成。

这份文档记录一次小样本实测：20 条指令（fetch / multi_fetch / fetch_dock / fetch_speak），warehouse 场景，1 次重复，关闭故障注入 / 消融 / 压力子集。目的是验证「模型 → 任务图 → 运行时」这条链路真的能闭环，而不是拿它当性能结论。

## 配置

| 环境变量 | 说明 | 默认 |
| --- | --- | --- |
| `AGENTOS_PLANNER_API_KEY` | Bearer token（必需；不要提交到仓库） | — |
| `AGENTOS_PLANNER_BASE_URL` | OpenAI 兼容 endpoint | 有 key 时用 DashScope compatible mode |
| `AGENTOS_PLANNER_MODEL` | 模型名 | 有 key 时用 `qwen3.8-flash` |
| `AGENTOS_PLANNER_THINKING` | `1` 开推理 token / `0` 关 | `1` |
| `AGENTOS_PLANNER_TIMEOUT_S` | 单次请求超时 | `30` |
| `AGENTOS_PLANNER_CACHE_DIR` | 响应缓存目录 | `.cache/llm-plans` |
| `AGENTOS_PLANNER_CACHE` | `0` 关闭缓存 | `1` |

换任何 OpenAI 兼容服务（vLLM、DashScope、自建网关）只要给三个核心变量即可。模型输出没通过技能 schema/依赖顺序校验时会回退到规则 planner，回退次数记在报告的 `planner_stats.fallbacks` 里；两次实验回退都是 0，说明任务图 JSON 稳定可解析。

## 复现

```bash
export AGENTOS_PLANNER_API_KEY=...                # 不要写进仓库
python scripts/run_eval.py --planner llm --episodes 20 --scenes warehouse --repeats 1 \
  --no-faults --no-ablation --no-stress --out reports/live_llm
# 关闭 thinking 再跑一组
AGENTOS_PLANNER_THINKING=0 AGENTOS_PLANNER_CACHE_DIR=.cache/llm-plans-nothink \
python scripts/run_eval.py --planner llm --episodes 20 --scenes warehouse --repeats 1 \
  --no-faults --no-ablation --no-stress --out reports/live_llm_nothink
```

## 结果

| 模式 | 成功率 | 规划调用 / 缓存命中 / 回退 | 规划时延 p50 / p95 | token（prompt + completion） | 墙钟 |
| --- | --- | --- | --- | --- | --- |
| thinking 开（`reports/live_llm/`） | **20/20（100%）** | 21 / 2 / 0 | 22.9 s / 42.5 s | 27,722 + 22,527 | 504 s |
| thinking 关（`reports/live_llm_nothink/`） | **17/20（85%）** | 29 / 2 / 0 | 7.1 s / 13.0 s | 38,435 + 13,426 | 236 s |

- 成功判定用的是评测器对世界状态的独立检查（`check_goal`），不是模型自报；两次实验「运行时/评测器不一致」都是 0。
- thinking 开的差距主要在延迟：推理 token 占了 completion 的大头（22.5k vs 13.4k），单次计划 20–40 s；关掉后 p95 降到 13 s，但重规划次数变多（21 → 29 次调用）。
- thinking 关时失败 3 条：L13、L15 在 `NOT_FOUND` 后连续重规划 3 次仍未找到目标，L17 是 `PRECONDITION_FAILED`。根因是当前 `LLMPlanner` 的 prompt **没有把 replan 的失败码 / 已搜索区域注入**——重规划时模型看到的上下文和首轮几乎一样，容易重复同一份计划。这是下一步要补的第一件事（见下）。

## 限制与下一步

- 20 条 × 单场景 × 单次，不构成统计结论；真实场景还要跑故障注入和 3 场景。
- LLM 只做任务图生成，不做感知；`find_object` 仍走仿真感知 / 可选 VLM 适配器。
- 没接真实 VLM：`perception/openai_vlm.py` 的接口已经留好，但没有用图像输入验证过。
- 下一步：
  1. 把 `replan_feedback`、`searched_zones`、失败节点注入 prompt，让重规划真的换策略；
  2. thinking 模式做自适应（首轮开、重规划关）压延迟；
  3. 用 `--planner llm` 跑 20 次故障注入，和规则 planner 的 20/20 对比；
  4. 接 Qwen-VL 系模型替换 `SimVLM`，把 `find_object` 换成真实视觉 grounding。

原始数据在 [`reports/live_llm/`](../reports/live_llm/) 和 [`reports/live_llm_nothink/`](../reports/live_llm_nothink/)（JSON 是唯一数据源，Markdown 为可读版）。
