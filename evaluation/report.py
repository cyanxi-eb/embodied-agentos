"""Report writer: JSON + Markdown + a small matplotlib chart.

``reports/eval_YYYYMMDD.json`` is the single source of truth; the Markdown report
and the README metric table are generated from it so the numbers can never drift.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _fmt(value: Any, suffix: str = "") -> str:
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_chart(report: dict[str, Any], path: Path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        by_scene = report.get("by_scene", {})
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), dpi=120)
        scenes = list(by_scene)
        rates = [by_scene[scene]["success_rate"] for scene in scenes]
        axes[0].bar(scenes, rates, color=["#4a7fd4", "#3fae6a", "#e3b23c"][: len(scenes)])
        axes[0].set_ylim(0, 105)
        axes[0].set_ylabel("success rate (%)")
        axes[0].set_title(f"Success by scene (n={report['overall']['episodes']})")
        for index, rate in enumerate(rates):
            axes[0].text(index, rate + 2, f"{rate:.1f}%", ha="center", fontsize=8)
        buckets = report["overall"].get("failure_buckets", {})
        if buckets:
            names = list(buckets)
            values = [buckets[name] for name in names]
            axes[1].bar(names, values, color="#c0392b")
            axes[1].set_title("Failure buckets")
            axes[1].tick_params(axis="x", rotation=20, labelsize=7)
        else:
            axes[1].text(0.5, 0.5, "no failures", ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_title("Failure buckets")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        return str(path)
    except Exception as error:  # pragma: no cover - chart is optional
        print(f"chart generation failed: {error}")
        return None


def write_report(report: dict[str, Any], output_dir: str | Path = "reports") -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = output / f"eval_{stamp}.json"
    md_path = output / f"eval_{stamp}.md"
    chart_path = write_chart(report, output / f"eval_{stamp}.png")
    report["report_paths"] = {"json": str(json_path), "markdown": str(md_path), "chart": chart_path}
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "chart": chart_path or ""}


def render_markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    config = report["config"]
    lines: list[str] = []
    lines.append("# Embodied AgentOS 评测报告\n")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- 主机：{report['host']['platform']} / Python {report['host']['python']} / {report['host']['cpu_count']} 核")
    lines.append(
        f"- 规模：{config['instructions']} 条指令 × {len(config['scenes'])} 场景 × {config['repeats']} 次重复 = "
        f"{config['episodes']} 幕；故障注入 {report.get('fault_injection', {}).get('cases', 0)} 幕；"
        f"总墙钟 {report.get('wall_time_s', 0)}s\n"
    )
    lines.append("## 1. 总体指标（实测）\n")
    lines.append(
        _table(
            ["指标", "数值"],
            [
                ["任务闭环成功率", f"{overall['success']}/{overall['episodes']} = {overall['success_rate']}%"],
                [
                    "运行时自报成功率",
                    f"{overall['runtime_declared_success']}/{overall['episodes']} = "
                    f"{overall['runtime_declared_success_rate']}%",
                ],
                ["运行时/评测器判定不一致", overall["runtime_evaluator_mismatch"]],
                [
                    "发生重规划的幕数",
                    f"{overall['replan_episodes']}（{overall['replan_rate']}%）",
                ],
                [
                    "重规划恢复率",
                    f"{overall['replan_recovered']}/{overall['replan_episodes']} = {overall['replan_recovery_rate']}%",
                ],
                ["规划时延 p50 / p95", f"{overall['plan_latency_ms']['p50']} / {overall['plan_latency_ms']['p95']} ms"],
                [
                    "技能决策时延 p50 / p95",
                    f"{overall['decision_latency_ms']['p50']} / {overall['decision_latency_ms']['p95']} ms",
                ],
                [
                    "单幕仿真时长 p50 / p95",
                    f"{overall['sim_time_s']['p50']} / {overall['sim_time_s']['p95']} s",
                ],
                [
                    "记忆命中幕数 / 命中率",
                    f"{overall['memory_hit_episodes']} / {overall['memory_hit_rate']}%"
                    f"（命中时成功率 {overall['memory_success_rate']}%，未命中 {overall['no_memory_success_rate']}%）",
                ],
            ],
        )
    )
    lines.append("\n> 说明：本机为 2 核 CPU、无 GPU；评测使用确定性规则规划器（离线），因此规划时延为毫秒级。")
    lines.append("> 接入 Qwen2.5-VL / 真实 LLM 规划器时，这一段时延会由模型 API 主导，本报告不能代表那种部署形态。\n")

    lines.append("## 2. 分场景结果\n")
    rows = []
    for scene, metrics in report.get("by_scene", {}).items():
        rows.append(
            [
                scene,
                metrics["episodes"],
                f"{metrics['success']} ({metrics['success_rate']}%)",
                metrics["replan_episodes"],
                metrics["replan_recovery_rate"],
                metrics["sim_time_s"]["p50"],
            ]
        )
    lines.append(_table(["场景", "幕数", "成功", "重规划幕数", "重规划恢复率(%)", "仿真时长 p50(s)"], rows))

    lines.append("\n## 3. 分意图结果\n")
    rows = []
    for intent, metrics in report.get("by_intent", {}).items():
        rows.append([intent, metrics["episodes"], f"{metrics['success_rate']}%", metrics["replan_episodes"]])
    lines.append(_table(["意图", "幕数", "成功率", "重规划幕数"], rows))

    lines.append("\n## 4. 失败 bucket\n")
    buckets = overall.get("failure_buckets", {})
    if buckets:
        lines.append(_table(["失败码", "次数"], [[code, count] for code, count in buckets.items()]))
    else:
        lines.append("本次评测无失败。")

    fault = report.get("fault_injection")
    if fault:
        lines.append("\n## 5. 故障注入（4 类 × 5 目标 = 20 幕）\n")
        summary = fault["summary"]
        lines.append(
            _table(
                ["指标", "数值"],
                [
                    ["故障恢复率", f"{summary['success']}/{summary['episodes']} = {summary['recovery_rate']}%"],
                    ["重规划恢复率", f"{summary['replan_recovery_rate']}%"],
                    ["失败 bucket", summary["failure_buckets"]],
                ],
            )
        )
        rows = []
        for episode in fault.get("episodes", []):
            fault_info = episode.get("fault") or {}
            rows.append(
                [
                    fault_info.get("kind", "-"),
                    fault_info.get("target", "-"),
                    episode["instruction_id"],
                    episode["status"],
                    episode["replans"],
                    "✅" if episode["success"] else "❌",
                ]
            )
        lines.append("\n" + _table(["故障", "目标", "用例", "状态", "重规划次数", "恢复"], rows))

    ablation = report.get("ablation")
    if ablation:
        lines.append("\n## 6. 消融：关闭后置条件校验 + 重规划\n")
        clean = ablation["clean"]
        fault_summary = ablation["fault_recovery"]
        lines.append(
            _table(
                ["配置", "成功", "成功率", "重规划恢复率"],
                [
                    ["完整系统（本报告主结果）", overall["success"], f"{overall['success_rate']}%", f"{overall['replan_recovery_rate']}%"],
                    [
                        "关闭 verify + replan（干净场景）",
                        clean["success"],
                        f"{clean['success_rate']}%",
                        f"{clean['replan_recovery_rate']}%",
                    ],
                    [
                        "关闭 verify + replan（故障注入）",
                        fault_summary["success"],
                        f"{fault_summary['recovery_rate']}%",
                        f"{fault_summary['replan_recovery_rate']}%",
                    ],
                ],
            )
        )
        lines.append(
            "\n> 干净场景下两者接近是符合预期的（技能大多一次成功，校验不会改变结果）；"
            "差异体现在故障注入：关闭 verify+replan 后恢复率降到基线水平。\n"
        )

    stress = report.get("stress")
    if stress:
        lines.append("\n## 7. 压力/泛化子集（30 条指令 × 3 类扰动场景 × 1 次 = 90 幕）\n")
        lines.append(
            "标准评测是确定性规划器 + 干净世界，因此成功率偏高；为了不让指标显得「过于完美」，"
            "另跑一组扰动场景（同一 runner、同一评测器判定），如实报告：\n"
        )
        overall_stress = stress["overall"]
        lines.append(
            _table(
                ["指标", "数值"],
                [
                    ["压力场景总成功率", f"{overall_stress['success']}/{overall_stress['episodes']} = {overall_stress['success_rate']}%"],
                    ["运行时/评测器判定不一致", overall_stress["runtime_evaluator_mismatch"]],
                    ["重规划恢复率", f"{overall_stress['replan_recovery_rate']}%（{overall_stress['replan_recovered']}/{overall_stress['replan_episodes']}）"],
                    ["失败 bucket", overall_stress["failure_buckets"]],
                ],
            )
        )
        rows = []
        for variant in stress.get("variants", []):
            rows.append(
                [
                    variant["name"],
                    variant["description"],
                    variant["episodes"],
                    f"{variant['success_rate']}%",
                    variant["replan_episodes"],
                    variant["replan_recovery_rate"],
                ]
            )
        lines.append("\n" + _table(["扰动场景", "说明", "幕数", "成功率", "重规划幕数", "重规划恢复率(%)"], rows))

    lines.append("\n## 8. 失败案例清单（最多 30 条）\n")
    failures = report.get("failures", [])
    if failures:
        rows = []
        for episode in failures[:30]:
            rows.append(
                [
                    episode["instruction_id"],
                    episode["scene"],
                    episode["instruction"][:24],
                    episode["status"],
                    episode.get("failure_code") or "-",
                    episode["replans"],
                    (episode.get("failure_detail") or episode.get("goal_detail") or "")[:60],
                ]
            )
        lines.append(_table(["用例", "场景", "指令", "状态", "失败码", "重规划", "细节"], rows))
    else:
        lines.append("本次评测无失败案例。")
    lines.append("")
    return "\n".join(lines)
