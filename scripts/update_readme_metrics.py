"""Inject the measured metrics from reports/eval_*.json into README.md.

The README metric table lives between ``<!-- METRICS:START -->`` and
``<!-- METRICS:END -->`` markers; this script guarantees the committed numbers are
exactly the numbers of a real evaluation run (no manual copy/paste drift).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
START = "<!-- METRICS:START -->"
END = "<!-- METRICS:END -->"


def build_table(report: dict) -> str:
    overall = report["overall"]
    fault = report.get("fault_injection", {}).get("summary", {})
    ablation = report.get("ablation", {})
    rows = [
        ("评测规模", f"{report['config']['instructions']} 条指令 × {len(report['config']['scenes'])} 场景 × {report['config']['repeats']} 次 = {overall['episodes']} 幕"),
        ("任务闭环成功率（评测器判定）", f"**{overall['success_rate']}%**（{overall['success']}/{overall['episodes']}）"),
        ("运行时自报成功率", f"{overall['runtime_declared_success_rate']}%（{overall['runtime_declared_success']}/{overall['episodes']}）"),
        ("重规划恢复率", f"**{overall['replan_recovery_rate']}%**（{overall['replan_recovered']}/{overall['replan_episodes']} 次重规划后成功）"),
        ("故障注入恢复率", f"**{fault.get('recovery_rate', 0)}%**（{fault.get('success', 0)}/{fault.get('episodes', 0)}，4 类故障 × 5 目标）"),
        ("消融：关闭 verify+replan（干净场景 / 故障注入）", f"{ablation.get('clean', {}).get('success_rate', 0)}% / **{ablation.get('fault_recovery', {}).get('recovery_rate', 0)}%**"),
        ("规划时延 p50 / p95", f"{overall['plan_latency_ms']['p50']} / {overall['plan_latency_ms']['p95']} ms"),
        ("技能决策时延 p50 / p95", f"{overall['decision_latency_ms']['p50']} / {overall['decision_latency_ms']['p95']} ms"),
        ("单幕仿真时长 p50 / p95", f"{overall['sim_time_s']['p50']} / {overall['sim_time_s']['p95']} s"),
        ("全量评测墙钟耗时", f"{report.get('wall_time_s', 0)} s"),
        ("失败 bucket", json.dumps(overall.get("failure_buckets", {}), ensure_ascii=False)),
        ("记忆命中率", f"{overall['memory_hit_rate']}%"),
    ]
    stress = report.get("stress")
    if stress:
        rows.append(
            (
                "压力/泛化场景成功率（30 指令 × 3 类扰动）",
                f"**{stress['overall']['success_rate']}%**（{stress['overall']['success']}/{stress['overall']['episodes']}）",
            )
        )
        for variant in stress.get("variants", []):
            rows.append((f"　· {variant['name']}", f"{variant['success_rate']}%（{variant['episodes']} 幕，重规划 {variant['replan_episodes']} 幕）"))
    lines = ["| 指标 | 实测值 |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    lines.append("")
    lines.append(
        f"<sub>数据来源：`{report.get('report_paths', {}).get('json', 'reports/eval_*.json')}`，"
        f"生成于 {report['generated_at']}，主机 {report['host']['platform']}（Python {report['host']['python']}）。"
        "本表由 `scripts/update_readme_metrics.py` 从 JSON 自动生成。</sub>"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=None, help="path to reports/eval_*.json (default: latest)")
    parser.add_argument("--readme", default=str(REPO_ROOT / "README.md"))
    args = parser.parse_args()
    path = args.report or sorted(glob.glob(str(REPO_ROOT / "reports" / "eval_*.json")))[-1]
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    table = build_table(report)
    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")
    start = text.index(START)
    end = text.index(END)
    new_text = text[: start + len(START)] + "\n\n" + table + "\n" + text[end:]
    readme.write_text(new_text, encoding="utf-8")
    print(f"updated {readme} from {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
