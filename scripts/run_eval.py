"""CLI: run the evaluation and write reports/eval_*.json/.md.

Examples
--------
    python scripts/run_eval.py --quick
    python scripts/run_eval.py --episodes 60 --scenes warehouse
    python scripts/run_eval.py                 # full 450 episodes + faults + ablation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from agentos.sim.scenes import SCENE_NAMES  # noqa: E402
from evaluation.instructions import INSTRUCTIONS  # noqa: E402
from evaluation.report import write_report  # noqa: E402
from evaluation.runner import EvalConfig, EpisodeResult, run_evaluation  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embodied AgentOS evaluation")
    parser.add_argument("--quick", action="store_true", help="fast CI mode: 10 instructions x 1 scene x 1 repeat")
    parser.add_argument("--episodes", type=int, default=None, help="limit the number of instructions")
    parser.add_argument("--scenes", nargs="+", default=list(SCENE_NAMES), choices=list(SCENE_NAMES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed-salt", type=int, default=0)
    parser.add_argument("--no-faults", action="store_true", help="skip the 20 fault-injection episodes")
    parser.add_argument("--no-ablation", action="store_true", help="skip the verify/replan ablation")
    parser.add_argument("--persist-events", action="store_true", help="write every trace event to SQLite")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    instructions = INSTRUCTIONS[: args.episodes] if args.episodes else None
    config = EvalConfig(
        scenes=tuple(args.scenes),
        repeats=args.repeats,
        instructions=instructions,
        run_faults=not args.no_faults,
        run_ablation=not args.no_ablation,
        persist_events=args.persist_events,
        cache_dir=args.cache_dir,
        output_dir=args.out,
        quick=args.quick,
    )

    started = time.perf_counter()
    state = {"last": 0.0}

    def progress(index: int, total: int, episode: EpisodeResult) -> None:
        if args.quiet:
            return
        now = time.perf_counter()
        if index == 1 or index == total or now - state["last"] > 2.0:
            state["last"] = now
            mark = "OK " if episode.success else "FAIL"
            print(
                f"[{index:4d}/{total}] {mark} {episode.instruction_id} {episode.scene:9s} "
                f"replans={episode.replans} sim={episode.sim_time_s:6.1f}s wall={episode.wall_time_s:.2f}s",
                flush=True,
            )

    report = run_evaluation(config, progress=progress)
    paths = write_report(report, args.out)
    overall = report["overall"]
    print("\n=== Evaluation summary ===")
    print(
        f"episodes: {overall['episodes']} | success: {overall['success']} ({overall['success_rate']}%) | "
        f"replan recovery: {overall['replan_recovery_rate']}% | "
        f"plan p95: {overall['plan_latency_ms']['p95']}ms | sim p95: {overall['sim_time_s']['p95']}s"
    )
    if "fault_injection" in report:
        print(
            f"fault recovery: {report['fault_injection']['summary']['recovery_rate']}% "
            f"({report['fault_injection']['summary']['success']}/{report['fault_injection']['cases']})"
        )
    if "ablation" in report:
        print(
            f"ablation (verify+replan off): clean {report['ablation']['clean']['success_rate']}% | "
            f"fault recovery {report['ablation']['fault_recovery']['recovery_rate']}%"
        )
    print(f"wall time: {report['wall_time_s']}s")
    print(f"json:     {paths['json']}")
    print(f"markdown: {paths['markdown']}")
    if paths["chart"]:
        print(f"chart:    {paths['chart']}")
    print(f"total measured wall time: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
