"""Replay a structured trace JSONL file (failure post-mortem, no simulator needed).

    python scripts/replay_trace.py .cache/traces/<session>.jsonl
    python scripts/replay_trace.py <file> --failures-only

The trace is the same artefact a real deployment would pair with a rosbag: it
contains the plan, each skill dispatch/result, the independent postcondition check,
replan decisions and the final status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="replay an AgentOS trace JSONL")
    parser.add_argument("path", help="trace jsonl path")
    parser.add_argument("--failures-only", action="store_true")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"trace not found: {path}")
        return 2
    interesting = {
        "plan",
        "plan_validated",
        "skill_dispatch",
        "skill_result",
        "postcondition",
        "skill_failed",
        "replan_scheduled",
        "safety_rejected",
        "finish",
        "pause_requested",
        "resumed",
        "unplannable",
    }
    failure_events = {"skill_failed", "safety_rejected", "unplannable", "replan_scheduled"}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            event_type = event.get("type", "")
            if args.failures_only and event_type not in failure_events:
                continue
            if event_type not in interesting:
                continue
            detail = {key: event.get(key) for key in ("skill", "code", "ok", "verified", "replan", "status") if key in event}
            print(f"[{event.get('seq'):>4}] {event_type:<18} {json.dumps(detail, ensure_ascii=False)}")
            if event_type == "plan":
                print(f"       goal={event.get('goal')} nodes={len(event.get('nodes') or [])} notes={event.get('notes')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
