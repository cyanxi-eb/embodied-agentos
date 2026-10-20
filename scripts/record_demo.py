"""Record the three demo GIFs from *real* simulator runs (not hand-drawn frames).

    python scripts/record_demo.py                # all three demos
    python scripts/record_demo.py --only 01      # one demo
    python scripts/record_demo.py --mp4          # also export mp4 via imageio-ffmpeg

Each frame is produced by ``SimRenderer`` while the LangGraph/behavior-tree stack
actually executes the task; banners are driven by emitted trace events, so what
you see is what the runtime did.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from agentos.config import RuntimeConfig  # noqa: E402
from agentos.graph.runtime import AgentRuntime  # noqa: E402
from agentos.sim.faults import FaultSpec  # noqa: E402
from agentos.sim.render import SimRenderer, save_gif  # noqa: E402
from agentos.sim.scenes import build_scene  # noqa: E402
from agentos.trace.recorder import TraceRecorder  # noqa: E402

COLORS = {
    "running": "#2d6cdf",
    "success": "#2e9d57",
    "fail": "#c0392b",
    "warn": "#d35400",
    "pause": "#e8a33d",
}

DEMOS = [
    {
        "name": "01_nav_pick_fetch",
        "file": "01_nav_pick_fetch.gif",
        "instruction": "把A区的红色盒子拿到工作站台",
        "title": "FETCH: red box @ zone A -> workbench",
        "scene": "warehouse",
        "seed": 3,
        "frame_every": 5,
        "pause_at_frame": None,
        "faults": [],
    },
    {
        "name": "02_interrupt_resume",
        "file": "02_interrupt_resume.gif",
        "instruction": "把B区的蓝色杯子拿到工作站台",
        "title": "INTERRUPT: pause mid-task, then resume",
        "scene": "warehouse",
        "seed": 5,
        "frame_every": 6,
        "pause_at_frame": 20,
        "faults": [],
    },
    {
        "name": "03_failure_replan",
        "file": "03_failure_replan.gif",
        "instruction": "把A区的红色盒子拿到工作站台",
        "title": "FAILURE: object missing -> replan -> success",
        "scene": "warehouse",
        "seed": 3,
        "frame_every": 8,
        "pause_at_frame": None,
        "faults": [FaultSpec(kind="object_missing", target="red:box").to_dict()],
    },
]


class DemoRecorder:
    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.name = spec["name"]
        self.instruction = spec["instruction"]
        self.title = spec.get("title", self.instruction)
        self.scene_name = spec["scene"]
        self.seed = spec["seed"]
        self.frame_every = spec["frame_every"]
        self.pause_at_frame = spec["pause_at_frame"]
        cache = REPO_ROOT / ".cache" / "demo-tmp" / f"{self.name}-{uuid.uuid4().hex[:8]}"
        cache.mkdir(parents=True, exist_ok=True)
        self.cache_dir = str(cache)
        self.runtime = AgentRuntime(
            RuntimeConfig(
                scene=self.scene_name,
                seed=self.seed,
                cache_dir=str(cache),
                persist_events=True,
                memory_enabled=False,
            )
        )
        self.renderer = SimRenderer(build_scene(self.scene_name))
        self.frames: list = []
        self.hud = {"skill": "-", "status": "planning", "replans": 0}
        self.banner: tuple[str, str] = ("PLAN · building task graph", COLORS["running"])
        self.session_id = f"demo-{self.name}-{uuid.uuid4().hex[:6]}"
        self.recorder = TraceRecorder(
            self.session_id,
            store=self.runtime.store,
            trace_dir=REPO_ROOT / "demos" / "traces",
        )
        self.recorder.subscribe(self._on_event)
        self.runtime.frame_hook = self._on_frame
        self.runtime.frame_every = self.frame_every
        self.pause_requested = False
        self.last_failure_code = ""
        self.run_info: dict = {}

    # --------------------------------------------------------------- events
    def _on_event(self, event: dict) -> None:
        event_type = event.get("type", "")
        if event_type == "skill_dispatch":
            self.hud["skill"] = event.get("skill", "-")
            self.hud["status"] = "executing"
            self.banner = (f"SKILL · {event.get('skill')}", COLORS["running"])
        elif event_type == "skill_result":
            self.hud["status"] = "verifying"
        elif event_type == "postcondition":
            self.hud["status"] = "verified" if event.get("verified") else "postcondition failed"
        elif event_type == "skill_failed":
            self.hud["status"] = "failed"
            self.last_failure_code = str(event.get("code", ""))
            self.banner = (f"{event.get('code')} → REPLAN", COLORS["fail"])
        elif event_type == "replan_scheduled":
            self.hud["replans"] = int(event.get("next_replans", self.hud["replans"]))
            failure = event.get("failure") or {}
            self.banner = (f"REPLAN #{self.hud['replans']} · {failure.get('code', '')}", COLORS["warn"])
        elif event_type == "pause_requested":
            self.banner = ("PAUSED · checkpoint saved, waiting for operator", COLORS["pause"])
        elif event_type == "resumed":
            self.banner = ("RESUMED · continuing from checkpoint", COLORS["running"])
        elif event_type == "finish":
            status = event.get("status")
            self.hud["status"] = status
            if status == "finished":
                self.banner = ("FINISHED · postconditions verified", COLORS["success"])
            elif status == "cancelled":
                self.banner = ("CANCELLED · goal + blackboard cleared", COLORS["warn"])
            else:
                self.banner = (f"FAILED · {status}", COLORS["fail"])

    # ---------------------------------------------------------------- frames
    def _capture(self, world, banner: str | None = None, banner_color: str | None = None, info: dict | None = None) -> None:
        info = info or {}
        self.frames.append(
            self.renderer.frame(
                world,
                extra_obstacles=info.get("obstacles", []),
                path=info.get("path", []),
                trail=info.get("trail", []),
                title=self.title,
                hud={**self.hud, "sim": world.sim_time},
                banner=self.banner[0] if banner is None else banner,
                banner_color=self.banner[1] if banner_color is None else banner_color,
            )
        )

    def _on_frame(self, world, info: dict) -> None:
        if len(self.frames) >= 170:
            return
        self._capture(world, info=info)
        if (
            self.pause_at_frame is not None
            and not self.pause_requested
            and len(self.frames) >= self.pause_at_frame
        ):
            self.pause_requested = True
            self.runtime.control.request_pause(self.session_id)

    def _capture_banner_sequence(self, count: int, banner: str, color: str, status: str) -> None:
        world = self.runtime.world(self.session_id)
        if world is None:
            return
        for index in range(count):
            self.frames.append(
                self.renderer.frame(
                    world,
                    title=self.title,
                    hud={
                        **self.hud,
                        "sim": world.sim_time,
                        "status": status,
                        "note": f"hold {index + 1}/{count}",
                    },
                    banner=banner,
                    banner_color=color,
                )
            )

    # ------------------------------------------------------------------- run
    def run(self) -> dict:
        started = time.perf_counter()
        result = self.runtime.start(
            self.instruction,
            scene=self.scene_name,
            seed=self.seed,
            faults=self.spec["faults"],
            session_id=self.session_id,
            recorder=self.recorder,
        )
        final = result
        if result.status == "paused":
            self._capture_banner_sequence(8, "PAUSED · checkpoint saved to SQLite", COLORS["pause"], "paused")
            final = self.runtime.resume(self.session_id, "resume")
            self._capture_banner_sequence(6, "RESUMED · replaying from checkpoint", COLORS["running"], "running")
        # make the replan explicit in the GIF (the banner is otherwise overwritten
        # by the next skill dispatch before a frame is rendered)
        if int(final.state.get("replans", 0)) > 0:
            self._capture_banner_sequence(
                10,
                f"{self.last_failure_code or 'FAILURE'} → REPLAN #{final.state.get('replans')} → recovered",
                COLORS["warn"],
                "replanning",
            )
        # hold the final status banner for a moment so it is readable in the GIF
        final_banner = self.banner[0] or "FINISHED"
        self._capture_banner_sequence(12, final_banner, self.banner[1], self.hud["status"])
        self.recorder.close()
        self.run_info = {
            "name": self.name,
            "session_id": self.session_id,
            "status": final.status,
            "goal_verified": bool(final.state.get("goal_verified")),
            "replans": int(final.state.get("replans", 0)),
            "frames": len(self.frames),
            "wall_s": round(time.perf_counter() - started, 2),
        }
        return self.run_info

    def save(self, path: Path, fps: int = 8) -> dict:
        info = save_gif(self.frames, path, fps=fps)
        info["run"] = self.run_info
        return info

    def close(self) -> None:
        self.renderer.close()
        self.runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="record AgentOS demo GIFs")
    parser.add_argument("--only", default=None, help="01 | 02 | 03")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--mp4", action="store_true", help="also export mp4 (imageio-ffmpeg)")
    args = parser.parse_args()

    demos_dir = REPO_ROOT / "demos"
    demos_dir.mkdir(exist_ok=True)
    (demos_dir / "traces").mkdir(exist_ok=True)
    results = []
    for spec in DEMOS:
        if args.only and not spec["name"].startswith(args.only):
            continue
        print(f"[demo] recording {spec['name']} ...", flush=True)
        demo = DemoRecorder(spec)
        run_info = demo.run()
        info = demo.save(demos_dir / spec["file"], fps=args.fps)
        if args.mp4:
            try:
                from record_demo_mp4 import frames_to_mp4  # same directory when run as a script

                frames_to_mp4(demo.frames, demos_dir / (Path(spec["file"]).stem + ".mp4"), fps=args.fps)
            except Exception as error:  # pragma: no cover - optional
                print(f"[demo] mp4 export skipped: {error}")
        demo.close()
        results.append(info)
        print(
            f"[demo] {spec['file']}: {info['frames']} frames, {info['size']}, "
            f"{info['megabytes']} MB, status={run_info['status']} "
            f"goal_verified={run_info['goal_verified']} replans={run_info['replans']}",
            flush=True,
        )

    print("\n=== demo summary ===")
    for info in results:
        run = info["run"]
        print(
            f"{Path(info['path']).name}: {info['frames']} frames @ {info['fps']}fps, {info['size']}, "
            f"{info['megabytes']} MB | status={run['status']} goal_verified={run['goal_verified']} "
            f"replans={run['replans']} session={run['session_id']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
