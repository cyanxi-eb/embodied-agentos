"""Run the FastAPI debug console.

    python scripts/run_api.py                 # http://127.0.0.1:8000
    python scripts/run_api.py --port 8100
    python scripts/run_api.py --ros2          # Linux + ROS2 only: use Ros2SkillBackend
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Embodied AgentOS debug console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scene", default="warehouse")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--ros2",
        action="store_true",
        help="use the rclpy backend (requires Ubuntu 24.04 + ROS2 Jazzy; not run on the dev host)",
    )
    args = parser.parse_args()

    from agentos.config import RuntimeConfig
    from agentos.graph.runtime import AgentRuntime

    if args.ros2:
        from agentos.ros2_bridge import ROS2_AVAILABLE

        if not ROS2_AVAILABLE:
            print("--ros2 requested but rclpy is not importable on this host; aborting honestly.")
            return 2
        raise SystemExit(
            "ROS2 backend wiring requires a running Nav2 stack; see docs/ros2_integration.md "
            "(this branch is not implemented as an automated path yet)."
        )

    config = RuntimeConfig(scene=args.scene, cache_dir=args.cache_dir)
    runtime = AgentRuntime(config)
    from agentos.api.server import create_app

    app = create_app(runtime)
    import uvicorn

    print(f"Embodied AgentOS console: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
