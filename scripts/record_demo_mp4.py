"""Optional MP4 export for demo frames (imageio-ffmpeg; not required for the GIF deliverable)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def frames_to_mp4(frames: list[Any], path: str | Path, fps: int = 8) -> str:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
    finally:
        writer.close()
    return str(path)


def main() -> int:
    import argparse

    from scripts.record_demo import main as record_main  # pragma: no cover

    parser = argparse.ArgumentParser(description="record demos and export mp4")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    return record_main(["--mp4", "--fps", str(args.fps)])


if __name__ == "__main__":
    raise SystemExit(main())
