"""Matplotlib top-down renderer for simulator frames -> PNG frames -> GIF.

Rendering is only used for demos and report figures; the evaluation itself never
renders, which keeps 450 episodes fast on a 2-core CPU.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

from agentos.sim.scenes import Scene  # noqa: E402
from agentos.types import WorldState  # noqa: E402

COLOR_MAP = {
    "red": "#e05252",
    "blue": "#4a7fd4",
    "green": "#3fae6a",
    "yellow": "#e3b23c",
    "black": "#333333",
    "white": "#f2f2f2",
    "orange": "#e08a3c",
}

ZONE_SHORT = {"zone_a": "A", "zone_b": "B", "zone_c": "C", "table": "TABLE", "dock": "DOCK"}

STATUS_COLORS = {
    "running": "#2d6cdf",
    "paused": "#e8a33d",
    "finished": "#2e9d57",
    "failed": "#c0392b",
    "cancelled": "#8e44ad",
    "replanning": "#d35400",
    "planning": "#2d6cdf",
}


class SimRenderer:
    def __init__(self, scene: Scene, figsize: tuple[float, float] = (4.8, 3.6), dpi: int = 100) -> None:
        self.scene = scene
        self.fig, self.ax = plt.subplots(figsize=figsize, dpi=dpi)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03)

    def close(self) -> None:
        plt.close(self.fig)

    def frame(
        self,
        world: WorldState,
        extra_obstacles: Iterable[Iterable[float]] = (),
        path: Iterable[tuple[float, float]] = (),
        trail: Iterable[tuple[float, float]] = (),
        title: str = "",
        hud: dict[str, Any] | None = None,
        banner: str | None = None,
        banner_color: str = "#2d6cdf",
        target: tuple[float, float] | None = None,
    ) -> Any:
        ax = self.ax
        ax.clear()
        ax.set_xlim(0, self.scene.width)
        ax.set_ylim(0, self.scene.height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#444444")
            spine.set_linewidth(1.2)
        ax.set_facecolor("#fafafa")
        ax.set_title(title or f"{self.scene.name}", fontsize=8, pad=4, color="#222222")

        # obstacles
        for rect in [*self.scene.obstacles, *[type(self.scene.obstacles[0])(*item) for item in extra_obstacles]]:
            ax.add_patch(
                Rectangle(
                    (rect.x0, rect.y0),
                    rect.width,
                    rect.height,
                    facecolor="#b8b8b8",
                    edgecolor="#555555",
                    linewidth=0.8,
                    zorder=2,
                )
            )

        # zones + dock
        for zone, (zx, zy) in self.scene.zones.items():
            ax.add_patch(Circle((zx, zy), 0.22, facecolor="none", edgecolor="#9aa6b2", linestyle="--", linewidth=0.8, zorder=3))
            # ASCII labels only: the default matplotlib font has no CJK glyphs
            ax.text(zx, zy + 0.34, ZONE_SHORT.get(zone, zone), fontsize=5, ha="center", color="#6b7785", zorder=3)
        dock_x, dock_y = self.scene.dock
        ax.add_patch(Rectangle((dock_x - 0.3, dock_y - 0.3), 0.6, 0.6, facecolor="#ffe9a8", edgecolor="#c9a227", linewidth=0.9, zorder=3))
        ax.text(dock_x, dock_y - 0.52, "DOCK", fontsize=5, ha="center", color="#8a6d1a", zorder=3)

        # path + trail
        if trail:
            xs, ys = zip(*trail)
            ax.plot(xs, ys, color="#9bbcf0", linewidth=1.0, alpha=0.8, zorder=4)
        if path:
            xs, ys = zip(*path)
            ax.plot(xs, ys, color="#2d6cdf", linewidth=1.1, linestyle="--", alpha=0.9, zorder=4)
        if target is not None:
            ax.plot([target[0]], [target[1]], marker="x", color="#c0392b", markersize=5, zorder=5)

        # objects
        for obj in world.objects.values():
            color = COLOR_MAP.get(obj.color, "#888888")
            ax.add_patch(
                Rectangle(
                    (obj.x - 0.11, obj.y - 0.11),
                    0.22,
                    0.22,
                    facecolor=color,
                    edgecolor="#333333",
                    linewidth=0.6,
                    zorder=5,
                )
            )
        if world.holding and world.held_meta:
            color = COLOR_MAP.get(str(world.held_meta.get("color", "")), "#888888")
            ax.add_patch(Circle((world.robot.x, world.robot.y), 0.14, facecolor=color, edgecolor="#111111", linewidth=0.7, zorder=7))

        # robot
        robot = world.robot
        length, width = 0.42, 0.28
        cos_y, sin_y = __import__("math").cos(robot.yaw), __import__("math").sin(robot.yaw)
        nose = (robot.x + cos_y * length * 0.6, robot.y + sin_y * length * 0.6)
        left = (robot.x - cos_y * length * 0.3 - sin_y * width * 0.5, robot.y - sin_y * length * 0.3 + cos_y * width * 0.5)
        right = (robot.x - cos_y * length * 0.3 + sin_y * width * 0.5, robot.y - sin_y * length * 0.3 - cos_y * width * 0.5)
        ax.add_patch(Polygon([nose, left, right], closed=True, facecolor="#1f4e9c", edgecolor="#0d2b5c", linewidth=0.9, zorder=8))

        # HUD
        hud = hud or {}
        lines = [
            f"skill: {hud.get('skill', '-')}",
            f"status: {hud.get('status', 'idle')}",
            f"replans: {hud.get('replans', 0)}   objects: {len(world.objects)}",
            f"sim: {world.sim_time:5.1f}s   battery: {world.battery:4.1f}%",
        ]
        if hud.get("note"):
            lines.append(str(hud["note"]))
        ax.text(
            0.02,
            0.03,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=5.2,
            color="#222222",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="#cccccc"),
            zorder=9,
        )
        if banner:
            ax.text(
                0.5,
                0.93,
                banner,
                transform=ax.transAxes,
                fontsize=8,
                color="white",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=banner_color, alpha=0.95, edgecolor="none"),
                zorder=10,
            )
        self.fig.canvas.draw()
        buffer = io.BytesIO()
        self.fig.savefig(buffer, format="png", dpi=self.fig.dpi)
        buffer.seek(0)
        from PIL import Image

        return Image.open(buffer).convert("RGB")


def save_gif(frames: list[Any], path: str | Path, fps: int = 10, max_colors: int = 32) -> dict[str, Any]:
    """Write an optimised GIF; returns size metadata for the report/README."""
    if not frames:
        raise ValueError("no frames to save")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000.0 / max(1, fps)))
    palette_frames = [frame.convert("P", palette=0, colors=max_colors) for frame in frames]
    palette_frames[0].save(
        path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size = path.stat().st_size
    return {
        "path": str(path),
        "frames": len(frames),
        "fps": fps,
        "bytes": size,
        "megabytes": round(size / (1024 * 1024), 3),
        "size": f"{frames[0].width}x{frames[0].height}",
    }
