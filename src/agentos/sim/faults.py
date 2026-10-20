"""Fault injection for the deterministic simulator.

Four faults map onto the failure codes the skills declare:

===============  =========================================================  ===================
fault            behaviour                                                  expected code
===============  =========================================================  ===================
object_missing   target object is silently relocated to another search zone  NOT_FOUND
path_blocked     an obstacle appears on the current path mid-navigation      BLOCKED
pick_slip        first grasp attempt slips                                   SLIP
nav_timeout      first navigation attempt is stalled below progress rate     TIMEOUT
===============  =========================================================  ===================

All faults are one-shot: after they fire the next attempt runs normally, which
is exactly what a replan/recovery evaluation needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FAULT_KINDS = ("none", "object_missing", "path_blocked", "pick_slip", "nav_timeout")


@dataclass
class FaultSpec:
    kind: str = "none"
    target: str = ""  # object_id, or "color:category", or "color category"
    magnitude: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}; choose from {FAULT_KINDS}")

    @property
    def active(self) -> bool:
        return self.kind != "none"

    def matches(self, obj: Any) -> bool:
        if not self.target:
            return True
        target = self.target.strip()
        object_id = getattr(obj, "object_id", "")
        if target == object_id:
            return True
        color = getattr(obj, "color", "")
        category = getattr(obj, "category", "")
        normalised = target.replace("_", " ").replace(":", " ").lower()
        return normalised in (f"{color} {category}", f"{color}:{category}", category, color)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "magnitude": self.magnitude, "note": self.note}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str | "FaultSpec" | None) -> "FaultSpec":
        if data is None or data == "" or data == "none":
            return cls()
        if isinstance(data, FaultSpec):
            return data
        if isinstance(data, str):
            return cls(kind=data)
        return cls(
            kind=str(data.get("kind", "none")),
            target=str(data.get("target", "")),
            magnitude=float(data.get("magnitude", 1.0)),
            note=str(data.get("note", "")),
        )


def fault_suite() -> list[FaultSpec]:
    """The 4 faults x 5 variants = 20 injections used by the evaluation report.

    Each variant targets a different object so we do not accidentally evaluate
    five copies of the same trajectory.
    """
    targets = ["red:box", "blue:cup", "green:ball", "yellow:package", "black:tool"]
    suite: list[FaultSpec] = []
    for kind in ("object_missing", "path_blocked", "pick_slip", "nav_timeout"):
        for target in targets:
            suite.append(FaultSpec(kind=kind, target=target, note=f"{kind} on {target}"))
    return suite
