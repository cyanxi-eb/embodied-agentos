"""Skill registry: name -> skill, plus metadata used by MCP / the planner."""

from __future__ import annotations

from typing import Any, Iterable

from agentos.skills.base import Skill
from agentos.skills.library import build_default_skills


class SkillRegistry:
    def __init__(self, skills: Iterable[Skill] | None = None) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills if skills is not None else build_default_skills():
            self.register(skill)

    def register(self, skill: Skill) -> None:
        if not skill.name:
            raise ValueError("skill must define a name")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def require(self, name: str) -> Skill:
        skill = self.get(name)
        if skill is None:
            raise KeyError(f"unknown skill {name!r}")
        return skill

    def names(self) -> list[str]:
        return sorted(self._skills)

    def all(self) -> list[Skill]:
        return [self._skills[name] for name in self.names()]

    def describe(self, name: str) -> dict[str, Any]:
        return self.require(name).describe()

    def list_descriptions(self) -> list[dict[str, Any]]:
        return [skill.describe() for skill in self.all()]

    def validate_call(self, name: str, params: dict[str, Any]) -> list[str]:
        skill = self.get(name)
        if skill is None:
            return [f"unknown skill {name!r}"]
        return skill.validate(params or {})

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "name": skill.name,
                "category": skill.category.value,
                "description": skill.description,
                "timeout_s": skill.timeout_s,
                "failure_codes": [code.value for code in skill.failure_codes],
            }
            for skill in self.all()
        ]


_DEFAULT: SkillRegistry | None = None


def default_registry() -> SkillRegistry:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SkillRegistry()
    return _DEFAULT
