"""Skill registry — loads hand-authored skill cards from skills/*.yaml.

A skill card is a small piece of domain knowledge that an agent can inject into
its prompts. Cards are grouped by domain (e.g. "wars", "glp1_trials"). Files
prefixed with `_` (such as `_suggestions.yaml` written by CriticAgent) are
skipped so agent-authored suggestions don't masquerade as hand-authored cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class SkillCard:
    name: str
    domain: str
    description: str
    prompt_fragment: str


class SkillRegistry:
    """In-memory registry of domain-scoped skill cards."""

    def __init__(self) -> None:
        self._by_domain: dict[str, list[SkillCard]] = {}

    def load_dir(self, skills_dir: Path | str) -> None:
        """Load all `*.yaml` files in the directory; tolerant of malformed files."""
        skills_dir = Path(skills_dir)
        if not skills_dir.is_dir():
            return
        for yaml_path in sorted(skills_dir.glob("*.yaml")):
            if yaml_path.name.startswith("_"):
                continue  # skip _suggestions.yaml etc.
            try:
                data = yaml.safe_load(yaml_path.read_text())
            except yaml.YAMLError:
                continue
            if not isinstance(data, dict):
                continue
            domain = data.get("domain", yaml_path.stem)
            cards = data.get("skills", [])
            if not isinstance(cards, list):
                continue
            parsed_cards: list[SkillCard] = []
            for c in cards:
                if not isinstance(c, dict):
                    continue
                parsed_cards.append(
                    SkillCard(
                        name=str(c.get("name", "")),
                        domain=str(domain),
                        description=str(c.get("description", "")),
                        prompt_fragment=str(c.get("prompt_fragment", "")),
                    )
                )
            self._by_domain[str(domain)] = parsed_cards

    def for_domain(self, domain: str) -> list[SkillCard]:
        return list(self._by_domain.get(domain, []))

    def all(self) -> list[SkillCard]:
        out: list[SkillCard] = []
        for cards in self._by_domain.values():
            out.extend(cards)
        return out
