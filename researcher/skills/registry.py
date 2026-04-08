"""Skill registry — loads hand-authored skill cards from skills/*.yaml.

A skill card is a small piece of domain knowledge that an agent can inject into
its prompts. Cards are grouped by domain (e.g. "wars", "glp1_trials"). Files
prefixed with `_` (such as `_suggestions.yaml` written by CriticAgent) are
skipped so agent-authored suggestions don't masquerade as hand-authored cards.

Cards may optionally declare an `input_schema` and/or `output_schema` — small
JSON-schema-like field maps. When an `output_schema` is present, native agents
can validate LLM-extracted field values against the declared types before
emitting `FactClaim`s. This adds BAML-style schema discipline without a new DSL.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, create_model

_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


def _resolve_field_type(type_str: str) -> type:
    type_str = type_str.strip()
    if type_str.startswith("list[") and type_str.endswith("]"):
        inner = type_str[5:-1].strip()
        return list[_TYPE_MAP.get(inner, str)]  # type: ignore[valid-type]
    return _TYPE_MAP.get(type_str, str)


def _build_schema_model(schema: dict[str, Any], required_only: bool) -> type[BaseModel]:
    """Build a Pydantic model from a declarative output_schema dict.

    Each entry in `schema` is `{field_name: {"type": "...", "required": bool, ...}}`.
    Unknown / malformed types fall back to `str` (matching ExpandAgent's behavior).
    When `required_only` is False, every field is `Optional[T] = None`. When True,
    fields whose entry has `required: true` are required (no default), and the
    rest stay optional with `None` defaults — used by `validate_output` for
    full-payload validation.
    """
    fields: dict[str, Any] = {}
    for name, spec in schema.items():
        if not isinstance(spec, dict):
            continue
        py_type = _resolve_field_type(str(spec.get("type", "str")))
        is_required = bool(spec.get("required", False)) if required_only else False
        if is_required:
            fields[name] = (py_type, Field(...))
        else:
            fields[name] = (Optional[py_type], Field(default=None))
    if not fields:
        fields["_placeholder"] = (Optional[str], Field(default=None))
    return create_model("SkillCardSchema", **fields)  # type: ignore[call-overload]


@dataclass
class SkillCard:
    name: str
    domain: str
    description: str
    prompt_fragment: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None

    def validate_output(self, payload: dict[str, Any]) -> tuple[bool, str]:
        """Validate a full payload against this card's `output_schema`.

        Returns `(True, "")` if no schema is declared or validation succeeds;
        `(False, reason)` otherwise. Required fields declared in the schema
        must be present and well-typed in the payload.
        """
        if not self.output_schema:
            return (True, "")
        try:
            model = _build_schema_model(self.output_schema, required_only=True)
        except Exception as exc:  # defensive: malformed schema shouldn't crash
            return (False, f"schema build failed: {exc}")
        try:
            model.model_validate(payload)
        except ValidationError as exc:
            return (False, _format_validation_error(exc))
        return (True, "")

    def validate_output_field(self, field_name: str, value: object) -> tuple[bool, str]:
        """Validate a single field value against this card's `output_schema`.

        Unknown fields (not declared by the schema) are silently accepted —
        the schema is opt-in field-level guardrail, not exhaustive coverage.
        Returns `(True, "")` on success or when the schema/field is absent;
        `(False, reason)` when validation fails.
        """
        if not self.output_schema:
            return (True, "")
        spec = self.output_schema.get(field_name)
        if not isinstance(spec, dict):
            return (True, "")
        try:
            single_schema = {field_name: spec}
            model = _build_schema_model(single_schema, required_only=False)
        except Exception as exc:
            return (False, f"schema build failed: {exc}")
        try:
            model.model_validate({field_name: value})
        except ValidationError as exc:
            return (False, _format_validation_error(exc))
        return (True, "")


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = err.get("msg", "validation error")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "validation error"


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
                input_schema = c.get("input_schema")
                if not isinstance(input_schema, dict):
                    input_schema = None
                output_schema = c.get("output_schema")
                if not isinstance(output_schema, dict):
                    output_schema = None
                parsed_cards.append(
                    SkillCard(
                        name=str(c.get("name", "")),
                        domain=str(domain),
                        description=str(c.get("description", "")),
                        prompt_fragment=str(c.get("prompt_fragment", "")),
                        input_schema=input_schema,
                        output_schema=output_schema,
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

    def cards_for_entity_type(self, entity_type: str) -> list[SkillCard]:
        """Return cards that plausibly apply to the given entity type.

        Heuristic: a card matches when its domain or name contains the
        entity_type as a case-insensitive substring (e.g. entity_type="War"
        matches domain "wars" and name "war_year_validator"). Empty entity
        types match nothing.
        """
        needle = entity_type.strip().lower()
        if not needle:
            return []
        out: list[SkillCard] = []
        for cards in self._by_domain.values():
            for card in cards:
                if needle in card.domain.lower() or needle in card.name.lower():
                    out.append(card)
        return out
