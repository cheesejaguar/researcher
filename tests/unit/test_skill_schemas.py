"""Tests for declarative input/output schemas on SkillCard (v1.2 #7)."""

from __future__ import annotations

from pathlib import Path

from researcher.skills.registry import SkillCard, SkillRegistry


def _write(path: Path, text: str) -> None:
    path.write_text(text)


def test_skill_card_without_output_schema_accepts_anything() -> None:
    card = SkillCard(
        name="no_schema",
        domain="wars",
        description="",
        prompt_fragment="",
    )
    ok, reason = card.validate_output({"whatever": 42, "x": "y"})
    assert ok is True
    assert reason == ""

    ok, reason = card.validate_output_field("anything", "value")
    assert ok is True
    assert reason == ""


def test_skill_card_validates_int_field() -> None:
    card = SkillCard(
        name="war_year",
        domain="wars",
        description="",
        prompt_fragment="",
        output_schema={"start_year": {"type": "int", "required": False}},
    )

    ok, _ = card.validate_output_field("start_year", 1939)
    assert ok is True

    ok, reason = card.validate_output_field("start_year", "not an int")
    assert ok is False
    assert reason  # non-empty reason


def test_skill_card_validates_list_field() -> None:
    card = SkillCard(
        name="war_belligerents",
        domain="wars",
        description="",
        prompt_fragment="",
        output_schema={"belligerents": {"type": "list[str]"}},
    )

    ok, _ = card.validate_output_field("belligerents", ["USA", "UK"])
    assert ok is True

    ok, reason = card.validate_output_field("belligerents", 42)
    assert ok is False
    assert reason


def test_skill_card_required_field_missing() -> None:
    card = SkillCard(
        name="war_required",
        domain="wars",
        description="",
        prompt_fragment="",
        output_schema={
            "start_year": {"type": "int", "required": True},
            "end_year": {"type": "int", "required": False},
        },
    )

    ok, reason = card.validate_output({"end_year": 1945})
    assert ok is False
    assert "start_year" in reason


def test_skill_card_ignores_unknown_field() -> None:
    card = SkillCard(
        name="war_year",
        domain="wars",
        description="",
        prompt_fragment="",
        output_schema={"start_year": {"type": "int"}},
    )

    ok, reason = card.validate_output_field("unknown", "anything")
    assert ok is True
    assert reason == ""


def test_skill_card_malformed_type_falls_back_to_str() -> None:
    card = SkillCard(
        name="war_weird",
        domain="wars",
        description="",
        prompt_fragment="",
        output_schema={"x": {"type": "banana"}},
    )

    # Schema builds without error and a string passes.
    ok, _ = card.validate_output_field("x", "hello")
    assert ok is True

    # Full payload validation also doesn't crash.
    ok, _ = card.validate_output({"x": "hello"})
    assert ok is True


def test_registry_loads_cards_with_schemas(tmp_path: Path) -> None:
    _write(
        tmp_path / "wars.yaml",
        """
domain: wars
skills:
  - name: war_year_validator
    description: validates start_year
    prompt_fragment: extract start_year
    input_schema:
      query:
        type: str
        required: true
    output_schema:
      start_year:
        type: int
        required: false
        min: 1000
        max: 2100
""",
    )

    reg = SkillRegistry()
    reg.load_dir(tmp_path)
    cards = reg.for_domain("wars")
    assert len(cards) == 1
    card = cards[0]
    assert isinstance(card.output_schema, dict)
    assert card.output_schema["start_year"]["type"] == "int"
    assert card.input_schema is not None
    assert card.input_schema["query"]["type"] == "str"


def test_registry_loads_cards_without_schemas_still_work(tmp_path: Path) -> None:
    _write(
        tmp_path / "wars.yaml",
        """
domain: wars
skills:
  - name: legacy
    description: a legacy card
    prompt_fragment: do something
""",
    )
    reg = SkillRegistry()
    reg.load_dir(tmp_path)
    cards = reg.for_domain("wars")
    assert len(cards) == 1
    card = cards[0]
    assert card.input_schema is None
    assert card.output_schema is None


def test_registry_cards_for_entity_type_matches_by_domain(tmp_path: Path) -> None:
    _write(
        tmp_path / "wars.yaml",
        """
domain: wars
skills:
  - name: war_year_validator
    description: ""
    prompt_fragment: ""
""",
    )
    reg = SkillRegistry()
    reg.load_dir(tmp_path)

    matched = reg.cards_for_entity_type("War")
    assert len(matched) == 1
    assert matched[0].name == "war_year_validator"


def test_registry_cards_for_entity_type_returns_empty_for_unknown(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "wars.yaml",
        """
domain: wars
skills:
  - name: war_year_validator
    description: ""
    prompt_fragment: ""
""",
    )
    reg = SkillRegistry()
    reg.load_dir(tmp_path)

    assert reg.cards_for_entity_type("Movie") == []
