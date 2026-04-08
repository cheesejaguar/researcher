"""Tests for SkillRegistry — loads hand-authored skill cards from skills/*.yaml."""

from __future__ import annotations

from pathlib import Path

from researcher.skills.registry import SkillCard, SkillRegistry


def _write(path: Path, text: str) -> None:
    path.write_text(text)


def test_loads_valid_yaml_file(tmp_path: Path) -> None:
    _write(
        tmp_path / "wars.yaml",
        """
domain: wars
skills:
  - name: wikipedia_infobox_first
    description: |
      Check infobox first.
    prompt_fragment: |
      For War entities, check the infobox first.
  - name: prefer_conclusion
    description: Use conclusion section for outcomes.
    prompt_fragment: Prefer conclusion phrasing.
""",
    )

    reg = SkillRegistry()
    reg.load_dir(tmp_path)

    cards = reg.for_domain("wars")
    assert len(cards) == 2
    assert all(isinstance(c, SkillCard) for c in cards)
    names = {c.name for c in cards}
    assert names == {"wikipedia_infobox_first", "prefer_conclusion"}
    # First card has a non-empty prompt fragment.
    first = next(c for c in cards if c.name == "wikipedia_infobox_first")
    assert "infobox" in first.prompt_fragment
    assert first.domain == "wars"


def test_skips_underscore_prefixed_files(tmp_path: Path) -> None:
    _write(
        tmp_path / "_suggestions.yaml",
        """
domain: suggestions
skills:
  - name: ignored
    description: should not load
    prompt_fragment: ignore
""",
    )
    _write(
        tmp_path / "real.yaml",
        """
domain: real
skills:
  - name: kept
    description: ok
    prompt_fragment: ok
""",
    )

    reg = SkillRegistry()
    reg.load_dir(tmp_path)

    assert reg.for_domain("suggestions") == []
    kept = reg.for_domain("real")
    assert len(kept) == 1
    assert kept[0].name == "kept"


def test_handles_malformed_yaml_gracefully(tmp_path: Path) -> None:
    _write(tmp_path / "bad.yaml", "this is: : : not valid: yaml: [[[")
    _write(
        tmp_path / "good.yaml",
        """
domain: good
skills:
  - name: ok
    description: ok
    prompt_fragment: ok
""",
    )

    reg = SkillRegistry()
    reg.load_dir(tmp_path)  # should not raise

    assert reg.for_domain("good") != []


def test_returns_empty_for_unknown_domain(tmp_path: Path) -> None:
    reg = SkillRegistry()
    reg.load_dir(tmp_path)  # empty dir
    assert reg.for_domain("nonexistent") == []
    assert reg.all() == []


def test_load_dir_missing_directory_is_noop(tmp_path: Path) -> None:
    reg = SkillRegistry()
    reg.load_dir(tmp_path / "does_not_exist")
    assert reg.all() == []
