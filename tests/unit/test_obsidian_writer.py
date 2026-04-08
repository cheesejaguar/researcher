"""Tests for ObsidianWriter and its helpers."""

from __future__ import annotations

from researcher.integrations.obsidian import _safe_filename


# ---------- Filename sanitization ----------


def test_safe_filename_basic():
    assert _safe_filename("World War II") == "World War II"


def test_safe_filename_strips_path_separators():
    result = _safe_filename("../../evil")
    assert ".." not in result
    assert "/" not in result
    assert "\\" not in result


def test_safe_filename_strips_windows_reserved():
    result = _safe_filename('a:b*c?d"e<f>g|h')
    for ch in ':*?"<>|':
        assert ch not in result


def test_safe_filename_empty_falls_back_to_hash():
    result = _safe_filename("")
    assert result.startswith("entity_")
    assert len(result) > len("entity_")


def test_safe_filename_all_special_falls_back_to_hash():
    result = _safe_filename("///")
    assert result.startswith("entity_")


def test_safe_filename_truncates_long_names():
    long = "W" * 500
    result = _safe_filename(long)
    assert len(result) <= 200
    # Truncated names get a hash suffix to disambiguate.
    assert "_" in result[-10:]


def test_safe_filename_strips_leading_dots():
    result = _safe_filename(".hidden")
    assert not result.startswith(".")


# ---------- Markdown rendering ----------

from datetime import datetime, timezone

from researcher.integrations.obsidian import (
    EntityState,
    FieldValue,
    _render_markdown,
)
from researcher.models import Provenance


def _sample_provenance(url: str = "https://example.com", span: str = "cli_0") -> Provenance:
    return Provenance(
        url=url,
        fetched_at=datetime(2026, 4, 8, 14, 23, 0, tzinfo=timezone.utc),
        snippet="began in 1939",
        extractor_model="claude_code/unknown",
        agent_id="sub-a1b2c3d4",
        task_id="task-1",
        span_id=span,
    )


def _sample_state() -> EntityState:
    state = EntityState(
        entity_type="War",
        entity_name="World War II",
        updated_at=datetime(2026, 4, 8, 14, 23, 0, tzinfo=timezone.utc),
    )
    state.run_ids.add("run-2026-04-08-wars-1")
    state.fields["start_year"] = FieldValue(
        value=1939,
        confidence=0.97,
        provenances=[_sample_provenance(span="cli_0")],
    )
    state.fields["belligerents"] = FieldValue(
        value=["Allies", "Axis"],
        confidence=0.96,
        provenances=[_sample_provenance(span="cli_1")],
    )
    return state


def test_render_contains_frontmatter_delimiters():
    md = _render_markdown(_sample_state())
    assert md.startswith("---\n")
    # Frontmatter closes with a --- before the body.
    assert "\n---\n" in md[4:]


def test_render_frontmatter_has_researcher_keys():
    md = _render_markdown(_sample_state())
    assert "researcher_type: War" in md
    assert "researcher_name: World War II" in md
    assert "researcher_runs:" in md
    assert "run-2026-04-08-wars-1" in md
    assert "researcher_updated: 2026-04-08T14:23:00" in md


def test_render_frontmatter_has_tags():
    md = _render_markdown(_sample_state())
    assert "tags:" in md
    assert "- researcher" in md
    assert "- researcher/War" in md


def test_render_frontmatter_has_entity_fields():
    md = _render_markdown(_sample_state())
    assert "start_year: 1939" in md


def test_render_list_field_as_yaml_sequence():
    md = _render_markdown(_sample_state())
    assert "belligerents:" in md
    assert "- Allies" in md
    assert "- Axis" in md


def test_render_contains_managed_warning():
    md = _render_markdown(_sample_state())
    assert "Managed by" in md
    assert "researcher" in md


def test_render_contains_fields_table_with_source_link():
    md = _render_markdown(_sample_state())
    assert "| Field |" in md
    assert "| start_year |" in md
    assert "https://example.com" in md


def test_render_contains_provenance_section_with_snippet():
    md = _render_markdown(_sample_state())
    assert "## Provenance" in md
    assert "### start_year = 1939" in md
    assert "> began in 1939" in md
    assert "claude_code/unknown" in md


def test_render_contains_seen_in_runs_section():
    md = _render_markdown(_sample_state())
    assert "## Seen in Runs" in md
    assert "- `run-2026-04-08-wars-1`" in md
