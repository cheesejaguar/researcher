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

from datetime import UTC, datetime

from researcher.integrations.obsidian import (
    EntityState,
    FieldValue,
    _render_markdown,
)
from researcher.models import Provenance


def _sample_provenance(url: str = "https://example.com", span: str = "cli_0") -> Provenance:
    return Provenance(
        url=url,
        fetched_at=datetime(2026, 4, 8, 14, 23, 0, tzinfo=UTC),
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
        updated_at=datetime(2026, 4, 8, 14, 23, 0, tzinfo=UTC),
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


# ---------- Lifecycle ----------

from pathlib import Path

import pytest

from researcher.integrations.obsidian import ObsidianVaultNotFound, ObsidianWriter


@pytest.mark.asyncio
async def test_start_creates_vault_subdir(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        assert (tmp_path / "researcher").is_dir()
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_start_raises_on_missing_vault_root(tmp_path: Path):
    bogus = tmp_path / "does-not-exist" / "vault"
    writer = ObsidianWriter(vault_path=bogus)
    with pytest.raises(ObsidianVaultNotFound):
        await writer.start()


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    await writer.start()  # should not raise
    try:
        assert (tmp_path / "researcher").is_dir()
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_stop_is_idempotent(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    await writer.stop()
    await writer.stop()  # should not raise


@pytest.mark.asyncio
async def test_stats_initially_zero(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    assert writer.stats == {
        "writes": 0,
        "coalesced_claims": 0,
        "errors": 0,
        "sanitized_names": 0,
    }


# ---------- on_fact + coalescing ----------

from typing import Any as _Any

from researcher.models import FactClaim


def _sample_claim(
    field_name: str = "start_year",
    value: _Any = 1939,
    confidence: float = 0.9,
    url: str = "https://example.com/wwii",
    span: str = "cli_0",
    entity_type: str = "War",
    entity_name: str = "World War II",
) -> FactClaim:
    return FactClaim(
        entity_type=entity_type,
        entity_name=entity_name,
        field=field_name,
        value=value,
        confidence=confidence,
        provenance=Provenance(
            url=url,
            fetched_at=datetime.now(UTC),
            snippet=f"{field_name}={value}",
            extractor_model="claude_code/unknown",
            agent_id="sub-1",
            task_id="t-1",
            span_id=span,
        ),
        emitted_by="sub-1",
        task_id="t-1",
    )


@pytest.mark.asyncio
async def test_on_fact_coalesces_into_pending_buffer(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(), run_id="run-1")
        assert len(writer._pending) == 1
        key = ("War", "World War II")
        assert key in writer._pending
        state = writer._pending[key]
        assert "start_year" in state.fields
        assert state.fields["start_year"].value == 1939
        assert "run-1" in state.run_ids
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_on_fact_merges_multiple_fields_for_same_entity(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(
            _sample_claim(field_name="start_year", value=1939, span="cli_0"),
            run_id="run-1",
        )
        await writer.on_fact(
            _sample_claim(field_name="end_year", value=1945, span="cli_1"),
            run_id="run-1",
        )
        await writer.on_fact(
            _sample_claim(field_name="belligerents", value=["Allies", "Axis"], span="cli_2"),
            run_id="run-1",
        )
        assert len(writer._pending) == 1
        state = list(writer._pending.values())[0]
        assert set(state.fields.keys()) == {"start_year", "end_year", "belligerents"}
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_on_fact_higher_confidence_wins(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(
            _sample_claim(value=1939, confidence=0.8, span="cli_0"), run_id="run-1"
        )
        await writer.on_fact(
            _sample_claim(value=1940, confidence=0.95, span="cli_1"), run_id="run-1"
        )
        state = writer._pending[("War", "World War II")]
        fv = state.fields["start_year"]
        assert fv.value == 1940
        assert fv.confidence == 0.95
        assert len(fv.provenances) == 2
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_on_fact_dedupes_provenance_by_url_and_span(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(
            _sample_claim(url="https://a.com", span="cli_0"), run_id="run-1"
        )
        await writer.on_fact(
            _sample_claim(url="https://a.com", span="cli_0"), run_id="run-1"
        )
        fv = writer._pending[("War", "World War II")].fields["start_year"]
        assert len(fv.provenances) == 1
        await writer.on_fact(
            _sample_claim(url="https://a.com", span="cli_1"), run_id="run-1"
        )
        assert len(fv.provenances) == 2
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_on_fact_separates_entity_types(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(entity_type="War"), run_id="run-1")
        await writer.on_fact(
            _sample_claim(entity_type="Trial", entity_name="NCT12345"), run_id="run-1"
        )
        assert ("War", "World War II") in writer._pending
        assert ("Trial", "NCT12345") in writer._pending
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_on_fact_after_stop_raises(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    await writer.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        await writer.on_fact(_sample_claim(), run_id="run-1")


@pytest.mark.asyncio
async def test_on_fact_marks_entity_dirty(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(), run_id="run-1")
        assert ("War", "World War II") in writer._dirty
    finally:
        await writer.stop()


# ---------- Flush + debounce + disk writes ----------

import asyncio


@pytest.mark.asyncio
async def test_flush_writes_entity_file_to_disk(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(), run_id="run-1")
        await writer.flush()
        expected = tmp_path / "researcher" / "War" / "World War II.md"
        assert expected.exists()
        content = expected.read_text()
        assert "researcher_type: War" in content
        assert "start_year: 1939" in content
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_flush_clears_dirty_set(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(), run_id="run-1")
        assert writer._dirty
        await writer.flush()
        assert not writer._dirty
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_flush_is_idempotent_when_nothing_dirty(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.flush()
        await writer.flush()
        assert writer.stats["writes"] == 0
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_flush_increments_write_count(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(entity_name="A"), run_id="run-1")
        await writer.on_fact(_sample_claim(entity_name="B"), run_id="run-1")
        await writer.flush()
        assert writer.stats["writes"] == 2
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_debounce_batches_bursts(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path, flush_interval_s=0.05)
    await writer.start()
    try:
        for i in range(10):
            await writer.on_fact(
                _sample_claim(field_name=f"f{i}", value=i, span=f"cli_{i}"),
                run_id="run-1",
            )
        await asyncio.sleep(0.15)
        # 10 claims for the same entity → 1 write.
        assert writer.stats["writes"] == 1
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_organizes_by_entity_type_subfolder(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(
            _sample_claim(entity_type="War", entity_name="WWII"), run_id="run-1"
        )
        await writer.on_fact(
            _sample_claim(
                entity_type="Trial",
                entity_name="NCT12345",
                field_name="phase",
                value=3,
            ),
            run_id="run-1",
        )
        await writer.flush()
        assert (tmp_path / "researcher" / "War" / "WWII.md").exists()
        assert (tmp_path / "researcher" / "Trial" / "NCT12345.md").exists()
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_stop_performs_final_flush(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    await writer.on_fact(_sample_claim(), run_id="run-1")
    await writer.stop()
    expected = tmp_path / "researcher" / "War" / "World War II.md"
    assert expected.exists()


@pytest.mark.asyncio
async def test_rerun_overwrites_existing_file(tmp_path: Path):
    writer1 = ObsidianWriter(vault_path=tmp_path)
    await writer1.start()
    await writer1.on_fact(_sample_claim(value=1939), run_id="run-1")
    await writer1.stop()
    file_path = tmp_path / "researcher" / "War" / "World War II.md"
    first_content = file_path.read_text()
    assert "start_year: 1939" in first_content

    writer2 = ObsidianWriter(vault_path=tmp_path)
    await writer2.start()
    await writer2.on_fact(_sample_claim(value=1914), run_id="run-2")
    await writer2.stop()
    second_content = file_path.read_text()
    assert "start_year: 1914" in second_content
    assert "1939" not in second_content


# ---------- Error paths (regression coverage) ----------

from unittest.mock import patch


@pytest.mark.asyncio
async def test_write_failure_does_not_raise_from_flush(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(), run_id="run-1")

        def bad_replace(src, dst):
            raise OSError("simulated disk full")

        with patch("os.replace", side_effect=bad_replace):
            await writer.flush()  # must NOT raise

        assert writer.stats["errors"] >= 1
        assert writer.stats["writes"] == 0
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_tempfile_cleaned_up_on_write_failure(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(), run_id="run-1")

        def bad_replace(src, dst):
            raise OSError("simulated failure")

        with patch("os.replace", side_effect=bad_replace):
            await writer.flush()

        target_dir = tmp_path / "researcher" / "War"
        if target_dir.exists():
            tmps = list(target_dir.glob("*.tmp"))
            assert tmps == [], f"leaked tempfiles: {tmps}"
            hidden_tmps = list(target_dir.glob(".*.tmp"))
            assert hidden_tmps == [], f"leaked hidden tempfiles: {hidden_tmps}"
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_stats_counts_coalesced_claims(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(field_name="start_year"), run_id="run-1")
        await writer.on_fact(
            _sample_claim(field_name="start_year", value=1940, span="cli_1"),
            run_id="run-1",
        )
        assert writer.stats["coalesced_claims"] == 1
    finally:
        await writer.stop()


# ---------- Dataview inline fields ----------


@pytest.mark.asyncio
async def test_render_emits_dataview_inline_fields(tmp_path: Path):
    """Each spec field should appear as both YAML frontmatter AND inline [field:: value]."""
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(field_name="start_year", value=1939), run_id="r1")
        await writer.on_fact(_sample_claim(field_name="end_year", value=1945, span="cli_1"), run_id="r1")
        await writer.flush()
    finally:
        await writer.stop()

    md = (tmp_path / "researcher" / "War" / "World War II.md").read_text()
    # Frontmatter (already present)
    assert "start_year: 1939" in md
    # Inline fields (new — Dataview-scrapable from body)
    assert "[start_year:: 1939]" in md
    assert "[end_year:: 1945]" in md


# ---------- Entity-type index note ----------


@pytest.mark.asyncio
async def test_writer_generates_index_note_per_entity_type(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(entity_type="War", entity_name="WWII"), run_id="r1")
        await writer.on_fact(
            _sample_claim(entity_type="Trial", entity_name="NCT12345", field_name="phase", value=3),
            run_id="r1",
        )
        await writer.flush()
        await writer.write_index_notes()
    finally:
        await writer.stop()

    war_index = tmp_path / "researcher" / "War" / "_index.md"
    trial_index = tmp_path / "researcher" / "Trial" / "_index.md"
    assert war_index.exists()
    assert trial_index.exists()

    war_content = war_index.read_text()
    assert "```dataview" in war_content
    assert "TABLE" in war_content
    # Should mention the entity type folder for the FROM clause
    assert "researcher/War" in war_content


# ---------- Wikilink injection ----------


@pytest.mark.asyncio
async def test_inject_wikilinks_wraps_known_entity_names(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        # Create two entities. The first mentions the second's name in a snippet.
        await writer.on_fact(
            FactClaim(
                entity_type="War",
                entity_name="World War II",
                field="related",
                value="World War I was a precursor",
                confidence=0.9,
                provenance=Provenance(
                    url="https://example.com",
                    fetched_at=datetime.now(UTC),
                    snippet="World War I was a precursor",
                    extractor_model="test",
                    agent_id="a1",
                    task_id="t1",
                    span_id="s0",
                ),
                emitted_by="a1",
                task_id="t1",
            ),
            run_id="r1",
        )
        await writer.on_fact(
            _sample_claim(entity_type="War", entity_name="World War I", field_name="start_year", value=1914),
            run_id="r1",
        )
        await writer.flush()
        # Now inject wikilinks: pass the set of known entity names.
        await writer.inject_wikilinks(known_names={"World War II", "World War I"})
    finally:
        await writer.stop()

    wwii_md = (tmp_path / "researcher" / "War" / "World War II.md").read_text()
    # The body should now contain a wikilink to World War I (in the snippet quote).
    assert "[[World War I]]" in wwii_md


@pytest.mark.asyncio
async def test_inject_wikilinks_does_not_wrap_self(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(entity_name="WWII"), run_id="r1")
        await writer.flush()
        await writer.inject_wikilinks(known_names={"WWII"})
    finally:
        await writer.stop()

    md = (tmp_path / "researcher" / "War" / "WWII.md").read_text()
    # Self-references should NOT become wikilinks.
    assert "[[WWII]]" not in md or md.count("[[WWII]]") == 0


# ---------- See Also section ----------


@pytest.mark.asyncio
async def test_see_also_section_added_when_neighbors_provided(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(entity_name="WWII"), run_id="r1")
        await writer.flush()
        # Inject neighbor mapping: WWII → [WWI, Korean War]
        await writer.write_see_also(
            neighbors={
                ("War", "WWII"): [("War", "WWI"), ("War", "Korean War")],
            }
        )
    finally:
        await writer.stop()

    md = (tmp_path / "researcher" / "War" / "WWII.md").read_text()
    assert "## See Also" in md
    assert "[[WWI]]" in md
    assert "[[Korean War]]" in md
