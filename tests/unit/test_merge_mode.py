"""Tests for cross-run merge mode in FactWriter + DuckDBKnowledgeStore."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from researcher.models import FactClaim, Provenance
from researcher.spec import EntitySpec, FieldSpec, RunSpec
from researcher.storage.duckdb_store import DuckDBKnowledgeStore
from researcher.storage.store import FieldCell


def _prov(url: str = "https://example.com", span: str = "s0") -> Provenance:
    return Provenance(
        url=url,
        fetched_at=datetime.now(UTC),
        snippet="snippet",
        extractor_model="test",
        agent_id="a1",
        task_id="t1",
        span_id=span,
    )


def _claim(
    field: str,
    value,
    entity_name: str = "WWII",
    confidence: float = 0.9,
) -> FactClaim:
    return FactClaim(
        entity_type="War",
        entity_name=entity_name,
        field=field,
        value=value,
        confidence=confidence,
        provenance=_prov(),
        emitted_by="a1",
        task_id="t1",
    )


def _cell(value, conf: float = 0.9) -> FieldCell:
    return FieldCell(
        value=value,
        confidence=conf,
        provenance_ids=[],
        updated_at=datetime.now(UTC),
    )


# ---------- RunSpec ----------


def test_runspec_mode_default_is_overwrite():
    s = RunSpec(
        spec_id="x",
        goal="g",
        entities=[
            EntitySpec(
                name="War",
                fields=[FieldSpec(name="name", type="str", required=True)],
                search_templates=[],
            )
        ],
        seeds=["s"],
        models={"fast": "m"},
    )
    assert s.mode == "overwrite"


def test_runspec_mode_accepts_merge():
    s = RunSpec(
        spec_id="x",
        goal="g",
        entities=[
            EntitySpec(
                name="War",
                fields=[FieldSpec(name="name", type="str", required=True)],
                search_templates=[],
            )
        ],
        seeds=["s"],
        models={"fast": "m"},
        mode="merge",
    )
    assert s.mode == "merge"


# ---------- DuckDBKnowledgeStore.merge_field ----------


@pytest.mark.asyncio
async def test_merge_field_inserts_when_missing(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        # Seed an entity row.
        eid = await store.upsert_entity("War", "WWII", {"name": _cell("WWII")})
        result = await store.merge_field(
            entity_id=eid,
            field_name="start_year",
            value=1939,
            confidence=0.9,
            run_id="run-1",
            provenance_ids=[],
        )
        assert result["status"] == "inserted"
        # Verify the field is now in the entity.
        ent = await store.get_entity(eid)
        assert ent is not None
        assert "start_year" in ent.fields
        assert ent.fields["start_year"].value == 1939
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_merge_field_same_value_updates_last_seen(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        eid = await store.upsert_entity("War", "WWII", {"name": _cell("WWII")})
        await store.merge_field(eid, "start_year", 1939, 0.9, "run-1", [])
        result = await store.merge_field(eid, "start_year", 1939, 0.85, "run-2", [])
        assert result["status"] == "confirmed"
        # last_seen_run should now be run-2.
        rows = await store.query(
            "SELECT first_seen_run, last_seen_run FROM fields WHERE entity_id = ? AND field_name = ?",
            (eid, "start_year"),
        )
        assert len(rows) == 1
        assert rows[0]["first_seen_run"] == "run-1"
        assert rows[0]["last_seen_run"] == "run-2"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_merge_field_higher_confidence_wins(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        eid = await store.upsert_entity("War", "WWII", {"name": _cell("WWII")})
        await store.merge_field(eid, "start_year", 1939, 0.7, "run-1", [])
        result = await store.merge_field(eid, "start_year", 1940, 0.95, "run-2", [])
        assert result["status"] == "superseded"
        ent = await store.get_entity(eid)
        assert ent is not None
        assert ent.fields["start_year"].value == 1940
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_merge_field_lower_confidence_loses_but_creates_conflict(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        eid = await store.upsert_entity("War", "WWII", {"name": _cell("WWII")})
        await store.merge_field(eid, "start_year", 1939, 0.95, "run-1", [])
        result = await store.merge_field(eid, "start_year", 1940, 0.7, "run-2", [])
        assert result["status"] == "conflict_kept_existing"
        ent = await store.get_entity(eid)
        assert ent is not None
        # Existing winner stays.
        assert ent.fields["start_year"].value == 1939
    finally:
        await store.close()


# ---------- FactWriter merge mode ----------


@pytest.mark.asyncio
async def test_writer_in_merge_mode_accumulates_across_runs(tmp_path):
    """Two sequential runs over the same entity name should accumulate fields."""
    from researcher.storage.resolver import (
        EntityResolver,
        ResolveDecision,
        ResolveResult,
    )
    from researcher.storage.writer import FactWriter

    class _StubResolver(EntityResolver):
        async def resolve(self, entity_type, name, context=None):
            return ResolveResult(
                decision=ResolveDecision.AUTO_MERGE,
                entity_id="stub",
                similarity=1.0,
                candidate_ids=[],
            )
        async def add_distinct_pair(self, a, b): return None
        async def is_blocked(self, a, b): return False

    schema = {
        "entity_type": "War",
        "fields": [
            {"name": "name", "type": "str", "required": True},
            {"name": "start_year", "type": "int"},
            {"name": "end_year", "type": "int"},
        ],
    }

    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        from researcher.spec import build_entity_class

        class War:
            pass  # placeholder; we'll use a Pydantic class via build_entity_class

        from researcher.spec import EntitySpec, FieldSpec
        e_spec = EntitySpec(
            name="War",
            fields=[
                FieldSpec(name="name", type="str", required=True),
                FieldSpec(name="start_year", type="int"),
                FieldSpec(name="end_year", type="int"),
            ],
            search_templates=[],
        )
        WarCls = build_entity_class(e_spec)
        await store.init_schema(WarCls)

        async def _emit_fact(_eid, _claim): return None
        async def _emit_conflict(_eid, _cells): return None

        # Run 1: add start_year.
        writer1 = FactWriter(
            store=store,
            resolver=_StubResolver(),
            entity_schema=schema,
            emit_fact=_emit_fact,
            emit_conflict=_emit_conflict,
            mode="merge",
            run_id="run-1",
        )
        writer1.start()
        try:
            await writer1.submit(_claim("start_year", 1939))
            await writer1.quiesce()
        finally:
            await writer1.drain()

        # Run 2: add end_year (different field on same entity).
        writer2 = FactWriter(
            store=store,
            resolver=_StubResolver(),
            entity_schema=schema,
            emit_fact=_emit_fact,
            emit_conflict=_emit_conflict,
            mode="merge",
            run_id="run-2",
        )
        writer2.start()
        try:
            await writer2.submit(_claim("end_year", 1945))
            await writer2.quiesce()
        finally:
            await writer2.drain()

        # The entity should now have BOTH fields.
        rows = await store.query(
            "SELECT id FROM entities WHERE entity_type = ? AND name = ?",
            ("War", "WWII"),
        )
        assert len(rows) == 1
        ent = await store.get_entity(rows[0]["id"])
        assert ent is not None
        assert "start_year" in ent.fields
        assert "end_year" in ent.fields
        assert ent.fields["start_year"].value == 1939
        assert ent.fields["end_year"].value == 1945
    finally:
        await store.close()
