"""Tests for DuckDBKnowledgeStore — the real persistent store."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel

from researcher.storage.duckdb_store import DuckDBKnowledgeStore
from researcher.storage.store import Conflict, FieldCell


class WarEntity(BaseModel):
    name: str
    start_year: int
    end_year: int | None = None


def _cell(value, conf: float = 0.9) -> FieldCell:
    return FieldCell(
        value=value,
        confidence=conf,
        provenance_ids=[],
        updated_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_open_close_creates_db_file(tmp_path: Path):
    db_path = tmp_path / "store.duckdb"
    store = DuckDBKnowledgeStore(db_path=db_path)
    await store.open()
    try:
        assert db_path.exists()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_init_schema_accepts_pydantic_class(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(WarEntity)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_upsert_and_get_entity_roundtrip(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(WarEntity)
        eid = await store.upsert_entity(
            entity_type="War",
            name="WWII",
            fields={"start_year": _cell(1939), "end_year": _cell(1945)},
        )
        assert eid
        ent = await store.get_entity(eid)
        assert ent is not None
        assert ent.type == "War"
        assert "start_year" in ent.fields
        assert ent.fields["start_year"].value == 1939
        assert ent.fields["end_year"].value == 1945
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_upsert_same_name_merges(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(WarEntity)
        eid1 = await store.upsert_entity("War", "WWII", {"start_year": _cell(1939)})
        eid2 = await store.upsert_entity("War", "WWII", {"end_year": _cell(1945)})
        assert eid1 == eid2  # same name → same id
        ent = await store.get_entity(eid1)
        assert ent is not None
        assert "start_year" in ent.fields
        assert "end_year" in ent.fields
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_snapshot_metrics_reports_counts(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(WarEntity)
        await store.upsert_entity("War", "WWII", {"start_year": _cell(1939)})
        await store.upsert_entity("War", "WWI", {"start_year": _cell(1914)})
        await store.upsert_entity("Trial", "NCT1", {"phase": _cell(3)})
        m = await store.snapshot_metrics()
        assert m.entities_total == 3
        assert m.by_type["War"] == 2
        assert m.by_type["Trial"] == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_find_similar_returns_knn_by_cosine(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(WarEntity)
        eid_a = await store.upsert_entity("War", "A", {"start_year": _cell(1)})
        eid_b = await store.upsert_entity("War", "B", {"start_year": _cell(2)})
        eid_c = await store.upsert_entity("War", "C", {"start_year": _cell(3)})
        store.set_vector(eid_a, [1.0, 0.0, 0.0])
        store.set_vector(eid_b, [0.9, 0.1, 0.0])  # very similar to A
        store.set_vector(eid_c, [0.0, 1.0, 0.0])  # orthogonal
        results = await store.find_similar("War", [1.0, 0.0, 0.0], k=3)
        # Should rank by cosine descending: A (1.0), B (~0.994), C (0.0)
        ids = [r[0] for r in results]
        assert ids[0] == eid_a
        assert ids[1] == eid_b
        assert ids[2] == eid_c
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_find_similar_filters_by_entity_type(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(WarEntity)
        w_id = await store.upsert_entity("War", "W", {"start_year": _cell(1)})
        t_id = await store.upsert_entity("Trial", "T", {"phase": _cell(1)})
        store.set_vector(w_id, [1.0, 0.0])
        store.set_vector(t_id, [1.0, 0.0])
        war_results = await store.find_similar("War", [1.0, 0.0], k=5)
        trial_results = await store.find_similar("Trial", [1.0, 0.0], k=5)
        assert [r[0] for r in war_results] == [w_id]
        assert [r[0] for r in trial_results] == [t_id]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_record_and_get_provenance(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.record_provenance(
            "prov-1",
            {
                "url": "https://example.com",
                "agent_id": "a1",
                "snippet": "hello",
            },
        )
        # No direct get_provenance in the ABC; verify via query if implemented,
        # or just confirm it didn't raise.
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_record_and_get_conflicts(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        c = Conflict(
            conflict_id="c-1",
            entity_id="ent-1",
            field="start_year",
            candidates=[_cell(1939), _cell(1940)],
            status="open",
        )
        await store.record_conflict(c)
        open_conflicts = await store.get_conflicts(status="open")
        assert len(open_conflicts) == 1
        assert open_conflicts[0].conflict_id == "c-1"
        resolved = await store.get_conflicts(status="resolved")
        assert resolved == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_write_run_summary(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.write_run_summary(
            run_id="run-1",
            summary={"reason": "plateau", "entities": 10, "cost_usd": 1.42},
        )
        # Verify via query — at minimum the row should exist.
        rows = await store.query("SELECT run_id FROM run_summary WHERE run_id = ?", ("run-1",))
        assert len(rows) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_async_context_manager(tmp_path: Path):
    db_path = tmp_path / "s.duckdb"
    async with DuckDBKnowledgeStore(db_path=db_path) as store:
        await store.init_schema(WarEntity)
        eid = await store.upsert_entity("War", "WWII", {"start_year": _cell(1939)})
        assert eid
    # After the context exits, the file should still be there
    assert db_path.exists()
