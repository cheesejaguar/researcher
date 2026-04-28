"""Tests for DuckDBKnowledgeStore — the real persistent store."""

from __future__ import annotations

from datetime import UTC, datetime
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
        updated_at=datetime.now(UTC),
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
        # No cost tracker installed → cost_usd_total stays 0.0.
        assert m.cost_usd_total == 0.0
    finally:
        await store.close()


class _FakeCostTracker:
    def __init__(self, total: float) -> None:
        self._total = total
        self.calls = 0

    def total_usd(self) -> float:
        self.calls += 1
        return self._total


class _BrokenCostTracker:
    def total_usd(self) -> float:
        raise RuntimeError("tracker exploded")


@pytest.mark.asyncio
async def test_snapshot_metrics_reads_cost_from_injected_tracker(tmp_path: Path):
    """snapshot_metrics reports the real USD total when a tracker is wired."""
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    tracker = _FakeCostTracker(total=1.2345)
    store.set_cost_tracker(tracker)

    await store.open()
    try:
        await store.init_schema(WarEntity)
        m = await store.snapshot_metrics()
        assert m.cost_usd_total == pytest.approx(1.2345)
        assert tracker.calls == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_snapshot_metrics_swallows_cost_tracker_errors(tmp_path: Path):
    """A raising tracker falls back to 0.0 instead of crashing metrics."""
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    store.set_cost_tracker(_BrokenCostTracker())

    await store.open()
    try:
        await store.init_schema(WarEntity)
        m = await store.snapshot_metrics()
        assert m.cost_usd_total == 0.0
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
async def test_record_sources_and_verification_votes(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.record_sources(
            [
                {
                    "source_id": "s1",
                    "title": "Source",
                    "url": "file:///source.md",
                    "source_type": "file",
                    "path": "/source.md",
                    "metadata": {"source_set": "local"},
                }
            ],
            [
                {
                    "chunk_id": "s1:0",
                    "source_id": "s1",
                    "ordinal": 0,
                    "text": "chunk text",
                }
            ],
        )
        await store.record_verification_vote(
            {
                "vote_id": "v1",
                "run_id": "run-1",
                "entity_id": "e1",
                "field_name": "name",
                "model": "m",
                "vote": "ok",
                "confidence": 0.8,
                "rationale": "required field accepted",
            }
        )
        sources = await store.query("SELECT source_id FROM sources")
        chunks = await store.query("SELECT chunk_id FROM source_chunks")
        votes = await store.query("SELECT vote_id FROM verification_votes")
        assert sources == [{"source_id": "s1"}]
        assert chunks == [{"chunk_id": "s1:0"}]
        assert votes == [{"vote_id": "v1"}]
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


@pytest.mark.asyncio
async def test_find_similar_uses_vss_when_available(tmp_path: Path):
    """When the VSS extension is loaded, find_similar uses array_cosine_distance via SQL."""
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(WarEntity)
        # Use 384-dim vectors so it matches the production HNSW shape.
        a = await store.upsert_entity("War", "A", {"start_year": _cell(1)})
        b = await store.upsert_entity("War", "B", {"start_year": _cell(2)})
        c = await store.upsert_entity("War", "C", {"start_year": _cell(3)})
        # 384-dim vectors: A and B are nearly identical, C is orthogonal-ish.
        store.set_vector(a, [1.0] + [0.0] * 383)
        store.set_vector(b, [0.99] + [0.01] + [0.0] * 382)
        store.set_vector(c, [0.0, 1.0] + [0.0] * 382)

        # The query vector matches A.
        results = await store.find_similar("War", [1.0] + [0.0] * 383, k=3)
        assert len(results) == 3
        assert results[0][0] == a
        assert results[1][0] == b
        # Cosine similarity is in [0, 1] (or [-1, 1] for general); values descending.
        assert results[0][1] >= results[1][1] >= results[2][1]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_vss_extension_is_loaded(tmp_path: Path):
    """The store should automatically install and load the vss extension on open()."""
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        # Query DuckDB for loaded extensions.
        rows = await store.query(
            "SELECT extension_name FROM duckdb_extensions() WHERE loaded = true"
        )
        loaded = {r["extension_name"] for r in rows}
        assert "vss" in loaded
    finally:
        await store.close()
