"""Tests for entity_relations table + DuckPGQ graph queries."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from researcher.spec import EntitySpec, FieldSpec, RelationSpec, RunSpec
from researcher.storage.duckdb_store import DuckDBKnowledgeStore
from researcher.storage.store import FieldCell


def _cell(v) -> FieldCell:
    return FieldCell(
        value=v,
        confidence=0.9,
        provenance_ids=[],
        updated_at=datetime.now(UTC),
    )


# ---------- RunSpec / RelationSpec ----------


def test_relation_spec_basic_construction():
    r = RelationSpec(
        name="ally_of",
        source_type="War",
        target_type="War",
    )
    assert r.name == "ally_of"
    assert r.source_type == "War"
    assert r.target_type == "War"


def test_runspec_relations_default_empty():
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
    assert s.relations == []


def test_runspec_relations_accepts_list():
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
        relations=[
            RelationSpec(name="ally_of", source_type="War", target_type="War"),
            RelationSpec(name="took_place_in", source_type="War", target_type="Region"),
        ],
    )
    assert len(s.relations) == 2
    assert s.relations[0].name == "ally_of"


# ---------- record_relation ----------


@pytest.mark.asyncio
async def test_record_relation_inserts_row(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        a = await store.upsert_entity("War", "WW1", {"name": _cell("WW1")})
        b = await store.upsert_entity("War", "WW2", {"name": _cell("WW2")})
        await store.record_relation(
            source_id=a,
            target_id=b,
            relation_label="precedes",
            confidence=0.95,
            run_id="run-1",
        )
        rows = await store.query(
            "SELECT source_id, target_id, relation_label, confidence FROM entity_relations"
        )
        assert len(rows) == 1
        assert rows[0]["source_id"] == a
        assert rows[0]["target_id"] == b
        assert rows[0]["relation_label"] == "precedes"
        assert rows[0]["confidence"] == 0.95
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_record_relation_dedupes_same_triple(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        a = await store.upsert_entity("War", "WW1", {"name": _cell("WW1")})
        b = await store.upsert_entity("War", "WW2", {"name": _cell("WW2")})
        await store.record_relation(a, b, "precedes", 0.9, "run-1")
        await store.record_relation(a, b, "precedes", 0.95, "run-2")  # higher confidence
        rows = await store.query("SELECT confidence FROM entity_relations")
        assert len(rows) == 1
        # Should keep the higher-confidence value.
        assert rows[0]["confidence"] == 0.95
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_record_relation_distinct_labels_coexist(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        a = await store.upsert_entity("War", "WW1", {"name": _cell("WW1")})
        b = await store.upsert_entity("War", "WW2", {"name": _cell("WW2")})
        await store.record_relation(a, b, "precedes", 0.9, "run-1")
        await store.record_relation(a, b, "related_to", 0.8, "run-1")
        rows = await store.query(
            "SELECT relation_label FROM entity_relations ORDER BY relation_label"
        )
        assert [r["relation_label"] for r in rows] == ["precedes", "related_to"]
    finally:
        await store.close()


# ---------- query_graph (DuckPGQ if available, otherwise SQL fallback) ----------


@pytest.mark.asyncio
async def test_query_graph_returns_neighbors(tmp_path):
    """A simple traversal: find all entities reachable from WW1 via 'precedes'."""
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        a = await store.upsert_entity("War", "WW1", {"name": _cell("WW1")})
        b = await store.upsert_entity("War", "WW2", {"name": _cell("WW2")})
        c = await store.upsert_entity("War", "WW3", {"name": _cell("WW3")})
        await store.record_relation(a, b, "precedes", 0.9, "run-1")
        await store.record_relation(b, c, "precedes", 0.9, "run-1")

        results = await store.find_neighbors(a, relation_label="precedes")
        # WW1 -> WW2 directly
        assert len(results) == 1
        assert results[0]["target_id"] == b
    finally:
        await store.close()
