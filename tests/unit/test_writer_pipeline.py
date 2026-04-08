"""Tests for real FactWriter pipeline stages."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from researcher.models import FactClaim, Provenance
from researcher.storage.duckdb_store import DuckDBKnowledgeStore
from researcher.storage.store import FieldCell
from researcher.storage.writer import (
    FactWriter,
    dedup,
    detect_conflict,
    score_confidence,
    validate_type,
)


WAR_SCHEMA = {
    "entity_type": "War",
    "fields": [
        {"name": "name", "type": "str", "required": True},
        {"name": "start_year", "type": "int", "required": True},
        {"name": "end_year", "type": "int", "required": False},
        {"name": "belligerents", "type": "list[str]"},
        {"name": "decisive", "type": "bool"},
    ],
}


def _prov(url: str = "https://example.com", span: str = "s0") -> Provenance:
    return Provenance(
        url=url,
        fetched_at=datetime.now(timezone.utc),
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
    conf: float = 0.9,
    url: str = "https://a.com",
) -> FactClaim:
    return FactClaim(
        entity_type="War",
        entity_name=entity_name,
        field=field,
        value=value,
        confidence=conf,
        provenance=_prov(url=url),
        emitted_by="a1",
        task_id="t1",
    )


# ---------- validate_type ----------


@pytest.mark.asyncio
async def test_validate_type_passes_through_matching_type():
    claim = _claim("start_year", 1939)
    result = await validate_type(claim, WAR_SCHEMA)
    assert result.value == 1939


@pytest.mark.asyncio
async def test_validate_type_coerces_string_to_int():
    claim = _claim("start_year", "1939")
    result = await validate_type(claim, WAR_SCHEMA)
    assert result.value == 1939
    assert isinstance(result.value, int)


@pytest.mark.asyncio
async def test_validate_type_rejects_non_numeric_int():
    claim = _claim("start_year", "not a year")
    with pytest.raises(ValueError, match="type_error"):
        await validate_type(claim, WAR_SCHEMA)


@pytest.mark.asyncio
async def test_validate_type_coerces_list_of_strings():
    claim = _claim("belligerents", ["Allies", "Axis"])
    result = await validate_type(claim, WAR_SCHEMA)
    assert result.value == ["Allies", "Axis"]


@pytest.mark.asyncio
async def test_validate_type_wraps_scalar_into_single_list_for_list_field():
    claim = _claim("belligerents", "Allies")
    result = await validate_type(claim, WAR_SCHEMA)
    assert result.value == ["Allies"]


@pytest.mark.asyncio
async def test_validate_type_bool_from_string():
    claim = _claim("decisive", "true")
    result = await validate_type(claim, WAR_SCHEMA)
    assert result.value is True


@pytest.mark.asyncio
async def test_validate_type_unknown_field_passes_through():
    claim = _claim("not_in_schema", 42)
    result = await validate_type(claim, WAR_SCHEMA)
    assert result.value == 42  # no schema entry -> pass through


# ---------- dedup (no-op in v1-B) ----------


@pytest.mark.asyncio
async def test_dedup_returns_false(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        assert await dedup(_claim("start_year", 1939), store) is False
    finally:
        await store.close()


# ---------- detect_conflict ----------


@pytest.mark.asyncio
async def test_detect_conflict_empty_store_returns_no_conflict(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        result = await detect_conflict(_claim("start_year", 1939), store)
        assert result == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_detect_conflict_same_value_returns_empty(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        existing_cell = FieldCell(
            value=1939,
            confidence=0.9,
            provenance_ids=[],
            updated_at=datetime.now(timezone.utc),
        )
        await store.upsert_entity("War", "WWII", {"start_year": existing_cell})
        result = await detect_conflict(_claim("start_year", 1939), store)
        assert result == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_detect_conflict_different_value_returns_existing_cell(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        existing_cell = FieldCell(
            value=1939,
            confidence=0.9,
            provenance_ids=[],
            updated_at=datetime.now(timezone.utc),
        )
        await store.upsert_entity("War", "WWII", {"start_year": existing_cell})
        result = await detect_conflict(_claim("start_year", 1940), store)
        assert len(result) == 1
        assert result[0].value == 1939
    finally:
        await store.close()


# ---------- score_confidence ----------


@pytest.mark.asyncio
async def test_score_confidence_uses_weighted_blend():
    claim = _claim("start_year", 1939, conf=0.8)
    # default source_authority = 0.5; 0.7*0.8 + 0.3*0.5 = 0.71
    score = await score_confidence(claim, {})
    assert abs(score - 0.71) < 0.001


@pytest.mark.asyncio
async def test_score_confidence_with_high_source_authority():
    claim = _claim("start_year", 1939, conf=0.8)
    score = await score_confidence(claim, {"source_authority": 1.0})
    # 0.7*0.8 + 0.3*1.0 = 0.86
    assert abs(score - 0.86) < 0.001


# ---------- FactWriter._process integration ----------


@pytest.mark.asyncio
async def test_writer_persists_fact_to_store(tmp_path):
    from researcher.storage.resolver import (
        EntityResolver,
        ResolveDecision,
        ResolveResult,
    )

    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:

        class _StubResolver(EntityResolver):
            async def resolve(self, entity_type, name, context=None):
                return ResolveResult(
                    decision=ResolveDecision.AUTO_MERGE,
                    entity_id="stub",
                    similarity=1.0,
                    candidate_ids=[],
                )

            async def add_distinct_pair(self, a, b):
                return None

            async def is_blocked(self, a, b):
                return False

        emitted_facts: list[tuple[str, str]] = []

        async def emit_fact(eid, claim):
            emitted_facts.append((eid, claim.field))

        async def emit_conflict(eid, cells):
            pass

        writer = FactWriter(
            store=store,
            resolver=_StubResolver(),
            entity_schema=WAR_SCHEMA,
            emit_fact=emit_fact,
            emit_conflict=emit_conflict,
        )
        writer.start()
        try:
            await writer.submit(_claim("start_year", 1939))
            await writer.quiesce()
            rows = await store.query(
                "SELECT id FROM entities WHERE entity_type = ? AND name = ?",
                ("War", "WWII"),
            )
            assert len(rows) == 1
            eid = rows[0]["id"]
            ent = await store.get_entity(eid)
            assert ent is not None
            assert "start_year" in ent.fields
            assert ent.fields["start_year"].value == 1939
            assert writer.metrics["written"] == 1
            assert len(emitted_facts) == 1
        finally:
            await writer.drain()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_writer_records_conflict_when_value_differs(tmp_path):
    from researcher.storage.resolver import (
        EntityResolver,
        ResolveDecision,
        ResolveResult,
    )

    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        existing_cell = FieldCell(
            value=1939,
            confidence=0.9,
            provenance_ids=[],
            updated_at=datetime.now(timezone.utc),
        )
        await store.upsert_entity("War", "WWII", {"start_year": existing_cell})

        class _StubResolver(EntityResolver):
            async def resolve(self, entity_type, name, context=None):
                return ResolveResult(
                    decision=ResolveDecision.AUTO_MERGE,
                    entity_id="stub",
                    similarity=1.0,
                    candidate_ids=[],
                )

            async def add_distinct_pair(self, a, b):
                return None

            async def is_blocked(self, a, b):
                return False

        conflict_calls: list[tuple[str, list[FieldCell]]] = []

        async def emit_conflict(eid, cells):
            conflict_calls.append((eid, cells))

        async def emit_fact(eid, claim):
            pass

        writer = FactWriter(
            store=store,
            resolver=_StubResolver(),
            entity_schema=WAR_SCHEMA,
            emit_fact=emit_fact,
            emit_conflict=emit_conflict,
        )
        writer.start()
        try:
            await writer.submit(_claim("start_year", 1940))
            await writer.quiesce()
            assert writer.metrics["conflicts"] == 1
            conflicts = await store.get_conflicts(status="open")
            assert len(conflicts) == 1
        finally:
            await writer.drain()
    finally:
        await store.close()
