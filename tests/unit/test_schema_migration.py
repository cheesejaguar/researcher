"""Tests for v1.2 schema evolution + migration."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from researcher.storage.duckdb_store import DuckDBKnowledgeStore
from researcher.storage.migrations import (
    SchemaMigrationError,
    compute_schema_hash,
    diff_schemas,
)


class WarV1(BaseModel):
    name: str
    start_year: int


class WarV2(BaseModel):
    """Additive change: adds end_year."""

    name: str
    start_year: int
    end_year: int | None = None


class WarV3(BaseModel):
    """Breaking change: renames start_year to began."""

    name: str
    began: int
    end_year: int | None = None


# ---------- compute_schema_hash ----------


def test_compute_schema_hash_deterministic():
    h1 = compute_schema_hash(WarV1)
    h2 = compute_schema_hash(WarV1)
    assert h1 == h2
    assert len(h1) > 0


def test_compute_schema_hash_changes_when_schema_changes():
    h1 = compute_schema_hash(WarV1)
    h2 = compute_schema_hash(WarV2)
    assert h1 != h2


# ---------- diff_schemas ----------


def test_diff_schemas_detects_no_change():
    diff = diff_schemas(WarV1, WarV1)
    assert diff.added == set()
    assert diff.removed == set()
    assert diff.is_additive is True
    assert diff.is_breaking is False


def test_diff_schemas_detects_additive_change():
    diff = diff_schemas(WarV1, WarV2)
    assert diff.added == {"end_year"}
    assert diff.removed == set()
    assert diff.is_additive is True
    assert diff.is_breaking is False


def test_diff_schemas_detects_breaking_change():
    diff = diff_schemas(WarV1, WarV3)
    # WarV3 removed start_year and added began.
    assert "start_year" in diff.removed
    assert "began" in diff.added
    # Breaking — fields were removed.
    assert diff.is_additive is False
    assert diff.is_breaking is True


# ---------- DuckDBKnowledgeStore.init_schema (new behavior) ----------


@pytest.mark.asyncio
async def test_init_schema_records_first_run(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(WarV1)
        # The schema_migrations table should have one row.
        rows = await store.query(
            "SELECT entity_class_name, schema_hash, migration_type "
            "FROM schema_migrations ORDER BY applied_at"
        )
        assert len(rows) == 1
        assert rows[0]["entity_class_name"] == "WarV1"
        assert rows[0]["migration_type"] == "initial"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_init_schema_with_same_schema_is_idempotent(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(WarV1)
        await store.init_schema(WarV1)  # second call, same schema
        rows = await store.query("SELECT entity_class_name FROM schema_migrations")
        # Still one row — no second migration recorded.
        assert len(rows) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_init_schema_records_additive_migration(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(WarV1)
        await store.init_schema(WarV2)  # add end_year
        rows = await store.query(
            "SELECT entity_class_name, migration_type "
            "FROM schema_migrations ORDER BY applied_at"
        )
        assert len(rows) == 2
        assert rows[0]["migration_type"] == "initial"
        assert rows[1]["migration_type"] == "additive"
        assert rows[1]["entity_class_name"] == "WarV2"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_init_schema_breaking_change_raises(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(WarV1)
        with pytest.raises(SchemaMigrationError):
            await store.init_schema(WarV3)  # removes start_year — breaking
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_init_schema_breaking_change_can_be_forced(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(WarV1)
        # force=True bypasses the breaking-change check.
        await store.init_schema(WarV3, force=True)
        rows = await store.query(
            "SELECT migration_type FROM schema_migrations ORDER BY applied_at"
        )
        assert len(rows) == 2
        assert rows[1]["migration_type"] == "breaking_forced"
    finally:
        await store.close()
