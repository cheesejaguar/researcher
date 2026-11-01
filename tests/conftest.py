"""Shared pytest fixtures: real store, event bus and resolver on temp dirs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from researcher.storage.duckdb_store import DuckDBKnowledgeStore
from researcher.storage.resolver import DefaultEntityResolver
from tests.support import RecordingEventBus, make_resolver


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[DuckDBKnowledgeStore]:
    s = DuckDBKnowledgeStore(db_path=tmp_path / "fixture-store.duckdb")
    await s.open()
    try:
        yield s
    finally:
        await s.close()


@pytest.fixture
async def bus(tmp_path: Path) -> AsyncIterator[RecordingEventBus]:
    b = RecordingEventBus(tmp_path / "fixture-events.jsonl")
    await b.start()
    try:
        yield b
    finally:
        await b.stop()


@pytest.fixture
def resolver(store: DuckDBKnowledgeStore) -> DefaultEntityResolver:
    return make_resolver(store)
