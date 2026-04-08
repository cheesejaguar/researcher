"""Tests for FactWriter.quiesce() — per-cycle drain without shutdown."""

import asyncio
from datetime import UTC, datetime

import pytest

from researcher.models import FactClaim, Provenance
from researcher.storage.writer import FactWriter
from tests.stubs.resolver import StubEntityResolver
from tests.stubs.store import StubKnowledgeStore


def _make_claim(i: int) -> FactClaim:
    return FactClaim(
        entity_type="War",
        entity_name=f"War {i}",
        field="start_year",
        value=1900 + i,
        confidence=0.9,
        provenance=Provenance(
            url=f"https://example.com/{i}",
            fetched_at=datetime.now(UTC),
            snippet=f"snippet {i}",
            extractor_model="stub",
            agent_id="agent-1",
            task_id=f"task-{i}",
            span_id=f"cli_{i}",
        ),
        emitted_by="agent-1",
        task_id=f"task-{i}",
    )


async def _make_writer() -> FactWriter:
    store = StubKnowledgeStore()
    await store.open()
    resolver = StubEntityResolver()

    async def _noop_fact(_entity_id, _claim):
        return None

    async def _noop_conflict(_entity_id, _cells):
        return None

    entity_schema = {"entity_type": "War", "fields": []}
    return FactWriter(
        store=store,
        resolver=resolver,
        entity_schema=entity_schema,
        emit_fact=_noop_fact,
        emit_conflict=_noop_conflict,
    )


@pytest.mark.asyncio
async def test_quiesce_waits_for_all_submitted_claims():
    writer = await _make_writer()
    writer.start()
    try:
        for i in range(10):
            await writer.submit(_make_claim(i))
        await writer.quiesce()
        # After quiesce, the metrics counter must have observed all 10 claims
        # (even though the no-op pipeline doesn't actually write anything).
        assert writer.metrics["submitted"] == 10
    finally:
        await writer.drain()


@pytest.mark.asyncio
async def test_quiesce_does_not_stop_writer_task():
    """After quiesce, the writer task should still be running and accepting more claims."""
    writer = await _make_writer()
    task = writer.start()
    try:
        for i in range(5):
            await writer.submit(_make_claim(i))
        await writer.quiesce()
        assert not task.done(), "writer task should still be running after quiesce"

        # Submit more claims; they must still be processed.
        for i in range(5, 10):
            await writer.submit(_make_claim(i))
        await writer.quiesce()
        assert writer.metrics["submitted"] == 10
    finally:
        await writer.drain()
    assert task.done()


@pytest.mark.asyncio
async def test_quiesce_on_empty_queue_returns_immediately():
    writer = await _make_writer()
    writer.start()
    try:
        # No submits — quiesce should return without hanging.
        await asyncio.wait_for(writer.quiesce(), timeout=1.0)
    finally:
        await writer.drain()


@pytest.mark.asyncio
async def test_quiesce_timeout_raises():
    """If the writer task is blocked somehow, quiesce should raise TimeoutError."""
    writer = await _make_writer()
    # Don't call writer.start() — no drain coroutine is consuming the queue.
    await writer.submit(_make_claim(0))
    with pytest.raises(asyncio.TimeoutError):
        await writer.quiesce(timeout_s=0.1)
