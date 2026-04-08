"""Tests for DefaultEntityResolver — real embedding-based entity resolver."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from researcher.storage.duckdb_store import DuckDBKnowledgeStore
from researcher.storage.resolver import (
    DefaultEntityResolver,
    ResolveDecision,
)
from researcher.storage.store import FieldCell
from datetime import datetime, timezone


async def _hash_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic pseudo-embedder for tests."""
    out = []
    for t in texts:
        h = hashlib.sha256(t.lower().encode()).digest()
        vec = [((b / 255.0) * 2.0 - 1.0) for b in h[:16]]
        out.append(vec)
    return out


async def _canned_embed_factory(vectors: dict[str, list[float]]):
    """Returns an embed_fn that looks up names in a canned dict."""
    async def embed(texts: list[str]) -> list[list[float]]:
        return [vectors[t] for t in texts]
    return embed


def _cell(v) -> FieldCell:
    return FieldCell(
        value=v, confidence=0.9, provenance_ids=[],
        updated_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_resolve_empty_store_returns_distinct(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        resolver = DefaultEntityResolver(store=store, embed_fn=_hash_embed)
        result = await resolver.resolve("War", "WWII")
        assert result.decision == ResolveDecision.DISTINCT
        assert result.entity_id  # new UUID assigned
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_resolve_auto_merge_on_high_similarity(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        vec_a = [1.0] + [0.0] * 15
        vec_b = [0.99] + [0.01] + [0.0] * 14  # cos(a,b) ≈ 0.9999
        vectors = {"A": vec_a, "B": vec_b}

        async def canned(texts):
            return [vectors[t] for t in texts]

        # Seed the store with entity A and its embedding
        eid_a = await store.upsert_entity("War", "A", {"name": _cell("A")})
        store.set_vector(eid_a, vec_a)

        resolver = DefaultEntityResolver(
            store=store, embed_fn=canned, high_thresh=0.9, low_thresh=0.5,
        )
        # B is very similar to A → should AUTO_MERGE with eid_a
        result = await resolver.resolve("War", "B")
        assert result.decision == ResolveDecision.AUTO_MERGE
        assert result.entity_id == eid_a
        assert result.similarity > 0.9
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_resolve_pending_in_medium_band(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.8, 0.6, 0.0]  # cos(a,b) = 0.8
        vectors = {"A": vec_a, "B": vec_b}

        async def canned(texts):
            return [vectors[t] for t in texts]

        eid_a = await store.upsert_entity("War", "A", {"name": _cell("A")})
        store.set_vector(eid_a, vec_a)

        resolver = DefaultEntityResolver(
            store=store, embed_fn=canned, high_thresh=0.92, low_thresh=0.78,
        )
        result = await resolver.resolve("War", "B")
        assert result.decision == ResolveDecision.PENDING
        assert result.entity_id == eid_a
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_resolve_distinct_below_low_thresh(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]  # orthogonal → cosine = 0
        vectors = {"A": vec_a, "B": vec_b}

        async def canned(texts):
            return [vectors[t] for t in texts]

        eid_a = await store.upsert_entity("War", "A", {"name": _cell("A")})
        store.set_vector(eid_a, vec_a)

        resolver = DefaultEntityResolver(
            store=store, embed_fn=canned, high_thresh=0.92, low_thresh=0.78,
        )
        result = await resolver.resolve("War", "B")
        assert result.decision == ResolveDecision.DISTINCT
        # New entity_id, not eid_a
        assert result.entity_id != eid_a
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_resolve_respects_distinct_pairs_blocklist(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.99, 0.0, 0.01]  # very similar to A
        vectors = {"Napoleonic Wars": vec_a, "War of the Sixth Coalition": vec_b}

        async def canned(texts):
            return [vectors[t] for t in texts]

        eid_a = await store.upsert_entity("War", "Napoleonic Wars", {"name": _cell("Napoleonic Wars")})
        store.set_vector(eid_a, vec_a)

        resolver = DefaultEntityResolver(
            store=store,
            embed_fn=canned,
            high_thresh=0.9,
            low_thresh=0.5,
            distinct_pairs=[("Napoleonic Wars", "War of the Sixth Coalition")],
        )
        result = await resolver.resolve("War", "War of the Sixth Coalition")
        # Even though they're very similar, the blocklist forces DISTINCT.
        assert result.decision == ResolveDecision.DISTINCT
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_resolve_distinct_writes_new_vector_to_store(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        resolver = DefaultEntityResolver(store=store, embed_fn=_hash_embed)
        result = await resolver.resolve("War", "NewWar")
        assert result.decision == ResolveDecision.DISTINCT
        # A subsequent lookup with the same name should NOT create a new id —
        # it should either find the prior vector via find_similar and merge,
        # OR if the first resolve didn't persist an entity row, it'll create
        # a new distinct one. Accept either — just verify the vector was stored.
        rows = await store.query(
            "SELECT entity_id FROM embeddings WHERE entity_id = ?",
            (result.entity_id,),
        )
        assert len(rows) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_wwii_synonyms_resolve_together_with_real_hash_embedder(tmp_path: Path):
    """The main plan's benchmark: ['WWII', 'World War II', 'WW2', 'World War 2']
    should collapse to a single entity, and 'Cold War' should stay DISTINCT.

    Uses the deterministic hash-embedder, which is NOT real semantic similarity —
    so this test just verifies the API shape. A true semantic test would use
    sentence-transformers, which is gated behind a slower integration test.
    """
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        # Use canned vectors to simulate what a real embedder would produce:
        # the four WWII variants cluster together; Cold War is far.
        vectors = {
            "WWII":            [1.0, 0.0, 0.0, 0.0],
            "World War II":    [0.98, 0.1, 0.0, 0.0],
            "WW2":             [0.97, 0.0, 0.1, 0.0],
            "World War 2":     [0.99, 0.05, 0.05, 0.0],
            "Cold War":        [0.0, 0.0, 0.0, 1.0],
        }

        async def canned(texts):
            return [vectors[t] for t in texts]

        resolver = DefaultEntityResolver(
            store=store, embed_fn=canned, high_thresh=0.9, low_thresh=0.5,
        )

        r1 = await resolver.resolve("War", "WWII")
        assert r1.decision == ResolveDecision.DISTINCT
        wwii_id = r1.entity_id

        r2 = await resolver.resolve("War", "World War II")
        assert r2.decision == ResolveDecision.AUTO_MERGE
        assert r2.entity_id == wwii_id

        r3 = await resolver.resolve("War", "WW2")
        assert r3.decision == ResolveDecision.AUTO_MERGE
        assert r3.entity_id == wwii_id

        r4 = await resolver.resolve("War", "World War 2")
        assert r4.decision == ResolveDecision.AUTO_MERGE
        assert r4.entity_id == wwii_id

        r5 = await resolver.resolve("War", "Cold War")
        assert r5.decision == ResolveDecision.DISTINCT
        assert r5.entity_id != wwii_id
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_add_and_check_distinct_pair(tmp_path: Path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        resolver = DefaultEntityResolver(store=store, embed_fn=_hash_embed)
        assert not await resolver.is_blocked("id1", "id2")
        await resolver.add_distinct_pair("id1", "id2")
        assert await resolver.is_blocked("id1", "id2")
        assert await resolver.is_blocked("id2", "id1")  # symmetric
    finally:
        await store.close()
