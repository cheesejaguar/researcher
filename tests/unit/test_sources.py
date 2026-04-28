from __future__ import annotations

import pytest

from researcher.search.base import SearchResult
from researcher.sources import (
    PolicySearchProvider,
    SourcePackSearchProvider,
    domain_allowed,
    ingest_source_sets,
    load_items,
    source_authority,
)
from researcher.spec import SourcePolicy, SourceSet


def test_domain_policy_allow_deny_and_trusted():
    policy = SourcePolicy(
        allow_domains=["example.com"],
        deny_domains=["bad.example.com"],
        trusted_domains=["example.com"],
    )
    assert domain_allowed("https://docs.example.com/a", policy)
    assert not domain_allowed("https://bad.example.com/a", policy)
    assert not domain_allowed("https://other.com/a", policy)
    assert source_authority("https://docs.example.com/a", policy) == 1.0


def test_ingest_source_sets_chunks_local_file(tmp_path):
    source = tmp_path / "note.md"
    source.write_text("World War II source text")
    records, chunks = ingest_source_sets([SourceSet(name="local", paths=[str(source)])])
    assert len(records) == 1
    assert records[0].url.startswith("file://")
    assert chunks
    assert "World War II" in chunks[0].text


@pytest.mark.asyncio
async def test_source_pack_searches_local_chunks_before_fallback(tmp_path):
    source = tmp_path / "note.md"
    source.write_text("World War II source text")
    records, chunks = ingest_source_sets([SourceSet(name="local", paths=[str(source)])])
    chunk_rows = [
        {
            "chunk_id": c.chunk_id,
            "source_id": c.source_id,
            "title": records[0].title,
            "url": records[0].url,
            "text": c.text,
        }
        for c in chunks
    ]
    provider = SourcePackSearchProvider(chunks=chunk_rows)
    results = await provider.search("World War", max_results=3)
    assert results
    assert results[0].url.startswith("file://")


@pytest.mark.asyncio
async def test_policy_search_provider_filters_fallback():
    class _Fallback:
        name = "fallback"

        async def search(self, query: str, max_results: int = 10):
            return [
                SearchResult(title="good", url="https://example.com/a"),
                SearchResult(title="bad", url="https://bad.com/a"),
            ]

    provider = PolicySearchProvider(
        fallback=_Fallback(),
        policy=SourcePolicy(allow_domains=["example.com"]),
    )
    results = await provider.search("x")
    assert [r.title for r in results] == ["good"]


def test_load_items_csv_and_jsonl(tmp_path):
    csv_path = tmp_path / "items.csv"
    csv_path.write_text("name,id\nA,1\n")
    assert load_items(csv_path, "name") == [{"name": "A", "id": "1"}]
    jsonl_path = tmp_path / "items.jsonl"
    jsonl_path.write_text('{"name": "B", "id": 2}\n')
    assert load_items(jsonl_path, "name") == [{"name": "B", "id": "2"}]
