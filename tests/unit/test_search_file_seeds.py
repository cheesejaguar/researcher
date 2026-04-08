"""Tests for FileSeedsProvider — offline fixture-backed search."""

from __future__ import annotations

import json
from pathlib import Path

from researcher.search.file_seeds import FileSeedsProvider


async def test_exact_match_returns_results() -> None:
    provider = FileSeedsProvider(
        seeds={
            "glp1 trials": [
                {"title": "A", "url": "https://a.example/", "snippet": "snip a"},
                {"title": "B", "url": "https://b.example/", "snippet": "snip b"},
            ]
        }
    )
    results = await provider.search("glp1 trials")
    assert len(results) == 2
    assert results[0].title == "A"
    assert results[0].rank == 0
    assert results[1].rank == 1


async def test_unknown_query_returns_empty() -> None:
    provider = FileSeedsProvider(seeds={"known": [{"title": "x", "url": "https://x"}]})
    results = await provider.search("unknown")
    assert results == []


async def test_respects_max_results() -> None:
    provider = FileSeedsProvider(
        seeds={
            "q": [{"title": f"t{i}", "url": f"https://x/{i}"} for i in range(10)]
        }
    )
    results = await provider.search("q", max_results=3)
    assert len(results) == 3
    assert [r.rank for r in results] == [0, 1, 2]


async def test_from_file(tmp_path: Path) -> None:
    seeds_file = tmp_path / "seeds.json"
    seeds_file.write_text(
        json.dumps({"hello": [{"title": "Hi", "url": "https://hi/", "snippet": ""}]})
    )
    provider = FileSeedsProvider.from_file(seeds_file)
    results = await provider.search("hello")
    assert len(results) == 1
    assert results[0].title == "Hi"
    assert provider.name == "file_seeds"
