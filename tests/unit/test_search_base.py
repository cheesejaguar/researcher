"""Tests for the SearchProvider Protocol and SearchResult model."""

from __future__ import annotations

from researcher.search.base import SearchProvider, SearchResult


def test_search_result_instantiates_with_defaults() -> None:
    r = SearchResult(title="t", url="https://example.com/")
    assert r.title == "t"
    assert r.url == "https://example.com/"
    assert r.snippet == ""
    assert r.rank == 0


def test_search_result_full_fields() -> None:
    r = SearchResult(title="Title", url="https://example.com/", snippet="snip", rank=3)
    assert r.snippet == "snip"
    assert r.rank == 3


def test_search_provider_is_a_protocol() -> None:
    # Anything implementing `.name` and async `.search(query, max_results)` is a SearchProvider.
    class Dummy:
        name = "dummy"

        async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
            return []

    d: SearchProvider = Dummy()
    assert d.name == "dummy"
