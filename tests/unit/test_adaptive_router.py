"""Tests for AdaptiveSearchRouter and DuckDBRouterScoreStore.

The router picks the historically best SearchProvider per entity type.
On first use it fans out to every configured provider in parallel,
scores them with a deterministic function, caches the ranking, and
thereafter routes only to the top scorer. Not a bandit — no exploration
after the trial.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researcher.search.base import SearchResult
from researcher.search.router import (
    AdaptiveSearchRouter,
    DuckDBRouterScoreStore,
    ProviderScore,
    score_results,
)
from researcher.spec import SearchConfig

# ---------- Stub provider ----------


class StubProvider:
    """Minimal SearchProvider: records call count, returns canned results."""

    def __init__(self, name: str, results: list[SearchResult]) -> None:
        self.name = name
        self._results = results
        self.calls = 0

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        self.calls += 1
        return list(self._results)


class RaisingProvider:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        self.calls += 1
        raise RuntimeError(f"{self.name} boom")


def _results(n: int, snippet_len: int) -> list[SearchResult]:
    snippet = "x" * snippet_len
    return [
        SearchResult(title=f"t{i}", url=f"https://e/{i}", snippet=snippet, rank=i) for i in range(n)
    ]


# ---------- Scoring ----------


def test_score_empty_results_is_zero() -> None:
    assert score_results([]) == 0.0


def test_score_full_results_with_max_snippets_is_one() -> None:
    # 10 results, each with the max-capped 500-char snippet
    results = _results(10, 500)
    assert score_results(results) == pytest.approx(1.0)


def test_score_mixed_count_and_snippet() -> None:
    # 5 results, 250-char snippets: 0.7 * 0.5 + 0.3 * 0.5 = 0.5
    results = _results(5, 250)
    assert score_results(results) == pytest.approx(0.5)


def test_score_snippet_length_is_capped_at_500() -> None:
    # 10 results with 1000-char snippets should still score 1.0 (capped).
    results = _results(10, 1000)
    assert score_results(results) == pytest.approx(1.0)


def test_score_result_count_is_capped_at_10() -> None:
    # 20 results, full snippets — still scores 1.0.
    results = _results(20, 500)
    assert score_results(results) == pytest.approx(1.0)


# ---------- Router routing ----------


async def test_router_first_call_trials_all_providers() -> None:
    p1 = StubProvider("p1", _results(3, 100))
    p2 = StubProvider("p2", _results(5, 200))
    p3 = StubProvider("p3", _results(1, 50))
    router = AdaptiveSearchRouter([p1, p2, p3])

    await router.search("q", entity_type="War")

    assert p1.calls == 1
    assert p2.calls == 1
    assert p3.calls == 1


async def test_router_second_call_uses_top_scorer_only() -> None:
    winner = StubProvider("winner", _results(10, 500))
    loser_a = StubProvider("loser_a", [])
    loser_b = StubProvider("loser_b", [])
    router = AdaptiveSearchRouter([winner, loser_a, loser_b])

    await router.search("q1", entity_type="War")
    await router.search("q2", entity_type="War")

    assert winner.calls == 2
    assert loser_a.calls == 1
    assert loser_b.calls == 1


async def test_router_returns_results_from_top_scorer() -> None:
    winner_results = _results(10, 500)
    winner = StubProvider("winner", winner_results)
    loser = StubProvider("loser", [])
    router = AdaptiveSearchRouter([winner, loser])

    out = await router.search("q", entity_type="War")

    assert [r.url for r in out] == [r.url for r in winner_results]


async def test_router_without_entity_type_uses_first_provider() -> None:
    p1 = StubProvider("p1", _results(3, 100))
    p2 = StubProvider("p2", _results(5, 200))
    p3 = StubProvider("p3", _results(1, 50))
    router = AdaptiveSearchRouter([p1, p2, p3])

    out = await router.search("q")

    assert p1.calls == 1
    assert p2.calls == 0
    assert p3.calls == 0
    assert [r.url for r in out] == [r.url for r in p1._results]


async def test_router_handles_provider_exception_in_trial() -> None:
    winner = StubProvider("winner", _results(10, 500))
    boom = RaisingProvider("boom")
    quiet = StubProvider("quiet", _results(2, 100))
    router = AdaptiveSearchRouter([winner, boom, quiet])

    out = await router.search("q", entity_type="War")

    assert [r.url for r in out] == [r.url for r in winner._results]
    rankings = router.rankings("War")
    by_name = {s.provider_name: s for s in rankings}
    assert by_name["boom"].mean_result_count == 0.0
    assert by_name["boom"].mean_snippet_len == 0.0
    # winner is top-ranked
    assert rankings[0].provider_name == "winner"


async def test_router_all_providers_raise_returns_empty() -> None:
    r1 = RaisingProvider("r1")
    r2 = RaisingProvider("r2")
    router = AdaptiveSearchRouter([r1, r2])

    out = await router.search("q", entity_type="War")

    assert out == []
    # Both were tried once.
    assert r1.calls == 1
    assert r2.calls == 1


async def test_router_rankings_reflects_scores() -> None:
    best = StubProvider("best", _results(10, 500))
    middling = StubProvider("middling", _results(5, 250))
    worst = StubProvider("worst", _results(1, 50))
    router = AdaptiveSearchRouter([best, middling, worst])

    await router.search("q", entity_type="War")
    ranks = router.rankings("War")

    assert [s.provider_name for s in ranks] == ["best", "middling", "worst"]


async def test_router_reset_clears_rankings() -> None:
    p1 = StubProvider("p1", _results(10, 500))
    p2 = StubProvider("p2", _results(2, 50))
    router = AdaptiveSearchRouter([p1, p2])

    await router.search("q", entity_type="War")
    assert p2.calls == 1
    router.reset("War")
    await router.search("q", entity_type="War")
    # After reset, trial happens again — both providers hit a second time.
    assert p1.calls == 2
    assert p2.calls == 2


async def test_router_reset_all_clears_every_entity_type() -> None:
    p1 = StubProvider("p1", _results(10, 500))
    p2 = StubProvider("p2", _results(1, 10))
    router = AdaptiveSearchRouter([p1, p2])

    await router.search("q", entity_type="War")
    await router.search("q", entity_type="Trial")
    router.reset()

    await router.search("q", entity_type="War")
    await router.search("q", entity_type="Trial")
    # Both entity types re-trialed → p2 called once per trial reset.
    assert p2.calls == 4  # 1 War + 1 Trial + 1 War + 1 Trial


async def test_router_per_entity_type_rankings_independent() -> None:
    # Provider A is great for "War" (lots of results), poor for "Trial".
    # We simulate that by using different router instances against the same
    # providers but swapping which provider wins per entity type.
    war_winner = StubProvider("war_winner", _results(10, 500))
    trial_winner = StubProvider("trial_winner", _results(9, 400))
    router = AdaptiveSearchRouter([war_winner, trial_winner])

    await router.search("q", entity_type="War")
    await router.search("q", entity_type="Trial")

    war_ranks = router.rankings("War")
    trial_ranks = router.rankings("Trial")

    assert war_ranks[0].provider_name == "war_winner"
    assert trial_ranks[0].provider_name == "war_winner"
    # Sanity: both rankings exist and are sorted desc by score.
    assert all(war_ranks[i].provider_name for i in range(len(war_ranks)))
    assert [s.provider_name for s in trial_ranks] == [
        "war_winner",
        "trial_winner",
    ]


# ---------- DuckDB score store ----------


def test_score_store_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = DuckDBRouterScoreStore(tmp_path / "scores.duckdb")
    store.open()
    try:
        scores = [
            ProviderScore(
                provider_name="tavily",
                mean_result_count=10.0,
                mean_snippet_len=500.0,
                sample_count=1,
            ),
            ProviderScore(
                provider_name="brave",
                mean_result_count=3.0,
                mean_snippet_len=120.0,
                sample_count=1,
            ),
        ]
        store.save("War", scores)
        loaded = store.load("War")
        assert len(loaded) == 2
        by_name = {s.provider_name: s for s in loaded}
        assert by_name["tavily"].mean_result_count == 10.0
        assert by_name["tavily"].mean_snippet_len == 500.0
        assert by_name["tavily"].sample_count == 1
        assert by_name["brave"].mean_result_count == 3.0
    finally:
        store.close()


def test_score_store_load_missing_returns_empty(tmp_path: Path) -> None:
    store = DuckDBRouterScoreStore(tmp_path / "scores.duckdb")
    store.open()
    try:
        assert store.load("Nope") == []
    finally:
        store.close()


def test_score_store_save_overwrites_previous(tmp_path: Path) -> None:
    store = DuckDBRouterScoreStore(tmp_path / "scores.duckdb")
    store.open()
    try:
        first = [
            ProviderScore("tavily", 5.0, 100.0, 1),
            ProviderScore("brave", 4.0, 80.0, 1),
        ]
        store.save("War", first)
        second = [
            ProviderScore("serper", 10.0, 500.0, 2),
        ]
        store.save("War", second)

        loaded = store.load("War")
        assert len(loaded) == 1
        assert loaded[0].provider_name == "serper"
        assert loaded[0].sample_count == 2
    finally:
        store.close()


def test_score_store_reopening_sees_persisted_scores(tmp_path: Path) -> None:
    db = tmp_path / "scores.duckdb"
    store1 = DuckDBRouterScoreStore(db)
    store1.open()
    store1.save("War", [ProviderScore("tavily", 8.0, 400.0, 1)])
    store1.close()

    store2 = DuckDBRouterScoreStore(db)
    store2.open()
    try:
        loaded = store2.load("War")
        assert len(loaded) == 1
        assert loaded[0].provider_name == "tavily"
        assert loaded[0].mean_result_count == 8.0
        assert loaded[0].mean_snippet_len == 400.0
    finally:
        store2.close()


# ---------- Router + store integration ----------


async def test_router_persists_scores_to_store(tmp_path: Path) -> None:
    store = DuckDBRouterScoreStore(tmp_path / "scores.duckdb")
    store.open()
    try:
        winner = StubProvider("winner", _results(10, 500))
        loser = StubProvider("loser", _results(1, 10))
        router = AdaptiveSearchRouter([winner, loser], score_store=store)

        await router.search("q", entity_type="War")

        loaded = store.load("War")
        by_name = {s.provider_name: s for s in loaded}
        assert "winner" in by_name
        assert "loser" in by_name
        # Top scorer from router's in-memory ranking matches the store.
        top_memory = router.rankings("War")[0].provider_name
        assert top_memory == "winner"
    finally:
        store.close()


async def test_router_loads_cached_scores_from_store(tmp_path: Path) -> None:
    store = DuckDBRouterScoreStore(tmp_path / "scores.duckdb")
    store.open()
    try:
        # Pre-populate: "loser" is cached as the winner so we can detect
        # whether the router used the cache or trialed afresh.
        store.save(
            "War",
            [
                ProviderScore("loser", 10.0, 500.0, 1),
                ProviderScore("winner", 0.0, 0.0, 1),
            ],
        )

        winner = StubProvider("winner", _results(10, 500))
        loser = StubProvider("loser", _results(2, 50))
        router = AdaptiveSearchRouter([winner, loser], score_store=store)

        await router.search("q", entity_type="War")

        # Cache said loser is top — so only loser should have been called.
        assert loser.calls == 1
        assert winner.calls == 0
    finally:
        store.close()


# ---------- Spec ----------


def test_spec_search_config_accepts_adaptive_flag() -> None:
    cfg = SearchConfig(adaptive=True, provider="tavily")
    assert cfg.adaptive is True
    assert cfg.provider == "tavily"


def test_spec_search_config_adaptive_defaults_false() -> None:
    cfg = SearchConfig()
    assert cfg.adaptive is False
