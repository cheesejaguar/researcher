"""Adaptive search-provider routing.

The :class:`AdaptiveSearchRouter` wraps a list of :class:`SearchProvider`
implementations and picks the historically best one for each entity type.
On first use for an entity type it fans out to every configured provider
in parallel, scores them with a simple deterministic function, caches
the ranking, and thereafter routes only to the top scorer.

This is *not* a Thompson-sampling bandit — there is no exploration after
the trial, and there are no reward updates from downstream fact quality.
The Thompson bandit re-entry criterion (100+ runs) still holds and is
explicitly out of scope for v1.3.

Score function (deterministic):

    score(results) = 0.7 * min(len(results), 10) / 10
                   + 0.3 * min(mean(snippet_len), 500) / 500

A provider that returns 10 results with 500-char snippets scores 1.0.
Snippet length is capped at 500 to match the ``SearchResult`` convention
and to keep long-form providers from dominating.

Persistence is optional: if an implementation of :class:`RouterScoreStore`
is passed to the router constructor, rankings are loaded lazily on first
use and written back after each trial. The in-repo
:class:`DuckDBRouterScoreStore` is a dedicated single-purpose DuckDB file
(it does *not* share a connection with :class:`KnowledgeStore`) so it can
live anywhere the caller likes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import duckdb

from researcher.search.base import SearchProvider, SearchResult

# Snippet cap mirrors the SearchResult convention across providers.
_SNIPPET_CAP = 500
# Result-count cap — more than 10 useful hits is diminishing returns.
_RESULT_COUNT_CAP = 10
# Score weights: keep them simple + testable.
_COUNT_WEIGHT = 0.7
_SNIPPET_WEIGHT = 0.3


@dataclass
class ProviderScore:
    """One provider's historical performance for a single entity type."""

    provider_name: str
    mean_result_count: float
    mean_snippet_len: float
    sample_count: int = 1

    @property
    def score(self) -> float:
        """Deterministic 0..1 ranking score."""
        count_norm = min(self.mean_result_count, _RESULT_COUNT_CAP) / _RESULT_COUNT_CAP
        snippet_norm = min(self.mean_snippet_len, _SNIPPET_CAP) / _SNIPPET_CAP
        return _COUNT_WEIGHT * count_norm + _SNIPPET_WEIGHT * snippet_norm


def score_results(results: list[SearchResult]) -> float:
    """Compute the deterministic routing score for a single provider call."""
    if not results:
        return 0.0
    mean_snippet = sum(len(r.snippet) for r in results) / len(results)
    ps = ProviderScore(
        provider_name="",
        mean_result_count=float(len(results)),
        mean_snippet_len=mean_snippet,
        sample_count=1,
    )
    return ps.score


def _score_from_results(provider_name: str, results: list[SearchResult]) -> ProviderScore:
    if not results:
        return ProviderScore(provider_name, 0.0, 0.0, 1)
    mean_snippet = sum(len(r.snippet) for r in results) / len(results)
    return ProviderScore(
        provider_name=provider_name,
        mean_result_count=float(len(results)),
        mean_snippet_len=mean_snippet,
        sample_count=1,
    )


@runtime_checkable
class RouterScoreStore(Protocol):
    """Optional persistence backend for AdaptiveSearchRouter rankings."""

    def load(self, entity_type: str) -> list[ProviderScore]: ...

    def save(self, entity_type: str, scores: list[ProviderScore]) -> None: ...


class AdaptiveSearchRouter:
    """Picks the historically best SearchProvider per entity type.

    See the module docstring for the full contract. Key points:

    - ``entity_type`` is a keyword-only optional kwarg on :meth:`search`,
      so the router still satisfies the :class:`SearchProvider` protocol
      at call sites that don't care about routing.
    - The first call for a new entity type fans out in parallel.
    - Exceptions during the trial are swallowed: the raising provider
      scores 0.0, the trial continues, and the top survivor's results
      are returned. If *every* provider raises, :meth:`search` returns
      ``[]`` without bubbling.
    - Subsequent calls for the same entity type hit only the top-ranked
      provider (cached in memory, optionally persisted).
    """

    name = "adaptive_router"

    def __init__(
        self,
        providers: list[SearchProvider],
        score_store: RouterScoreStore | None = None,
    ) -> None:
        if not providers:
            raise ValueError("AdaptiveSearchRouter requires at least one provider")
        self._providers = list(providers)
        self._by_name = {p.name: p for p in self._providers}
        self._score_store = score_store
        # entity_type -> sorted list[ProviderScore] (desc by score).
        self._rankings: dict[str, list[ProviderScore]] = {}
        # entity_types we've already attempted to load from the store, so
        # an empty store doesn't re-hit DuckDB on every call.
        self._loaded_from_store: set[str] = set()

    # ---- public protocol -----------------------------------------------

    async def search(
        self,
        query: str,
        max_results: int = 10,
        *,
        entity_type: str | None = None,
    ) -> list[SearchResult]:
        if entity_type is None:
            # No routing signal — fall back to the first provider.
            return await self._providers[0].search(query, max_results)

        # Lazy load from store the first time we see this entity type.
        if (
            self._score_store is not None
            and entity_type not in self._rankings
            and entity_type not in self._loaded_from_store
        ):
            cached = await asyncio.to_thread(self._score_store.load, entity_type)
            self._loaded_from_store.add(entity_type)
            if cached:
                self._rankings[entity_type] = sorted(cached, key=lambda s: s.score, reverse=True)

        cached_rank = self._rankings.get(entity_type)
        if cached_rank:
            # Cached: only call the top scorer.
            top_name = cached_rank[0].provider_name
            top = self._by_name.get(top_name)
            if top is None:
                # Top provider no longer in the configured list — fall
                # back to first configured provider. Don't touch the cache.
                return await self._providers[0].search(query, max_results)
            try:
                return await top.search(query, max_results)
            except Exception:
                return []

        # Trial: fan out to every provider in parallel.
        return await self._trial(query, max_results, entity_type)

    def rankings(self, entity_type: str) -> list[ProviderScore]:
        """Return the cached ranking for ``entity_type`` (desc by score)."""
        return list(self._rankings.get(entity_type, []))

    def reset(self, entity_type: str | None = None) -> None:
        """Drop cached rankings.

        - ``reset(None)`` clears every entity type (and any store-load
          memoization).
        - ``reset("War")`` drops just that entity type.
        """
        if entity_type is None:
            self._rankings.clear()
            self._loaded_from_store.clear()
            return
        self._rankings.pop(entity_type, None)
        self._loaded_from_store.discard(entity_type)

    # ---- trial helpers -------------------------------------------------

    async def _trial(self, query: str, max_results: int, entity_type: str) -> list[SearchResult]:
        tasks = [p.search(query, max_results) for p in self._providers]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        scored: list[tuple[ProviderScore, list[SearchResult]]] = []
        for provider, outcome in zip(self._providers, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                scored.append(
                    (
                        ProviderScore(provider.name, 0.0, 0.0, 1),
                        [],
                    )
                )
                continue
            scored.append((_score_from_results(provider.name, outcome), outcome))

        scored.sort(key=lambda pair: pair[0].score, reverse=True)
        ranking = [pair[0] for pair in scored]
        self._rankings[entity_type] = ranking

        if self._score_store is not None:
            await asyncio.to_thread(self._score_store.save, entity_type, ranking)
            self._loaded_from_store.add(entity_type)

        # Return top scorer's results — if every provider raised or
        # returned [], this is [] and that's the trial signal.
        return list(scored[0][1]) if scored else []


class DuckDBRouterScoreStore:
    """Persists adaptive router scores in a dedicated DuckDB file.

    Creates its own ``router_scores`` table on :meth:`open`. Small and
    single-purpose — it does NOT share a connection with
    :class:`KnowledgeStore`, so it can live in a separate file or
    alongside the main knowledge store at the caller's discretion.

    Synchronous on purpose: the router wraps its :meth:`load` and
    :meth:`save` calls in ``asyncio.to_thread`` so the event loop is
    not blocked on DuckDB I/O.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._conn: duckdb.DuckDBPyConnection | None = None

    def open(self) -> None:
        if self._conn is not None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS router_scores (
                entity_type VARCHAR NOT NULL,
                provider_name VARCHAR NOT NULL,
                mean_result_count DOUBLE NOT NULL,
                mean_snippet_len DOUBLE NOT NULL,
                sample_count INTEGER NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                PRIMARY KEY (entity_type, provider_name)
            )
            """
        )

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def load(self, entity_type: str) -> list[ProviderScore]:
        if self._conn is None:
            raise RuntimeError("DuckDBRouterScoreStore is not open")
        rows = self._conn.execute(
            """
            SELECT provider_name, mean_result_count, mean_snippet_len, sample_count
            FROM router_scores
            WHERE entity_type = ?
            """,
            [entity_type],
        ).fetchall()
        return [
            ProviderScore(
                provider_name=r[0],
                mean_result_count=float(r[1]),
                mean_snippet_len=float(r[2]),
                sample_count=int(r[3]),
            )
            for r in rows
        ]

    def save(self, entity_type: str, scores: list[ProviderScore]) -> None:
        if self._conn is None:
            raise RuntimeError("DuckDBRouterScoreStore is not open")
        now = datetime.now(UTC)
        # Replace all rows for this entity_type — simpler than per-row
        # upsert at v1 scale.
        self._conn.execute(
            "DELETE FROM router_scores WHERE entity_type = ?",
            [entity_type],
        )
        for s in scores:
            self._conn.execute(
                """
                INSERT INTO router_scores (
                    entity_type,
                    provider_name,
                    mean_result_count,
                    mean_snippet_len,
                    sample_count,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    entity_type,
                    s.provider_name,
                    float(s.mean_result_count),
                    float(s.mean_snippet_len),
                    int(s.sample_count),
                    now,
                ],
            )


__all__ = [
    "AdaptiveSearchRouter",
    "DuckDBRouterScoreStore",
    "ProviderScore",
    "RouterScoreStore",
    "score_results",
]
