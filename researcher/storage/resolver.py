"""EntityResolver ABC — canonicalizes candidate entity names to stable ids.

Two-threshold policy + distinct-pairs blocklist. Concrete impl uses DuckDB vss
for similarity lookup and sentence-transformers MiniLM for embeddings. Lives in
Wave 1-B.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Optional

from researcher.storage.store import KnowledgeStore


class ResolveDecision(str, Enum):
    AUTO_MERGE = "auto_merge"  # sim >= high_thresh: merge into existing entity
    PENDING = "pending"  # low_thresh <= sim < high_thresh: kept as provisional
    DISTINCT = "distinct"  # sim < low_thresh OR blocklisted pair: new entity


@dataclass(frozen=True)
class ResolveResult:
    decision: ResolveDecision
    entity_id: str  # existing id if merged, freshly-minted id otherwise
    similarity: float
    candidate_ids: list[str]  # top-k considered, for auditability


class EntityResolver(ABC):
    """Resolves a candidate entity name to a stable entity id.

    Default thresholds: AUTO_MERGE >= 0.92, DISTINCT < 0.78, PENDING in between.
    Composite keys should include year range / other disambiguators where the
    spec's schema has date fields.
    """

    @abstractmethod
    async def resolve(
        self, entity_type: str, name: str, context: Optional[dict] = None
    ) -> ResolveResult: ...

    @abstractmethod
    async def add_distinct_pair(self, id_a: str, id_b: str) -> None: ...

    @abstractmethod
    async def is_blocked(self, id_a: str, id_b: str) -> bool: ...


EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


class DefaultEntityResolver(EntityResolver):
    """Real entity resolver using embedding cosine similarity via store.find_similar.

    Two-threshold policy:
      - sim >= high_thresh → AUTO_MERGE with the best candidate
      - low_thresh <= sim < high_thresh → PENDING (flagged but not merged)
      - sim < low_thresh → DISTINCT (new entity)

    Distinct-pairs blocklist: a set of (name_a, name_b) tuples that must NEVER
    merge, regardless of similarity. Used to prevent known hierarchical
    conflations (e.g., "Napoleonic Wars" vs "War of the Sixth Coalition").
    """

    def __init__(
        self,
        store: KnowledgeStore,
        embed_fn: EmbedFn,
        high_thresh: float = 0.92,
        low_thresh: float = 0.78,
        distinct_pairs: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        self._store = store
        self._embed_fn = embed_fn
        self._high = high_thresh
        self._low = low_thresh
        # Name-level blocklist (symmetric).
        self._blocked_names: set[frozenset[str]] = set()
        if distinct_pairs:
            for a, b in distinct_pairs:
                self._blocked_names.add(frozenset({a, b}))
        # Id-level blocklist (for runtime add_distinct_pair calls).
        self._blocked_ids: set[frozenset[str]] = set()

    async def resolve(
        self, entity_type: str, name: str, context: Optional[dict] = None
    ) -> ResolveResult:
        embeddings = await self._embed_fn([name])
        embedding = embeddings[0]

        candidates = await self._store.find_similar(entity_type, embedding, k=5)
        # Filter out any candidate whose entity name is in the blocklist with `name`.
        filtered: list[tuple[str, float]] = []
        for candidate_id, sim in candidates:
            ent = await self._store.get_entity(candidate_id)
            if ent is None:
                continue
            candidate_name_cell = ent.fields.get("name")
            candidate_name = (
                candidate_name_cell.value if candidate_name_cell else candidate_id
            )
            if frozenset({name, str(candidate_name)}) in self._blocked_names:
                continue
            filtered.append((candidate_id, sim))

        candidate_ids = [cid for cid, _ in filtered[:5]]

        if filtered:
            best_id, best_sim = filtered[0]
            if best_sim >= self._high:
                return ResolveResult(
                    decision=ResolveDecision.AUTO_MERGE,
                    entity_id=best_id,
                    similarity=best_sim,
                    candidate_ids=candidate_ids,
                )
            if best_sim >= self._low:
                return ResolveResult(
                    decision=ResolveDecision.PENDING,
                    entity_id=best_id,
                    similarity=best_sim,
                    candidate_ids=candidate_ids,
                )
            # Fall through to DISTINCT.
            similarity_value = best_sim
        else:
            similarity_value = 0.0

        # DISTINCT: create a new entity row and persist the embedding so
        # future resolutions can find this entity via find_similar (which
        # joins embeddings against entities by type).
        new_id = await self._store.upsert_entity(entity_type, name, {})
        self._store.set_vector(new_id, embedding)
        return ResolveResult(
            decision=ResolveDecision.DISTINCT,
            entity_id=new_id,
            similarity=similarity_value,
            candidate_ids=candidate_ids,
        )

    async def add_distinct_pair(self, id_a: str, id_b: str) -> None:
        self._blocked_ids.add(frozenset({id_a, id_b}))

    async def is_blocked(self, id_a: str, id_b: str) -> bool:
        return frozenset({id_a, id_b}) in self._blocked_ids
