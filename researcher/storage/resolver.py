"""EntityResolver ABC — canonicalizes candidate entity names to stable ids.

Two-threshold policy + distinct-pairs blocklist. Concrete impl uses DuckDB vss
for similarity lookup and sentence-transformers MiniLM for embeddings. Lives in
Wave 1-B.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
