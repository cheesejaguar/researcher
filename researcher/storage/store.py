"""KnowledgeStore ABC — the typed object store that agents write into.

Concrete impl (DuckDBKnowledgeStore) lives in Wave 1-B. The ABC is the contract
the FactWriter, EntityResolver, and Orchestrator all code against.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class FieldCell(BaseModel):
    """One cell of an entity row: value + confidence + provenance pointers."""

    value: Any
    confidence: float
    provenance_ids: list[str]
    updated_at: datetime


class Entity(BaseModel):
    id: str
    type: str
    fields: dict[str, FieldCell]


class Conflict(BaseModel):
    """Record of two or more incompatible values for the same (entity, field)."""

    conflict_id: str
    entity_id: str
    field: str
    candidates: list[FieldCell]
    status: str  # "open" | "resolved"
    winning_value: Any = None
    reason: Optional[str] = None


class StoreMetrics(BaseModel):
    entities_total: int
    by_type: dict[str, int]
    fields_filled_pct: float
    conflicts_open: int
    cost_usd_total: float


class KnowledgeStore(ABC):
    """Async-context-managed typed object store.

    `init_schema` introspects a Pydantic entity class and builds DuckDB tables +
    a vss index. All writes funnel through the FactWriter — the store itself
    exposes the low-level primitives the writer composes.
    """

    @abstractmethod
    async def open(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def __aenter__(self) -> KnowledgeStore:
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    @abstractmethod
    async def init_schema(self, entity_class: type[BaseModel]) -> None:
        """Create tables + vss index from a dynamically-constructed Pydantic model."""

    @abstractmethod
    async def upsert_entity(
        self, entity_type: str, name: str, fields: dict[str, FieldCell]
    ) -> str:
        """Create or update an entity row; returns its id."""

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Entity]: ...

    @abstractmethod
    async def query(self, sql: str, params: tuple = ()) -> list[dict]: ...

    @abstractmethod
    async def find_similar(
        self, entity_type: str, embedding: list[float], k: int = 5
    ) -> list[tuple[str, float]]:
        """k-NN search via DuckDB vss; returns (entity_id, cosine_similarity) tuples."""

    @abstractmethod
    async def record_provenance(self, provenance_id: str, data: dict) -> None: ...

    @abstractmethod
    async def get_conflicts(self, status: str = "open") -> list[Conflict]: ...

    @abstractmethod
    async def record_conflict(self, conflict: Conflict) -> None: ...

    @abstractmethod
    async def snapshot_metrics(self) -> StoreMetrics: ...

    @abstractmethod
    async def write_run_summary(self, run_id: str, summary: dict) -> None: ...

    async def merge_field(
        self,
        entity_id: str,
        field_name: str,
        value: Any,
        confidence: float,
        run_id: str,
        provenance_ids: list[str],
    ) -> dict:
        """Merge a single field value with cross-run conflict semantics.

        Default: not supported. Concrete stores (DuckDBKnowledgeStore) override.
        Returning ``{"status": "not_supported"}`` lets the FactWriter degrade
        gracefully when a stub store is used in higher-layer tests.
        """
        return {"status": "not_supported"}

    async def record_relation(
        self,
        source_id: str,
        target_id: str,
        relation_label: str,
        confidence: float,
        run_id: str,
    ) -> None:
        """Record a typed edge between two entities.

        Default: no-op. The DuckDB store overrides this with a real
        ``entity_relations`` upsert (higher-confidence wins on conflict).
        """
        return None
