"""In-memory stub KnowledgeStore for Wave 1 agent/orchestrator tests."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel

from researcher.storage.store import (
    Conflict,
    Entity,
    FieldCell,
    KnowledgeStore,
    StoreMetrics,
)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class StubKnowledgeStore(KnowledgeStore):
    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._provenance: dict[str, dict] = {}
        self._conflicts: list[Conflict] = []
        self._vectors: dict[str, list[float]] = {}  # entity_id -> embedding
        self._schema: Optional[dict] = None
        self._run_summaries: list[tuple[str, dict]] = []
        self._cost_usd = 0.0

    async def open(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def init_schema(self, entity_class: type[BaseModel]) -> None:
        self._schema = entity_class.model_json_schema()

    async def upsert_entity(
        self, entity_type: str, name: str, fields: dict[str, FieldCell]
    ) -> str:
        # Match by name within type, else create.
        for eid, ent in self._entities.items():
            if ent.type == entity_type and ent.fields.get("name", FieldCell(
                value="", confidence=0, provenance_ids=[], updated_at=datetime.now(UTC)
            )).value == name:
                ent.fields.update(fields)
                return eid
        eid = uuid4().hex
        name_cell = FieldCell(
            value=name,
            confidence=1.0,
            provenance_ids=[],
            updated_at=datetime.now(UTC),
        )
        merged = {"name": name_cell, **fields}
        self._entities[eid] = Entity(id=eid, type=entity_type, fields=merged)
        return eid

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    async def query(self, sql: str, params: tuple = ()) -> list[dict]:
        raise NotImplementedError("StubKnowledgeStore.query — use typed APIs in tests")

    async def find_similar(
        self, entity_type: str, embedding: list[float], k: int = 5
    ) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for eid, vec in self._vectors.items():
            ent = self._entities.get(eid)
            if ent is None or ent.type != entity_type:
                continue
            scored.append((eid, _cosine(embedding, vec)))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:k]

    def set_vector(self, entity_id: str, vec: list[float]) -> None:
        """Test helper: attach an embedding to an existing entity."""
        self._vectors[entity_id] = vec

    async def record_provenance(self, provenance_id: str, data: dict) -> None:
        self._provenance[provenance_id] = data

    async def get_conflicts(self, status: str = "open") -> list[Conflict]:
        return [c for c in self._conflicts if c.status == status]

    async def record_conflict(self, conflict: Conflict) -> None:
        # Upsert by conflict_id so the stub matches DuckDBKnowledgeStore semantics.
        self._conflicts = [
            c for c in self._conflicts if c.conflict_id != conflict.conflict_id
        ]
        self._conflicts.append(conflict)

    async def snapshot_metrics(self) -> StoreMetrics:
        by_type: dict[str, int] = {}
        for e in self._entities.values():
            by_type[e.type] = by_type.get(e.type, 0) + 1
        # Field-fill: average over all entities of non-None field cell count.
        if self._entities:
            filled = 0
            total = 0
            for e in self._entities.values():
                for cell in e.fields.values():
                    total += 1
                    if cell.value is not None:
                        filled += 1
            pct = filled / total if total else 0.0
        else:
            pct = 0.0
        return StoreMetrics(
            entities_total=len(self._entities),
            by_type=by_type,
            fields_filled_pct=pct,
            conflicts_open=sum(1 for c in self._conflicts if c.status == "open"),
            cost_usd_total=self._cost_usd,
        )

    async def write_run_summary(self, run_id: str, summary: dict) -> None:
        self._run_summaries.append((run_id, summary))

    def set_cost(self, usd: float) -> None:
        """Test helper so orchestrator stop-checks can observe synthetic cost."""
        self._cost_usd = usd
