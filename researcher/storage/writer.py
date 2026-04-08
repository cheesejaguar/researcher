"""FactWriter — the single-writer reduce step.

**Serial reduce, parallel map.** Agents (map) run concurrently and submit
FactClaim objects to this queue. A single drain coroutine (the writer) runs the
`validate_type -> dedup -> detect_conflict -> score_confidence` pipeline on each
claim and commits to the store. Making the reduce step serial is correct, not a
compromise — DuckDB is single-writer and the reduce pipeline is CPU-bound + fast
relative to network latency.

Wave 1-B replaces the no-op stages with real implementations:

- ``validate_type``    coerces ``claim.value`` against the entity schema's
                       declared field type.
- ``dedup``            deferred no-op; ``store.upsert_entity`` is the actual
                       same-name merge point.
- ``detect_conflict``  queries the store by ``(entity_type, entity_name)`` and
                       returns the existing :class:`FieldCell` if its value
                       differs from the new claim.
- ``score_confidence`` blends ``0.7 * agent_confidence + 0.3 * source_authority``.

The four stages remain module-level functions so each can be tested in isolation
without spinning up the writer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional
from uuid import uuid4

from researcher.models import FactClaim
from researcher.storage.resolver import EntityResolver
from researcher.storage.store import Conflict, FieldCell, KnowledgeStore


@dataclass
class WriteOutcome:
    written: bool
    entity_id: Optional[str]
    reason: str  # "ok" | "duplicate" | "conflict" | "type_error" | "low_confidence"
    conflict_id: Optional[str] = None


# ---------- Pipeline stages ----------


async def validate_type(claim: FactClaim, entity_schema: dict) -> FactClaim:
    """Coerce ``claim.value`` to match the declared type for ``claim.field``.

    The ``entity_schema`` arg is a dict like
    ``{"entity_type": "War", "fields": [{"name": "start_year", "type": "int"}]}``.
    Unknown fields pass through unchanged. Coercion failures raise
    ``ValueError("type_error: ...")`` so the writer can route them to the
    type_errors metric.
    """
    fields = entity_schema.get("fields", [])
    field_spec = next((f for f in fields if f.get("name") == claim.field), None)
    if field_spec is None:
        return claim
    type_str = field_spec.get("type", "")
    if not type_str:
        return claim
    try:
        coerced = _coerce_value(claim.value, type_str)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"type_error: field={claim.field} type={type_str}: {exc}"
        ) from exc
    return claim.model_copy(update={"value": coerced})


def _coerce_value(value: Any, type_str: str) -> Any:
    type_str = type_str.strip()
    if type_str.startswith("list[") and type_str.endswith("]"):
        inner = type_str[5:-1].strip()
        if not isinstance(value, (list, tuple)):
            value = [value]
        return [_coerce_scalar(v, inner) for v in value]
    return _coerce_scalar(value, type_str)


def _coerce_scalar(value: Any, type_str: str) -> Any:
    if type_str == "str":
        return str(value)
    if type_str == "int":
        # bool is an int subclass — accept transparently.
        if isinstance(value, bool):
            return int(value)
        return int(value)
    if type_str == "float":
        return float(value)
    if type_str == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "1", "yes", "y"):
                return True
            if v in ("false", "0", "no", "n"):
                return False
            raise ValueError(f"cannot parse bool from {value!r}")
        raise ValueError(f"cannot parse bool from {type(value).__name__}")
    if type_str == "date":
        from datetime import date as _date
        from datetime import datetime as _dt

        if isinstance(value, _dt):
            return value.date()
        if isinstance(value, _date):
            return value
        if isinstance(value, str):
            return _dt.fromisoformat(value).date()
        raise ValueError(f"cannot parse date from {type(value).__name__}")
    if type_str == "datetime":
        from datetime import datetime as _dt

        if isinstance(value, _dt):
            return value
        if isinstance(value, str):
            return _dt.fromisoformat(value)
        raise ValueError(f"cannot parse datetime from {type(value).__name__}")
    raise ValueError(f"unsupported type: {type_str!r}")


async def dedup(claim: FactClaim, store: KnowledgeStore) -> bool:
    """No-op in Wave 1-B.

    ``store.upsert_entity`` already merges by ``(entity_type, name)`` so we
    don't need a separate dedup step. Kept as a hook for v1.1 when we may
    want claim-level idempotency keyed by ``(entity_id, field, value, source_url)``.
    """
    return False


async def detect_conflict(
    claim: FactClaim, store: KnowledgeStore
) -> list[FieldCell]:
    """Return existing :class:`FieldCell` rows that contradict ``claim``.

    Looks up the entity by ``(entity_type, entity_name)`` via the store's
    ``query`` API. If the store doesn't support ``query`` (e.g. the in-memory
    stub used by orchestrator tests) we silently degrade to "no conflict
    detected" — real conflict tracking happens against
    :class:`DuckDBKnowledgeStore`.
    """
    try:
        rows = await store.query(
            "SELECT id FROM entities WHERE entity_type = ? AND name = ?",
            (claim.entity_type, claim.entity_name),
        )
    except NotImplementedError:
        return []
    except Exception:
        # Be defensive: a malformed query (e.g. backend doesn't speak SQL at all)
        # must not bring down the writer pipeline.
        return []
    if not rows:
        return []
    entity_id = rows[0]["id"]
    entity = await store.get_entity(entity_id)
    if entity is None:
        return []
    existing = entity.fields.get(claim.field)
    if existing is None:
        return []
    if existing.value == claim.value:
        return []
    return [existing]


async def score_confidence(claim: FactClaim, context: dict) -> float:
    """Blend agent confidence with source authority.

    ``0.7 * claim.confidence + 0.3 * context.get("source_authority", 0.5)``,
    clamped to [0, 1]. Source authority defaults to 0.5 (neutral) when the
    context doesn't carry one.
    """
    source_authority = float(context.get("source_authority", 0.5))
    blended = 0.7 * float(claim.confidence) + 0.3 * source_authority
    return max(0.0, min(1.0, blended))


# ---------- Writer ----------

_SENTINEL = object()  # queue sentinel for shutdown


class FactWriter:
    """Single-writer async queue drain loop.

    Agents call `submit(claim)` (non-blocking); the writer coroutine awaits
    items off the queue and runs the reduce pipeline serially. On shutdown,
    `drain()` sends a sentinel and awaits the drain task to completion.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        resolver: EntityResolver,
        entity_schema: dict,
        emit_fact: Callable[[str, FactClaim], Awaitable[None]],
        emit_conflict: Callable[[str, list[FieldCell]], Awaitable[None]],
        queue_maxsize: int = 2048,
    ) -> None:
        self._store = store
        self._resolver = resolver
        self._schema = entity_schema
        self._emit_fact = emit_fact
        self._emit_conflict = emit_conflict
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=queue_maxsize)
        self._task: Optional[asyncio.Task] = None
        self._metrics = {
            "submitted": 0,
            "written": 0,
            "duplicates": 0,
            "conflicts": 0,
            "type_errors": 0,
            "drops_queue_full": 0,
        }

    # ---------- Public API ----------

    async def submit(self, claim: FactClaim) -> None:
        """Enqueue a claim for the drain loop.

        Non-blocking: raises asyncio.QueueFull if the queue is saturated so the
        caller can log and retry on the next cycle. This back-pressure signal
        is intentional — we never silently drop facts.
        """
        self._metrics["submitted"] += 1
        try:
            self._queue.put_nowait(claim)
        except asyncio.QueueFull:
            self._metrics["drops_queue_full"] += 1
            raise

    async def run(self) -> None:
        """Drain loop. Exits cleanly when the sentinel is received."""
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                self._queue.task_done()
                return
            try:
                await self._process(item)
            finally:
                self._queue.task_done()

    async def drain(self, timeout_s: float = 30.0) -> None:
        """Signal the drain loop to stop and await completion."""
        await self._queue.put(_SENTINEL)
        if self._task is not None:
            await asyncio.wait_for(self._task, timeout=timeout_s)

    async def quiesce(self, timeout_s: float = 30.0) -> None:
        """Wait for all currently-queued claims to be processed.

        Unlike `drain()`, this does NOT stop the writer task — the drain loop
        stays running and can accept more claims afterward. Used by the
        orchestrator to enforce a per-cycle barrier between map and the next
        cycle's plan without tearing down the writer.
        """
        await asyncio.wait_for(self._queue.join(), timeout=timeout_s)

    def start(self) -> asyncio.Task:
        """Spawn the drain coroutine as a background task."""
        if self._task is not None:
            raise RuntimeError("FactWriter already started")
        self._task = asyncio.create_task(self.run())
        return self._task

    @property
    def metrics(self) -> dict:
        return dict(self._metrics)

    # ---------- Internal pipeline ----------

    async def _process(self, claim: FactClaim) -> WriteOutcome:
        # 1. validate type
        try:
            claim = await validate_type(claim, self._schema)
        except Exception:
            self._metrics["type_errors"] += 1
            return WriteOutcome(
                written=False, entity_id=None, reason="type_error"
            )

        # 2. dedup (no-op in v1-B; store.upsert_entity handles same-name merge)
        if await dedup(claim, self._store):
            self._metrics["duplicates"] += 1
            return WriteOutcome(
                written=False, entity_id=None, reason="duplicate"
            )

        # 3. conflict detect
        existing_cells = await detect_conflict(claim, self._store)
        if existing_cells:
            self._metrics["conflicts"] += 1
            return await self._record_conflict(claim, existing_cells)

        # 4. score confidence
        scored_confidence = await score_confidence(claim, {})
        claim = claim.model_copy(update={"confidence": scored_confidence})

        # 5. persist provenance + field cell
        return await self._persist(claim)

    async def _persist(self, claim: FactClaim) -> WriteOutcome:
        """Record provenance, upsert the entity field, and emit fact_written."""
        prov_id = uuid4().hex
        try:
            await self._store.record_provenance(
                prov_id, claim.provenance.model_dump(mode="json")
            )
        except Exception:
            # Best-effort: provenance write failure shouldn't sink the field
            # write — but the field cell will still link to prov_id, so the
            # downstream auditor will flag the dangling reference.
            pass

        cell = FieldCell(
            value=claim.value,
            confidence=float(claim.confidence),
            provenance_ids=[prov_id],
            updated_at=datetime.now(timezone.utc),
        )

        try:
            entity_id = await self._store.upsert_entity(
                claim.entity_type,
                claim.entity_name,
                {claim.field: cell},
            )
        except Exception:
            # Persist failure is non-fatal to the writer loop; surface via
            # the type_errors counter for now (Wave 1-C will add a dedicated
            # store_errors counter once we wire structured errors).
            self._metrics["type_errors"] += 1
            return WriteOutcome(
                written=False, entity_id=None, reason="store_error"
            )

        self._metrics["written"] += 1
        try:
            await self._emit_fact(entity_id, claim)
        except Exception:
            # Emitter failures must not corrupt the writer; the fact is
            # already persisted at this point.
            pass
        return WriteOutcome(written=True, entity_id=entity_id, reason="ok")

    async def _record_conflict(
        self, claim: FactClaim, existing_cells: list[FieldCell]
    ) -> WriteOutcome:
        """Build a Conflict row, persist it, and notify the emitter."""
        new_cell = FieldCell(
            value=claim.value,
            confidence=float(claim.confidence),
            provenance_ids=[],
            updated_at=datetime.now(timezone.utc),
        )

        # Look up the entity id so the conflict row references the right row.
        entity_id = ""
        try:
            rows = await self._store.query(
                "SELECT id FROM entities WHERE entity_type = ? AND name = ?",
                (claim.entity_type, claim.entity_name),
            )
            if rows:
                entity_id = rows[0]["id"]
        except Exception:
            entity_id = ""

        conflict = Conflict(
            conflict_id=uuid4().hex,
            entity_id=entity_id,
            field=claim.field,
            candidates=list(existing_cells) + [new_cell],
            status="open",
        )

        try:
            await self._store.record_conflict(conflict)
        except Exception:
            pass

        try:
            await self._emit_conflict(entity_id, existing_cells)
        except Exception:
            pass

        return WriteOutcome(
            written=False,
            entity_id=entity_id or None,
            reason="conflict",
            conflict_id=conflict.conflict_id,
        )
