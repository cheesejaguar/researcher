"""FactWriter — the single-writer reduce step.

**Serial reduce, parallel map.** Agents (map) run concurrently and submit
FactClaim objects to this queue. A single drain coroutine (the writer) runs the
`validate_type -> dedup -> detect_conflict -> score_confidence` pipeline on each
claim and commits to the store. Making the reduce step serial is correct, not a
compromise — DuckDB is single-writer and the reduce pipeline is CPU-bound + fast
relative to network latency.

Wave 0 provides the queue scaffolding and a no-op pipeline so Wave 1-B can
replace the stages one at a time with real logic. The scaffold tests
(test_writer_pipeline.py) land in Wave 1-B.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from researcher.models import FactClaim
from researcher.storage.resolver import EntityResolver
from researcher.storage.store import FieldCell, KnowledgeStore


@dataclass
class WriteOutcome:
    written: bool
    entity_id: Optional[str]
    reason: str  # "ok" | "duplicate" | "conflict" | "type_error" | "low_confidence"
    conflict_id: Optional[str] = None


# ---------- Pipeline stages ----------
# Wave 1-B replaces the no-op implementations with real logic. The stages are
# kept as small composable functions so each can be tested in isolation.


async def validate_type(claim: FactClaim, entity_schema: dict) -> FactClaim:
    """Coerce/validate claim.value against the entity schema's declared field type."""
    return claim  # no-op for Wave 0


async def dedup(claim: FactClaim, store: KnowledgeStore) -> bool:
    """Return True iff this exact (entity, field, value, source) is already recorded."""
    return False  # no-op for Wave 0 — nothing is a duplicate yet


async def detect_conflict(claim: FactClaim, store: KnowledgeStore) -> list[FieldCell]:
    """Return any existing FieldCells that contradict this claim; empty list = no conflict."""
    return []  # no-op for Wave 0


async def score_confidence(claim: FactClaim, context: dict) -> float:
    """Blend source authority + agent self-report into a final confidence score."""
    return claim.confidence  # no-op for Wave 0 — pass through


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
            return WriteOutcome(written=False, entity_id=None, reason="type_error")

        # 2. dedup
        if await dedup(claim, self._store):
            self._metrics["duplicates"] += 1
            return WriteOutcome(written=False, entity_id=None, reason="duplicate")

        # 3. conflict detect
        conflicts = await detect_conflict(claim, self._store)
        if conflicts:
            self._metrics["conflicts"] += 1
            # TODO Wave 1-B: store conflict, route to ReconcilerAgent.
            return WriteOutcome(written=False, entity_id=None, reason="conflict")

        # 4. score confidence + write
        claim = claim.model_copy(update={"confidence": await score_confidence(claim, {})})
        # TODO Wave 1-B: resolve entity id, record provenance, upsert field cell.
        self._metrics["written"] += 1
        return WriteOutcome(written=True, entity_id=None, reason="ok")
