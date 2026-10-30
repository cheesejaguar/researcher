"""Scheduler — decides which tasks to run each cycle and detects plateau.

Wave 0 ships a trivial FIFO implementation: seeds go in, tasks come out, plateau
is based on a 2-cycle sliding window of entity-count deltas. Wave 1-D/E replaces
the seeding logic with planner-agent output and tunes the plateau thresholds.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC

from researcher.models import AgentResult, Task, TaskKind
from researcher.spec import RunSpec
from researcher.storage.store import KnowledgeStore, StoreMetrics


class Scheduler:
    """FIFO task queue with plateau detection over a sliding 2-cycle window."""

    def __init__(
        self,
        spec: RunSpec,
        store: KnowledgeStore,
        plateau_min_new_entities: int = 3,
    ) -> None:
        self._spec = spec
        self._store = store
        self._queue: deque[Task] = deque()
        self._entity_history: list[int] = []  # entity count per cycle
        self._plateau_min = plateau_min_new_entities
        self._cycle = 0
        # Dedupe keys for scheduler-generated follow-up tasks. Persisted in
        # checkpoints so a resume cannot re-enqueue the same follow-up.
        self._scheduled_keys: set[str] = set()
        self._critic_scheduled = False

    async def seed(self) -> list[Task]:
        """Turn spec.seeds into initial DISCOVER tasks."""
        from datetime import datetime, timedelta

        deadline = datetime.now(UTC) + timedelta(seconds=self._spec.wall_limit_s)
        item_queries = [_query_for_item(item) for item in getattr(self._spec, "items", [])]
        queries = [*self._spec.seeds, *item_queries]
        per_task_budget = self._per_task_budget(len(queries))
        tasks: list[Task] = []
        for seed in queries:
            tasks.append(
                Task(
                    kind=TaskKind.DISCOVER,
                    spec_ref=self._spec.spec_id,
                    seed_query=seed,
                    budget_usd=per_task_budget,
                    deadline_ts=deadline,
                )
            )
        self._queue.extend(tasks)
        return list(tasks)

    async def next_batch(self, max_n: int) -> list[Task]:
        """Pop up to max_n tasks off the front of the queue."""
        batch: list[Task] = []
        while self._queue and len(batch) < max_n:
            batch.append(self._queue.popleft())
        return batch

    async def update_from_metrics(self, metrics: StoreMetrics) -> None:
        """Record post-cycle metrics and enqueue the next cycle's work.

        Order of preference:
          1. EXPAND sparse entities (only when the queue is otherwise idle),
          2. VERIFY every open conflict,
          3. ENRICH declared relations,
          4. one optional CRITIC task per run.

        Every follow-up is deduped by ``(kind, target, query, fields)`` so a
        conflict or relation that persists across cycles cannot create an
        unbounded task loop.
        """
        self._entity_history.append(metrics.entities_total)
        self._cycle += 1

        if not self._queue:
            await self._enqueue_sparse_expands()
        await self._enqueue_conflict_verifications()
        await self._enqueue_relation_enrichment()
        self._maybe_enqueue_critic()

    async def _enqueue_sparse_expands(self) -> None:
        """Emit EXPAND tasks for entities that only have their name field."""
        try:
            sparse = await self._store.query(
                """
                SELECT e.id AS id
                FROM entities e
                WHERE e.id NOT IN (
                    SELECT DISTINCT entity_id FROM fields
                    WHERE value_json != 'null' AND field_name != 'name'
                )
                LIMIT ?
                """,
                (self._spec.max_entities_per_cycle,),
            )
        except Exception:
            return

        per_task_budget = self._per_task_budget(len(sparse))
        for row in sparse:
            entity_id = row.get("id")
            if not entity_id:
                continue
            self._enqueue_task(
                Task(
                    kind=TaskKind.EXPAND,
                    spec_ref=self._spec.spec_id,
                    target_entity_id=entity_id,
                    budget_usd=per_task_budget,
                    deadline_ts=self._deadline(),
                )
            )

    async def _enqueue_conflict_verifications(self) -> None:
        """Queue one VERIFY task per open conflict, at most once per field."""
        get_conflicts = getattr(self._store, "get_conflicts", None)
        if not callable(get_conflicts):
            return
        try:
            conflicts = await get_conflicts(status="open")
        except TypeError:
            try:
                conflicts = await get_conflicts()
            except Exception:
                return
        except Exception:
            return

        per_task_budget = self._per_task_budget(max(len(conflicts), 1))
        for conflict in conflicts:
            entity_id = getattr(conflict, "entity_id", "")
            field = getattr(conflict, "field", "")
            if not entity_id or not field:
                continue
            self._enqueue_task(
                Task(
                    kind=TaskKind.VERIFY,
                    spec_ref=self._spec.spec_id,
                    target_entity_id=entity_id,
                    field_hints=[field],
                    budget_usd=per_task_budget,
                    deadline_ts=self._deadline(),
                )
            )

    async def _enqueue_relation_enrichment(self) -> None:
        """Queue at most ``max_relations_per_cycle`` ENRICH tasks per cycle."""
        if not self._spec.relations:
            return
        cap = self._spec.max_relations_per_cycle
        if cap <= 0:
            return
        try:
            rows = await self._store.query(
                "SELECT id FROM entities ORDER BY updated_at ASC LIMIT ?",
                (cap * 4,),
            )
        except Exception:
            return

        relation_names = [r.name for r in self._spec.relations]
        per_task_budget = self._per_task_budget(max(len(rows), 1))
        enqueued = 0
        for row in rows:
            entity_id = row.get("id")
            if not entity_id:
                continue
            added = self._enqueue_task(
                Task(
                    kind=TaskKind.ENRICH,
                    spec_ref=self._spec.spec_id,
                    target_entity_id=entity_id,
                    field_hints=relation_names,
                    budget_usd=per_task_budget,
                    deadline_ts=self._deadline(),
                )
            )
            if added:
                enqueued += 1
            if enqueued >= cap:
                break

    def _maybe_enqueue_critic(self) -> None:
        if not self._spec.enable_critic or self._critic_scheduled:
            return
        self._critic_scheduled = True
        self._enqueue_task(
            Task(
                kind=TaskKind.CRITIC,
                spec_ref=self._spec.spec_id,
                field_hints=["trajectory"],
                budget_usd=self._per_task_budget(1),
                deadline_ts=self._deadline(),
            )
        )

    def _enqueue_task(self, task: Task, *, dedupe: bool = True) -> bool:
        key = self._task_key(task)
        if dedupe and key in self._scheduled_keys:
            return False
        self._scheduled_keys.add(key)
        self._queue.append(task)
        return True

    @staticmethod
    def _task_key(task: Task) -> str:
        fields = ",".join(sorted(task.field_hints))
        return f"{task.kind.value}|{task.target_entity_id or ''}|{task.seed_query or ''}|{fields}"

    def _deadline(self):
        from datetime import datetime, timedelta

        return datetime.now(UTC) + timedelta(seconds=self._spec.wall_limit_s)

    def _per_task_budget(self, task_count: int) -> float:
        """Per-task allocation that respects the entity/task sub-cap budget.

        Scheduler tasks used to be handed ~10% of the total budget each, which
        is above the default 5% per-task and 3% per-entity ceilings. Allocate
        at most 2% so the orchestrator's hard gates are reachable in practice.
        """
        share = self._spec.budget_usd / max(task_count, 1) / 10
        return max(0.0, min(share, self._spec.budget_usd * 0.02))

    async def mark_done(self, task_id: str, result: AgentResult) -> None:
        """Reroute spawned tasks through the dedupe-aware queue."""
        for spawned in result.spawned_tasks:
            self._enqueue_task(spawned)

    def serialize(self) -> dict:
        """Snapshot the scheduler's control state for checkpointing.

        Captures the pending task queue, the entity-count history used by
        plateau detection, and the cycle index. Tasks are dumped via
        :meth:`pydantic.BaseModel.model_dump` so they round-trip through
        JSON. Embeddings, store rows, and any other heavy state are NOT
        included — those live in DuckDB and survive a crash on their own.
        """
        return {
            "cycle": self._cycle,
            "entity_history": list(self._entity_history),
            "queue": [t.model_dump(mode="json") for t in self._queue],
            "scheduled_keys": sorted(self._scheduled_keys),
            "critic_scheduled": self._critic_scheduled,
        }

    def restore(self, data: dict) -> None:
        """Re-hydrate the scheduler from a :meth:`serialize` blob."""
        self._cycle = int(data.get("cycle", 0))
        self._entity_history = list(data.get("entity_history", []))
        self._queue = deque(Task.model_validate(t) for t in data.get("queue", []))
        self._scheduled_keys = set(data.get("scheduled_keys", []))
        self._critic_scheduled = bool(data.get("critic_scheduled", False))

    @property
    def plateau(self) -> bool:
        """True iff the last 2 cycles each added fewer than `plateau_min_new_entities`."""
        if len(self._entity_history) < 3:
            return False  # need at least 2 deltas
        deltas = [
            self._entity_history[-1] - self._entity_history[-2],
            self._entity_history[-2] - self._entity_history[-3],
        ]
        return all(d < self._plateau_min for d in deltas)

    @property
    def pending(self) -> int:
        return len(self._queue)


def _query_for_item(item: dict) -> str:
    for key in ("name", "title", "id"):
        value = item.get(key)
        if value:
            return str(value)
    return " ".join(str(v) for v in item.values() if v)
