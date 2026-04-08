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

    async def seed(self) -> list[Task]:
        """Turn spec.seeds into initial DISCOVER tasks."""
        from datetime import datetime, timedelta

        deadline = datetime.now(UTC) + timedelta(seconds=self._spec.wall_limit_s)
        per_task_budget = self._spec.budget_usd / max(len(self._spec.seeds), 1) / 10
        tasks: list[Task] = []
        for seed in self._spec.seeds:
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
        """Record post-cycle metrics; used by `plateau` to decide stopping."""
        self._entity_history.append(metrics.entities_total)
        self._cycle += 1

    async def mark_done(self, task_id: str, result: AgentResult) -> None:
        """Record that a task finished; for v1 this is a no-op hook."""
        # Wave 1-E may reroute spawned tasks, retry failed ones, etc.
        for spawned in result.spawned_tasks:
            self._queue.append(spawned)

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
