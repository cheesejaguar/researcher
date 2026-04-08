"""Scheduler auto-EXPAND after a DISCOVER cycle.

DiscoverAgent emits one FactClaim per new entity (just the `name` field).
The scheduler notices that entities now exist with only `name` filled and
auto-enqueues EXPAND tasks so the next cycle can populate the full schema.
Capped at spec.max_entities_per_cycle and gated on the queue being empty
so in-progress work is never preempted.
"""

from __future__ import annotations

from typing import Any

import pytest

from researcher.models import TaskKind
from researcher.scheduler import Scheduler
from researcher.spec import EntitySpec, FieldSpec, RunSpec, SearchConfig
from researcher.storage.store import StoreMetrics


def _spec(max_entities_per_cycle: int = 60, seeds: list[str] | None = None) -> RunSpec:
    return RunSpec(
        spec_id="test",
        goal="test",
        entities=[
            EntitySpec(
                name="Widget",
                fields=[
                    FieldSpec(name="name", type="str", required=True),
                    FieldSpec(name="color", type="str"),
                ],
            )
        ],
        seeds=seeds or ["a", "b"],
        search=SearchConfig(),
        budget_usd=1.0,
        wall_limit_s=60,
        max_cycles=3,
        max_entities_per_cycle=max_entities_per_cycle,
        models={"fast": "stub", "smart": "stub", "heavy": "stub"},
    )


class _StubStore:
    """Minimal KnowledgeStore stub that just answers the sparse-entity query."""

    def __init__(self, sparse_ids: list[str]) -> None:
        self._sparse_ids = sparse_ids
        self.query_calls: list[tuple[str, tuple]] = []

    async def query(self, sql: str, params: tuple = ()) -> list[dict]:
        self.query_calls.append((sql, params))
        # The scheduler issues a LIMIT query; respect the limit param if it's
        # the last positional argument.
        limit = params[-1] if params else len(self._sparse_ids)
        return [{"id": sid} for sid in self._sparse_ids[:limit]]


class _EmptyQueryStore:
    async def query(self, sql: str, params: tuple = ()) -> list[dict]:
        return []


class _RaisingStore:
    async def query(self, sql: str, params: tuple = ()) -> list[dict]:
        raise RuntimeError("no such table: fields")


def _metrics(entities: int) -> StoreMetrics:
    return StoreMetrics(
        entities_total=entities,
        by_type={"Widget": entities},
        fields_filled_pct=0.0,
        conflicts_open=0,
        cost_usd_total=0.0,
    )


@pytest.mark.asyncio
async def test_update_from_metrics_enqueues_expand_for_sparse_entities() -> None:
    store: Any = _StubStore(sparse_ids=["e1", "e2", "e3"])
    sched = Scheduler(spec=_spec(max_entities_per_cycle=10), store=store)

    await sched.update_from_metrics(_metrics(entities=3))

    batch = await sched.next_batch(max_n=10)
    assert len(batch) == 3
    assert all(t.kind == TaskKind.EXPAND for t in batch)
    assert {t.target_entity_id for t in batch} == {"e1", "e2", "e3"}


@pytest.mark.asyncio
async def test_update_from_metrics_respects_max_entities_per_cycle() -> None:
    # 100 sparse entities in the store, cap is 7 → we enqueue 7 only.
    store: Any = _StubStore(sparse_ids=[f"e{i}" for i in range(100)])
    sched = Scheduler(spec=_spec(max_entities_per_cycle=7), store=store)

    await sched.update_from_metrics(_metrics(entities=100))

    batch = await sched.next_batch(max_n=50)
    assert len(batch) == 7


@pytest.mark.asyncio
async def test_update_from_metrics_skips_when_queue_nonempty() -> None:
    """In-progress work must not be preempted by auto-expand."""
    store: Any = _StubStore(sparse_ids=["e1", "e2"])
    sched = Scheduler(spec=_spec(), store=store)

    # Seed work manually, then trigger update_from_metrics — nothing should
    # be added to the queue because it already has pending work.
    await sched.seed()
    pre = sched.pending
    await sched.update_from_metrics(_metrics(entities=2))

    assert sched.pending == pre
    # And the store's query method should not have been hit.
    assert store.query_calls == []


@pytest.mark.asyncio
async def test_update_from_metrics_noop_when_no_sparse_entities() -> None:
    store: Any = _EmptyQueryStore()
    sched = Scheduler(spec=_spec(), store=store)

    await sched.update_from_metrics(_metrics(entities=0))

    batch = await sched.next_batch(max_n=10)
    assert batch == []


@pytest.mark.asyncio
async def test_update_from_metrics_swallows_query_exceptions() -> None:
    """A store without the `fields` table (stub stores) must not crash."""
    store: Any = _RaisingStore()
    sched = Scheduler(spec=_spec(), store=store)

    # Must not raise.
    await sched.update_from_metrics(_metrics(entities=0))

    batch = await sched.next_batch(max_n=10)
    assert batch == []


@pytest.mark.asyncio
async def test_enqueued_expand_tasks_carry_budget_and_deadline() -> None:
    store: Any = _StubStore(sparse_ids=["e1", "e2"])
    sched = Scheduler(spec=_spec(), store=store)

    await sched.update_from_metrics(_metrics(entities=2))
    batch = await sched.next_batch(max_n=10)

    for task in batch:
        assert task.budget_usd > 0
        assert task.deadline_ts is not None
        assert task.kind == TaskKind.EXPAND
        assert task.spec_ref == "test"
