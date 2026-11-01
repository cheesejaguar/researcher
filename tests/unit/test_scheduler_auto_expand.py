"""Scheduler auto-EXPAND after a DISCOVER cycle.

DiscoverAgent emits one FactClaim per new entity (just the `name` field).
The scheduler notices that entities now exist with only `name` filled and
auto-enqueues EXPAND tasks so the next cycle can populate the full schema.
Capped at spec.max_entities_per_cycle and gated on the queue being empty
so in-progress work is never preempted.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from researcher.models import TaskKind
from researcher.scheduler import Scheduler
from researcher.spec import EntitySpec, FieldSpec, RunSpec, SearchConfig
from researcher.storage.duckdb_store import DuckDBKnowledgeStore
from researcher.storage.store import FieldCell, StoreMetrics


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


def _cell(value: Any) -> FieldCell:
    return FieldCell(
        value=value, confidence=0.9, provenance_ids=[], updated_at=datetime.now(UTC)
    )


async def _seed_sparse(store: DuckDBKnowledgeStore, names: list[str]) -> list[str]:
    """Insert name-only entities (what DiscoverAgent leaves behind)."""
    return [await store.upsert_entity("Widget", n, {}) for n in names]


class _QueryCountingStore(DuckDBKnowledgeStore):
    """Real store that also records whether the scheduler queried it."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.query_calls = 0

    async def query(self, sql: str, params: tuple = ()) -> list[dict]:
        self.query_calls += 1
        return await super().query(sql, params)


def _metrics(entities: int) -> StoreMetrics:
    return StoreMetrics(
        entities_total=entities,
        by_type={"Widget": entities},
        fields_filled_pct=0.0,
        conflicts_open=0,
        cost_usd_total=0.0,
    )


@pytest.mark.asyncio
async def test_update_from_metrics_enqueues_expand_for_sparse_entities(store) -> None:
    sparse_ids = await _seed_sparse(store, ["w1", "w2", "w3"])
    # An entity with a non-name field filled is NOT sparse; one whose only
    # extra field is an explicit null still counts as sparse.
    await store.upsert_entity("Widget", "filled", {"color": _cell("red")})
    null_id = await store.upsert_entity("Widget", "nulled", {"color": _cell(None)})
    sched = Scheduler(spec=_spec(max_entities_per_cycle=10), store=store)

    await sched.update_from_metrics(_metrics(entities=5))

    batch = await sched.next_batch(max_n=10)
    assert all(t.kind == TaskKind.EXPAND for t in batch)
    assert {t.target_entity_id for t in batch} == {*sparse_ids, null_id}


@pytest.mark.asyncio
async def test_update_from_metrics_respects_max_entities_per_cycle(store) -> None:
    # 20 sparse entities in the store, cap is 7 -> we enqueue 7 only.
    await _seed_sparse(store, [f"w{i}" for i in range(20)])
    sched = Scheduler(spec=_spec(max_entities_per_cycle=7), store=store)

    await sched.update_from_metrics(_metrics(entities=20))

    batch = await sched.next_batch(max_n=50)
    assert len(batch) == 7


@pytest.mark.asyncio
async def test_update_from_metrics_skips_when_queue_nonempty(tmp_path: Path) -> None:
    """In-progress work must not be preempted by auto-expand."""
    store = _QueryCountingStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await _seed_sparse(store, ["w1", "w2"])
        sched = Scheduler(spec=_spec(), store=store)

        # Seed work manually, then trigger update_from_metrics — nothing should
        # be added to the queue because it already has pending work.
        await sched.seed()
        pre = sched.pending
        await sched.update_from_metrics(_metrics(entities=2))

        assert sched.pending == pre
        # And the store's query method should not have been hit.
        assert store.query_calls == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_update_from_metrics_noop_when_no_sparse_entities(store) -> None:
    await store.upsert_entity("Widget", "filled", {"color": _cell("red")})
    sched = Scheduler(spec=_spec(), store=store)

    await sched.update_from_metrics(_metrics(entities=1))

    batch = await sched.next_batch(max_n=10)
    assert batch == []


@pytest.mark.asyncio
async def test_update_from_metrics_swallows_query_exceptions(tmp_path: Path) -> None:
    """A store whose query fails (e.g. a missing table) must not crash."""
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await _seed_sparse(store, ["w1"])
        await store.query("DROP TABLE fields")
        sched = Scheduler(spec=_spec(), store=store)

        # Must not raise.
        await sched.update_from_metrics(_metrics(entities=1))

        batch = await sched.next_batch(max_n=10)
        assert batch == []
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_enqueued_expand_tasks_carry_budget_and_deadline(store) -> None:
    await _seed_sparse(store, ["w1", "w2"])
    spec = _spec()
    sched = Scheduler(spec=spec, store=store)

    before = datetime.now(UTC)
    await sched.update_from_metrics(_metrics(entities=2))
    batch = await sched.next_batch(max_n=10)

    assert len(batch) == 2
    for task in batch:
        # budget_usd / n_sparse / 10
        assert task.budget_usd == pytest.approx(spec.budget_usd / 2 / 10)
        assert task.deadline_ts >= before + timedelta(seconds=spec.wall_limit_s)
        assert task.kind == TaskKind.EXPAND
        assert task.spec_ref == "test"
