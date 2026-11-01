"""Scheduler follow-up scheduling: VERIFY, ENRICH, CRITIC, and dedupe."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from researcher.models import TaskKind
from researcher.scheduler import Scheduler
from researcher.spec import EntitySpec, FieldSpec, RelationSpec, RunSpec
from researcher.storage.store import StoreMetrics


def _spec(**overrides: Any) -> RunSpec:
    kwargs: dict[str, Any] = {
        "spec_id": "test",
        "goal": "test",
        "entities": [
            EntitySpec(
                name="War",
                fields=[
                    FieldSpec(name="name", type="str", required=True),
                    FieldSpec(name="start_year", type="int"),
                ],
            )
        ],
        "seeds": [],
        "models": {"fast": "stub"},
        "budget_usd": 1.0,
        "wall_limit_s": 60,
        "max_cycles": 3,
    }
    kwargs.update(overrides)
    return RunSpec(**kwargs)


class _Store:
    def __init__(
        self,
        sparse: list[str] | None = None,
        entities: list[str] | None = None,
        conflicts: list[Any] | None = None,
    ) -> None:
        self.sparse = sparse or []
        self.entities = entities or []
        self.conflicts = conflicts or []

    async def query(self, sql: str, params: tuple = ()) -> list[dict]:
        if "NOT IN" in sql:
            return [{"id": entity_id} for entity_id in self.sparse]
        if "updated_at" in sql:
            limit = params[-1] if params else len(self.entities)
            return [{"id": entity_id} for entity_id in self.entities[:limit]]
        return []

    async def get_conflicts(self, status: str = "open") -> list[Any]:
        return list(self.conflicts)


def _metrics(entities: int = 0) -> StoreMetrics:
    return StoreMetrics(
        entities_total=entities,
        by_type={"War": entities},
        fields_filled_pct=0.0,
        conflicts_open=0,
        cost_usd_total=0.0,
    )


@pytest.mark.asyncio
async def test_open_conflicts_enqueue_verify_tasks_once() -> None:
    conflicts = [
        SimpleNamespace(entity_id="e1", field="start_year"),
        SimpleNamespace(entity_id="e2", field="end_year"),
    ]
    sched = Scheduler(spec=_spec(), store=_Store(conflicts=conflicts))

    await sched.update_from_metrics(_metrics(2))
    first = await sched.next_batch(max_n=10)
    assert [t.kind for t in first] == [TaskKind.VERIFY, TaskKind.VERIFY]
    assert {t.target_entity_id for t in first} == {"e1", "e2"}
    assert [t.field_hints for t in first] == [["start_year"], ["end_year"]]

    # Same conflicts next cycle must not duplicate work.
    await sched.update_from_metrics(_metrics(2))
    assert await sched.next_batch(max_n=10) == []


@pytest.mark.asyncio
async def test_declared_relations_enqueue_bounded_enrich_tasks_once() -> None:
    relations = [RelationSpec(name="ally_of", source_type="War", target_type="War")]
    sched = Scheduler(
        spec=_spec(relations=relations, max_relations_per_cycle=2),
        store=_Store(entities=["e1", "e2", "e3"]),
    )

    await sched.update_from_metrics(_metrics(3))
    first = await sched.next_batch(max_n=10)
    assert len(first) == 2
    assert all(t.kind == TaskKind.ENRICH for t in first)
    assert all(t.field_hints == ["ally_of"] for t in first)

    # The cap defers e3 to the next cycle rather than dropping it.
    await sched.update_from_metrics(_metrics(3))
    second = await sched.next_batch(max_n=10)
    assert [t.target_entity_id for t in second] == ["e3"]
    assert second[0].kind == TaskKind.ENRICH

    # Once every entity has been enriched, no duplicate tasks are emitted.
    await sched.update_from_metrics(_metrics(3))
    assert await sched.next_batch(max_n=10) == []


@pytest.mark.asyncio
async def test_critic_is_scheduled_at_most_once_per_run() -> None:
    sched = Scheduler(spec=_spec(enable_critic=True), store=_Store())

    await sched.update_from_metrics(_metrics(0))
    first = await sched.next_batch(max_n=10)
    assert [t.kind for t in first] == [TaskKind.CRITIC]

    await sched.update_from_metrics(_metrics(0))
    assert await sched.next_batch(max_n=10) == []


@pytest.mark.asyncio
async def test_critic_disabled_by_default() -> None:
    sched = Scheduler(spec=_spec(), store=_Store())
    await sched.update_from_metrics(_metrics(0))
    assert await sched.next_batch(max_n=10) == []


@pytest.mark.asyncio
async def test_followup_dedupe_survives_checkpoint_roundtrip() -> None:
    conflicts = [SimpleNamespace(entity_id="e1", field="start_year")]
    store = _Store(conflicts=conflicts)
    sched = Scheduler(spec=_spec(), store=store)
    await sched.update_from_metrics(_metrics(1))
    snapshot = sched.serialize()
    assert await sched.next_batch(max_n=10)  # clear the live queue

    restored = Scheduler(spec=_spec(), store=store)
    restored.restore(snapshot)
    assert await restored.next_batch(max_n=10)  # restored pending VERIFY
    await restored.update_from_metrics(_metrics(1))
    assert await restored.next_batch(max_n=10) == []


@pytest.mark.asyncio
async def test_scheduler_followups_stay_under_per_task_subcap() -> None:
    spec = _spec(
        budget_usd=10.0,
        relations=[RelationSpec(name="ally_of", source_type="War", target_type="War")],
    )
    sched = Scheduler(spec=spec, store=_Store(entities=["e1"], sparse=["e1"]))
    await sched.seed()
    await sched.update_from_metrics(_metrics(1))
    batch = await sched.next_batch(max_n=10)
    assert batch
    for task in batch:
        assert task.budget_usd <= spec.budget_usd * 0.02 + 1e-9
