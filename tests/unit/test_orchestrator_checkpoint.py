"""Tests for orchestrator crash-safe checkpointing + resume."""

from __future__ import annotations

import pytest

from researcher.budget import Budget
from researcher.scheduler import Scheduler
from researcher.spec import EntitySpec, FieldSpec, RunSpec
from researcher.storage.duckdb_store import DuckDBKnowledgeStore


def _sample_spec() -> RunSpec:
    return RunSpec(
        spec_id="wars",
        goal="Wars",
        entities=[
            EntitySpec(
                name="War",
                fields=[FieldSpec(name="name", type="str", required=True)],
                search_templates=[],
            )
        ],
        seeds=["seed1", "seed2"],
        models={"fast": "m"},
        max_cycles=3,
    )


@pytest.mark.asyncio
async def test_save_and_load_checkpoint_roundtrip(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.save_checkpoint(
            run_id="run-1",
            cycle=2,
            data={
                "scheduler": {"pending": [], "history": [10, 20]},
                "budget": {"spent": 0.5, "subagent_calls_total": 5},
                "started_at": "2026-04-08T12:00:00Z",
            },
        )
        loaded = await store.load_checkpoint("run-1")
        assert loaded is not None
        assert loaded["cycle"] == 2
        assert loaded["data"]["scheduler"]["history"] == [10, 20]
        assert loaded["data"]["budget"]["spent"] == 0.5
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_load_checkpoint_returns_none_when_missing(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        loaded = await store.load_checkpoint("nonexistent-run")
        assert loaded is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_save_checkpoint_overwrites_previous(tmp_path):
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.save_checkpoint("run-1", cycle=1, data={"x": 1})
        await store.save_checkpoint("run-1", cycle=2, data={"x": 2})
        loaded = await store.load_checkpoint("run-1")
        assert loaded is not None
        assert loaded["cycle"] == 2
        assert loaded["data"]["x"] == 2
    finally:
        await store.close()


def test_scheduler_serialize_roundtrip():
    spec = _sample_spec()

    class _StubStore:
        async def snapshot_metrics(self):
            from researcher.storage.store import StoreMetrics
            return StoreMetrics(
                entities_total=0, by_type={}, fields_filled_pct=0.0,
                conflicts_open=0, cost_usd_total=0.0,
            )

    sched = Scheduler(spec=spec, store=_StubStore())  # type: ignore[arg-type]
    sched._entity_history = [5, 10, 15]
    sched._cycle = 3
    snap = sched.serialize()
    assert snap["cycle"] == 3
    assert snap["entity_history"] == [5, 10, 15]
    # Restore into a fresh scheduler.
    sched2 = Scheduler(spec=spec, store=_StubStore())  # type: ignore[arg-type]
    sched2.restore(snap)
    assert sched2._cycle == 3
    assert sched2._entity_history == [5, 10, 15]


def test_budget_serialize_roundtrip():
    b = Budget(usd_cap=3.0, wall_cap_s=600, max_subagent_calls=200)
    b.spend(1.25)
    b.record_subagent_call()
    b.record_subagent_call()
    snap = b.serialize()
    assert snap["spent"] == 1.25
    assert snap["subagent_calls_total"] == 2
    b2 = Budget(usd_cap=3.0, wall_cap_s=600, max_subagent_calls=200)
    b2.restore(snap)
    assert b2.total_spent() == 1.25
    assert b2.subagent_calls_total == 2


@pytest.mark.asyncio
async def test_orchestrator_saves_checkpoint_each_cycle(tmp_path):
    """End-to-end: run an orchestrator (with stub agents) for 1 cycle and verify a checkpoint row exists."""
    from researcher.orchestrator import Orchestrator
    from researcher.storage.writer import FactWriter
    from tests.stubs.bus import StubEventBus
    from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
    from tests.stubs.llm import StubLLMClient
    from tests.stubs.resolver import StubEntityResolver

    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    spec = _sample_spec()
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        from researcher.spec import build_entity_class
        await store.init_schema(build_entity_class(spec.entities[0]))
        bus = StubEventBus()
        llm = StubLLMClient()
        resolver = StubEntityResolver()
        entity_schema = {
            "entity_type": "War",
            "fields": [{"name": "name", "type": "str", "required": True}],
        }

        async def _noop_fact(_eid, _claim): return None
        async def _noop_conflict(_eid, _cells): return None

        writer = FactWriter(
            store=store, resolver=resolver, entity_schema=entity_schema,
            emit_fact=_noop_fact, emit_conflict=_noop_conflict,
        )
        scheduler = Scheduler(spec=spec, store=store)
        budget = Budget(usd_cap=3.0, wall_cap_s=600)

        from researcher.backends.resolver import BackendResolver
        backend_resolver = BackendResolver(which_fn=lambda c: "/usr/local/bin/claude" if c == "claude" else None)

        orch = Orchestrator(
            spec=spec,
            store=store,  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
            bus=bus,  # type: ignore[arg-type]
            writer=writer,
            scheduler=scheduler,
            budget=budget,
            run_id="ckpt-test",
            max_parallel_agents=2,
        )
        orch.set_backend_resolver(backend_resolver)
        orch.set_cli_runner_factory(lambda kind: runner)

        await orch.run()

        # After the run completes, a checkpoint row should exist.
        ckpt = await store.load_checkpoint("ckpt-test")
        assert ckpt is not None
        assert ckpt["cycle"] >= 1
    finally:
        await store.close()
