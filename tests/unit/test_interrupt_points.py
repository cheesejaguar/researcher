"""Tests for HITL interrupt points in the orchestrator."""

from __future__ import annotations

import pytest

from researcher.interrupts import InterruptDecision, StubInterruptHandler
from researcher.spec import EntitySpec, FieldSpec, RunSpec


def _sample_spec(interrupts: list[str] | None = None) -> RunSpec:
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
        seeds=["seed1"],
        models={"fast": "m"},
        max_cycles=1,
        interrupt_points=interrupts or [],
    )


# ---------- Spec ----------


def test_runspec_interrupt_points_default_empty():
    s = _sample_spec()
    assert s.interrupt_points == []


def test_runspec_interrupt_points_accepts_list():
    s = _sample_spec(["after_initial_seed", "after_cycle_end"])
    assert s.interrupt_points == ["after_initial_seed", "after_cycle_end"]


# ---------- StubInterruptHandler ----------


@pytest.mark.asyncio
async def test_stub_interrupt_handler_records_calls():
    handler = StubInterruptHandler(default_decision=InterruptDecision.CONTINUE)
    decision = await handler.handle(point="after_cycle_end", context={"cycle": 1})
    assert decision == InterruptDecision.CONTINUE
    assert len(handler.calls) == 1
    assert handler.calls[0]["point"] == "after_cycle_end"
    assert handler.calls[0]["context"]["cycle"] == 1


@pytest.mark.asyncio
async def test_stub_interrupt_handler_can_abort():
    handler = StubInterruptHandler(default_decision=InterruptDecision.ABORT)
    decision = await handler.handle(point="after_initial_seed", context={})
    assert decision == InterruptDecision.ABORT


# ---------- Orchestrator integration ----------


@pytest.mark.asyncio
async def test_orchestrator_calls_interrupt_handler_at_declared_points(tmp_path):
    """End-to-end: with after_cycle_end declared, the handler is called per cycle."""
    from researcher.backends.resolver import BackendResolver
    from researcher.budget import Budget
    from researcher.orchestrator import Orchestrator
    from researcher.scheduler import Scheduler
    from researcher.spec import build_entity_class
    from researcher.storage.duckdb_store import DuckDBKnowledgeStore
    from researcher.storage.writer import FactWriter
    from tests.stubs.bus import StubEventBus
    from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
    from tests.stubs.llm import StubLLMClient
    from tests.stubs.resolver import StubEntityResolver

    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    spec = _sample_spec(interrupts=["after_cycle_end"])
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(build_entity_class(spec.entities[0]))
        bus = StubEventBus()
        llm = StubLLMClient()
        resolver = StubEntityResolver()
        entity_schema = {
            "entity_type": "War",
            "fields": [{"name": "name", "type": "str", "required": True}],
        }

        async def _noop(*args, **kwargs):
            return None

        writer = FactWriter(
            store=store,
            resolver=resolver,
            entity_schema=entity_schema,
            emit_fact=_noop,
            emit_conflict=_noop,
        )
        scheduler = Scheduler(spec=spec, store=store)
        budget = Budget(usd_cap=3.0, wall_cap_s=600)
        backend_resolver = BackendResolver(
            which_fn=lambda c: "/usr/local/bin/claude" if c == "claude" else None
        )

        handler = StubInterruptHandler(default_decision=InterruptDecision.CONTINUE)

        orch = Orchestrator(
            spec=spec,
            store=store,  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
            bus=bus,  # type: ignore[arg-type]
            writer=writer,
            scheduler=scheduler,
            budget=budget,
            run_id="interrupt-test",
            max_parallel_agents=2,
        )
        orch.set_backend_resolver(backend_resolver)
        orch.set_cli_runner_factory(lambda kind: runner)
        orch.set_interrupt_handler(handler)

        await orch.run()

        # Handler should have been called at least once for after_cycle_end.
        cycle_end_calls = [c for c in handler.calls if c["point"] == "after_cycle_end"]
        assert len(cycle_end_calls) >= 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_orchestrator_aborts_when_handler_returns_abort(tmp_path):
    """If the handler returns ABORT, the run terminates with StopReason.CTRL_C."""
    from researcher.backends.resolver import BackendResolver
    from researcher.budget import Budget
    from researcher.orchestrator import Orchestrator, StopReason
    from researcher.scheduler import Scheduler
    from researcher.spec import build_entity_class
    from researcher.storage.duckdb_store import DuckDBKnowledgeStore
    from researcher.storage.writer import FactWriter
    from tests.stubs.bus import StubEventBus
    from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
    from tests.stubs.llm import StubLLMClient
    from tests.stubs.resolver import StubEntityResolver

    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    spec = _sample_spec(interrupts=["after_initial_seed"])
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(build_entity_class(spec.entities[0]))
        bus = StubEventBus()
        llm = StubLLMClient()
        resolver = StubEntityResolver()

        async def _noop(*args, **kwargs):
            return None

        writer = FactWriter(
            store=store,
            resolver=resolver,
            entity_schema={
                "entity_type": "War",
                "fields": [{"name": "name", "type": "str", "required": True}],
            },
            emit_fact=_noop,
            emit_conflict=_noop,
        )
        scheduler = Scheduler(spec=spec, store=store)
        budget = Budget(usd_cap=3.0, wall_cap_s=600)
        backend_resolver = BackendResolver(
            which_fn=lambda c: "/usr/local/bin/claude" if c == "claude" else None
        )

        handler = StubInterruptHandler(default_decision=InterruptDecision.ABORT)
        orch = Orchestrator(
            spec=spec,
            store=store,  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
            bus=bus,  # type: ignore[arg-type]
            writer=writer,
            scheduler=scheduler,
            budget=budget,
            run_id="abort-test",
            max_parallel_agents=2,
        )
        orch.set_backend_resolver(backend_resolver)
        orch.set_cli_runner_factory(lambda kind: runner)
        orch.set_interrupt_handler(handler)

        reason = await orch.run()
        assert reason == StopReason.CTRL_C
    finally:
        await store.close()
