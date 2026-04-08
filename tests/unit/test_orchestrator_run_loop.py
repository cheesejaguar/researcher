"""Tests for the Orchestrator.run() loop — agent dispatch, claim submission, stop reasons."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from researcher.backends.models import CliKind, CliResult, SubagentResponse
from researcher.backends.resolver import BackendResolver
from researcher.budget import Budget
from researcher.events import CycleEnd, CycleStart, RunComplete, SubagentCall
from researcher.models import AgentResult, Task, TaskKind
from researcher.orchestrator import Orchestrator, StopReason
from researcher.scheduler import Scheduler
from researcher.spec import EntitySpec, FieldSpec, RunSpec
from researcher.storage.writer import FactWriter
from tests.stubs.bus import StubEventBus
from tests.stubs.cli_runner import (
    StubCliRunner,
    make_empty_result,
    make_wars_discover_result,
)
from tests.stubs.llm import StubLLMClient
from tests.stubs.resolver import StubEntityResolver
from tests.stubs.store import StubKnowledgeStore


def _which_claude(cmd: str) -> str | None:
    return "/usr/local/bin/claude" if cmd == "claude" else None


def _which_none(_: str) -> str | None:
    return None


def _sample_spec(policy: str = "auto", max_cycles: int = 1) -> RunSpec:
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
        seeds=["major wars"],
        models={"fast": "m", "smart": "m", "heavy": "m"},
        backend_policy=policy,
        max_cycles=max_cycles,
    )


async def _noop_fact(_entity_id, _claim):
    return None


async def _noop_conflict(_entity_id, _cells):
    return None


async def _make_orchestrator(
    spec: RunSpec,
    which_fn=_which_claude,
    runner: StubCliRunner | None = None,
    budget: Budget | None = None,
) -> tuple[Orchestrator, StubEventBus, StubKnowledgeStore, FactWriter]:
    store = StubKnowledgeStore()
    await store.open()
    llm = StubLLMClient()
    bus = StubEventBus()
    resolver = StubEntityResolver()
    entity_schema = {
        "entity_type": "War",
        "fields": [{"name": "name", "type": "str", "required": True}],
    }
    writer = FactWriter(
        store=store,
        resolver=resolver,
        entity_schema=entity_schema,
        emit_fact=_noop_fact,
        emit_conflict=_noop_conflict,
    )
    scheduler = Scheduler(spec=spec, store=store)
    if budget is None:
        budget = Budget(usd_cap=3.0, wall_cap_s=600)
    backend_resolver = BackendResolver(which_fn=which_fn)

    orch = Orchestrator(
        spec=spec,
        store=store,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        bus=bus,  # type: ignore[arg-type]
        writer=writer,
        scheduler=scheduler,
        budget=budget,
        run_id="run-x",
        max_parallel_agents=2,
    )
    orch.set_backend_resolver(backend_resolver)
    if runner is not None:
        orch.set_cli_runner_factory(lambda kind: runner)
    return orch, bus, store, writer


@pytest.mark.asyncio
async def test_run_dispatches_subagent_and_submits_claims_to_writer():
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    spec = _sample_spec(policy="auto")
    orch, bus, _store, writer = await _make_orchestrator(spec, runner=runner)

    reason = await orch.run()

    # The subagent was called.
    assert len(runner.calls) >= 1
    # Claims submitted to writer. The fixture has 4 extractions, so at least 4 submits.
    assert writer.metrics["submitted"] >= 4
    # No exceptions propagated.
    assert reason in (StopReason.NO_TASKS, StopReason.PLATEAU, StopReason.BUDGET, StopReason.DEADLINE)
    # At least one SubagentCall event on the bus.
    subagent_events = [e for e in bus.events if isinstance(e, SubagentCall)]
    assert len(subagent_events) >= 1


@pytest.mark.asyncio
async def test_run_emits_cycle_and_run_events():
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    spec = _sample_spec(policy="auto")
    orch, bus, _store, _writer = await _make_orchestrator(spec, runner=runner)

    await orch.run()

    cycle_starts = [e for e in bus.events if isinstance(e, CycleStart)]
    cycle_ends = [e for e in bus.events if isinstance(e, CycleEnd)]
    run_completes = [e for e in bus.events if isinstance(e, RunComplete)]

    assert len(cycle_starts) == 1
    assert len(cycle_ends) == 1
    assert len(run_completes) == 1
    assert cycle_starts[0].payload.cycle == 1
    assert cycle_ends[0].payload.cycle == 1


@pytest.mark.asyncio
async def test_run_native_path_raises_not_implemented():
    """With no CLI detected and backend_policy=auto, _spawn_agent should raise.

    Wave 1-D (native agents) isn't implemented yet; we fail loud with a clear
    message rather than silently producing bogus results.
    """
    spec = _sample_spec(policy="auto")
    orch, _bus, _store, _writer = await _make_orchestrator(spec, which_fn=_which_none)

    reason = await orch.run()
    # The run should end with ERROR (the NotImplementedError was caught and stored).
    assert reason == StopReason.ERROR


@pytest.mark.asyncio
async def test_run_records_subagent_failure_and_circuit_breaks():
    """After 3 spawn/timeout failures in one cycle, the resolver is circuit-broken."""
    runner = StubCliRunner()
    runner.add_response_for_any(
        CliResult(ok=False, error="timeout", wall_ms=100, exit_code=None)
    )
    # Need enough tasks to trigger the circuit break (3+ in one cycle).
    spec = _sample_spec(policy="auto")
    # Force multiple tasks by expanding the seed list.
    spec = RunSpec(
        spec_id="wars",
        goal="Wars",
        entities=spec.entities,
        seeds=["seed1", "seed2", "seed3", "seed4"],
        models={"fast": "m", "smart": "m", "heavy": "m"},
        backend_policy="auto",
        max_cycles=1,
    )
    orch, _bus, _store, _writer = await _make_orchestrator(spec, runner=runner)

    await orch.run()

    # After >= 3 failures, the resolver should have been circuit-broken.
    # Either detected is empty OR the cleared_reason is set.
    resolver = orch._backend_resolver  # type: ignore[attr-defined]
    assert resolver is not None
    assert resolver.detected == []  # circuit break cleared it


@pytest.mark.asyncio
async def test_run_stops_with_subagent_cap_reason():
    """If the budget's subagent cap is hit, the run stops with SUBAGENT_CAP."""
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    budget = Budget(usd_cap=3.0, wall_cap_s=600, max_subagent_calls=1)
    spec = _sample_spec(policy="auto")
    spec = RunSpec(
        spec_id="wars",
        goal="Wars",
        entities=spec.entities,
        seeds=["seed1", "seed2"],  # two tasks; the second should hit the cap
        models={"fast": "m", "smart": "m", "heavy": "m"},
        backend_policy="auto",
        max_cycles=1,
    )
    orch, _bus, _store, _writer = await _make_orchestrator(
        spec, runner=runner, budget=budget
    )

    reason = await orch.run()
    # After the cap is hit, the run should end with SUBAGENT_CAP (possibly after
    # the cycle finishes).
    assert reason == StopReason.SUBAGENT_CAP
    assert budget.subagent_calls_total == 1  # only one call allowed


@pytest.mark.asyncio
async def test_run_quiesces_writer_between_cycles():
    """After run() returns, the writer task should have fully drained."""
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    spec = _sample_spec(policy="auto")
    orch, _bus, _store, writer = await _make_orchestrator(spec, runner=runner)

    await orch.run()

    # Writer task should be done (drained at end of run).
    assert writer._task is not None
    assert writer._task.done()
