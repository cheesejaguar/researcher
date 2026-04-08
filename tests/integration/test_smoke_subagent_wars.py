"""Offline integration smoke for the CLI subagent path.

Builds a minimal Orchestrator with StubCliRunner pre-populated for the
seed query in wars.yaml, runs one cycle, and asserts the subagent path
produces FactClaims that flow through the writer and the event bus
records a subagent_call event.
"""

from datetime import datetime, timedelta, timezone

import pytest

from researcher.agents.subagent import SubagentResearcher
from researcher.backends.models import CliKind
from researcher.backends.resolver import BackendResolver
from researcher.budget import Budget
from researcher.events import CycleEnd, CycleStart, RunComplete, SubagentCall
from researcher.models import Task, TaskKind
from researcher.orchestrator import Orchestrator, StopReason
from researcher.scheduler import Scheduler
from researcher.spec import EntitySpec, FieldSpec, RunSpec
from researcher.storage.writer import FactWriter
from tests.stubs.bus import StubEventBus
from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
from tests.stubs.llm import StubLLMClient
from tests.stubs.resolver import StubEntityResolver
from tests.stubs.store import StubKnowledgeStore


@pytest.mark.asyncio
async def test_subagent_smoke_end_to_end_offline():
    # Arrange
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())

    store = StubKnowledgeStore()
    await store.open()
    bus = StubEventBus()
    llm = StubLLMClient()
    budget = Budget(usd_cap=3.0, wall_cap_s=600)
    entity_schema = {
        "entity_type": "War",
        "fields": [
            {"name": "name", "type": "str", "required": True},
            {"name": "start_year", "type": "int", "required": True},
            {"name": "end_year", "type": "int", "required": False},
            {"name": "belligerents", "type": "list[str]", "required": False},
        ],
    }

    agent = SubagentResearcher(
        agent_id="smoke-1",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="smoke-run",
        runner=runner,
        cli_kind=CliKind.CLAUDE_CODE,
        entity_schema=entity_schema,
        budget=budget,
        goal="Major interstate wars since 1500",
    )

    task = Task(
        kind=TaskKind.DISCOVER,
        spec_ref="wars",
        seed_query="Major wars since 1500",
        field_hints=["name", "start_year", "end_year", "belligerents"],
        budget_usd=0.01,
        deadline_ts=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    # Act
    result = await agent.run(task)

    # Assert — agent result
    assert result.state.value == "done"
    assert len(result.claims) == 4
    assert result.cost_usd == 0.0
    assert result.tokens_in == 500
    assert result.tokens_out == 300

    # Assert — budget
    assert budget.subagent_calls_total == 1
    assert budget.total_spent() == 0.0
    assert not budget.exceeded()

    # Assert — event bus has a subagent_call event with 4 claims
    subagent_events = [e for e in bus.events if isinstance(e, SubagentCall)]
    assert len(subagent_events) == 1
    assert subagent_events[0].payload.claims_emitted == 4
    assert subagent_events[0].payload.cli_kind == "claude_code"

    # Assert — runner was called exactly once
    assert len(runner.calls) == 1


# ---------- Full orchestrator smoke ----------


def _which_claude(cmd: str) -> str | None:
    return "/usr/local/bin/claude" if cmd == "claude" else None


async def _noop_fact(_entity_id, _claim):
    return None


async def _noop_conflict(_entity_id, _cells):
    return None


@pytest.mark.asyncio
async def test_orchestrator_full_run_smoke_offline():
    """Full Orchestrator.run() loop with StubCliRunner — the integration smoke for Wave 1-E."""
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())

    spec = RunSpec(
        spec_id="wars",
        goal="Major interstate wars since 1500",
        entities=[
            EntitySpec(
                name="War",
                fields=[
                    FieldSpec(name="name", type="str", required=True),
                    FieldSpec(name="start_year", type="int", required=True),
                    FieldSpec(name="end_year", type="int"),
                    FieldSpec(name="belligerents", type="list[str]"),
                ],
                search_templates=[],
            )
        ],
        seeds=["Major wars since 1500", "Wars of the 20th century"],
        models={"fast": "stub", "smart": "stub", "heavy": "stub"},
        backend_policy="auto",
        max_cycles=1,
    )

    store = StubKnowledgeStore()
    await store.open()
    bus = StubEventBus()
    llm = StubLLMClient()
    budget = Budget(usd_cap=3.0, wall_cap_s=600)
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
    backend_resolver = BackendResolver(which_fn=_which_claude)

    orch = Orchestrator(
        spec=spec,
        store=store,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        bus=bus,  # type: ignore[arg-type]
        writer=writer,
        scheduler=scheduler,
        budget=budget,
        run_id="orch-smoke",
        max_parallel_agents=2,
    )
    orch.set_backend_resolver(backend_resolver)
    orch.set_cli_runner_factory(lambda kind: runner)

    # Act
    reason = await orch.run()

    # Assert — stop reason is one of the expected terminal states
    assert reason in (
        StopReason.PLATEAU,
        StopReason.NO_TASKS,
        StopReason.BUDGET,
        StopReason.DEADLINE,
    ), f"unexpected stop reason: {reason}"

    # Assert — two seeds became two tasks, each dispatched once
    assert len(runner.calls) == 2

    # Assert — claims submitted to the writer (4 per task × 2 tasks = 8)
    assert writer.metrics["submitted"] == 8

    # Assert — at least one CycleStart + CycleEnd + exactly one RunComplete
    cycle_starts = [e for e in bus.events if isinstance(e, CycleStart)]
    cycle_ends = [e for e in bus.events if isinstance(e, CycleEnd)]
    run_completes = [e for e in bus.events if isinstance(e, RunComplete)]
    assert len(cycle_starts) >= 1
    assert len(cycle_ends) >= 1
    assert len(run_completes) == 1

    # Assert — two SubagentCall events (one per task)
    subagent_events = [e for e in bus.events if isinstance(e, SubagentCall)]
    assert len(subagent_events) == 2
    assert all(e.payload.claims_emitted == 4 for e in subagent_events)
    assert all(e.payload.cli_kind == "claude_code" for e in subagent_events)

    # Assert — budget: zero cost, counter == 2
    assert budget.total_spent() == 0.0
    assert budget.subagent_calls_total == 2

    # Assert — writer is fully drained (task done)
    assert writer._task is not None
    assert writer._task.done()
