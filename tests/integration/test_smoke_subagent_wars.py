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
from researcher.budget import Budget
from researcher.events import SubagentCall
from researcher.models import Task, TaskKind
from tests.stubs.bus import StubEventBus
from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
from tests.stubs.llm import StubLLMClient
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
