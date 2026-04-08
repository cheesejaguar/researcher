"""Tests for SubagentResearcher — the Agent subclass that drives CLI runners."""

from datetime import UTC, datetime, timedelta

import pytest

from researcher.agents.subagent import SubagentResearcher
from researcher.backends.models import CliKind
from researcher.budget import Budget
from researcher.events import AgentLog, AgentStateChange, SubagentCall
from researcher.models import AgentState, Task, TaskKind
from tests.stubs.bus import StubEventBus
from tests.stubs.cli_runner import (
    StubCliRunner,
    make_empty_result,
    make_timeout_result,
    make_wars_discover_result,
)
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubKnowledgeStore


def _sample_task() -> Task:
    return Task(
        kind=TaskKind.DISCOVER,
        spec_ref="wars",
        seed_query="major wars since 1500",
        field_hints=["name", "start_year", "end_year", "belligerents"],
        budget_usd=0.01,
        deadline_ts=datetime.now(UTC) + timedelta(minutes=5),
    )


def _make_agent(runner: StubCliRunner, budget: Budget | None = None) -> tuple[SubagentResearcher, StubEventBus]:
    bus = StubEventBus()
    llm = StubLLMClient()
    store = StubKnowledgeStore()
    entity_schema = {
        "entity_type": "War",
        "fields": [
            {"name": "name", "type": "str", "required": True},
            {"name": "start_year", "type": "int", "required": True},
            {"name": "end_year", "type": "int", "required": False},
            {"name": "belligerents", "type": "list[str]", "required": False},
        ],
    }
    if budget is None:
        budget = Budget(usd_cap=3.0, wall_cap_s=600)
    agent = SubagentResearcher(
        agent_id="sub-1",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        runner=runner,
        cli_kind=CliKind.CLAUDE_CODE,
        entity_schema=entity_schema,
        budget=budget,
        goal="Major interstate wars",
    )
    return agent, bus


@pytest.mark.asyncio
async def test_run_maps_extractions_to_fact_claims():
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    agent, bus = _make_agent(runner)

    result = await agent.run(_sample_task())

    assert result.state == AgentState.DONE
    assert len(result.claims) == 4
    fields = {c.field for c in result.claims}
    assert fields == {"name", "start_year", "end_year", "belligerents"}
    for claim in result.claims:
        assert claim.entity_type == "War"
        assert claim.entity_name == "World War II"
        assert claim.provenance.url.startswith("https://")
        assert 0.0 <= claim.confidence <= 1.0


@pytest.mark.asyncio
async def test_run_reports_zero_cost():
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    agent, _ = _make_agent(runner)
    result = await agent.run(_sample_task())
    assert result.cost_usd == 0.0
    # Tokens are reported for observability.
    assert result.tokens_in == 500
    assert result.tokens_out == 300


@pytest.mark.asyncio
async def test_run_emits_state_transitions_and_subagent_call_event():
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    agent, bus = _make_agent(runner)
    await agent.run(_sample_task())

    state_events = [e for e in bus.events if isinstance(e, AgentStateChange)]
    transitions = [(e.payload.old, e.payload.new) for e in state_events]
    # IDLE -> PLANNING -> FETCHING -> DONE
    assert ("idle", "planning") in transitions
    assert ("planning", "fetching") in transitions
    assert ("fetching", "done") in transitions

    subagent_events = [e for e in bus.events if isinstance(e, SubagentCall)]
    assert len(subagent_events) == 1
    assert subagent_events[0].payload.claims_emitted == 4
    assert subagent_events[0].payload.cli_kind == "claude_code"
    assert subagent_events[0].payload.exit_code == 0


@pytest.mark.asyncio
async def test_run_returns_failed_on_timeout():
    runner = StubCliRunner()
    runner.add_response_for_any(make_timeout_result())
    agent, bus = _make_agent(runner)
    result = await agent.run(_sample_task())

    assert result.state == AgentState.FAILED
    assert result.error == "timeout"
    error_logs = [e for e in bus.events if isinstance(e, AgentLog) and e.payload.level == "error"]
    assert len(error_logs) >= 1


@pytest.mark.asyncio
async def test_run_emits_info_log_on_empty_extractions():
    runner = StubCliRunner()
    runner.add_response_for_any(make_empty_result())
    agent, bus = _make_agent(runner)
    result = await agent.run(_sample_task())

    assert result.state == AgentState.DONE
    assert result.claims == []
    info_logs = [e for e in bus.events if isinstance(e, AgentLog) and e.payload.level == "info"]
    assert any("0 extractions" in log.payload.msg or "empty" in log.payload.msg for log in info_logs)


@pytest.mark.asyncio
async def test_run_respects_subagent_cap():
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    budget = Budget(usd_cap=3.0, wall_cap_s=600, max_subagent_calls=0)
    agent, _ = _make_agent(runner, budget=budget)

    result = await agent.run(_sample_task())

    assert result.state == AgentState.FAILED
    assert result.error == "subagent_cap"
    # Runner was NOT called — the cap blocks before dispatch.
    assert runner.calls == []


@pytest.mark.asyncio
async def test_run_records_subagent_call_on_success():
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    budget = Budget(usd_cap=3.0, wall_cap_s=600)
    agent, _ = _make_agent(runner, budget=budget)

    await agent.run(_sample_task())
    assert budget.subagent_calls_total == 1
