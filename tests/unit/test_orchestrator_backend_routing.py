"""Tests for Orchestrator._pick_agent_for_task — backend routing logic."""

from datetime import datetime, timedelta, timezone

import pytest

from researcher.backends.models import BackendChoice, CliKind
from researcher.backends.resolver import BackendResolver
from researcher.budget import Budget
from researcher.models import AgentResult, AgentState, Task, TaskKind
from researcher.orchestrator import Orchestrator, StopReason
from researcher.scheduler import Scheduler
from researcher.spec import EntitySpec, FieldSpec, RunSpec
from researcher.storage.writer import FactWriter
from tests.stubs.bus import StubEventBus
from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
from tests.stubs.llm import StubLLMClient
from tests.stubs.resolver import StubEntityResolver
from tests.stubs.store import StubKnowledgeStore


def _which_none(_: str) -> str | None:
    return None


def _which_claude(cmd: str) -> str | None:
    return "/usr/local/bin/claude" if cmd == "claude" else None


def _sample_spec(policy: str = "auto") -> RunSpec:
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
    )


async def _noop_fact(_entity_id, _claim):
    return None


async def _noop_conflict(_entity_id, _cells):
    return None


async def _make_orchestrator(
    spec: RunSpec, which_fn, runner: StubCliRunner | None = None
) -> Orchestrator:
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
    return orch


def _sample_task() -> Task:
    return Task(
        kind=TaskKind.DISCOVER,
        spec_ref="wars",
        seed_query="wars",
        budget_usd=0.01,
        deadline_ts=datetime.now(timezone.utc) + timedelta(minutes=5),
    )


@pytest.mark.asyncio
async def test_auto_policy_with_no_cli_routes_native():
    orch = await _make_orchestrator(_sample_spec("auto"), _which_none)
    choice = orch._resolver_pick(_sample_task())
    assert choice.kind is None


@pytest.mark.asyncio
async def test_auto_policy_with_claude_detected_routes_subagent():
    orch = await _make_orchestrator(_sample_spec("auto"), _which_claude)
    choice = orch._resolver_pick(_sample_task())
    assert choice.kind == CliKind.CLAUDE_CODE


@pytest.mark.asyncio
async def test_api_policy_overrides_detection():
    orch = await _make_orchestrator(_sample_spec("api"), _which_claude)
    choice = orch._resolver_pick(_sample_task())
    assert choice.kind is None


@pytest.mark.asyncio
async def test_circuit_break_clears_detected_after_three_failures():
    orch = await _make_orchestrator(_sample_spec("auto"), _which_claude)
    assert orch._resolver_pick(_sample_task()).kind == CliKind.CLAUDE_CODE
    orch._record_subagent_failure("spawn_failed: ENOENT")
    orch._record_subagent_failure("timeout")
    assert orch._resolver_pick(_sample_task()).kind == CliKind.CLAUDE_CODE  # 2 failures, not tripped
    orch._record_subagent_failure("exit 1: oops")
    # Third failure should trip the break.
    assert orch._resolver_pick(_sample_task()).kind is None


@pytest.mark.asyncio
async def test_usage_limit_clears_detected_immediately():
    orch = await _make_orchestrator(_sample_spec("auto"), _which_claude)
    assert orch._resolver_pick(_sample_task()).kind == CliKind.CLAUDE_CODE
    orch._record_subagent_failure("usage_limit_reached")
    assert orch._resolver_pick(_sample_task()).kind is None


@pytest.mark.asyncio
async def test_auth_required_clears_detected_immediately():
    orch = await _make_orchestrator(_sample_spec("auto"), _which_claude)
    orch._record_subagent_failure("auth_required")
    assert orch._resolver_pick(_sample_task()).kind is None


@pytest.mark.asyncio
async def test_failure_counter_resets_between_cycles():
    orch = await _make_orchestrator(_sample_spec("auto"), _which_claude)
    orch._record_subagent_failure("timeout")
    orch._record_subagent_failure("timeout")
    orch._reset_cycle_failure_counters()  # simulate cycle boundary
    orch._record_subagent_failure("timeout")
    # Only 1 failure in the current cycle — not tripped.
    assert orch._resolver_pick(_sample_task()).kind == CliKind.CLAUDE_CODE


@pytest.mark.asyncio
async def test_stop_reason_subagent_cap_exists():
    assert hasattr(StopReason, "SUBAGENT_CAP")
    assert StopReason.SUBAGENT_CAP.value == "subagent_cap"


@pytest.mark.asyncio
async def test_auto_after_circuit_break_returns_unprefixed_reason():
    """I-2 fix: don't double-prefix 'circuit_break: ' in BackendChoice.reason."""
    orch = await _make_orchestrator(_sample_spec("auto"), _which_claude)
    orch._record_subagent_failure("usage_limit_reached")
    choice = orch._resolver_pick(_sample_task())
    assert choice.kind is None
    # Should contain the reason once, not twice.
    assert choice.reason.count("circuit_break") <= 1
