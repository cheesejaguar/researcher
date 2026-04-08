"""Tests for v1.2 conditional revision on objective signals."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from researcher.budget import Budget
from researcher.models import AgentResult, AgentState, Task, TaskKind
from researcher.spec import EntitySpec, FieldSpec, RunSpec


def _sample_spec(enable_conditional_revision: bool = False) -> RunSpec:
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
        enable_conditional_revision=enable_conditional_revision,
    )


def _task(attempt: int = 0) -> Task:
    return Task(
        kind=TaskKind.DISCOVER,
        spec_ref="x",
        seed_query="seed1",
        budget_usd=0.05,
        deadline_ts=datetime.now(UTC) + timedelta(minutes=5),
        attempt=attempt,
    )


# ---------- RunSpec ----------


def test_runspec_enable_conditional_revision_default_false():
    s = _sample_spec()
    assert s.enable_conditional_revision is False


def test_runspec_enable_conditional_revision_accepts_true():
    s = _sample_spec(enable_conditional_revision=True)
    assert s.enable_conditional_revision is True


# ---------- AgentResult ----------


def test_agent_result_needs_revision_default_false():
    r = AgentResult(task_id="t1", agent_id="a1", state=AgentState.DONE)
    assert r.needs_revision is False


def test_agent_result_needs_revision_can_be_set():
    r = AgentResult(
        task_id="t1", agent_id="a1", state=AgentState.DONE, needs_revision=True
    )
    assert r.needs_revision is True


# ---------- Budget revisions counter ----------


def test_budget_revisions_total_starts_at_zero():
    b = Budget(usd_cap=10.0, wall_cap_s=600)
    assert b.revisions_total == 0


def test_budget_record_revision_increments():
    b = Budget(usd_cap=10.0, wall_cap_s=600)
    b.record_revision()
    b.record_revision()
    assert b.revisions_total == 2


# ---------- Orchestrator: re-dispatch on needs_revision ----------


@pytest.mark.asyncio
async def test_orchestrator_redispatches_when_needs_revision_set(tmp_path):
    """A task that returns needs_revision=True with attempt=0 should be re-dispatched once with attempt=1."""
    from researcher.backends.resolver import BackendResolver
    from researcher.orchestrator import Orchestrator
    from researcher.scheduler import Scheduler
    from researcher.spec import build_entity_class
    from researcher.storage.duckdb_store import DuckDBKnowledgeStore
    from researcher.storage.writer import FactWriter
    from tests.stubs.bus import StubEventBus
    from tests.stubs.cli_runner import StubCliRunner
    from tests.stubs.llm import StubLLMClient
    from tests.stubs.resolver import StubEntityResolver

    spec = _sample_spec(enable_conditional_revision=True)
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

        async def _noop(*args, **kwargs): return None

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

        # Mock subagent runner that returns needs_revision=True on attempt=0,
        # and a normal (empty-but-no-revision) result on attempt=1.
        runner = StubCliRunner()
        from researcher.backends.models import CliResult, SubagentResponse
        runner.add_response_for_any(
            CliResult(
                ok=True,
                data=SubagentResponse(entity_name="WWII", extractions=[], diagnostics=""),
                wall_ms=100,
                exit_code=0,
                raw_usage={"input_tokens": 50, "output_tokens": 25},
            )
        )

        orch = Orchestrator(
            spec=spec,
            store=store,  # type: ignore[arg-type]
            llm=llm,  # type: ignore[arg-type]
            bus=bus,  # type: ignore[arg-type]
            writer=writer,
            scheduler=scheduler,
            budget=budget,
            run_id="rev-test",
            max_parallel_agents=2,
        )
        orch.set_backend_resolver(backend_resolver)
        orch.set_cli_runner_factory(lambda kind: runner)

        await orch.run()

        # The runner should have been called more than once for the same task
        # because the empty-extraction result triggered a revision.
        assert len(runner.calls) >= 2
        assert budget.revisions_total >= 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_orchestrator_does_not_redispatch_when_revision_disabled(tmp_path):
    """When enable_conditional_revision is False, no revision happens even if needs_revision=True."""
    from researcher.backends.resolver import BackendResolver
    from researcher.orchestrator import Orchestrator
    from researcher.scheduler import Scheduler
    from researcher.spec import build_entity_class
    from researcher.storage.duckdb_store import DuckDBKnowledgeStore
    from researcher.storage.writer import FactWriter
    from tests.stubs.bus import StubEventBus
    from tests.stubs.cli_runner import StubCliRunner
    from tests.stubs.llm import StubLLMClient
    from tests.stubs.resolver import StubEntityResolver

    spec = _sample_spec(enable_conditional_revision=False)  # disabled
    store = DuckDBKnowledgeStore(db_path=tmp_path / "s.duckdb")
    await store.open()
    try:
        await store.init_schema(build_entity_class(spec.entities[0]))
        bus = StubEventBus()
        llm = StubLLMClient()
        resolver = StubEntityResolver()
        entity_schema = {"entity_type": "War", "fields": [{"name": "name", "type": "str", "required": True}]}

        async def _noop(*args, **kwargs): return None

        writer = FactWriter(
            store=store, resolver=resolver, entity_schema=entity_schema,
            emit_fact=_noop, emit_conflict=_noop,
        )
        scheduler = Scheduler(spec=spec, store=store)
        budget = Budget(usd_cap=3.0, wall_cap_s=600)
        backend_resolver = BackendResolver(which_fn=lambda c: "/usr/local/bin/claude" if c == "claude" else None)

        runner = StubCliRunner()
        from researcher.backends.models import CliResult, SubagentResponse
        runner.add_response_for_any(
            CliResult(
                ok=True,
                data=SubagentResponse(entity_name="WWII", extractions=[], diagnostics=""),
                wall_ms=100, exit_code=0,
                raw_usage={"input_tokens": 50, "output_tokens": 25},
            )
        )

        orch = Orchestrator(
            spec=spec,
            store=store, llm=llm, bus=bus, writer=writer,  # type: ignore[arg-type]
            scheduler=scheduler, budget=budget,
            run_id="no-rev-test", max_parallel_agents=2,
        )
        orch.set_backend_resolver(backend_resolver)
        orch.set_cli_runner_factory(lambda kind: runner)

        await orch.run()

        # Without conditional revision, only one call per task.
        assert budget.revisions_total == 0
    finally:
        await store.close()
