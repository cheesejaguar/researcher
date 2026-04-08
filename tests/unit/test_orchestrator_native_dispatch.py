"""Tests for Orchestrator native-agent dispatch (Wave 1-D)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from researcher.agents.native_deps import NativeAgentDeps
from researcher.backends.resolver import BackendResolver
from researcher.budget import Budget
from researcher.fetch.http import FetchResult
from researcher.llm.prompts import default_registry
from researcher.orchestrator import Orchestrator, StopReason
from researcher.scheduler import Scheduler
from researcher.search.base import SearchResult
from researcher.spec import EntitySpec, FieldSpec, RunSpec
from researcher.storage.writer import FactWriter
from tests.stubs.bus import StubEventBus
from tests.stubs.llm import StubLLMClient
from tests.stubs.resolver import StubEntityResolver
from tests.stubs.store import StubKnowledgeStore


def _which_none(_: str) -> str | None:
    return None


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
        seeds=["major wars"],
        models={"fast": "m", "smart": "m", "heavy": "m"},
        backend_policy="auto",
        max_cycles=1,
    )


async def _noop_fact(_entity_id, _claim):
    return None


async def _noop_conflict(_entity_id, _cells):
    return None


def _make_native_deps_with_canned_discover() -> NativeAgentDeps:
    search = MagicMock()
    search.name = "mock"
    search.search = AsyncMock(
        return_value=[
            SearchResult(
                title="WW2",
                url="https://example.com/ww2",
                snippet="",
                rank=0,
            )
        ]
    )
    http = MagicMock()
    http.fetch = AsyncMock(
        return_value=FetchResult(
            url="https://example.com/ww2",
            status=200,
            content="<html><body><p>World War II was fought 1939-1945.</p></body></html>",
            headers={},
            ok=True,
        )
    )
    return NativeAgentDeps(
        search=search, http=http, prompts=default_registry(), max_fetch_per_task=2
    )


async def _build_orchestrator(
    native_deps: NativeAgentDeps | None,
    patched_llm_entities: list[str] | None = None,
) -> tuple[Orchestrator, StubEventBus, StubKnowledgeStore, FactWriter]:
    store = StubKnowledgeStore()
    await store.open()
    llm = StubLLMClient()
    if patched_llm_entities is not None:
        async def fake_structured(messages, schema, tier, task_id, temperature=0.0, schema_retry=True):
            # DiscoverAgent expects `entities` field.
            if "entities" in schema.model_fields:
                return schema(entities=patched_llm_entities)
            return schema()

        llm.complete_structured = fake_structured  # type: ignore[method-assign]
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
    spec = _sample_spec()
    scheduler = Scheduler(spec=spec, store=store)
    budget = Budget(usd_cap=3.0, wall_cap_s=600)

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
    orch.set_backend_resolver(BackendResolver(which_fn=_which_none))
    if native_deps is not None:
        orch.set_native_deps(native_deps)
    return orch, bus, store, writer


@pytest.mark.asyncio
async def test_native_dispatch_runs_discover_agent_and_submits_claims() -> None:
    deps = _make_native_deps_with_canned_discover()
    orch, _bus, _store, writer = await _build_orchestrator(
        native_deps=deps, patched_llm_entities=["World War II"]
    )

    reason = await orch.run()

    # The run should have completed without error, and the writer should have
    # received at least one claim from the native DiscoverAgent.
    assert reason in (StopReason.NO_TASKS, StopReason.PLATEAU, StopReason.BUDGET, StopReason.DEADLINE)
    assert writer.metrics["submitted"] >= 1


@pytest.mark.asyncio
async def test_native_dispatch_without_deps_falls_back_to_not_implemented() -> None:
    """Backwards-compat: without set_native_deps(), the native path raises."""
    orch, _bus, _store, _writer = await _build_orchestrator(native_deps=None)

    reason = await orch.run()

    # The NotImplementedError is caught by the orchestrator and mapped to ERROR.
    assert reason == StopReason.ERROR


@pytest.mark.asyncio
async def test_native_dispatch_propagates_enum_into_entity_schema() -> None:
    """Regression guard: orchestrator._spawn_native_agent must carry the
    FieldSpec.enum metadata into the entity_schema dict it passes to the
    native agent. If this lookup drops `enum`, ExpandAgent's response
    model silently loses its Literal constraints and free-form LLM output
    lands in the store unchecked — the exact bug that silently bypassed
    enum validation on the Israel-Hamas munitions test drive.
    """
    spec = RunSpec(
        spec_id="wars",
        goal="Wars",
        entities=[
            EntitySpec(
                name="War",
                fields=[
                    FieldSpec(name="name", type="str", required=True),
                    FieldSpec(
                        name="outcome",
                        type="str",
                        enum=["decisive_victory", "stalemate", "treaty"],
                    ),
                ],
                search_templates=[],
            )
        ],
        seeds=["x"],
        models={"fast": "m", "smart": "m", "heavy": "m"},
        backend_policy="auto",
        max_cycles=1,
    )
    # Capture the entity_schema the agent constructor receives.
    captured: dict = {}

    # Patch ExpandAgent in-module to snapshot the kwargs.
    import researcher.agents.expand as expand_mod

    real_init = expand_mod.ExpandAgent.__init__

    def spy_init(self, *, entity_schema, **kwargs):
        captured["entity_schema"] = entity_schema
        real_init(self, entity_schema=entity_schema, **kwargs)

    expand_mod.ExpandAgent.__init__ = spy_init  # type: ignore[method-assign]
    try:
        store = StubKnowledgeStore()
        await store.open()
        llm = StubLLMClient()
        bus = StubEventBus()
        resolver = StubEntityResolver()
        entity_schema_stub = {
            "entity_type": "War",
            "fields": [{"name": "name", "type": "str", "required": True}],
        }
        writer = FactWriter(
            store=store,
            resolver=resolver,
            entity_schema=entity_schema_stub,
            emit_fact=_noop_fact,
            emit_conflict=_noop_conflict,
        )
        scheduler = Scheduler(spec=spec, store=store)
        budget = Budget(usd_cap=1.0, wall_cap_s=60)
        orch = Orchestrator(
            spec=spec,
            store=store,
            llm=llm,
            bus=bus,
            writer=writer,
            scheduler=scheduler,
            budget=budget,
            run_id="r",
        )
        backend_resolver = BackendResolver(which_fn=_which_none)
        orch.set_backend_resolver(backend_resolver)
        orch.set_native_deps(_make_native_deps_with_canned_discover())

        # Manually dispatch an EXPAND task so the native path hits ExpandAgent.
        from datetime import UTC, datetime, timedelta

        from researcher.models import Task, TaskKind
        from researcher.storage.store import FieldCell

        eid = await store.upsert_entity("War", "WWII", {
            "name": FieldCell(
                value="WWII",
                confidence=1.0,
                provenance_ids=[],
                updated_at=datetime.now(UTC),
            )
        })
        task = Task(
            kind=TaskKind.EXPAND,
            spec_ref="wars",
            target_entity_id=eid,
            budget_usd=0.01,
            deadline_ts=datetime.now(UTC) + timedelta(minutes=5),
        )
        await orch._spawn_agent(task)

        # The captured schema must include enum metadata for outcome.
        assert "entity_schema" in captured
        fields = captured["entity_schema"]["fields"]
        outcome = next(f for f in fields if f["name"] == "outcome")
        assert outcome.get("enum") == ["decisive_victory", "stalemate", "treaty"]
    finally:
        expand_mod.ExpandAgent.__init__ = real_init  # type: ignore[method-assign]
