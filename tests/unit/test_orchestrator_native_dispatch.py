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
