"""Tests for Orchestrator native-agent dispatch (Wave 1-D)."""

from __future__ import annotations

import pytest

from researcher.agents.native_deps import NativeAgentDeps
from researcher.backends.resolver import BackendResolver
from researcher.budget import Budget
from researcher.llm.prompts import default_registry
from researcher.orchestrator import Orchestrator, StopReason
from researcher.scheduler import Scheduler
from researcher.search.file_seeds import FileSeedsProvider
from researcher.spec import EntitySpec, FieldSpec, RunSpec
from researcher.storage.duckdb_store import DuckDBKnowledgeStore
from researcher.storage.writer import FactWriter
from tests.stubs.llm import StubLLMClient
from tests.support import RecordingEventBus, make_resolver, mock_http


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


WW2_URL = "https://example.com/ww2"


def _make_native_deps_with_canned_discover() -> NativeAgentDeps:
    # Search results exist only for the spec's seed query.
    search = FileSeedsProvider(seeds={"major wars": [{"title": "WW2", "url": WW2_URL}]})
    http = mock_http(
        {WW2_URL: (200, "<html><body><p>World War II was fought 1939-1945.</p></body></html>")}
    )
    return NativeAgentDeps(
        search=search, http=http, prompts=default_registry(), max_fetch_per_task=2
    )


async def _build_orchestrator(
    store: DuckDBKnowledgeStore,
    bus: RecordingEventBus,
    native_deps: NativeAgentDeps | None,
    patched_llm_entities: list[str] | None = None,
) -> tuple[Orchestrator, RecordingEventBus, DuckDBKnowledgeStore, FactWriter]:
    llm = StubLLMClient()
    if patched_llm_entities is not None:

        async def fake_structured(
            messages, schema, tier, task_id, temperature=0.0, schema_retry=True
        ):
            # DiscoverAgent expects `entities` field.
            if "entities" in schema.model_fields:
                return schema(entities=patched_llm_entities)
            return schema()

        llm.complete_structured = fake_structured  # type: ignore[method-assign]
    resolver = make_resolver(store)
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
async def test_native_dispatch_runs_discover_agent_and_submits_claims(store, bus) -> None:
    deps = _make_native_deps_with_canned_discover()
    orch, _bus, _store, writer = await _build_orchestrator(
        store, bus, native_deps=deps, patched_llm_entities=["World War II"]
    )

    reason = await orch.run()

    # The run completed without error and the native DiscoverAgent's claim
    # went through the writer into the store.
    assert reason == StopReason.PLATEAU
    assert writer.metrics["submitted"] == 1
    assert writer.metrics["written"] == 1
    rows = await store.query("SELECT name FROM entities WHERE entity_type = 'War'")
    assert [r["name"] for r in rows] == ["World War II"]


@pytest.mark.asyncio
async def test_native_dispatch_without_deps_falls_back_to_not_implemented(store, bus) -> None:
    """Backwards-compat: without set_native_deps(), the native path raises."""
    orch, _bus, _store, _writer = await _build_orchestrator(store, bus, native_deps=None)

    reason = await orch.run()

    # The NotImplementedError is caught by the orchestrator and mapped to ERROR.
    assert reason == StopReason.ERROR


@pytest.mark.asyncio
async def test_native_dispatch_propagates_enum_into_entity_schema(store, bus) -> None:
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
        llm = StubLLMClient()
        resolver = make_resolver(store)
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

        eid = await store.upsert_entity(
            "War",
            "WWII",
            {
                "name": FieldCell(
                    value="WWII",
                    confidence=1.0,
                    provenance_ids=[],
                    updated_at=datetime.now(UTC),
                )
            },
        )
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
