"""Tests for EnrichAgent — known entity -> spawned DISCOVER tasks for relations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from researcher.agents.enrich import EnrichAgent
from researcher.agents.native_deps import NativeAgentDeps
from researcher.llm.prompts import default_registry
from researcher.models import AgentState, Task, TaskKind
from researcher.search.file_seeds import FileSeedsProvider
from tests.stubs.llm import StubLLMClient
from tests.support import mock_http


def _entity_schema() -> dict:
    return {
        "entity_type": "War",
        "fields": [
            {"name": "name", "type": "str", "required": True},
        ],
    }


def _enrich_task(entity_id: str) -> Task:
    return Task(
        kind=TaskKind.ENRICH,
        spec_ref="wars",
        target_entity_id=entity_id,
        budget_usd=0.01,
        deadline_ts=datetime.now(UTC) + timedelta(minutes=5),
    )


def _make_deps() -> NativeAgentDeps:
    return NativeAgentDeps(
        search=FileSeedsProvider(seeds={}),
        http=mock_http({}),
        prompts=default_registry(),
        max_fetch_per_task=2,
    )


def _llm_suggesting(related: list[str]) -> StubLLMClient:
    llm = StubLLMClient()

    async def fake_structured(messages, schema, tier, task_id, temperature=0.0, schema_retry=True):
        return schema(related=related)

    llm.complete_structured = fake_structured  # type: ignore[method-assign]
    return llm


@pytest.mark.asyncio
async def test_enrich_spawns_discover_tasks_for_related_entities(store, bus) -> None:
    entity_id = await store.upsert_entity(entity_type="War", name="World War II", fields={})
    deps = _make_deps()
    llm = _llm_suggesting(["Korean War", "Cold War"])

    agent = EnrichAgent(
        agent_id="native-enr",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
        goal="Major wars",
    )

    result = await agent.run(_enrich_task(entity_id))

    assert result.state == AgentState.DONE
    assert result.claims == []
    assert len(result.spawned_tasks) == 2
    for spawned in result.spawned_tasks:
        assert spawned.kind == TaskKind.DISCOVER
        assert spawned.parent_task_id is not None
        assert spawned.depth == 1  # enrich was depth=0
    queries = {t.seed_query for t in result.spawned_tasks}
    assert "Korean War" in queries
    assert "Cold War" in queries


@pytest.mark.asyncio
async def test_enrich_entity_not_found_returns_empty_spawned(store, bus) -> None:
    deps = _make_deps()
    llm = _llm_suggesting([])

    agent = EnrichAgent(
        agent_id="native-enr",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
        goal="Wars",
    )

    task = Task(
        kind=TaskKind.ENRICH,
        spec_ref="wars",
        target_entity_id="nope",
        budget_usd=0.01,
        deadline_ts=datetime.now(UTC) + timedelta(minutes=5),
    )
    result = await agent.run(task)

    assert result.state == AgentState.FAILED
    assert result.error == "entity_not_found"
    assert result.spawned_tasks == []


def _llm_with_relations(related: list[str], relations: list[dict]) -> StubLLMClient:
    llm = StubLLMClient()

    async def fake_structured(
        messages, schema, tier, task_id, temperature=0.0, schema_retry=True, model=None
    ):
        return schema(related=related, relations=relations)

    llm.complete_structured = fake_structured  # type: ignore[method-assign]
    return llm


@pytest.mark.asyncio
async def test_enrich_records_declared_relations_and_spawns_targets(store, bus) -> None:
    from researcher.spec import RelationSpec

    entity_id = await store.upsert_entity(entity_type="War", name="World War II", fields={})
    deps = _make_deps()
    deps.relations = [RelationSpec(name="ally_of", source_type="War", target_type="War")]
    deps.max_depth = 3
    llm = _llm_with_relations(
        related=[],
        relations=[
            {"relation": "ally_of", "target_name": "France", "confidence": 0.8},
            {"relation": "not_declared", "target_name": "Nowhere", "confidence": 0.9},
        ],
    )
    agent = EnrichAgent(
        agent_id="native-enrich",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
        goal="Wars",
    )

    result = await agent.run(_enrich_task(entity_id))

    assert result.state == AgentState.DONE
    # Only the relation declared in the spec is persisted, pointing at a
    # newly created target entity.
    neighbors = await store.find_neighbors(entity_id)
    assert len(neighbors) == 1
    assert neighbors[0]["relation_label"] == "ally_of"
    assert neighbors[0]["confidence"] == pytest.approx(0.8)
    target_rows = await store.query(
        "SELECT entity_type, name FROM entities WHERE id = ?",
        (neighbors[0]["target_id"],),
    )
    assert [(r["entity_type"], r["name"]) for r in target_rows] == [("War", "France")]
    assert [t.seed_query for t in result.spawned_tasks] == ["France"]
    assert result.spawned_tasks[0].kind == TaskKind.DISCOVER
    assert result.spawned_tasks[0].depth == 1


@pytest.mark.asyncio
async def test_enrich_stops_spawning_at_max_depth(store, bus) -> None:
    from researcher.spec import RelationSpec

    entity_id = await store.upsert_entity(entity_type="War", name="World War II", fields={})
    deps = _make_deps()
    deps.relations = [RelationSpec(name="ally_of", source_type="War", target_type="War")]
    deps.max_depth = 2
    llm = _llm_with_relations(
        related=["Cold War"],
        relations=[{"relation": "ally_of", "target_name": "France", "confidence": 0.8}],
    )
    agent = EnrichAgent(
        agent_id="native-enrich",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
        goal="Wars",
    )
    task = _enrich_task(entity_id).model_copy(update={"depth": 2})

    result = await agent.run(task)

    assert result.state == AgentState.DONE
    assert result.spawned_tasks == []
