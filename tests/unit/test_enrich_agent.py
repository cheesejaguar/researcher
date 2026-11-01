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
    entity_id = await store.upsert_entity(
        entity_type="War", name="World War II", fields={}
    )
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
