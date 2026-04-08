"""Tests for DiscoverAgent — seed query -> search -> fetch -> extract -> claims."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from researcher.agents.discover import DiscoverAgent
from researcher.agents.native_deps import NativeAgentDeps
from researcher.events import AgentStateChange
from researcher.fetch.http import FetchResult
from researcher.llm.prompts import default_registry
from researcher.models import AgentState, FactClaim, Task, TaskKind
from researcher.search.base import SearchResult
from tests.stubs.bus import StubEventBus
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubKnowledgeStore


def _sample_task() -> Task:
    return Task(
        kind=TaskKind.DISCOVER,
        spec_ref="wars",
        seed_query="major wars since 1500",
        budget_usd=0.01,
        deadline_ts=datetime.now(timezone.utc) + timedelta(minutes=5),
    )


def _entity_schema() -> dict:
    return {
        "entity_type": "War",
        "fields": [
            {"name": "name", "type": "str", "required": True},
            {"name": "start_year", "type": "int", "required": False},
        ],
    }


def _make_deps(
    search_results: list[SearchResult],
    fetch_results: list[FetchResult] | None = None,
) -> NativeAgentDeps:
    search = MagicMock()
    search.name = "mock"
    search.search = AsyncMock(return_value=search_results)

    http = MagicMock()
    if fetch_results is None:
        fetch_results = [
            FetchResult(
                url=r.url,
                status=200,
                content="<html><body><p>World War II was fought 1939-1945.</p></body></html>",
                headers={},
                ok=True,
            )
            for r in search_results
        ]
    http.fetch = AsyncMock(side_effect=fetch_results)

    return NativeAgentDeps(
        search=search,
        http=http,
        prompts=default_registry(),
        max_fetch_per_task=3,
    )


def _make_llm_returning(entities: list[str]) -> Any:
    llm = StubLLMClient()
    original = llm.complete_structured

    async def fake_structured(messages, schema, tier, task_id, temperature=0.0, schema_retry=True):
        return schema(entities=entities)

    llm.complete_structured = fake_structured  # type: ignore[method-assign]
    return llm


@pytest.mark.asyncio
async def test_discover_happy_path_emits_claims() -> None:
    search_results = [
        SearchResult(title="WW2", url="https://en.wikipedia.org/wiki/World_War_II", snippet="", rank=0),
        SearchResult(title="WW1", url="https://en.wikipedia.org/wiki/World_War_I", snippet="", rank=1),
    ]
    deps = _make_deps(search_results)
    llm = _make_llm_returning(["World War II", "World War I"])
    bus = StubEventBus()
    store = StubKnowledgeStore()

    agent = DiscoverAgent(
        agent_id="native-abc",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
        goal="Major interstate wars",
    )

    result = await agent.run(_sample_task())

    assert result.state == AgentState.DONE
    assert len(result.claims) == 2
    names = {c.value for c in result.claims}
    assert names == {"World War II", "World War I"}
    for claim in result.claims:
        assert isinstance(claim, FactClaim)
        assert claim.entity_type == "War"
        assert claim.field == "name"
        assert claim.provenance.url.startswith("https://")


@pytest.mark.asyncio
async def test_discover_empty_search_returns_no_claims() -> None:
    deps = _make_deps(search_results=[])
    llm = _make_llm_returning([])
    bus = StubEventBus()
    store = StubKnowledgeStore()

    agent = DiscoverAgent(
        agent_id="native-abc",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
        goal="Wars",
    )

    result = await agent.run(_sample_task())

    assert result.state == AgentState.DONE
    assert result.claims == []


@pytest.mark.asyncio
async def test_discover_skips_failed_fetches() -> None:
    search_results = [
        SearchResult(title="WW2", url="https://example.com/a", snippet="", rank=0),
        SearchResult(title="WW1", url="https://example.com/b", snippet="", rank=1),
    ]
    fetch_results = [
        FetchResult(url="https://example.com/a", status=500, content="", headers={}, ok=False, error="http 500"),
        FetchResult(
            url="https://example.com/b",
            status=200,
            content="<html><body><p>World War I 1914-1918.</p></body></html>",
            headers={},
            ok=True,
        ),
    ]
    deps = _make_deps(search_results, fetch_results=fetch_results)
    llm = _make_llm_returning(["World War I"])
    bus = StubEventBus()
    store = StubKnowledgeStore()

    agent = DiscoverAgent(
        agent_id="native-abc",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
        goal="Wars",
    )

    result = await agent.run(_sample_task())

    assert result.state == AgentState.DONE
    assert len(result.claims) == 1
    assert result.claims[0].value == "World War I"
    # Provenance URL points at the one source that succeeded.
    assert "example.com/b" in result.claims[0].provenance.url


@pytest.mark.asyncio
async def test_discover_emits_state_transitions() -> None:
    search_results = [
        SearchResult(title="WW2", url="https://example.com/a", snippet="", rank=0),
    ]
    deps = _make_deps(search_results)
    llm = _make_llm_returning(["World War II"])
    bus = StubEventBus()
    store = StubKnowledgeStore()

    agent = DiscoverAgent(
        agent_id="native-abc",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
        goal="Wars",
    )

    await agent.run(_sample_task())

    state_events = [e for e in bus.events if isinstance(e, AgentStateChange)]
    transitions = [(e.payload.old, e.payload.new) for e in state_events]
    assert ("idle", "planning") in transitions
    assert ("planning", "fetching") in transitions
    assert ("fetching", "extracting") in transitions
    assert ("extracting", "done") in transitions
