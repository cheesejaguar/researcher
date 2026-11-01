"""Tests for DiscoverAgent — seed query -> search -> fetch -> extract -> claims."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from researcher.agents.discover import DiscoverAgent
from researcher.agents.native_deps import NativeAgentDeps
from researcher.events import AgentStateChange
from researcher.llm.prompts import default_registry
from researcher.models import AgentState, FactClaim, Task, TaskKind
from researcher.search.base import SearchResult
from researcher.search.file_seeds import FileSeedsProvider
from researcher.skills.registry import SkillRegistry
from tests.stubs.llm import StubLLMClient
from tests.support import mock_http

SEED_QUERY = "major wars since 1500"


def _sample_task() -> Task:
    return Task(
        kind=TaskKind.DISCOVER,
        spec_ref="wars",
        seed_query=SEED_QUERY,
        budget_usd=0.01,
        deadline_ts=datetime.now(UTC) + timedelta(minutes=5),
    )


def _entity_schema() -> dict:
    return {
        "entity_type": "War",
        "fields": [
            {"name": "name", "type": "str", "required": True},
            {"name": "start_year", "type": "int", "required": False},
        ],
    }


_DEFAULT_PAGE = "<html><body><p>World War II was fought 1939-1945.</p></body></html>"


def _make_deps(
    search_results: list[SearchResult],
    pages: dict[str, tuple[int, str]] | None = None,
    skill_registry: SkillRegistry | None = None,
) -> NativeAgentDeps:
    # Results are only returned for the task's exact seed query, so the
    # tests also pin down the query DiscoverAgent builds.
    search = FileSeedsProvider(
        seeds={SEED_QUERY: [r.model_dump() for r in search_results]}
    )
    if pages is None:
        pages = {r.url: (200, _DEFAULT_PAGE) for r in search_results}
    return NativeAgentDeps(
        search=search,
        http=mock_http(pages),
        prompts=default_registry(),
        max_fetch_per_task=3,
        skill_registry=skill_registry,
    )


def _make_llm_returning(entities: list[str]) -> Any:
    llm = StubLLMClient()

    async def fake_structured(messages, schema, tier, task_id, temperature=0.0, schema_retry=True):
        return schema(entities=entities)

    llm.complete_structured = fake_structured  # type: ignore[method-assign]
    return llm


@pytest.mark.asyncio
async def test_discover_happy_path_emits_claims(store, bus) -> None:
    search_results = [
        SearchResult(title="WW2", url="https://en.wikipedia.org/wiki/World_War_II", snippet="", rank=0),
        SearchResult(title="WW1", url="https://en.wikipedia.org/wiki/World_War_I", snippet="", rank=1),
    ]
    deps = _make_deps(search_results)
    llm = _make_llm_returning(["World War II", "World War I"])

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
        # Provenance points at the first source that was actually fetched.
        assert claim.provenance.url == "https://en.wikipedia.org/wiki/World_War_II"
        assert "1939-1945" in claim.provenance.snippet


@pytest.mark.asyncio
async def test_discover_empty_search_returns_no_claims(store, bus) -> None:
    deps = _make_deps(search_results=[])
    # Even if the LLM would hallucinate an entity, no sources means the agent
    # must short-circuit before extraction.
    llm = _make_llm_returning(["Phantom War"])

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
async def test_discover_skips_failed_fetches(store, bus) -> None:
    search_results = [
        SearchResult(title="WW2", url="https://example.com/a", snippet="", rank=0),
        SearchResult(title="WW1", url="https://example.com/b", snippet="", rank=1),
    ]
    pages = {
        "https://example.com/a": (500, ""),
        "https://example.com/b": (200, "<html><body><p>World War I 1914-1918.</p></body></html>"),
    }
    deps = _make_deps(search_results, pages=pages)
    llm = _make_llm_returning(["World War I"])

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
async def test_discover_emits_state_transitions(store, bus) -> None:
    search_results = [
        SearchResult(title="WW2", url="https://example.com/a", snippet="", rank=0),
    ]
    deps = _make_deps(search_results)
    llm = _make_llm_returning(["World War II"])

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


@pytest.mark.asyncio
async def test_discover_prompt_includes_matching_skill_guidance(store, bus) -> None:
    search_results = [
        SearchResult(title="WW2", url="https://example.com/a", snippet="", rank=0),
    ]
    registry = SkillRegistry()
    registry.load_dir("skills")
    deps = _make_deps(search_results, skill_registry=registry)
    llm = StubLLMClient()

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

    assert llm.calls
    user_messages = [
        m["content"]
        for m in llm.calls[0]["messages"]
        if m.get("role") == "user"
    ]
    assert any("Wikipedia infobox" in msg for msg in user_messages)
