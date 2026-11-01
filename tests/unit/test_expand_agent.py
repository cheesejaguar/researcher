"""Tests for ExpandAgent — fill fields for a known entity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from researcher.agents.expand import ExpandAgent
from researcher.agents.native_deps import NativeAgentDeps
from researcher.llm.prompts import default_registry
from researcher.models import AgentState, Task, TaskKind
from researcher.search.file_seeds import FileSeedsProvider
from researcher.storage.duckdb_store import DuckDBKnowledgeStore
from tests.stubs.llm import StubLLMClient
from tests.support import mock_http

WW2_URL = "https://en.wikipedia.org/wiki/World_War_II"


def _entity_schema() -> dict:
    return {
        "entity_type": "War",
        "fields": [
            {"name": "name", "type": "str", "required": True},
            {"name": "start_year", "type": "int", "required": False},
            {"name": "end_year", "type": "int", "required": False},
            {"name": "belligerents", "type": "list[str]", "required": False},
        ],
    }


def _make_task(target_id: str, field_hints: list[str] | None = None) -> Task:
    return Task(
        kind=TaskKind.EXPAND,
        spec_ref="wars",
        target_entity_id=target_id,
        field_hints=field_hints or ["start_year", "end_year"],
        budget_usd=0.01,
        deadline_ts=datetime.now(UTC) + timedelta(minutes=5),
    )


def _make_deps() -> NativeAgentDeps:
    # ExpandAgent searches for "<entity name> <field hints>"; any other
    # query finds nothing and the agent produces no claims.
    search = FileSeedsProvider(
        seeds={"World War II start_year end_year": [{"title": "WW2", "url": WW2_URL}]}
    )
    http = mock_http(
        {WW2_URL: (200, "<html><body><p>World War II ran from 1939 to 1945.</p></body></html>")}
    )
    return NativeAgentDeps(
        search=search,
        http=http,
        prompts=default_registry(),
        max_fetch_per_task=2,
    )


def _llm_returning(fields: dict[str, Any]) -> StubLLMClient:
    llm = StubLLMClient()

    async def fake_structured(messages, schema, tier, task_id, temperature=0.0, schema_retry=True):
        llm.calls.append({"messages": messages})
        # Only populate the fields that are present in the schema.
        payload: dict[str, Any] = {}
        for name in schema.model_fields:
            if name in fields:
                payload[name] = fields[name]
        return schema.model_validate(payload)

    llm.complete_structured = fake_structured  # type: ignore[method-assign]
    return llm


async def _seed_entity(store: DuckDBKnowledgeStore, name: str) -> str:
    return await store.upsert_entity(entity_type="War", name=name, fields={})


@pytest.mark.asyncio
async def test_expand_happy_path_fills_fields(store, bus) -> None:
    entity_id = await _seed_entity(store, "World War II")
    deps = _make_deps()
    llm = _llm_returning({"start_year": 1939, "end_year": 1945})

    agent = ExpandAgent(
        agent_id="native-exp",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
        goal="Wars",
    )

    result = await agent.run(_make_task(entity_id))

    assert result.state == AgentState.DONE
    assert len(result.claims) == 2
    by_field = {c.field: c for c in result.claims}
    assert by_field["start_year"].value == 1939
    assert by_field["end_year"].value == 1945
    for claim in result.claims:
        assert claim.entity_name == "World War II"
        assert claim.entity_type == "War"
        assert claim.confidence == 0.8
    # The agent searched for "<name> <hints>", fetched the hit through the
    # real HttpFetcher + readability pipeline, and fed it to the LLM.
    prompt = "\n".join(m["content"] for m in llm.calls[0]["messages"])
    assert f"SOURCE: {WW2_URL}" in prompt
    assert "ran from 1939 to 1945" in prompt


@pytest.mark.asyncio
async def test_expand_entity_not_found_returns_failed(store, bus) -> None:
    deps = _make_deps()
    llm = _llm_returning({})

    agent = ExpandAgent(
        agent_id="native-exp",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
        goal="Wars",
    )

    result = await agent.run(_make_task(target_id="nonexistent"))

    assert result.state == AgentState.FAILED
    assert result.error == "entity_not_found"
    assert result.claims == []


@pytest.mark.asyncio
async def test_expand_partial_fields_only_claims_populated_ones(store, bus) -> None:
    entity_id = await _seed_entity(store, "World War I")
    deps = _make_deps()
    # Only start_year populated; end_year is None and should be skipped.
    llm = _llm_returning({"start_year": 1914})

    agent = ExpandAgent(
        agent_id="native-exp",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
        goal="Wars",
    )

    result = await agent.run(_make_task(entity_id, field_hints=["start_year", "end_year"]))

    assert result.state == AgentState.DONE
    assert len(result.claims) == 1
    assert result.claims[0].field == "start_year"
    assert result.claims[0].value == 1914
