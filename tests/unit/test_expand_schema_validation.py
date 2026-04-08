"""Tests for ExpandAgent consulting SkillRegistry to validate extracted fields."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from researcher.agents.expand import ExpandAgent
from researcher.agents.native_deps import NativeAgentDeps
from researcher.events import AgentLog
from researcher.fetch.http import FetchResult
from researcher.llm.prompts import default_registry
from researcher.models import AgentState, Task, TaskKind
from researcher.search.base import SearchResult
from researcher.skills.registry import SkillRegistry
from tests.stubs.bus import StubEventBus
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubKnowledgeStore


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


def _make_deps(
    skill_registry: SkillRegistry | None = None,
) -> NativeAgentDeps:
    search = MagicMock()
    search.name = "mock"
    search.search = AsyncMock(
        return_value=[
            SearchResult(
                title="WW2",
                url="https://en.wikipedia.org/wiki/World_War_II",
                snippet="",
                rank=0,
            )
        ]
    )
    http = MagicMock()
    http.fetch = AsyncMock(
        return_value=FetchResult(
            url="https://en.wikipedia.org/wiki/World_War_II",
            status=200,
            content="<html><body><p>World War II ran from 1939 to 1945.</p></body></html>",
            headers={},
            ok=True,
        )
    )
    return NativeAgentDeps(
        search=search,
        http=http,
        prompts=default_registry(),
        max_fetch_per_task=2,
        skill_registry=skill_registry,
    )


class _RawDict(dict):
    """Dict that quacks just enough to skip the BaseModel branch in expand."""


def _llm_returning(fields: dict[str, Any]) -> StubLLMClient:
    llm = StubLLMClient()

    async def fake_structured(messages, schema, tier, task_id, temperature=0.0, schema_retry=True):
        # Return a plain dict so ExpandAgent's `dict(response)` branch fires —
        # this bypasses Pydantic's type coercion and lets us inject a string
        # into an int-typed field for the validation test.
        payload: dict[str, Any] = {}
        for name in schema.model_fields:
            if name in fields:
                payload[name] = fields[name]
        return _RawDict(payload)

    llm.complete_structured = fake_structured  # type: ignore[method-assign]
    return llm


async def _seed_entity(store: StubKnowledgeStore, name: str) -> str:
    return await store.upsert_entity(entity_type="War", name=name, fields={})


def _registry_with_year_int(tmp_path: Path) -> SkillRegistry:
    (tmp_path / "wars.yaml").write_text(
        """
domain: wars
skills:
  - name: war_year_validator
    description: ""
    prompt_fragment: ""
    output_schema:
      start_year:
        type: int
        required: false
"""
    )
    reg = SkillRegistry()
    reg.load_dir(tmp_path)
    return reg


@pytest.mark.asyncio
async def test_expand_agent_emits_claim_when_no_skill_registry() -> None:
    store = StubKnowledgeStore()
    entity_id = await _seed_entity(store, "World War II")
    deps = _make_deps(skill_registry=None)
    llm = _llm_returning({"start_year": 1939, "end_year": 1945})
    bus = StubEventBus()

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


@pytest.mark.asyncio
async def test_expand_agent_rejects_invalid_field_when_schema_declared(
    tmp_path: Path,
) -> None:
    store = StubKnowledgeStore()
    entity_id = await _seed_entity(store, "World War II")
    registry = _registry_with_year_int(tmp_path)
    deps = _make_deps(skill_registry=registry)
    llm = _llm_returning({"start_year": "nineteen thirty nine", "end_year": 1945})
    bus = StubEventBus()

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
    fields = {c.field for c in result.claims}
    assert "end_year" in fields
    assert "start_year" not in fields  # rejected by schema


@pytest.mark.asyncio
async def test_expand_agent_accepts_valid_field_when_schema_declared(
    tmp_path: Path,
) -> None:
    store = StubKnowledgeStore()
    entity_id = await _seed_entity(store, "World War II")
    registry = _registry_with_year_int(tmp_path)
    deps = _make_deps(skill_registry=registry)
    llm = _llm_returning({"start_year": 1939, "end_year": 1945})
    bus = StubEventBus()

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
    by_field = {c.field: c for c in result.claims}
    assert by_field["start_year"].value == 1939
    assert by_field["end_year"].value == 1945


@pytest.mark.asyncio
async def test_expand_agent_logs_rejection(tmp_path: Path) -> None:
    store = StubKnowledgeStore()
    entity_id = await _seed_entity(store, "World War II")
    registry = _registry_with_year_int(tmp_path)
    deps = _make_deps(skill_registry=registry)
    llm = _llm_returning({"start_year": "nineteen thirty nine", "end_year": 1945})
    bus = StubEventBus()

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

    await agent.run(_make_task(entity_id))

    warn_logs = [e for e in bus.events if isinstance(e, AgentLog) and e.payload.level == "warn"]
    assert any("start_year" in log.payload.msg for log in warn_logs)
