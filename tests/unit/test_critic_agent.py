"""Tests for CriticAgent — appends skill card suggestions to skills/_suggestions.md."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from researcher.agents.critic import CriticAgent
from researcher.agents.native_deps import NativeAgentDeps
from researcher.llm.prompts import default_registry
from researcher.models import AgentState, Task, TaskKind
from tests.stubs.bus import StubEventBus
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubKnowledgeStore


def _entity_schema() -> dict:
    return {"entity_type": "War", "fields": [{"name": "name", "type": "str", "required": True}]}


def _critic_task(trigger_agent: str = "native-abc") -> Task:
    return Task(
        kind=TaskKind.VERIFY,  # critic is triggered out-of-band; kind is unused
        spec_ref="wars",
        field_hints=[trigger_agent],
        budget_usd=0.01,
        deadline_ts=datetime.now(timezone.utc) + timedelta(minutes=5),
    )


def _make_deps() -> NativeAgentDeps:
    search = MagicMock()
    search.name = "mock"
    search.search = AsyncMock(return_value=[])
    http = MagicMock()
    http.fetch = AsyncMock()
    return NativeAgentDeps(
        search=search, http=http, prompts=default_registry(), max_fetch_per_task=2
    )


def _llm_suggesting_card(name: str, description: str, prompt: str) -> StubLLMClient:
    llm = StubLLMClient()

    async def fake_structured(messages, schema, tier, task_id, temperature=0.0, schema_retry=True):
        return schema(name=name, description=description, prompt_fragment=prompt)

    llm.complete_structured = fake_structured  # type: ignore[method-assign]
    return llm


@pytest.mark.asyncio
async def test_critic_appends_skill_card_to_suggestions_file(tmp_path: Path) -> None:
    suggestions_path = tmp_path / "_suggestions.md"
    deps = _make_deps()
    llm = _llm_suggesting_card(
        name="check_infobox",
        description="Check the Wikipedia infobox first for structured fields.",
        prompt="For War entities, prefer the infobox.",
    )
    bus = StubEventBus()
    store = StubKnowledgeStore()

    agent = CriticAgent(
        agent_id="native-critic",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
        suggestions_path=suggestions_path,
    )

    result = await agent.run(_critic_task())

    assert result.state == AgentState.DONE
    assert result.claims == []
    assert suggestions_path.exists()
    content = suggestions_path.read_text()
    assert "check_infobox" in content
    assert "infobox" in content  # from the description / prompt fragment


@pytest.mark.asyncio
async def test_critic_creates_parent_directory_if_missing(tmp_path: Path) -> None:
    suggestions_path = tmp_path / "nested" / "dir" / "_suggestions.md"
    deps = _make_deps()
    llm = _llm_suggesting_card(
        name="skill_x",
        description="description",
        prompt="fragment",
    )
    bus = StubEventBus()
    store = StubKnowledgeStore()

    agent = CriticAgent(
        agent_id="native-critic",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
        suggestions_path=suggestions_path,
    )

    result = await agent.run(_critic_task())
    assert result.state == AgentState.DONE
    assert suggestions_path.exists()
    assert "skill_x" in suggestions_path.read_text()
