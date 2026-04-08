"""Tests for CriticAgent — appends skill card suggestions to skills/_suggestions.md."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
        deadline_ts=datetime.now(UTC) + timedelta(minutes=5),
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


# ---------- CoVe independence (v1.2) ----------


@pytest.mark.asyncio
async def test_critic_does_not_receive_verifier_output_in_prompt(tmp_path):
    """The Critic's LLM call should not contain any text suggesting prior verifier conditioning."""
    from datetime import datetime, timedelta

    from researcher.fetch.http import HttpFetcher
    from researcher.fetch.politeness import PolitenessLimiter
    from researcher.search.file_seeds import FileSeedsProvider

    captured_messages: list = []

    class _SpyLLM(StubLLMClient):
        async def complete_structured(self, messages, schema, tier, task_id, **kwargs):
            captured_messages.append(list(messages))
            return schema(name="test_skill", description="x", prompt_fragment="y")

    llm = _SpyLLM()
    store = StubKnowledgeStore()
    await store.open()
    bus = StubEventBus()
    deps = NativeAgentDeps(
        search=FileSeedsProvider(seeds={}),
        http=HttpFetcher(politeness=PolitenessLimiter()),
        prompts=default_registry(),
    )

    agent = CriticAgent(
        agent_id="c1",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="r1",
        deps=deps,
        entity_schema={"entity_type": "War", "fields": [{"name": "name", "type": "str"}]},
        suggestions_path=tmp_path / "_suggestions.md",
    )

    task = Task(
        kind=TaskKind.VERIFY,  # critic might be triggered after verify
        spec_ref="x",
        target_entity_id="ent-1",
        seed_query="critique the verify pass for ent-1",
        budget_usd=0.01,
        deadline_ts=datetime.now(UTC) + timedelta(minutes=5),
    )

    await agent.run(task)

    # The captured messages should NOT contain any text indicating
    # the critic was conditioned on a verifier's output. Specifically,
    # they should not include phrases like "verifier said", "verifier's",
    # "winning_value:", "previous answer", etc.
    forbidden = (
        "verifier said",
        "verifier's",
        "winning_value:",
        "previous answer",
        "the answer was",
    )
    for msg_list in captured_messages:
        text = " ".join(
            str(m.get("content", "")) for m in msg_list
        ).lower()
        for f in forbidden:
            assert f not in text, f"critic prompt contains forbidden phrase: {f!r}"


@pytest.mark.asyncio
async def test_critic_run_still_writes_skill_suggestion_file(tmp_path):
    """Make sure the existing critic functionality (writing _suggestions.md) is preserved."""
    from datetime import datetime, timedelta

    from researcher.fetch.http import HttpFetcher
    from researcher.fetch.politeness import PolitenessLimiter
    from researcher.search.file_seeds import FileSeedsProvider

    class _MockLLM(StubLLMClient):
        async def complete_structured(self, messages, schema, tier, task_id, **kwargs):
            return schema(
                name="prefer_infobox",
                description="Use the Wikipedia infobox first.",
                prompt_fragment="Check the infobox before the article body.",
            )

    suggestions_path = tmp_path / "_suggestions.md"
    agent = CriticAgent(
        agent_id="c1",
        llm=_MockLLM(),
        store=StubKnowledgeStore(),
        emit=StubEventBus().emit,
        run_id="r1",
        deps=NativeAgentDeps(
            search=FileSeedsProvider(seeds={}),
            http=HttpFetcher(politeness=PolitenessLimiter()),
            prompts=default_registry(),
        ),
        entity_schema={"entity_type": "War", "fields": []},
        suggestions_path=suggestions_path,
    )

    task = Task(
        kind=TaskKind.VERIFY,
        spec_ref="x",
        target_entity_id="ent-1",
        budget_usd=0.01,
        deadline_ts=datetime.now(UTC) + timedelta(minutes=5),
    )
    await agent.run(task)

    assert suggestions_path.exists()
    content = suggestions_path.read_text()
    assert "prefer_infobox" in content
