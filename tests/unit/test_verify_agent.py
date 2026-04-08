"""Tests for VerifyAgent — read open conflict, pick winner, emit resolved claim."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from researcher.agents.native_deps import NativeAgentDeps
from researcher.agents.verify import VerifyAgent
from researcher.llm.prompts import default_registry
from researcher.models import AgentState, Task, TaskKind
from researcher.storage.store import Conflict, FieldCell
from tests.stubs.bus import StubEventBus
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubKnowledgeStore


def _entity_schema() -> dict:
    return {
        "entity_type": "War",
        "fields": [
            {"name": "name", "type": "str", "required": True},
            {"name": "start_year", "type": "int", "required": False},
        ],
    }


def _verify_task(entity_id: str, field: str) -> Task:
    return Task(
        kind=TaskKind.VERIFY,
        spec_ref="wars",
        target_entity_id=entity_id,
        field_hints=[field],
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


def _llm_picking(winning_value: Any, reason: str = "more authoritative") -> StubLLMClient:
    llm = StubLLMClient()

    async def fake_structured(messages, schema, tier, task_id, temperature=0.0, schema_retry=True):
        return schema(winning_value=winning_value, reason=reason)

    llm.complete_structured = fake_structured  # type: ignore[method-assign]
    return llm


@pytest.mark.asyncio
async def test_verify_picks_winner_and_emits_claim() -> None:
    store = StubKnowledgeStore()
    entity_id = await store.upsert_entity(
        entity_type="War", name="World War II", fields={}
    )
    now = datetime.now(UTC)
    cell_a = FieldCell(value=1939, confidence=0.9, provenance_ids=["p1"], updated_at=now)
    cell_b = FieldCell(value=1940, confidence=0.6, provenance_ids=["p2"], updated_at=now)
    conflict = Conflict(
        conflict_id="c1",
        entity_id=entity_id,
        field="start_year",
        candidates=[cell_a, cell_b],
        status="open",
    )
    await store.record_conflict(conflict)

    deps = _make_deps()
    llm = _llm_picking(winning_value=1939)
    bus = StubEventBus()

    agent = VerifyAgent(
        agent_id="native-ver",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
    )

    result = await agent.run(_verify_task(entity_id, "start_year"))

    assert result.state == AgentState.DONE
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.field == "start_year"
    assert claim.value == 1939
    assert claim.confidence == 0.9
    # The conflict should be marked resolved.
    open_conflicts = await store.get_conflicts(status="open")
    resolved_conflicts = await store.get_conflicts(status="resolved")
    assert open_conflicts == []
    assert len(resolved_conflicts) == 1
    assert resolved_conflicts[0].winning_value == 1939


@pytest.mark.asyncio
async def test_verify_no_matching_conflict_returns_done_with_zero_claims() -> None:
    store = StubKnowledgeStore()
    entity_id = await store.upsert_entity(
        entity_type="War", name="World War I", fields={}
    )
    deps = _make_deps()
    llm = _llm_picking(winning_value=None)
    bus = StubEventBus()

    agent = VerifyAgent(
        agent_id="native-ver",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        deps=deps,
        entity_schema=_entity_schema(),
    )

    result = await agent.run(_verify_task(entity_id, "start_year"))

    assert result.state == AgentState.DONE
    assert result.claims == []
