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


# ---------- BoN-MAV (v1.2) ----------


@pytest.mark.asyncio
async def test_verify_agent_runs_three_aspect_verifiers_in_parallel() -> None:
    """VerifyAgent should call complete_structured 3 times (once per aspect) and majority vote."""
    call_aspects: list[str] = []

    llm = StubLLMClient()

    async def fake_structured(
        messages, schema, tier, task_id, temperature=0.0, schema_retry=True
    ):
        sys_text = ""
        for m in messages:
            if m.get("role") == "system":
                content = m.get("content", "")
                if isinstance(content, list):
                    sys_text = " ".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
                else:
                    sys_text = content
                break
        matched = "unknown"
        for aspect in (
            "factual_consistency",
            "source_quality",
            "entity_resolution",
        ):
            if aspect in sys_text:
                matched = aspect
                break
        call_aspects.append(matched)
        return schema(winning_value=1939, reason=f"aspect-vote:{matched}")

    llm.complete_structured = fake_structured  # type: ignore[method-assign]

    store = StubKnowledgeStore()
    entity_id = await store.upsert_entity(
        entity_type="War", name="World War II", fields={}
    )
    now = datetime.now(UTC)
    conflict = Conflict(
        conflict_id="c-mav-1",
        entity_id=entity_id,
        field="start_year",
        candidates=[
            FieldCell(value=1939, confidence=0.9, provenance_ids=[], updated_at=now),
            FieldCell(value=1940, confidence=0.85, provenance_ids=[], updated_at=now),
        ],
        status="open",
    )
    await store.record_conflict(conflict)

    bus = StubEventBus()
    deps = _make_deps()

    agent = VerifyAgent(
        agent_id="v1",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="r1",
        deps=deps,
        entity_schema=_entity_schema(),
    )

    result = await agent.run(_verify_task(entity_id, "start_year"))

    # All three aspects should have been queried.
    assert "factual_consistency" in call_aspects
    assert "source_quality" in call_aspects
    assert "entity_resolution" in call_aspects
    assert len(call_aspects) == 3
    # Result should be DONE with at least one claim emitted.
    assert result.state == AgentState.DONE
    assert len(result.claims) == 1
    assert result.claims[0].value == 1939
    # Conflict marked resolved.
    resolved = await store.get_conflicts(status="resolved")
    assert len(resolved) == 1
    assert resolved[0].winning_value == 1939


@pytest.mark.asyncio
async def test_verify_agent_majority_vote_across_aspects() -> None:
    """Two aspects vote for value A, one votes for B -> majority wins."""
    store = StubKnowledgeStore()
    entity_id = await store.upsert_entity(
        entity_type="War", name="World War II", fields={}
    )
    now = datetime.now(UTC)
    conflict = Conflict(
        conflict_id="c-mav-2",
        entity_id=entity_id,
        field="start_year",
        candidates=[
            FieldCell(value=1939, confidence=0.9, provenance_ids=[], updated_at=now),
            FieldCell(value=1940, confidence=0.85, provenance_ids=[], updated_at=now),
        ],
        status="open",
    )
    await store.record_conflict(conflict)

    llm = StubLLMClient()

    async def fake_structured(
        messages, schema, tier, task_id, temperature=0.0, schema_retry=True
    ):
        sys_text = ""
        for m in messages:
            if m.get("role") == "system":
                sys_text = m.get("content", "")
                break
        # factual_consistency + source_quality vote 1939; entity_resolution votes 1940
        if "entity_resolution" in sys_text:
            return schema(winning_value=1940, reason="entity_resolution says 1940")
        return schema(winning_value=1939, reason="majority picks 1939")

    llm.complete_structured = fake_structured  # type: ignore[method-assign]

    bus = StubEventBus()
    deps = _make_deps()

    agent = VerifyAgent(
        agent_id="v2",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="r2",
        deps=deps,
        entity_schema=_entity_schema(),
    )

    result = await agent.run(_verify_task(entity_id, "start_year"))

    assert result.state == AgentState.DONE
    assert len(result.claims) == 1
    # Majority vote wins: 1939 got 2 votes, 1940 got 1 vote.
    assert result.claims[0].value == 1939
    resolved = await store.get_conflicts(status="resolved")
    assert len(resolved) == 1
    assert resolved[0].winning_value == 1939
