"""Tests for the v1.2 difficulty-aware compute gate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from researcher.budget import Budget
from researcher.difficulty import DifficultyEstimate, DifficultyGate, StubDifficultyGate
from researcher.models import Task, TaskKind


def _task(query: str = "list major wars") -> Task:
    return Task(
        kind=TaskKind.DISCOVER,
        spec_ref="x",
        seed_query=query,
        budget_usd=0.05,
        deadline_ts=datetime.now(UTC) + timedelta(minutes=5),
    )


# ---------- Budget.allows_task_with_difficulty ----------


def test_allows_task_with_difficulty_easy():
    """Easy tasks (difficulty=1) get a tighter budget cap (~ 1/5 of normal)."""
    b = Budget(usd_cap=10.0, wall_cap_s=600, per_task_fraction=0.1)
    # Normal allows up to 1.0 (10% of 10).
    # Easy should get 1/5 of that = 0.2.
    assert b.allows_task_with_difficulty(0.15, difficulty=1)
    assert not b.allows_task_with_difficulty(0.30, difficulty=1)


def test_allows_task_with_difficulty_hard():
    """Hard tasks (difficulty=5) get the full per-task fraction."""
    b = Budget(usd_cap=10.0, wall_cap_s=600, per_task_fraction=0.1)
    # Hard tasks can use the full per-task fraction.
    assert b.allows_task_with_difficulty(0.95, difficulty=5)
    assert not b.allows_task_with_difficulty(1.5, difficulty=5)


def test_allows_task_with_difficulty_medium():
    """Medium difficulty (3) gets ~3/5 of the per-task cap."""
    b = Budget(usd_cap=10.0, wall_cap_s=600, per_task_fraction=0.1)
    # 3/5 of 1.0 = 0.6.
    assert b.allows_task_with_difficulty(0.55, difficulty=3)
    assert not b.allows_task_with_difficulty(0.7, difficulty=3)


def test_allows_task_with_difficulty_invalid_falls_back():
    """Difficulty values outside 1-5 should fall back to the standard allows_task."""
    b = Budget(usd_cap=10.0, wall_cap_s=600, per_task_fraction=0.1)
    assert b.allows_task_with_difficulty(0.95, difficulty=10) == b.allows_task(0.95)


# ---------- StubDifficultyGate ----------


@pytest.mark.asyncio
async def test_stub_gate_returns_canned_difficulty():
    gate = StubDifficultyGate(default=3)
    est = await gate.estimate(_task())
    assert est.difficulty == 3
    assert isinstance(est, DifficultyEstimate)
    assert est.confidence > 0


@pytest.mark.asyncio
async def test_stub_gate_records_calls():
    gate = StubDifficultyGate(default=4)
    await gate.estimate(_task("query A"))
    await gate.estimate(_task("query B"))
    assert len(gate.calls) == 2
    assert gate.calls[0]["query"] == "query A"


@pytest.mark.asyncio
async def test_stub_gate_per_query_overrides():
    gate = StubDifficultyGate(default=2, overrides={"hard query": 5})
    easy = await gate.estimate(_task("simple query"))
    hard = await gate.estimate(_task("hard query"))
    assert easy.difficulty == 2
    assert hard.difficulty == 5


# ---------- LLM-backed DifficultyGate ----------


@pytest.mark.asyncio
async def test_llm_difficulty_gate_extracts_int_from_response():
    """The LLM-backed gate should parse a number 1-5 from the structured response."""
    from tests.stubs.llm import StubLLMClient

    class _MockLLM(StubLLMClient):
        async def complete_structured(self, messages, schema, tier, task_id, **kwargs):
            return schema(difficulty=4, reason="three sources needed")

    gate = DifficultyGate(llm=_MockLLM())
    est = await gate.estimate(_task("research the entire war of 1812"))
    assert est.difficulty == 4
