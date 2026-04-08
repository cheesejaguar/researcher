"""Tests for researcher.models — the core Pydantic data types."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from researcher.models import (
    AgentResult,
    AgentState,
    FactClaim,
    LLMTier,
    Provenance,
    Task,
    TaskKind,
)

# ---------- Enums ----------

def test_agent_state_values():
    assert [s.value for s in AgentState] == [
        "idle",
        "planning",
        "fetching",
        "extracting",
        "done",
        "failed",
    ]


def test_task_kind_values():
    assert [k.value for k in TaskKind] == ["discover", "expand", "verify", "enrich"]


def test_llm_tier_values():
    assert [t.value for t in LLMTier] == ["fast", "smart", "heavy"]


# ---------- Provenance ----------

def _sample_provenance() -> Provenance:
    return Provenance(
        url="https://example.com/a",
        fetched_at=datetime(2026, 4, 8, 12, 0, tzinfo=UTC),
        snippet="the sample says hello",
        extractor_model="openrouter/hermes-3-8b",
        agent_id="agent-1",
        task_id="task-1",
        span_id="ch_0-42",
    )


def test_provenance_is_frozen():
    p = _sample_provenance()
    with pytest.raises(ValidationError):
        p.url = "https://other.example.com"  # type: ignore[misc]


def test_provenance_requires_span_id():
    # span_id is load-bearing: lets a single source contribute multiple conflicting values.
    with pytest.raises(ValidationError):
        Provenance(
            url="https://example.com/a",
            fetched_at=datetime.now(UTC),
            snippet="s",
            extractor_model="m",
            agent_id="a",
            task_id="t",
        )  # type: ignore[call-arg]


# ---------- Task ----------

def test_task_defaults():
    deadline = datetime(2026, 4, 8, 13, 0, tzinfo=UTC)
    t = Task(
        kind=TaskKind.DISCOVER,
        spec_ref="specs/wars.yaml",
        budget_usd=0.10,
        deadline_ts=deadline,
    )
    assert t.id  # auto-generated
    assert len(t.id) >= 8
    assert t.depth == 0
    assert t.priority == 0
    assert t.attempt == 0
    assert t.field_hints == []
    assert t.parent_task_id is None
    assert t.target_entity_id is None
    assert t.seed_query is None


def test_task_ids_are_unique():
    deadline = datetime.now(UTC)
    a = Task(kind=TaskKind.DISCOVER, spec_ref="s", budget_usd=0.01, deadline_ts=deadline)
    b = Task(kind=TaskKind.DISCOVER, spec_ref="s", budget_usd=0.01, deadline_ts=deadline)
    assert a.id != b.id


# ---------- FactClaim ----------

def test_fact_claim_confidence_bounds():
    with pytest.raises(ValidationError):
        FactClaim(
            entity_type="War",
            entity_name="WWII",
            field="start_year",
            value=1939,
            confidence=1.5,
            provenance=_sample_provenance(),
            emitted_by="agent-1",
            task_id="task-1",
        )
    with pytest.raises(ValidationError):
        FactClaim(
            entity_type="War",
            entity_name="WWII",
            field="start_year",
            value=1939,
            confidence=-0.1,
            provenance=_sample_provenance(),
            emitted_by="agent-1",
            task_id="task-1",
        )


def test_fact_claim_accepts_any_json_value():
    # value is Any — must accept scalars AND lists/dicts without validation error
    for v in [1939, "Victory", ["Allies", "Axis"], {"by": "treaty"}, None]:
        c = FactClaim(
            entity_type="War",
            entity_name="WWII",
            field="x",
            value=v,
            confidence=0.9,
            provenance=_sample_provenance(),
            emitted_by="agent-1",
            task_id="task-1",
        )
        assert c.value == v


def test_fact_claim_roundtrip():
    c = FactClaim(
        entity_type="War",
        entity_name="WWII",
        field="start_year",
        value=1939,
        confidence=0.9,
        provenance=_sample_provenance(),
        emitted_by="agent-1",
        task_id="task-1",
    )
    data = c.model_dump(mode="json")
    c2 = FactClaim.model_validate(data)
    assert c2.value == 1939
    assert c2.provenance.url == "https://example.com/a"
    assert c2.claim_id == c.claim_id


# ---------- AgentResult ----------

def test_agent_result_defaults():
    r = AgentResult(task_id="task-1", agent_id="agent-1", state=AgentState.DONE)
    assert r.claims == []
    assert r.spawned_tasks == []
    assert r.tokens_in == 0
    assert r.tokens_out == 0
    assert r.cost_usd == 0.0
    assert r.wall_ms == 0
    assert r.error is None
