"""Council verification must call each configured model independently."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from researcher.budget import Budget
from researcher.models import LLMTier
from researcher.orchestrator import Orchestrator
from researcher.spec import (
    EntitySpec,
    FieldSpec,
    RunSpec,
    VerificationConfig,
)
from researcher.storage.store import FieldCell
from tests.stubs.llm import StubLLMClient


def _spec() -> RunSpec:
    return RunSpec(
        spec_id="wars",
        goal="Wars",
        entities=[
            EntitySpec(
                name="War",
                fields=[FieldSpec(name="name", type="str", required=True)],
            )
        ],
        seeds=["wars"],
        models={"fast": "m-fast", "smart": "m-smart", "heavy": "m-heavy"},
        verification=VerificationConfig(
            mode="council",
            models=["model-a", "model-b"],
            confidence_threshold=0.65,
        ),
    )


@pytest.mark.asyncio
async def test_council_calls_every_model_and_records_its_vote(store, bus) -> None:
    # One stored entity whose required ``name`` field is up for review.
    entity_id = await store.upsert_entity(
        "War",
        "World War II",
        {
            "name": FieldCell(
                value="World War II",
                confidence=0.9,
                provenance_ids=[],
                updated_at=datetime.now(UTC),
            )
        },
    )
    llm = StubLLMClient()
    calls: list[tuple[str | None, LLMTier]] = []

    async def fake_structured(
        messages,
        schema,
        tier,
        task_id,
        temperature=0.0,
        schema_retry=True,
        model=None,
    ):
        calls.append((model, tier))
        return schema(
            vote="World War II",
            confidence=0.91,
            rationale=f"verified by {model}",
        )

    llm.complete_structured = fake_structured  # type: ignore[method-assign]
    orch = Orchestrator(
        spec=_spec(),
        store=store,
        llm=llm,
        bus=bus,
        writer=None,  # type: ignore[arg-type]
        scheduler=None,  # type: ignore[arg-type]
        budget=Budget(usd_cap=1.0, wall_cap_s=60),
        run_id="run-council",
    )

    count = await orch._maybe_record_council_votes()

    assert count == 2
    assert calls == [
        ("model-a", LLMTier.SMART),
        ("model-b", LLMTier.SMART),
    ]
    votes = await store.query(
        "SELECT run_id, entity_id, field_name, model, vote_json, confidence, rationale "
        "FROM verification_votes ORDER BY model"
    )
    assert [v["model"] for v in votes] == ["model-a", "model-b"]
    assert all(v["run_id"] == "run-council" for v in votes)
    assert all(v["entity_id"] == entity_id for v in votes)
    assert all(v["field_name"] == "name" for v in votes)
    assert all(json.loads(v["vote_json"]) == "World War II" for v in votes)
    assert all(v["confidence"] == pytest.approx(0.91) for v in votes)
    assert votes[0]["rationale"] == "verified by model-a"
    assert votes[1]["rationale"] == "verified by model-b"
