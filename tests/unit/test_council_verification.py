"""Council verification must call each configured model independently."""

from __future__ import annotations

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
from tests.stubs.bus import StubEventBus
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubKnowledgeStore


class _CouncilStore(StubKnowledgeStore):
    def __init__(self) -> None:
        super().__init__()
        self.votes: list[dict] = []

    async def query(self, sql: str, params: tuple = ()) -> list[dict]:
        return [
            {
                "entity_id": "e1",
                "field_name": "name",
                "value_json": '"World War II"',
                "confidence": 0.9,
            }
        ]

    async def record_verification_vote(self, vote: dict) -> None:
        self.votes.append(vote)


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
async def test_council_calls_every_model_and_records_its_vote() -> None:
    store = _CouncilStore()
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
        bus=StubEventBus(),
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
    assert [v["model"] for v in store.votes] == ["model-a", "model-b"]
    assert all(v["vote"] == "World War II" for v in store.votes)
    assert all(v["confidence"] == pytest.approx(0.91) for v in store.votes)
    assert store.votes[0]["rationale"] == "verified by model-a"
    assert store.votes[1]["rationale"] == "verified by model-b"
