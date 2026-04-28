"""Deterministic zero-cost LLM for offline CLI acceptance runs."""

from __future__ import annotations

import hashlib
import json
from typing import TypeVar

from pydantic import BaseModel

from researcher.llm.client import LLMClient, LLMResponse, LLMUsage
from researcher.models import LLMTier

T = TypeVar("T", bound=BaseModel)


class OfflineFixtureLLM(LLMClient):
    """Small deterministic client used by `researcher run --offline`.

    It is intentionally domain-light: enough to prove the orchestration,
    storage, inspect, and acceptance surfaces without live API calls.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def content_hash(self, messages: list[dict], tier: LLMTier) -> str:
        blob = json.dumps({"tier": tier.value, "messages": messages}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    async def complete(
        self,
        messages: list[dict],
        tier: LLMTier,
        task_id: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append({"kind": "complete", "tier": tier.value, "task_id": task_id})
        return LLMResponse(
            text="{}",
            usage=LLMUsage(tokens_in=0, tokens_out=0, cost_usd=0.0, model="offline"),
        )

    async def complete_structured(
        self,
        messages: list[dict],
        schema: type[T],
        tier: LLMTier,
        task_id: str,
        temperature: float = 0.0,
        schema_retry: bool = True,
    ) -> T:
        self.calls.append(
            {
                "kind": "structured",
                "tier": tier.value,
                "task_id": task_id,
                "messages": messages,
            }
        )
        payload = self._payload_for_schema(messages, schema)
        return schema.model_validate(payload)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            out.append([((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(384)])
        return out

    def _payload_for_schema(self, messages: list[dict], schema: type[T]) -> dict:
        fields = set(schema.model_fields)
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        if "entities" in fields:
            if "ClinicalTrial" in prompt:
                return {"entities": ["STEP 1", "SURPASS-2"]}
            return {"entities": ["World War I", "World War II"]}

        payload: dict[str, object] = {}
        for field in fields:
            if field == "_placeholder":
                continue
            payload[field] = self._value_for_field(field, prompt)
        return payload

    @staticmethod
    def _value_for_field(field: str, prompt: str) -> object:
        is_trial = "ClinicalTrial" in prompt or "STEP" in prompt or "SURPASS" in prompt
        trial_values: dict[str, object] = {
            "name": "STEP 1",
            "nct_id": "NCT03548935",
            "phase": "Phase 3",
            "drug": "semaglutide",
            "sponsor": "Novo Nordisk",
            "status": "completed",
            "start_date": "2018-06-01",
            "completion_date": "2021-03-01",
            "indication": "obesity",
            "enrollment": 1961,
            "primary_endpoint": "body weight change",
            "results_url": "https://offline.local/glp1/step-1",
        }
        war_values: dict[str, object] = {
            "name": "World War II",
            "start_year": 1939,
            "end_year": 1945,
            "belligerents": ["Allies", "Axis"],
            "theater": "global",
            "estimated_deaths": 70000000,
            "outcome": "Allied victory",
            "primary_causes": ["German invasion of Poland"],
            "wikipedia_url": "https://offline.local/wars/world-war-ii",
        }
        values = trial_values if is_trial else war_values
        return values.get(field, "")
