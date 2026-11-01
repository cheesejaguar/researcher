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

    def content_hash(self, messages: list[dict], tier: LLMTier, model: str | None = None) -> str:
        blob = json.dumps(
            {"tier": tier.value, "model": model, "messages": messages}, sort_keys=True
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    async def complete(
        self,
        messages: list[dict],
        tier: LLMTier,
        task_id: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "kind": "complete",
                "tier": tier.value,
                "task_id": task_id,
                "model": model,
            }
        )
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
        model: str | None = None,
    ) -> T:
        self.calls.append(
            {
                "kind": "structured",
                "tier": tier.value,
                "task_id": task_id,
                "messages": messages,
                "model": model,
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
        field_sources: dict[str, str] = {}
        for field, info in schema.model_fields.items():
            if field == "_placeholder":
                continue
            if field == "field_sources":
                payload[field] = field_sources
                continue
            value = self._value_for_field(field, prompt)
            if value == "":
                # Unknown schema fields still need a type-correct default so
                # the response validates; metadata must never get an empty
                # string where a dict/list is expected.
                value = self._default_for_annotation(info.annotation)
            payload[field] = value
            if value not in (None, "", [], {}):
                suffix = "glp1/step-1" if "ClinicalTrial" in prompt else "wars/world-war-ii"
                field_sources[field] = f"https://offline.local/{suffix}"
        return payload

    @staticmethod
    def _default_for_annotation(annotation: object) -> object:
        import types
        import typing

        origin = typing.get_origin(annotation)
        if origin in (list, set, tuple):
            return []
        if origin is dict:
            return {}
        if origin in (typing.Union, types.UnionType):
            members = [arg for arg in typing.get_args(annotation) if arg is not types.NoneType]
            if len(members) == 1:
                return OfflineFixtureLLM._default_for_annotation(members[0])
            return None
        if annotation is bool:
            return False
        if annotation is int:
            return 0
        if annotation is float:
            return 0.0
        if annotation is str:
            return ""
        return None

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
