"""OpenRouter-backed LLMClient using the openai Python SDK.

Wave 1-C: real implementation of the LLMClient ABC. Caches responses
via LLMResponseCache, tracks usage via SimpleCostTracker, and lazy-
imports the openai SDK so the rest of the codebase doesn't pay the
import cost until first use.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any, Optional, TypeVar

from pydantic import BaseModel, ValidationError

from researcher.llm.cache import LLMResponseCache
from researcher.llm.client import (
    CostTracker,
    LLMClient,
    LLMResponse,
    LLMUsage,
)
from researcher.llm.cost_tracker import SimpleCostTracker
from researcher.llm.embedder import embed_texts
from researcher.models import LLMTier

T = TypeVar("T", bound=BaseModel)


# Pricing table — USD per million input/output tokens, best-effort defaults.
# OpenRouter passes through to provider; `default` is used for unknown models.
_PRICING: dict[str, tuple[float, float]] = {
    "default": (0.50, 1.50),
    "openrouter/hermes-3-8b": (0.10, 0.20),
    "openrouter/hermes-3-70b": (0.50, 1.20),
    "anthropic/claude-sonnet-4-6": (3.00, 15.00),
    "anthropic/claude-opus-4-6": (15.00, 75.00),
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    pi, po = _PRICING.get(model, _PRICING["default"])
    return (tokens_in * pi + tokens_out * po) / 1_000_000


class OpenRouterClient(LLMClient):
    """Calls OpenRouter's OpenAI-compatible API via the openai Python SDK.

    Tiered model selection: FAST / SMART / HEAVY map to concrete OpenRouter
    model ids. All calls are tagged with a `task_id` so the CostTracker can
    attribute spend to the originating orchestrator task.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model_map: Optional[dict[LLMTier, str]] = None,
        cache: Optional[LLMResponseCache] = None,
        cost_tracker: Optional[CostTracker] = None,
        request_timeout_s: float = 60.0,
        async_openai_cls: Any = None,  # test seam — inject a fake class
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = base_url
        self._model_map = model_map or {
            LLMTier.FAST: "openrouter/hermes-3-8b",
            LLMTier.SMART: "openrouter/hermes-3-70b",
            LLMTier.HEAVY: "anthropic/claude-sonnet-4-6",
        }
        self._cache = cache
        self._cost_tracker: CostTracker = cost_tracker or SimpleCostTracker()
        self._timeout = request_timeout_s
        self._async_openai_cls = async_openai_cls
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._async_openai_cls is not None:
            cls = self._async_openai_cls
        else:
            from openai import AsyncOpenAI

            cls = AsyncOpenAI
        self._client = cls(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def content_hash(self, messages: list[dict], tier: LLMTier) -> str:
        model = self._model_map[tier]
        blob = json.dumps(
            {"model": model, "messages": messages}, sort_keys=True
        ).encode()
        return hashlib.sha256(blob).hexdigest()

    async def complete(
        self,
        messages: list[dict],
        tier: LLMTier,
        task_id: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        key = self.content_hash(messages, tier)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                self._cost_tracker.add(task_id, cached.usage)
                return cached

        model = self._model_map[tier]
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        raw = await asyncio.wait_for(
            client.chat.completions.create(**kwargs),
            timeout=self._timeout,
        )
        text = raw.choices[0].message.content or ""
        usage_obj = getattr(raw, "usage", None)
        tokens_in = int(getattr(usage_obj, "prompt_tokens", 0) or 0) if usage_obj else 0
        tokens_out = (
            int(getattr(usage_obj, "completion_tokens", 0) or 0) if usage_obj else 0
        )
        cost = _estimate_cost(model, tokens_in, tokens_out)

        response = LLMResponse(
            text=text,
            usage=LLMUsage(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                model=model,
                cache_hit=False,
            ),
        )
        if self._cache is not None:
            self._cache.put(key, response)
        self._cost_tracker.add(task_id, response.usage)
        return response

    async def complete_structured(
        self,
        messages: list[dict],
        schema: type[T],
        tier: LLMTier,
        task_id: str,
        temperature: float = 0.0,
        schema_retry: bool = True,
    ) -> T:
        # Append a schema instruction so the model returns ONLY JSON.
        schema_json = json.dumps(schema.model_json_schema(), separators=(",", ":"))
        augmented = list(messages)
        augmented.append(
            {
                "role": "user",
                "content": (
                    "Return ONLY a single JSON object matching this schema, "
                    "no prose, no markdown:\n" + schema_json
                ),
            }
        )
        response = await self.complete(
            augmented, tier, task_id, temperature=temperature
        )
        try:
            data = json.loads(response.text)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            if not schema_retry:
                raise
            # One retry with a stricter nudge.
            stricter = list(augmented)
            stricter.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous response could not be parsed. "
                        "Return ONLY a single JSON object matching the schema. "
                        "No prose, no markdown."
                    ),
                }
            )
            retry_response = await self.complete(
                stricter, tier, task_id, temperature=temperature
            )
            data = json.loads(retry_response.text)
            return schema.model_validate(data)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await embed_texts(texts)

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker
