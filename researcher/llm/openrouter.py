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
# Note: OpenRouter uses dot-separated version strings ("claude-haiku-4.5"),
# not hyphenated ones. Both conventions appear here for legacy compatibility.
_PRICING: dict[str, tuple[float, float]] = {
    "default": (0.50, 1.50),
    "openrouter/hermes-3-8b": (0.10, 0.20),
    "openrouter/hermes-3-70b": (0.50, 1.20),
    # Legacy hyphen-separated IDs (retained for existing test fixtures).
    "anthropic/claude-sonnet-4-6": (3.00, 15.00),
    "anthropic/claude-opus-4-6": (15.00, 75.00),
    # Current OpenRouter dot-separated Claude 4.x IDs (verified via
    # /v1/models API, April 2026).
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "anthropic/claude-opus-4.6": (5.00, 25.00),
    "anthropic/claude-opus-4.6-fast": (30.00, 150.00),
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    pi, po = _PRICING.get(model, _PRICING["default"])
    return (tokens_in * pi + tokens_out * po) / 1_000_000


def _strip_json_fence(text: str) -> str:
    """Strip markdown code fences (```json ... ``` or ``` ... ```) from an LLM
    response. Anthropic models routinely wrap JSON in fences even when told
    not to. Returns the inner content, stripped, or the original text if no
    fence is detected.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    # Drop the opening fence line (either ``` or ```json).
    nl = s.find("\n")
    if nl == -1:
        return s
    inner = s[nl + 1 :]
    # Drop trailing fence.
    if inner.rstrip().endswith("```"):
        inner = inner.rstrip()[:-3]
    return inner.strip()


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

    def _apply_prompt_caching(self, messages: list[dict]) -> list[dict]:
        """Mark the shared system prefix as a cacheable content part.

        Anthropic's prompt caching (and OpenRouter's pass-through) accepts
        ``cache_control`` on a message-content part. The cacheable content
        must be in list-of-parts format. Only the LAST system message is
        marked so the provider's cache hits when the same shared system
        context is reused across many parallel workers.

        Per Anthropic's June 2025 engineering blog, this yields ~90% input
        cost reduction and ~75% latency reduction for orchestrators sharing
        a system prompt across many parallel workers.

        Pure: returns a new list and does not mutate the input.
        """
        if not messages:
            return list(messages)
        out = [dict(m) for m in messages]
        last_system_idx: int | None = None
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "system":
                last_system_idx = i
                break
        if last_system_idx is None:
            return out
        sys_msg = out[last_system_idx]
        content = sys_msg.get("content")
        if isinstance(content, str):
            out[last_system_idx] = {
                **sys_msg,
                "content": [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        elif isinstance(content, list) and content:
            new_content = [dict(p) if isinstance(p, dict) else p for p in content]
            first = new_content[0]
            if isinstance(first, dict):
                first["cache_control"] = {"type": "ephemeral"}
            out[last_system_idx] = {**sys_msg, "content": new_content}
        return out

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
        cached_messages = self._apply_prompt_caching(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": cached_messages,
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
        cache_read_input_tokens = 0
        if usage_obj is not None:
            details = getattr(usage_obj, "prompt_tokens_details", None)
            if details is not None:
                try:
                    cache_read_input_tokens = int(
                        getattr(details, "cached_tokens", 0) or 0
                    )
                except (TypeError, ValueError):
                    cache_read_input_tokens = 0
        cost = _estimate_cost(model, tokens_in, tokens_out)

        response = LLMResponse(
            text=text,
            usage=LLMUsage(
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=cost,
                model=model,
                cache_hit=False,
                cache_read_input_tokens=cache_read_input_tokens,
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
            data = json.loads(_strip_json_fence(response.text))
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
                        "No prose, no markdown, no code fences."
                    ),
                }
            )
            retry_response = await self.complete(
                stricter, tier, task_id, temperature=temperature
            )
            data = json.loads(_strip_json_fence(retry_response.text))
            return schema.model_validate(data)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await embed_texts(texts)

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker
