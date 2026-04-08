"""LLMClient ABC + response/usage types.

The concrete OpenRouter-backed implementation lives in Wave 1-C. Wave 0 only
locks the contract so agents, the orchestrator, and the stub client can all be
written against it in parallel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from researcher.models import LLMTier


T = TypeVar("T", bound=BaseModel)


class LLMUsage(BaseModel):
    tokens_in: int
    tokens_out: int
    cost_usd: float
    model: str
    cache_hit: bool = False


class LLMResponse(BaseModel):
    text: str
    usage: LLMUsage


class StructuredResponse(BaseModel):
    data: Any
    usage: LLMUsage


class CostTracker(Protocol):
    """Per-task token accounting. Implementations live alongside LLMClient."""

    def add(self, task_id: str, usage: LLMUsage) -> None: ...

    def total_usd(self) -> float: ...

    def by_task(self, task_id: str) -> LLMUsage: ...


class LLMClient(ABC):
    """Tiered LLM client. fast/smart/heavy map to concrete OpenRouter model ids.

    All calls are tagged with a `task_id` so the CostTracker can attribute spend
    to the originating orchestrator task.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        tier: LLMTier,
        task_id: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def complete_structured(
        self,
        messages: list[dict],
        schema: type[T],
        tier: LLMTier,
        task_id: str,
        temperature: float = 0.0,
        schema_retry: bool = True,
    ) -> T:
        """Structured-output call. On validation failure, retries once iff schema_retry."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Local CPU embedding (sentence-transformers MiniLM). NOT routed to OpenRouter."""
        ...

    @abstractmethod
    def content_hash(self, messages: list[dict], tier: LLMTier) -> str:
        """Deterministic hash used as the cache key for prefix caching."""
        ...
