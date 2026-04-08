"""Simple in-memory cost tracker that aggregates LLMUsage per task_id."""

from __future__ import annotations

from researcher.llm.client import LLMUsage


class SimpleCostTracker:
    """In-memory aggregator of LLMUsage records, bucketed by task_id.

    Implements the CostTracker Protocol from researcher.llm.client. Not
    thread-safe — intended to be owned by a single orchestrator run.
    """

    def __init__(self) -> None:
        self._by_task: dict[str, LLMUsage] = {}
        self._total_usd: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0

    def add(self, task_id: str, usage: LLMUsage) -> None:
        existing = self._by_task.get(task_id)
        if existing is None:
            self._by_task[task_id] = LLMUsage(
                tokens_in=usage.tokens_in,
                tokens_out=usage.tokens_out,
                cost_usd=usage.cost_usd,
                model=usage.model,
                cache_hit=usage.cache_hit,
            )
        else:
            self._by_task[task_id] = LLMUsage(
                tokens_in=existing.tokens_in + usage.tokens_in,
                tokens_out=existing.tokens_out + usage.tokens_out,
                cost_usd=existing.cost_usd + usage.cost_usd,
                model=existing.model,
                cache_hit=existing.cache_hit or usage.cache_hit,
            )
        self._total_usd += usage.cost_usd
        self._total_tokens_in += usage.tokens_in
        self._total_tokens_out += usage.tokens_out

    def total_usd(self) -> float:
        return self._total_usd

    def by_task(self, task_id: str) -> LLMUsage:
        return self._by_task.get(
            task_id,
            LLMUsage(tokens_in=0, tokens_out=0, cost_usd=0.0, model="unknown"),
        )

    @property
    def total_tokens_in(self) -> int:
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self._total_tokens_out
