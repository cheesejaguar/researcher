"""Difficulty-aware compute gate (v1.2).

Per CODA (arXiv 2603.08659) and TALE-EP, classifying each subtask as
easy/medium/hard before dispatching the heavy worker prevents the
"overthinking on simple tasks" failure mode that the 2025 TTC survey
identified as the dominant inefficiency in current agent systems.

This module provides:
- ``DifficultyEstimate`` — output dataclass
- ``DifficultyGate`` — LLM-backed estimator (uses fast tier)
- ``StubDifficultyGate`` — deterministic test stub
- ``DifficultyGateProtocol`` — a Protocol for swap-in implementations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field

from researcher.llm.client import LLMClient
from researcher.models import LLMTier, Task


@dataclass
class DifficultyEstimate:
    """Output of a difficulty estimator.

    ``difficulty`` is an integer in ``[1, 5]`` where 1 is trivial and 5 is
    open-ended deep research. ``confidence`` is the estimator's self-reported
    confidence in ``[0, 1]``.
    """

    difficulty: int  # 1 (trivial) to 5 (very hard)
    reason: str = ""
    confidence: float = 0.9


class DifficultyGateProtocol(Protocol):
    """Structural type that any difficulty gate must satisfy."""

    async def estimate(self, task: Task) -> DifficultyEstimate: ...


class _DifficultyResponse(BaseModel):
    """Pydantic schema for the LLM's structured difficulty response."""

    difficulty: int = Field(ge=1, le=5)
    reason: str = ""


class DifficultyGate:
    """LLM-backed difficulty estimator.

    Uses the FAST tier so it stays cheap relative to the worker it gates.
    A single structured call per task returns ``{difficulty, reason}``; any
    error falls back conservatively to ``difficulty=3`` (medium).
    """

    def __init__(
        self,
        llm: LLMClient,
        tier: LLMTier = LLMTier.FAST,
    ) -> None:
        self._llm = llm
        self._tier = tier

    async def estimate(self, task: Task) -> DifficultyEstimate:
        prompt = (
            "Rate the difficulty of the following research task on a scale of 1 to 5, "
            "where 1 = trivial single-fact lookup, 3 = moderate multi-source "
            "aggregation, and 5 = open-ended deep research requiring many sources. "
            "Return JSON with the difficulty (1-5) and a one-sentence reason."
        )
        user = (
            f"Task kind: {task.kind.value}\n"
            f"Seed query: {task.seed_query or '(none)'}\n"
            f"Field hints: {task.field_hints or '(none)'}"
        )
        try:
            response = await self._llm.complete_structured(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user},
                ],
                schema=_DifficultyResponse,
                tier=self._tier,
                task_id=task.id,
            )
            return DifficultyEstimate(
                difficulty=int(response.difficulty),
                reason=response.reason,
                confidence=0.85,
            )
        except Exception:
            # Conservative fallback on any error: assume medium.
            return DifficultyEstimate(
                difficulty=3, reason="estimator_error", confidence=0.5
            )


class StubDifficultyGate:
    """Deterministic test stub.

    Returns ``default`` for any task, unless the task's ``seed_query`` matches
    a key in ``overrides``. Every call is recorded on ``self.calls`` so tests
    can assert call counts and arguments.
    """

    def __init__(
        self,
        default: int = 3,
        overrides: Optional[dict[str, int]] = None,
    ) -> None:
        self._default = default
        self._overrides = overrides or {}
        self.calls: list[dict[str, Any]] = []

    async def estimate(self, task: Task) -> DifficultyEstimate:
        query = task.seed_query or ""
        diff = self._overrides.get(query, self._default)
        self.calls.append(
            {"query": query, "difficulty": diff, "task_id": task.id}
        )
        return DifficultyEstimate(
            difficulty=diff, reason=f"stub default={self._default}"
        )
