"""Human-in-the-loop interrupt handlers for the orchestrator.

When a `RunSpec.interrupt_points` includes a named point, the
Orchestrator calls the registered InterruptHandler at that point and
awaits a decision (CONTINUE or ABORT). Stub for tests; stdin reader
for the CLI; future TUI handler is a drop-in replacement.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Callable, Protocol


class InterruptDecision(str, Enum):
    CONTINUE = "continue"
    ABORT = "abort"


class InterruptHandler(Protocol):
    async def handle(
        self, point: str, context: dict[str, Any]
    ) -> InterruptDecision: ...


class StubInterruptHandler:
    """Always returns the configured default. Records every call for assertions."""

    def __init__(
        self, default_decision: InterruptDecision = InterruptDecision.CONTINUE
    ) -> None:
        self._default = default_decision
        self.calls: list[dict[str, Any]] = []

    async def handle(
        self, point: str, context: dict[str, Any]
    ) -> InterruptDecision:
        self.calls.append({"point": point, "context": dict(context)})
        return self._default


class StdinInterruptHandler:
    """Reads y/n from stdin at each interrupt. Used by --interactive CLI."""

    def __init__(self, prompt_fn: Callable[[str], str] | None = None) -> None:
        self._prompt_fn = prompt_fn

    async def handle(
        self, point: str, context: dict[str, Any]
    ) -> InterruptDecision:
        # Render a brief summary of the context, then ask the user.
        summary_lines = [f"  - {k}: {v}" for k, v in context.items()]
        summary = "\n".join(summary_lines) if summary_lines else "  (no context)"
        prompt = (
            f"\n[interrupt @ {point}]\n{summary}\n"
            f"Continue? [Y/n] "
        )
        if self._prompt_fn is not None:
            response = self._prompt_fn(prompt)
        else:
            # Run the blocking input() in a thread so we don't block the loop.
            response = await asyncio.to_thread(input, prompt)
        response = response.strip().lower()
        if response in ("", "y", "yes"):
            return InterruptDecision.CONTINUE
        return InterruptDecision.ABORT
