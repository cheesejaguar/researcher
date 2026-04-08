"""CriticAgent — appends hand-authored skill card suggestions to a Markdown log.

Wave 1-D native agent. Given a Task whose field_hints[0] identifies the
triggering agent_id, asks the LLM for a short skill card (name / description /
prompt fragment) and appends it to `skills/_suggestions.md` (or whatever path
is passed in). The file is curated by humans; CriticAgent only writes, never
reads, so skill card quality stays under human review.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from researcher.agents.base import Agent, EventEmitter
from researcher.agents.native_deps import NativeAgentDeps
from researcher.llm.client import LLMClient
from researcher.models import (
    AgentResult,
    AgentState,
    LLMTier,
    Task,
)
from researcher.storage.store import KnowledgeStore


class _SkillCard(BaseModel):
    name: str = ""
    description: str = ""
    prompt_fragment: str = ""


class CriticAgent(Agent):
    """Writes LLM-authored skill card suggestions to a Markdown log."""

    kind = "critic"

    def __init__(
        self,
        agent_id: str,
        llm: LLMClient,
        store: KnowledgeStore,
        emit: EventEmitter,
        run_id: str,
        deps: NativeAgentDeps,
        entity_schema: dict,
        suggestions_path: Path | str = Path("skills/_suggestions.md"),
    ) -> None:
        super().__init__(
            agent_id=agent_id, llm=llm, store=store, emit=emit, run_id=run_id
        )
        self._deps = deps
        self._entity_schema = entity_schema
        self._suggestions_path = Path(suggestions_path)

    async def run(self, task: Task) -> AgentResult:
        start = time.monotonic()
        await self.set_state(AgentState.PLANNING)

        trigger_agent = task.field_hints[0] if task.field_hints else "unknown"
        entity_type = self._entity_schema.get("entity_type", "Entity")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a critic that reviews research trajectories. "
                    "Suggest improvements as a short skill card with a name, "
                    "description, and a reusable prompt fragment."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Entity type under research: {entity_type}\n"
                    f"Triggering agent: {trigger_agent}\n\n"
                    f"Return a short skill card (name, description, prompt_fragment)."
                ),
            },
        ]

        await self.set_state(AgentState.EXTRACTING)
        try:
            card = await self.llm.complete_structured(
                messages=messages,
                schema=_SkillCard,
                tier=LLMTier.SMART,
                task_id=task.id,
            )
        except Exception as e:
            await self.log("error", f"llm critic failed: {e}")
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error=f"llm_failed: {e}",
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        try:
            self._append_card(card, trigger_agent)
        except OSError as e:
            await self.log("error", f"failed to append suggestion: {e}")
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error=f"write_failed: {e}",
                wall_ms=int((time.monotonic() - start) * 1000),
            )

        await self.set_state(AgentState.DONE)
        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            state=AgentState.DONE,
            claims=[],
            wall_ms=int((time.monotonic() - start) * 1000),
        )

    def _append_card(self, card: _SkillCard, trigger_agent: str) -> None:
        self._suggestions_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).isoformat()
        entry = (
            f"\n## {card.name or 'unnamed_skill'} ({ts})\n"
            f"Triggered by: `{trigger_agent}`\n\n"
            f"**Description:** {card.description}\n\n"
            f"**Prompt fragment:**\n\n> {card.prompt_fragment}\n"
        )
        with self._suggestions_path.open("a", encoding="utf-8") as fh:
            fh.write(entry)
