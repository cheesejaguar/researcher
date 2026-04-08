"""Agent ABC — the unit of parallel work in the map phase.

Concrete agents (DiscoverAgent, ExpandAgent, VerifyAgent, EnrichAgent,
CriticAgent) live in Wave 1-D. All of them share the same signature: take a
Task, emit FactClaims, return an AgentResult.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Awaitable, Callable, ClassVar

from researcher.events import (
    AgentLog,
    AgentLogPayload,
    AgentStateChange,
    AgentStateChangePayload,
    Event,
)
from researcher.llm.client import LLMClient
from researcher.models import AgentResult, AgentState, Task
from researcher.storage.store import KnowledgeStore

EventEmitter = Callable[[Event], Awaitable[None]]


class Agent(ABC):
    """Base class for all agents.

    Subclasses set a class-level `kind` (e.g. "discover") and implement
    `run(task) -> AgentResult`. The base class provides small helpers for
    emitting state-change and log events so every agent reports uniformly to
    the TUI.
    """

    kind: ClassVar[str] = "base"

    def __init__(
        self,
        agent_id: str,
        llm: LLMClient,
        store: KnowledgeStore,
        emit: EventEmitter,
        run_id: str,
    ) -> None:
        self.agent_id = agent_id
        self.llm = llm
        self.store = store
        self._emit = emit
        self._run_id = run_id
        self._state: AgentState = AgentState.IDLE

    @property
    def state(self) -> AgentState:
        return self._state

    async def set_state(self, new: AgentState) -> None:
        old = self._state
        self._state = new
        await self._emit(
            AgentStateChange(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=AgentStateChangePayload(
                    agent_id=self.agent_id, old=old.value, new=new.value
                ),
            )
        )

    async def log(self, level: str, msg: str) -> None:
        await self._emit(
            AgentLog(
                seq=0,
                ts=datetime.now(UTC),
                run_id=self._run_id,
                payload=AgentLogPayload(
                    agent_id=self.agent_id,
                    level=level,  # type: ignore[arg-type]
                    msg=msg,
                ),
            )
        )

    @abstractmethod
    async def run(self, task: Task) -> AgentResult: ...
