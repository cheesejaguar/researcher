"""Core data models — the types that flow through the orchestrator, agents, and storage."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    FETCHING = "fetching"
    EXTRACTING = "extracting"
    DONE = "done"
    FAILED = "failed"


class TaskKind(str, Enum):
    DISCOVER = "discover"  # seed search for candidate entities
    EXPAND = "expand"  # fill fields for a known entity
    VERIFY = "verify"  # resolve conflicts
    ENRICH = "enrich"  # follow relations


class LLMTier(str, Enum):
    FAST = "fast"  # cheap/small model for extraction
    SMART = "smart"  # mid model for planning/critique
    HEAVY = "heavy"  # expensive, used sparingly


class Provenance(BaseModel):
    """Immutable record of where a single fact was sourced from.

    span_id is a character offset / chunk id identifying the specific span of the source;
    it lets a single source contribute multiple (potentially conflicting) values.
    """

    model_config = ConfigDict(frozen=True)

    url: str
    fetched_at: datetime
    snippet: str
    extractor_model: str
    agent_id: str
    task_id: str
    span_id: str


class Task(BaseModel):
    """A unit of work placed on the orchestrator queue."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    kind: TaskKind
    spec_ref: str  # path to loaded spec
    target_entity_id: Optional[str] = None
    seed_query: Optional[str] = None
    field_hints: list[str] = Field(default_factory=list)
    parent_task_id: Optional[str] = None
    depth: int = 0
    budget_usd: float
    deadline_ts: datetime
    priority: int = 0  # higher runs first
    attempt: int = 0  # schema-retry counter


class FactClaim(BaseModel):
    """A single assertion about one field of one entity from one source."""

    claim_id: str = Field(default_factory=lambda: uuid4().hex)
    entity_type: str
    entity_name: str
    entity_id_hint: Optional[str] = None
    field: str
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: Provenance
    emitted_by: str  # agent_id
    task_id: str


class AgentResult(BaseModel):
    """The return value of Agent.run."""

    task_id: str
    agent_id: str
    state: AgentState
    claims: list[FactClaim] = Field(default_factory=list)
    spawned_tasks: list[Task] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    wall_ms: int = 0
    error: Optional[str] = None
