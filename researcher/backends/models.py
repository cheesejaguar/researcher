"""Pydantic types + errors shared across the CLI subagent backend."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class CliKind(str, Enum):
    CLAUDE_CODE = "claude_code"
    CODEX = "codex"


class Extraction(BaseModel):
    """One extracted field value for one entity, sourced from one URL."""

    field: str
    value: Any
    source_url: str
    snippet: str
    confidence: float = Field(ge=0.0, le=1.0)


class SubagentResponse(BaseModel):
    """The JSON shape the CLI is required to return — enforced by --json-schema."""

    entity_name: str
    extractions: list[Extraction] = Field(default_factory=list)
    diagnostics: str = ""


class CliResult(BaseModel):
    """Everything the runner produces from one subprocess invocation."""

    ok: bool
    data: Optional[SubagentResponse] = None
    error: Optional[str] = None
    wall_ms: int
    exit_code: Optional[int] = None
    raw_usage: Optional[dict] = None


class BackendChoice(BaseModel):
    """What BackendResolver.pick returns — which backend a given task should use."""

    kind: Optional[CliKind] = None  # None means "native API path"
    reason: str


class BackendUnavailableError(RuntimeError):
    """Raised when policy='cli' but no CLI is detected at run start."""
