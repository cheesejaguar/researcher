"""Pydantic types + errors shared across the CLI subagent backend."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


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


class SubagentEntity(BaseModel):
    """One entity plus all field extractions sourced by the CLI subagent."""

    entity_name: str
    extractions: list[Extraction] = Field(default_factory=list)


class SubagentResponse(BaseModel):
    """The JSON shape the CLI is required to return — enforced by --json-schema.

    v1.4 accepts the new multi-entity shape:

        {"entities": [{"entity_name": "...", "extractions": [...]}]}

    The validator also accepts the older single-entity shape so existing
    fixtures, per-run caches, and post-mortem files keep parsing.
    """

    entities: list[SubagentEntity] = Field(default_factory=list)
    diagnostics: str = ""

    @model_validator(mode="before")
    @classmethod
    def _upgrade_single_entity_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "entities" in data:
            return data
        if "entity_name" not in data and "extractions" not in data:
            return data
        upgraded = dict(data)
        upgraded["entities"] = [
            {
                "entity_name": data.get("entity_name", ""),
                "extractions": data.get("extractions", []),
            }
        ]
        return upgraded

    @property
    def entity_name(self) -> str:
        """Backward-compatible first entity name for older tests/callers."""
        return self.entities[0].entity_name if self.entities else ""

    @property
    def extractions(self) -> list[Extraction]:
        """Backward-compatible flattened extraction list."""
        out: list[Extraction] = []
        for entity in self.entities:
            out.extend(entity.extractions)
        return out


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
