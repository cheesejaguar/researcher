"""Obsidian vault write-through sink.

Materializes FactClaims into Markdown files in a user-provided Obsidian
vault. Designed as a secondary sink: failures here must NOT disrupt the
orchestrator's primary reduce path. See doc/2026-04-08-obsidian-integration-design.md.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from researcher.models import FactClaim, Provenance


class ObsidianWriterError(RuntimeError):
    """Base class for Obsidian writer errors that abort a run at start time."""


class ObsidianVaultNotFound(ObsidianWriterError):
    """Vault root directory does not exist. Fail loud — probably a typo."""


@dataclass
class FieldValue:
    """One field's value plus every source that contributed a claim."""

    value: Any
    confidence: float
    provenances: list[Provenance] = field(default_factory=list)


@dataclass
class EntityState:
    """In-memory coalescing buffer for one entity's pending writes."""

    entity_type: str
    entity_name: str
    fields: dict[str, FieldValue] = field(default_factory=dict)
    run_ids: set[str] = field(default_factory=set)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
