"""Obsidian vault write-through sink.

Materializes FactClaims into Markdown files in a user-provided Obsidian
vault. Designed as a secondary sink: failures here must NOT disrupt the
orchestrator's primary reduce path. See doc/2026-04-08-obsidian-integration-design.md.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
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


# ---------- Helpers ----------


_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|]')
_DOT_RUN = re.compile(r"\.{2,}")
_WHITESPACE_RUN = re.compile(r"\s+")
_MAX_FILENAME_LEN = 200


def _safe_filename(name: str) -> str:
    """Sanitize an entity name for use as a filename.

    - Strips path separators and Windows-reserved characters.
    - Collapses whitespace and removes path-traversal dot runs.
    - Falls back to `entity_<sha1_8>` for empty or all-special names.
    - Truncates to 200 characters with an 8-char hash suffix for collision safety.
    """
    original = name
    # Kill path-traversal dot runs before anything else so `../` can't survive.
    cleaned = _DOT_RUN.sub("_", name)
    cleaned = _UNSAFE_CHARS.sub("_", cleaned)
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned).strip()
    # Strip leading/trailing dots (Obsidian hides dotfiles).
    cleaned = cleaned.strip(".")
    # Treat all-separator / all-underscore residue as empty so it falls back.
    if not cleaned or not cleaned.strip("_ "):
        digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]
        return f"entity_{digest}"
    if len(cleaned) > _MAX_FILENAME_LEN:
        digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:7]
        cleaned = cleaned[: _MAX_FILENAME_LEN - 9] + "_" + digest
    return cleaned
