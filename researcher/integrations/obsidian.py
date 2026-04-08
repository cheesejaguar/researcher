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


def _yaml_scalar(value: Any) -> str:
    """Serialize a scalar for YAML frontmatter (strings, ints, floats, bools)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    # Strings: only quote if they contain YAML special characters.
    s = str(value)
    if any(c in s for c in ':#&*!|>\'"%@`') or s.strip() != s:
        escaped = s.replace('"', '\\"')
        return f'"{escaped}"'
    return s


def _yaml_field(name: str, value: Any, indent: int = 0) -> str:
    """Render one frontmatter line for a field (scalar) or list."""
    pad = " " * indent
    if isinstance(value, (list, tuple)):
        if not value:
            return f"{pad}{name}: []"
        lines = [f"{pad}{name}:"]
        for item in value:
            lines.append(f"{pad}  - {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{name}: {_yaml_scalar(value)}"


def _render_fields_table(state: EntityState) -> str:
    """Render a Markdown table summarizing each field's value, confidence, source."""
    rows = ["| Field | Value | Confidence | Source |", "|---|---|---|---|"]
    for field_name in sorted(state.fields):
        fv = state.fields[field_name]
        value_str = (
            ", ".join(_yaml_scalar(v) for v in fv.value)
            if isinstance(fv.value, (list, tuple))
            else str(fv.value)
        )
        source = fv.provenances[0].url if fv.provenances else ""
        source_link = f"[source]({source})" if source else "—"
        rows.append(
            f"| {field_name} | {value_str} | {fv.confidence:.2f} | {source_link} |"
        )
    return "\n".join(rows)


def _render_provenance(state: EntityState) -> str:
    """Render the Provenance section grouped by field."""
    blocks: list[str] = []
    for field_name in sorted(state.fields):
        fv = state.fields[field_name]
        value_str = (
            ", ".join(str(v) for v in fv.value)
            if isinstance(fv.value, (list, tuple))
            else str(fv.value)
        )
        blocks.append(f"### {field_name} = {value_str}")
        for prov in fv.provenances:
            blocks.append(f"> {prov.snippet}")
            blocks.append("")
            blocks.append(
                f"- Source: {prov.url}\n"
                f"- Agent: `{prov.agent_id}` · Extractor: `{prov.extractor_model}` · "
                f"Span: `{prov.span_id}`"
            )
            blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def _render_markdown(state: EntityState) -> str:
    """Produce the full Markdown file content for one entity."""
    # Frontmatter
    fm_lines = [
        "---",
        "# This file is managed by researcher. Edits will be overwritten on the next run.",
        _yaml_field("researcher_type", state.entity_type),
        _yaml_field("researcher_name", state.entity_name),
        _yaml_field("researcher_runs", sorted(state.run_ids)),
        # ISO 8601 timestamps are unambiguous in YAML; emit raw to keep the
        # frontmatter queryable as a literal string by Dataview and humans.
        f"researcher_updated: {state.updated_at.isoformat().replace('+00:00', 'Z')}",
        _yaml_field(
            "tags",
            ["researcher", f"researcher/{state.entity_type}"],
        ),
    ]
    for field_name in sorted(state.fields):
        fv = state.fields[field_name]
        fm_lines.append(_yaml_field(field_name, fv.value))
    fm_lines.append("---")

    # Body
    body_parts = [
        "",
        f"# {state.entity_name}",
        "",
        "> Managed by `researcher`. Re-running the job will overwrite this file.",
        "",
        "## Fields",
        "",
        _render_fields_table(state),
        "",
        "## Provenance",
        "",
        _render_provenance(state),
        "## Seen in Runs",
        "",
    ]
    for run_id in sorted(state.run_ids):
        body_parts.append(f"- `{run_id}`")
    body_parts.append("")

    return "\n".join(fm_lines) + "\n" + "\n".join(body_parts)


# ---------- ObsidianWriter ----------


class ObsidianWriter:
    """Secondary write-through sink that materializes FactClaims into Markdown.

    Thread-safety: not thread-safe. Designed for a single asyncio event loop.
    The orchestrator owns the writer's lifecycle.
    """

    def __init__(
        self,
        vault_path: Path | str,
        subdir: str = "researcher",
        flush_interval_s: float = 0.1,
    ) -> None:
        self._vault_path = Path(vault_path).expanduser()
        self._subdir = subdir
        self._flush_interval_s = flush_interval_s
        self._pending: dict[tuple[str, str], EntityState] = {}
        self._dirty: set[tuple[str, str]] = set()
        self._lock = asyncio.Lock()
        self._debounce_task: Optional[asyncio.Task] = None
        self._started: bool = False
        self._stopped: bool = False
        self._stats: dict[str, int] = {
            "writes": 0,
            "coalesced_claims": 0,
            "errors": 0,
            "sanitized_names": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def root_dir(self) -> Path:
        return self._vault_path / self._subdir

    async def start(self) -> None:
        if self._started:
            return
        # Validate that the vault root (the user-provided directory) exists.
        if not self._vault_path.exists():
            raise ObsidianVaultNotFound(
                f"Obsidian vault root does not exist: {self._vault_path}"
            )
        try:
            self.root_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ObsidianWriterError(
                f"could not create Obsidian subdir {self.root_dir}: {e}"
            ) from e
        self._debounce_task = asyncio.create_task(self._debounce_loop())
        self._started = True

    async def stop(self) -> None:
        if self._stopped:
            return
        # Final flush — even if start() was never called, flush is cheap.
        try:
            await self.flush()
        except Exception:
            pass
        self._stopped = True
        if self._debounce_task is not None:
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except (asyncio.CancelledError, Exception):
                pass
            self._debounce_task = None

    async def flush(self) -> None:
        """Placeholder — real implementation in Task 6."""
        return

    async def on_fact(self, claim: FactClaim, run_id: str) -> None:
        """Coalesce a claim into the pending buffer and mark the entity dirty.

        Never raises OSError — all disk I/O happens in the flush path. May
        raise RuntimeError if called after stop(), which is misuse.
        """
        if self._stopped:
            raise RuntimeError("ObsidianWriter is stopped")

        key = (claim.entity_type, claim.entity_name)
        async with self._lock:
            state = self._pending.get(key)
            if state is None:
                state = EntityState(
                    entity_type=claim.entity_type,
                    entity_name=claim.entity_name,
                )
                self._pending[key] = state

            state.run_ids.add(run_id)
            state.updated_at = datetime.now(timezone.utc)

            existing = state.fields.get(claim.field)
            if existing is None:
                state.fields[claim.field] = FieldValue(
                    value=claim.value,
                    confidence=claim.confidence,
                    provenances=[claim.provenance],
                )
            else:
                self._stats["coalesced_claims"] += 1
                # Higher-confidence value wins; always append provenance (deduped).
                if claim.confidence > existing.confidence:
                    existing.value = claim.value
                    existing.confidence = claim.confidence
                # Dedup by (url, span_id) tuple.
                seen = {(p.url, p.span_id) for p in existing.provenances}
                if (claim.provenance.url, claim.provenance.span_id) not in seen:
                    existing.provenances.append(claim.provenance)

            self._dirty.add(key)

    async def _debounce_loop(self) -> None:
        """Placeholder — real implementation in Task 6."""
        while not self._stopped:
            await asyncio.sleep(self._flush_interval_s)
