"""Obsidian vault write-through sink.

Materializes FactClaims into Markdown files in a user-provided Obsidian
vault. Designed as a secondary sink: failures here must NOT disrupt the
orchestrator's primary reduce path. See doc/2026-04-08-obsidian-integration-design.md.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


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


def _format_inline_field(name: str, value: Any) -> str:
    """Serialize one field as a Dataview-scrapable inline `[name:: value]` annotation."""
    if isinstance(value, (list, tuple)):
        formatted = ", ".join(str(v) for v in value)
    elif value is None:
        formatted = ""
    else:
        formatted = str(value)
    return f"[{name}:: {formatted}]"


def _render_inline_fields(state: EntityState) -> str:
    """Render an inline-fields block that Dataview can scrape from the body."""
    lines = ["<!-- dataview-scrapable inline fields below -->"]
    for field_name in sorted(state.fields):
        fv = state.fields[field_name]
        lines.append(_format_inline_field(field_name, fv.value))
    return "\n".join(lines)


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
        _render_inline_fields(state),
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


def _snapshot_state(state: EntityState) -> EntityState:
    """Shallow-deep copy of an EntityState for rendering outside the lock."""
    copied_fields = {
        name: FieldValue(
            value=fv.value,
            confidence=fv.confidence,
            provenances=list(fv.provenances),
        )
        for name, fv in state.fields.items()
    }
    return EntityState(
        entity_type=state.entity_type,
        entity_name=state.entity_name,
        fields=copied_fields,
        run_ids=set(state.run_ids),
        updated_at=state.updated_at,
    )


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
        """Synchronously drain all pending writes to disk.

        Secondary-sink discipline: swallows OSError and logs via stats,
        never raises (except CancelledError).
        """
        async with self._lock:
            to_write = list(self._dirty)
            self._dirty.clear()
        for key in to_write:
            await self._flush_one(key)

    async def _flush_one(self, key: tuple[str, str]) -> None:
        async with self._lock:
            state = self._pending.get(key)
            if state is None:
                return
            snapshot = _snapshot_state(state)

        try:
            target = self._path_for(snapshot)
            content = _render_markdown(snapshot)
            await asyncio.to_thread(self._atomic_write, target, content)
            self._stats["writes"] += 1
        except asyncio.CancelledError:
            raise
        except OSError as e:
            self._stats["errors"] += 1
            try:
                import logging
                logging.getLogger("researcher.obsidian").warning(
                    "obsidian write failed for %s: %s", key, e
                )
            except Exception:
                pass

    def _path_for(self, state: EntityState) -> Path:
        safe_name = _safe_filename(state.entity_name)
        if safe_name.startswith("entity_") and state.entity_name:
            self._stats["sanitized_names"] += 1
        return self.root_dir / state.entity_type / f"{safe_name}.md"

    def _atomic_write(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path_str = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp_path, target)
        except OSError:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            raise

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
            state.updated_at = datetime.now(UTC)

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
        """Periodically flush dirty entities to disk."""
        try:
            while not self._stopped:
                try:
                    await asyncio.sleep(self._flush_interval_s)
                except asyncio.CancelledError:
                    break
                if self._dirty:
                    try:
                        await self.flush()
                    except Exception:
                        self._stats["errors"] += 1
        except asyncio.CancelledError:
            pass

    # ---------- Post-flush surface-up hooks ----------

    async def write_index_notes(self) -> None:
        """Generate one `_index.md` per entity-type folder with a Dataview TABLE block.

        Lists every entity in that folder so users can navigate the typed
        collection inside Obsidian without leaving the app. Safe to call
        repeatedly; scans both the pending buffer and the on-disk layout so
        previously-flushed cycles get indices too.
        """
        if not self._started:
            return
        types_seen: set[str] = set()
        for entity_type, _ in self._pending.keys():
            types_seen.add(entity_type)
        type_root = self.root_dir
        if type_root.exists():
            for child in type_root.iterdir():
                if child.is_dir():
                    types_seen.add(child.name)

        for et in sorted(types_seen):
            target = self.root_dir / et / "_index.md"
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                content = self._render_index_note(et)
                await asyncio.to_thread(self._atomic_write, target, content)
                self._stats["writes"] += 1
            except OSError as e:
                self._stats["errors"] += 1
                try:
                    import logging
                    logging.getLogger("researcher.obsidian").warning(
                        "obsidian index write failed for %s: %s", et, e
                    )
                except Exception:
                    pass

    def _render_index_note(self, entity_type: str) -> str:
        """Render an `_index.md` for one entity type with a Dataview TABLE block."""
        return (
            "---\n"
            "# This file is managed by researcher. Edits will be overwritten on the next run.\n"
            f"researcher_type: {entity_type}\n"
            "researcher_index: true\n"
            "tags:\n"
            "  - researcher\n"
            f"  - researcher/{entity_type}\n"
            "  - researcher/index\n"
            "---\n"
            "\n"
            f"# {entity_type} index\n"
            "\n"
            f"> Generated by `researcher`. Lists all `{entity_type}` entities in this vault.\n"
            "\n"
            "```dataview\n"
            'TABLE WITHOUT ID file.link AS "Entity", researcher_type AS "Type"\n'
            f'FROM "researcher/{entity_type}"\n'
            "WHERE researcher_index != true\n"
            "SORT file.name ASC\n"
            "```\n"
        )

    async def inject_wikilinks(self, known_names: set[str]) -> None:
        """Post-process every entity file and wrap known entity names in `[[wikilinks]]`.

        Skips the entity's own self-reference (a note about World War II
        doesn't get its own name turned into a wikilink in its own body),
        leaves the YAML frontmatter untouched, and is safe to call repeatedly
        (word-boundary regex plus an explicit `(?<!\\[)` guard keep existing
        wikilinks from being double-wrapped).
        """
        if not self._started:
            return
        if not known_names:
            return
        # Longest-first so that "World War II" wins over "World War I" when
        # both appear in the same line.
        sorted_names = sorted(known_names, key=len, reverse=True)
        type_root = self.root_dir
        if not type_root.exists():
            return
        for type_dir in type_root.iterdir():
            if not type_dir.is_dir():
                continue
            for md_file in type_dir.glob("*.md"):
                if md_file.name == "_index.md":
                    continue
                self_name = md_file.stem
                other_names = [n for n in sorted_names if n != self_name]
                if not other_names:
                    continue
                # `(?<!\[)` avoids double-wrapping existing `[[...]]` links.
                # Word-boundary lookarounds prevent partial matches like
                # "WW1" inside "WW1234".
                pattern = re.compile(
                    r"(?<!\[)(?<!\w)("
                    + "|".join(re.escape(n) for n in other_names)
                    + r")(?!\w)(?!\])"
                )
                try:
                    content = await asyncio.to_thread(
                        md_file.read_text, encoding="utf-8"
                    )
                except OSError:
                    continue
                # Don't touch the YAML frontmatter (between the first two `---`).
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = "---" + parts[1] + "---"
                    body = parts[2]
                else:
                    frontmatter = ""
                    body = content
                new_body = pattern.sub(r"[[\1]]", body)
                if new_body == body:
                    continue
                new_content = frontmatter + new_body if frontmatter else new_body
                try:
                    await asyncio.to_thread(
                        self._atomic_write, md_file, new_content
                    )
                except OSError:
                    self._stats["errors"] += 1

    async def write_see_also(
        self,
        neighbors: dict[tuple[str, str], list[tuple[str, str]]],
    ) -> None:
        """Append a `## See Also` section of semantic neighbors to each entity's note.

        ``neighbors`` maps ``(entity_type, entity_name) -> [(target_type, target_name), ...]``.
        Each target is rendered as a ``[[wikilink]]``. A file that already
        contains a ``## See Also`` heading is left alone so this hook is safe
        to call repeatedly.
        """
        if not self._started:
            return
        for (entity_type, entity_name), targets in neighbors.items():
            if not targets:
                continue
            safe_name = _safe_filename(entity_name)
            target_path = self.root_dir / entity_type / f"{safe_name}.md"
            if not target_path.exists():
                continue
            try:
                content = await asyncio.to_thread(
                    target_path.read_text, encoding="utf-8"
                )
            except OSError:
                continue
            if "## See Also" in content:
                continue
            wikilinks = "\n".join(
                f"- [[{_safe_filename(name)}]]" for _, name in targets
            )
            see_also_block = f"\n\n## See Also\n\n{wikilinks}\n"
            new_content = content.rstrip() + see_also_block
            try:
                await asyncio.to_thread(
                    self._atomic_write, target_path, new_content
                )
            except OSError:
                self._stats["errors"] += 1
