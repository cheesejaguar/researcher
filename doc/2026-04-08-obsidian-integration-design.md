# Obsidian Integration — Design

**Status:** draft for review
**Date:** 2026-04-08
**Author:** Aaron (+ Claude)
**Follows:** Wave 1-E orchestrator wiring (`c6a7a8a`)

## Context

`researcher` produces typed entities with provenance-tracked facts during a run. Today those facts live in a `KnowledgeStore` (currently `StubKnowledgeStore`, Wave 1-B will ship the real DuckDB backend). Users who already run Obsidian for personal knowledge management want those entities to show up as browsable Markdown notes in their vault so they can use backlinks, the graph view, Dataview queries, tags, and all the other PKM affordances Obsidian provides.

This doc specifies an additive, optional Obsidian integration that materializes `FactClaim`s into Markdown files as they're produced during a run, without disturbing the primary `KnowledgeStore` path.

## Goals

1. Emit one Markdown file per entity into a user-configured Obsidian vault, live during the run.
2. Each file is valid Obsidian Markdown with YAML frontmatter, tags, and a body rendered for human reading.
3. Failures in the Obsidian writer MUST NOT disrupt the orchestrator's primary reduce path.
4. Zero impact when not configured — runs without `obsidian_vault` are byte-identical to today.

## Non-goals

- Bidirectional sync. The writer owns the managed files; user edits to those files are overwritten on the next run.
- Reading existing files on startup to merge with new facts. Each run's view is what the writer produces this run.
- Cross-run accumulation beyond "same file path." Accumulation lives in the typed store; the vault is a view.
- Obsidian plugin development. The integration is pure file writes — no plugin, no communication with a running Obsidian process.
- Reading or writing Obsidian's `.obsidian/` config directory.
- Prompt-injection defense. If a web-scraped snippet contains Obsidian-flavored syntax, it renders as Obsidian-flavored syntax. Documented.

## Decisions locked from brainstorming

| Dimension | Decision | Rationale |
|---|---|---|
| Role | Secondary write-through sink | Keeps typed store as source of truth; vault is derived |
| Update trigger | Per-event live updates on `on_fact`, debounced 100ms, hard-flushed at cycle end + run end | Live feel without thrashing the filesystem |
| Vault layout | Merge by entity type: `<vault>/researcher/<EntityType>/<SafeName>.md` | Accumulates across runs in the same canonical location |
| Editability | Writer owns the whole file | User explicitly accepted that hand edits are overwritten |
| Architecture | Explicit orchestrator hook (approach A) | Smallest change; no pub/sub infrastructure needed |
| Failure mode | Swallow-and-log inside the writer | Secondary sink must never crash the primary path |

## Architecture

A new `ObsidianWriter` class lives at `researcher/integrations/obsidian.py`. The orchestrator takes it as an optional constructor parameter and calls:

- `await writer.start()` once at the beginning of `run()` (alongside `self._writer.start()`).
- `await writer.on_fact(claim, run_id)` inside the `_spawn_agent` result loop, immediately after `self._writer.submit(claim)`.
- `await writer.flush()` at the end of each cycle, after `self._writer.quiesce()`.
- `await writer.stop()` in `_graceful_stop()`, before emitting `run_complete`.

When `obsidian_writer` is `None`, the orchestrator skips all four call sites. The integration is strictly additive — no change to `FactWriter`, `EventBus`, `KnowledgeStore`, or any other contract.

Inside the writer, a per-entity in-memory coalescing buffer accepts claims without touching disk. A background debounce task wakes every `flush_interval_s` (default 100ms) and writes any dirty entities to disk atomically via `tempfile` + `os.replace`. Explicit `flush()` calls run the same drain synchronously so cycle boundaries produce a consistent on-disk snapshot.

```
Orchestrator.run()
├── writer.start()                   # FactWriter
├── obsidian_writer.start()          # NEW
├── for cycle:
│     for agent_result in gather:
│         for claim in agent_result.claims:
│             writer.submit(claim)
│             obsidian_writer.on_fact(claim, run_id)   # NEW
│     writer.quiesce()
│     obsidian_writer.flush()        # NEW
└── _graceful_stop:
      writer.drain()
      obsidian_writer.stop()         # NEW
```

## Components

### New files

- `researcher/integrations/__init__.py` — empty package marker.
- `researcher/integrations/obsidian.py` — `ObsidianWriter`, `EntityState`, `FieldValue`, `ObsidianWriterError`, `ObsidianVaultNotFound`.
- `tests/unit/test_obsidian_writer.py` — unit tests (23 tests).
- `tests/unit/test_orchestrator_obsidian_hook.py` — orchestrator-integration tests (5 tests).
- `tests/integration/test_smoke_obsidian_wars.py` — offline end-to-end smoke (1 test).

### Touched files

- `researcher/orchestrator.py` — add optional `obsidian_writer` constructor parameter; call `start`/`on_fact`/`flush`/`stop` at the four documented sites.
- `researcher/spec.py` — add `obsidian_vault: Optional[str] = None` to `RunSpec`.
- `researcher/cli.py` — add `--obsidian-vault` option to the `run` subcommand; precedence: CLI flag > spec field > None.

### Class surface

```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from researcher.models import FactClaim, Provenance


class ObsidianWriterError(RuntimeError): ...
class ObsidianVaultNotFound(ObsidianWriterError): ...


@dataclass
class FieldValue:
    value: Any
    confidence: float
    provenances: list[Provenance] = field(default_factory=list)


@dataclass
class EntityState:
    entity_type: str
    entity_name: str
    fields: dict[str, FieldValue] = field(default_factory=dict)
    run_ids: set[str] = field(default_factory=set)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class ObsidianWriter:
    def __init__(
        self,
        vault_path: Path | str,
        subdir: str = "researcher",
        flush_interval_s: float = 0.1,
    ) -> None: ...

    async def start(self) -> None: ...
    async def on_fact(self, claim: FactClaim, run_id: str) -> None: ...
    async def flush(self) -> None: ...
    async def stop(self) -> None: ...

    @property
    def stats(self) -> dict[str, int]: ...
```

### Internal state

- `_pending: dict[tuple[str, str], EntityState]` — coalescing buffer keyed by `(entity_type, entity_name)`.
- `_dirty: set[tuple[str, str]]` — entities with pending on-disk writes.
- `_lock: asyncio.Lock` — serializes mutation of `_pending` and `_dirty`.
- `_debounce_task: asyncio.Task | None` — background coroutine started in `start()`, cancelled in `stop()`.
- `_started: bool` — guards double-start and use-after-stop.
- `_stopped: bool` — after `stop()`, further `on_fact` raises.
- `_stats: dict[str, int]` — counters: `writes`, `coalesced_claims`, `errors`, `sanitized_names`.

### Coalescing semantics

When `on_fact` receives a claim, it looks up or creates the `EntityState` for `(claim.entity_type, claim.entity_name)`. The claim's field is merged into `state.fields[claim.field]`:

- **New field:** create a `FieldValue` with the claim's value/confidence/provenance.
- **Same field, same value:** append the new provenance if `(url, span_id)` pair isn't already present. Keep the higher confidence.
- **Same field, different value:** if new confidence > existing confidence, replace value + confidence; always append the new provenance. Body renders every distinct value under that field in the Provenance section so the conflict is visible.

### Filename sanitization

`_safe_filename(name: str) -> str`:
1. `name = name.strip()`; if empty, return `entity_<sha1(original)[:8]>`.
2. Replace every character in `os.sep + "/\\:*?\"<>|"` with `_`.
3. Collapse consecutive whitespace to single `_`.
4. Strip leading/trailing dots (Obsidian hides dotfiles).
5. If the result is empty after sanitization, return `entity_<sha1(original)[:8]>`.
6. If length > 200, truncate to 192 + `_<sha1(original)[:7]>`.
7. Filename becomes `<safe>.md`.

### Markdown rendering

`_render_markdown(state: EntityState) -> str` produces the full file content:

```markdown
---
# This file is managed by researcher. Edits will be overwritten on the next run.
researcher_type: War
researcher_name: World War II
researcher_runs:
  - run-2026-04-08-wars-1
researcher_updated: 2026-04-08T14:23:00Z
tags:
  - researcher
  - researcher/War
start_year: 1939
end_year: 1945
belligerents:
  - Allies
  - Axis
---

# World War II

> Managed by `researcher`. Re-running the job will overwrite this file.

## Fields

| Field | Value | Confidence | Source |
|---|---|---|---|
| start_year | 1939 | 0.97 | [wiki](https://en.wikipedia.org/wiki/World_War_II) |
| end_year | 1945 | 0.97 | [wiki](https://en.wikipedia.org/wiki/World_War_II) |
| belligerents | Allies, Axis | 0.96 | [wiki](https://en.wikipedia.org/wiki/World_War_II) |

## Provenance

### start_year = 1939
> began in 1939

- Source: https://en.wikipedia.org/wiki/World_War_II
- Agent: `sub-a1b2c3d4` · Extractor: `claude_code/unknown` · Span: `cli_0`

### end_year = 1945
> lasted from 1939 to 1945

- Source: https://en.wikipedia.org/wiki/World_War_II
- Agent: `sub-a1b2c3d4` · Extractor: `claude_code/unknown` · Span: `cli_1`

## Seen in Runs

- `run-2026-04-08-wars-1`
```

**Frontmatter conventions:**
- Keys prefixed with `researcher_` are writer-managed metadata and chosen to avoid collision with entity schema field names.
- `tags` always includes a generic `researcher` tag and a type-scoped `researcher/<EntityType>` tag.
- Entity schema fields live at the top level of frontmatter so Dataview queries (`WHERE researcher_type = "War"`, `WHERE start_year > 1900`) work without prefixing.
- Lists serialize as YAML sequences. Strings, ints, floats, bools stay scalar. Unknown types fall back to `repr()` with a comment.
- `researcher_updated` uses ISO 8601 UTC with trailing `Z`.

**Body conventions:**
- Managed warning quote at the top — visible in Obsidian preview, not just when reading the raw file.
- Fields table is a quick-scan view. Full provenance lives in the section below.
- Provenance section is headed by `field = value` so Obsidian's outline view shows the grouping.
- "Seen in Runs" section is a bulleted list of run IDs that contributed facts to this entity across its lifetime (populated via `state.run_ids`). Since the writer doesn't read existing files, this list only reflects the current run — documented as a limitation.

## Data flow

1. `Orchestrator.run()` enters. `self._writer.start()`; if `self._obsidian_writer is not None`, `await self._obsidian_writer.start()`.
2. `start()` calls `vault_path.expanduser().resolve()`, checks the vault root exists (raise `ObsidianVaultNotFound` if not), creates `<vault>/researcher/` with `mkdir(parents=True, exist_ok=True)`, spawns the debounce task, sets `_started = True`.
3. For each cycle, agents return `AgentResult`s with `claims`. For each claim:
   - `await self._writer.submit(claim)` (primary path, unchanged).
   - If `self._obsidian_writer`, `await self._obsidian_writer.on_fact(claim, run_id=self._run_id)`.
4. `on_fact` acquires `_lock`, merges the claim into `_pending` per the coalescing rules, adds the key to `_dirty`, releases the lock. No disk I/O. Increments `_stats["coalesced_claims"]` if this claim merged into an existing entity-field.
5. Meanwhile, the debounce task runs `while not _stopped:`:
   - `await asyncio.sleep(flush_interval_s)`.
   - Acquire `_lock`, copy `_dirty` to a local list, clear `_dirty`, release lock.
   - For each dirty key, call `_flush_one(key)` which: re-acquires `_lock` to snapshot the entity state, renders Markdown, writes atomically via tempfile + `os.replace`, increments `_stats["writes"]`.
   - If the dirty set is empty, loop continues (cheap no-op).
6. At cycle end, the orchestrator calls `await self._obsidian_writer.flush()` which:
   - Acquires `_lock`, snapshots `_dirty`, clears it, releases.
   - For each dirty key, calls `_flush_one(key)` synchronously.
   - Returns when all have been written (or errored).
7. Run ends via `_graceful_stop()`, which calls `await self._obsidian_writer.stop()`:
   - Calls `flush()` to ensure all pending writes are on disk.
   - Sets `_stopped = True`.
   - Cancels `_debounce_task` if alive; awaits it.
   - Further `on_fact` calls raise `RuntimeError("ObsidianWriter is stopped")`.

### Atomic write

```python
async def _flush_one(self, key: tuple[str, str]) -> None:
    async with self._lock:
        state = self._pending.get(key)
        if state is None:
            return
        state_snapshot = _snapshot(state)  # deep-ish copy so rendering runs outside the lock
    content = _render_markdown(state_snapshot)
    target = self._path_for(state_snapshot)  # <vault>/researcher/<Type>/<safe>.md
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, target)
        self._stats["writes"] += 1
    except OSError as e:
        self._stats["errors"] += 1
        try:
            if "tmp_path" in locals() and tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        # Do not re-raise: secondary sink must not disrupt the primary path.
```

Rendering happens outside `_lock` so long runs don't block `on_fact` callers. The snapshot is shallow-deep enough to prevent races on mutable lists (provenance list, run_ids set).

## Error handling

| Failure | Detection | Response |
|---|---|---|
| Vault root doesn't exist | `start()` | Raise `ObsidianVaultNotFound` with the resolved path. Run aborts at start — the user typo'd the path, fail loud. |
| `<vault>/researcher/` can't be created | `start()` | Wrap `OSError` in `ObsidianWriterError`. Run aborts. |
| Invalid entity name | `_safe_filename` | Sanitize; increment `_stats["sanitized_names"]`; never raise. |
| Filesystem error during write | `_flush_one` | Catch `OSError`, increment `_stats["errors"]`, clean up tempfile, do NOT re-raise. Entity stays dirty and is retried on the next debounce tick. |
| `on_fact` before `start` | `on_fact` | Accept the claim into `_pending` (store-forward). First `flush` or `start` drains it. |
| `on_fact` after `stop` | `on_fact` | Raise `RuntimeError("ObsidianWriter is stopped")`. This is a misuse, not a filesystem problem — raising is appropriate. The orchestrator never does this (stop is in the finally block); only test misuse or future-caller bugs would hit it. |
| Debounce task exception | `add_done_callback` | Log via the standard Python logger; set `_debounce_task = None` so subsequent `flush` calls don't hang. |
| `stop()` called twice | `stop()` | Idempotent: first call stops, second is a no-op. |
| Orchestrator crash mid-cycle | `_graceful_stop` | The existing `finally` clause in `Orchestrator.run()` now also calls `await self._obsidian_writer.stop()`. Any exception from `stop` is swallowed and logged — we're already on the way down. |
| `CancelledError` during `_flush_one` | `_flush_one` | Re-raise. Cancellation is the one exception the writer does not swallow. |
| Concurrent writes to the same entity | `_lock` | Serialized by `_lock`. Rendering happens outside the lock but a snapshot guarantees consistency. |

**Secondary sink discipline** — the guiding invariant: **the primary orchestrator path must never be disrupted by an Obsidian failure.** Every public method catches `OSError` and logs via `_stats["errors"]`. Only `CancelledError` and explicit misuse (`on_fact` after `stop`) propagate.

## Testing

### Unit tests (`tests/unit/test_obsidian_writer.py`, 23 tests)

| # | Test | Asserts |
|---|---|---|
| 1 | `test_start_creates_vault_subdir` | `<vault>/researcher/` exists after start |
| 2 | `test_start_raises_on_missing_vault_root` | `ObsidianVaultNotFound` with bad path |
| 3 | `test_start_is_idempotent` | double start is a no-op |
| 4 | `test_on_fact_writes_entity_file_after_flush` | file exists at expected path |
| 5 | `test_on_fact_coalesces_multiple_fields_for_same_entity` | ONE file with all 4 fields |
| 6 | `test_on_fact_organizes_by_entity_type` | two subfolders for two types |
| 7 | `test_markdown_contains_frontmatter_and_provenance` | content structure |
| 8 | `test_frontmatter_tags_include_type_scoped_tag` | `researcher/War` tag present |
| 9 | `test_list_field_serializes_as_yaml_list` | YAML sequence, not string |
| 10 | `test_higher_confidence_wins_on_conflict` | value is higher-confidence one |
| 11 | `test_duplicate_provenance_deduped_by_url_and_span` | provenance list has one entry |
| 12 | `test_unsafe_entity_name_sanitized` | path traversal prevented |
| 13 | `test_empty_entity_name_falls_back_to_hash` | `entity_<hash>.md` |
| 14 | `test_long_entity_name_truncated` | ≤200 char filename |
| 15 | `test_debounce_batches_bursts` | one write for 10 rapid claims |
| 16 | `test_flush_drains_pending_synchronously` | file exists immediately after flush |
| 17 | `test_flush_is_idempotent_when_nothing_dirty` | second flush is no-op |
| 18 | `test_stop_calls_final_flush_and_cancels_task` | pending writes land, task done |
| 19 | `test_stop_is_idempotent` | double stop is a no-op |
| 20 | `test_on_fact_after_stop_raises` | `RuntimeError` |
| 21 | `test_write_failure_does_not_raise_from_on_fact` | `stats["errors"]` incremented, no exception |
| 22 | `test_atomic_write_cleans_up_tempfile_on_error` | no `.tmp` left behind |
| 23 | `test_rerun_overwrites_existing_file` | stale content replaced |

### Orchestrator integration (`tests/unit/test_orchestrator_obsidian_hook.py`, 5 tests)

| # | Test | Asserts |
|---|---|---|
| 1 | `test_orchestrator_forwards_claims_to_writer` | `on_fact` called for each claim |
| 2 | `test_orchestrator_flushes_at_cycle_end` | `flush` called once per cycle |
| 3 | `test_orchestrator_stops_writer_in_graceful_stop` | `stop` called once |
| 4 | `test_orchestrator_without_obsidian_writer_is_unchanged` | baseline still passes |
| 5 | `test_obsidian_write_failure_does_not_crash_run` | `on_fact` raising doesn't abort the run |

Uses a `MockObsidianWriter` spy in the same file.

### Offline integration smoke (`tests/integration/test_smoke_obsidian_wars.py`, 1 test)

Full `Orchestrator.run()` with a real `ObsidianWriter` pointed at `tmp_path`, `StubCliRunner` returning the wars fixture, two seed queries. Asserts:
- Exactly one `.md` file exists under `<tmp_path>/researcher/War/`
- YAML frontmatter contains `researcher_type: War`, `researcher_name: "World War II"`, `start_year: 1939`, `belligerents: [Allies, Axis]`
- Body contains the Provenance section with the Wikipedia URL
- `stats["writes"]` ≥ 1, `stats["errors"] == 0`

### Total

29 new tests (23 unit + 5 orchestrator-integration + 1 offline smoke), all deterministic.

## CLI ergonomics

```bash
# Via spec YAML (obsidian_vault: ~/Documents/ObsidianVault):
uv run researcher run specs/wars.yaml

# Via CLI flag (overrides spec):
uv run researcher run specs/wars.yaml --obsidian-vault ~/Documents/ObsidianVault

# Disable despite spec:
uv run researcher run specs/wars.yaml --obsidian-vault ""
```

Precedence: CLI flag > spec field > None.

Empty-string sentinel on the CLI disables the writer even if the spec declares one. Useful for debug runs.

Path resolution: `Path(value).expanduser().resolve()`. Validation happens in `ObsidianWriter.start()` not at config load time — the writer owns the "does this directory exist?" check and fails the run cleanly if it doesn't.

## Scope cuts

- **Reading existing files.** The writer never opens existing `.md` files. It can't recover a user's prior-run fields, custom sections, or hand edits. Documented as intentional.
- **Cross-run accumulation of facts in a single note.** The vault file reflects one run's view. True accumulation lives in the typed store (DuckDB, Wave 1-B) — a future feature could snapshot the store's accumulated view into the vault at run end.
- **Backlink generation between entities.** `[[Wikilink]]` auto-generation (e.g., linking "WWII" to the "Allies" entity note) is deferred. Obsidian can do fuzzy linking itself via the "unlinked mentions" feature.
- **Obsidian-specific features** — callouts, footnotes, embedded queries, Dataview inline fields. The rendered body uses plain Markdown + tables. Users who want these can post-process the vault.
- **Attachment handling.** If a source has an image, we don't download or embed it. Only text snippets land in the vault.
- **Obsidian plugin.** No JS/TS plugin, no communication with a running Obsidian instance. Pure file writes.

## Open risks

- **Large vaults:** Obsidian's index rebuild time grows with vault size. A research run producing 1000 entities is fine; 100k entities would stress Obsidian's indexer. Not a concern for v1 targets.
- **Filesystem watchers:** If Obsidian is running and watching the vault, our atomic tempfile writes will trigger its change-detection fast enough that mid-run preview might occasionally show half-written content. `os.replace` is atomic at the POSIX level, so the file is never empty, but Obsidian might re-read the file before we've written the next entity. Acceptable.
- **Case-insensitive filesystems:** On macOS (case-insensitive by default), `WWII.md` and `wwii.md` collide. Filename sanitization does not normalize case; collisions would overwrite each other. Documented; unlikely to hit in practice.
- **Prompt injection in snippets:** If a scraped page contains `[[evil]]` or `$SOMETHING` Obsidian syntax, it renders as-is. We do NOT escape Markdown metacharacters in snippets. Documented.
- **Rendering markdown within snippets:** Same category — snippets may contain raw HTML or Markdown that renders unexpectedly. Not a security issue, but aesthetically surprising.

## Success criteria

1. `uv run pytest -q` passes with 29 new tests green.
2. Running `uv run researcher run specs/wars.yaml --obsidian-vault /tmp/vault` (with `claude` on PATH) populates `/tmp/vault/researcher/War/` with `.md` files that open cleanly in Obsidian.
3. Each file has valid YAML frontmatter, a Fields table, a Provenance section, and a "Seen in Runs" section.
4. Re-running the same spec twice produces the same filenames (merged location), with the second run's content replacing the first.
5. Running without `--obsidian-vault` produces byte-identical behavior to pre-integration runs (baseline test suite still green).
6. Simulated write failure (read-only vault mid-run) does NOT crash the orchestrator; the run completes with `stats["errors"] > 0` logged.
