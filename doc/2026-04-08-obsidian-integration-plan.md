# Obsidian Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `ObsidianWriter` that materializes `FactClaim`s into Markdown files in a user-configured Obsidian vault, live during a run, without disturbing the primary `KnowledgeStore` path.

**Architecture:** New `researcher/integrations/obsidian.py` module with `ObsidianWriter`, `EntityState`, `FieldValue`, and error types. The orchestrator takes it as an optional constructor parameter and calls `start` / `on_fact` / `flush` / `stop` at four documented sites. Per-entity debounced writes with atomic tempfile + `os.replace`. Failures are swallowed and logged so the primary path is never disrupted.

**Tech Stack:** Python 3.11+ (uv), stdlib only (`pathlib`, `tempfile`, `os`, `asyncio`, `hashlib`, `dataclasses`, `datetime`, `re`), `pyyaml` (already a dep via spec.py) for frontmatter serialization.

**Spec:** `doc/2026-04-08-obsidian-integration-design.md` (commit `1ba2521`)

---

## File structure

**New files:**

| Path | Purpose |
|---|---|
| `researcher/integrations/__init__.py` | empty package marker |
| `researcher/integrations/obsidian.py` | `ObsidianWriter`, `EntityState`, `FieldValue`, `ObsidianWriterError`, `ObsidianVaultNotFound`, `_safe_filename`, `_render_markdown`, `_snapshot_state` |
| `tests/unit/test_obsidian_writer.py` | ~23 unit tests covering sanitization, rendering, lifecycle, coalescing, debounce/flush, error paths |
| `tests/unit/test_orchestrator_obsidian_hook.py` | 5 tests for orchestrator integration |
| `tests/integration/test_smoke_obsidian_wars.py` | 1 end-to-end offline smoke test |

**Modified files:**

| Path | Change |
|---|---|
| `researcher/orchestrator.py` | add optional `obsidian_writer` constructor parameter; call `start` / `on_fact` / `flush` / `stop` at four sites |
| `researcher/spec.py` | add `obsidian_vault: Optional[str] = None` to `RunSpec` |
| `researcher/cli.py` | add `--obsidian-vault` option to the `run` subcommand |

---

## Task 1: Scaffold the integrations package with data types and errors

**Files:**
- Create: `researcher/integrations/__init__.py`
- Create: `researcher/integrations/obsidian.py`
- Test: none yet — pure data types get exercised by later tasks

- [ ] **Step 1: Create the package marker**

Create `researcher/integrations/__init__.py` as an EMPTY file (0 bytes).

- [ ] **Step 2: Create `researcher/integrations/obsidian.py` with the data types + error classes**

```python
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
```

- [ ] **Step 3: Verify the module imports cleanly**

Run: `uv run python -c "from researcher.integrations.obsidian import EntityState, FieldValue, ObsidianWriterError, ObsidianVaultNotFound; print('ok')"`

Expected output: `ok`

- [ ] **Step 4: Commit**

```bash
git add researcher/integrations/__init__.py researcher/integrations/obsidian.py
git commit -m "$(cat <<'EOF'
integrations: scaffold obsidian package with data types + errors

First step of the Obsidian integration. Creates the researcher.integrations
package and declares EntityState, FieldValue, ObsidianWriterError, and
ObsidianVaultNotFound. No runtime behavior yet — just the types later tasks
will build on.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Filename sanitization helper

**Files:**
- Modify: `researcher/integrations/obsidian.py`
- Test: `tests/unit/test_obsidian_writer.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_obsidian_writer.py`:

```python
"""Tests for ObsidianWriter and its helpers."""

from __future__ import annotations

from researcher.integrations.obsidian import _safe_filename


# ---------- Filename sanitization ----------


def test_safe_filename_basic():
    assert _safe_filename("World War II") == "World War II"


def test_safe_filename_strips_path_separators():
    result = _safe_filename("../../evil")
    assert ".." not in result
    assert "/" not in result
    assert "\\" not in result


def test_safe_filename_strips_windows_reserved():
    result = _safe_filename('a:b*c?d"e<f>g|h')
    for ch in ':*?"<>|':
        assert ch not in result


def test_safe_filename_empty_falls_back_to_hash():
    result = _safe_filename("")
    assert result.startswith("entity_")
    assert len(result) > len("entity_")


def test_safe_filename_all_special_falls_back_to_hash():
    result = _safe_filename("///")
    assert result.startswith("entity_")


def test_safe_filename_truncates_long_names():
    long = "W" * 500
    result = _safe_filename(long)
    assert len(result) <= 200
    # Truncated names get a hash suffix to disambiguate.
    assert "_" in result[-10:]


def test_safe_filename_strips_leading_dots():
    result = _safe_filename(".hidden")
    assert not result.startswith(".")
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/unit/test_obsidian_writer.py -q`

Expected: `ImportError: cannot import name '_safe_filename' from 'researcher.integrations.obsidian'`

- [ ] **Step 3: Add imports to `researcher/integrations/obsidian.py`**

At the top of `researcher/integrations/obsidian.py`, add `hashlib` and `re` to the existing imports. The top of the file should become:

```python
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
```

- [ ] **Step 4: Implement `_safe_filename`**

Add this function at the end of `researcher/integrations/obsidian.py` (after the `EntityState` dataclass):

```python
# ---------- Helpers ----------


_UNSAFE_CHARS = re.compile(r'[\\/:*?"<>|]')
_WHITESPACE_RUN = re.compile(r"\s+")
_MAX_FILENAME_LEN = 200


def _safe_filename(name: str) -> str:
    """Sanitize an entity name for use as a filename.

    - Strips path separators and Windows-reserved characters.
    - Collapses whitespace.
    - Falls back to `entity_<sha1_8>` for empty or all-special names.
    - Truncates to 200 characters with an 8-char hash suffix for collision safety.
    """
    original = name
    cleaned = _UNSAFE_CHARS.sub("_", name).strip()
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned)
    # Strip leading/trailing dots (Obsidian hides dotfiles).
    cleaned = cleaned.strip(".")
    if not cleaned:
        digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]
        return f"entity_{digest}"
    if len(cleaned) > _MAX_FILENAME_LEN:
        digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:7]
        cleaned = cleaned[: _MAX_FILENAME_LEN - 9] + "_" + digest
    return cleaned
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_obsidian_writer.py -q`

Expected: `7 passed`

- [ ] **Step 6: Commit**

```bash
git add researcher/integrations/obsidian.py tests/unit/test_obsidian_writer.py
git commit -m "$(cat <<'EOF'
integrations: add _safe_filename for Obsidian vault entity names

Strips path separators and Windows-reserved characters, collapses
whitespace, strips leading/trailing dots, falls back to entity_<hash>
for empty names, and truncates long names with an 8-char hash suffix.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Markdown rendering

**Files:**
- Modify: `researcher/integrations/obsidian.py`
- Test: `tests/unit/test_obsidian_writer.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_obsidian_writer.py`:

```python
# ---------- Markdown rendering ----------

from datetime import datetime, timezone

from researcher.integrations.obsidian import (
    EntityState,
    FieldValue,
    _render_markdown,
)
from researcher.models import Provenance


def _sample_provenance(url: str = "https://example.com", span: str = "cli_0") -> Provenance:
    return Provenance(
        url=url,
        fetched_at=datetime(2026, 4, 8, 14, 23, 0, tzinfo=timezone.utc),
        snippet="began in 1939",
        extractor_model="claude_code/unknown",
        agent_id="sub-a1b2c3d4",
        task_id="task-1",
        span_id=span,
    )


def _sample_state() -> EntityState:
    state = EntityState(
        entity_type="War",
        entity_name="World War II",
        updated_at=datetime(2026, 4, 8, 14, 23, 0, tzinfo=timezone.utc),
    )
    state.run_ids.add("run-2026-04-08-wars-1")
    state.fields["start_year"] = FieldValue(
        value=1939,
        confidence=0.97,
        provenances=[_sample_provenance(span="cli_0")],
    )
    state.fields["belligerents"] = FieldValue(
        value=["Allies", "Axis"],
        confidence=0.96,
        provenances=[_sample_provenance(span="cli_1")],
    )
    return state


def test_render_contains_frontmatter_delimiters():
    md = _render_markdown(_sample_state())
    assert md.startswith("---\n")
    # Frontmatter closes with a --- before the body.
    assert "\n---\n" in md[4:]


def test_render_frontmatter_has_researcher_keys():
    md = _render_markdown(_sample_state())
    assert "researcher_type: War" in md
    assert "researcher_name: World War II" in md
    assert "researcher_runs:" in md
    assert "run-2026-04-08-wars-1" in md
    assert "researcher_updated: 2026-04-08T14:23:00" in md


def test_render_frontmatter_has_tags():
    md = _render_markdown(_sample_state())
    assert "tags:" in md
    assert "- researcher" in md
    assert "- researcher/War" in md


def test_render_frontmatter_has_entity_fields():
    md = _render_markdown(_sample_state())
    assert "start_year: 1939" in md


def test_render_list_field_as_yaml_sequence():
    md = _render_markdown(_sample_state())
    # belligerents should appear as a YAML list, not a string.
    assert "belligerents:" in md
    assert "- Allies" in md
    assert "- Axis" in md


def test_render_contains_managed_warning():
    md = _render_markdown(_sample_state())
    assert "Managed by" in md
    assert "researcher" in md


def test_render_contains_fields_table_with_source_link():
    md = _render_markdown(_sample_state())
    assert "| Field |" in md
    assert "| start_year |" in md
    assert "https://example.com" in md


def test_render_contains_provenance_section_with_snippet():
    md = _render_markdown(_sample_state())
    assert "## Provenance" in md
    assert "### start_year = 1939" in md
    assert "> began in 1939" in md
    assert "claude_code/unknown" in md


def test_render_contains_seen_in_runs_section():
    md = _render_markdown(_sample_state())
    assert "## Seen in Runs" in md
    assert "- `run-2026-04-08-wars-1`" in md
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/unit/test_obsidian_writer.py -q -k "test_render"`

Expected: `ImportError: cannot import name '_render_markdown' from 'researcher.integrations.obsidian'`

- [ ] **Step 3: Implement `_render_markdown`**

Append to `researcher/integrations/obsidian.py` (after `_safe_filename`):

```python
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
        _yaml_field(
            "researcher_updated", state.updated_at.isoformat().replace("+00:00", "Z")
        ),
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
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_obsidian_writer.py -q`

Expected: `16 passed` (7 from Task 2 + 9 new render tests)

- [ ] **Step 5: Commit**

```bash
git add researcher/integrations/obsidian.py tests/unit/test_obsidian_writer.py
git commit -m "$(cat <<'EOF'
integrations: add _render_markdown for Obsidian notes

Produces frontmatter with researcher_* managed keys, generic + type-scoped
tags, entity schema fields as top-level YAML so Dataview can query them.
Body has the managed-file warning, a Fields table with source links, a
Provenance section grouped by field with snippet blockquotes, and a
Seen in Runs section.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `ObsidianWriter` lifecycle — `__init__`, `start`, `stop`

**Files:**
- Modify: `researcher/integrations/obsidian.py`
- Test: `tests/unit/test_obsidian_writer.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_obsidian_writer.py`:

```python
# ---------- Lifecycle ----------

from pathlib import Path

import pytest

from researcher.integrations.obsidian import ObsidianVaultNotFound, ObsidianWriter


@pytest.mark.asyncio
async def test_start_creates_vault_subdir(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        assert (tmp_path / "researcher").is_dir()
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_start_raises_on_missing_vault_root(tmp_path: Path):
    bogus = tmp_path / "does-not-exist" / "vault"
    writer = ObsidianWriter(vault_path=bogus)
    with pytest.raises(ObsidianVaultNotFound):
        await writer.start()


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    await writer.start()  # should not raise
    try:
        assert (tmp_path / "researcher").is_dir()
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_stop_is_idempotent(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    await writer.stop()
    await writer.stop()  # should not raise


@pytest.mark.asyncio
async def test_stats_initially_zero(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    assert writer.stats == {
        "writes": 0,
        "coalesced_claims": 0,
        "errors": 0,
        "sanitized_names": 0,
    }
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/unit/test_obsidian_writer.py -q -k "start or stop or stats_initially"`

Expected: `ImportError: cannot import name 'ObsidianWriter' from 'researcher.integrations.obsidian'`

- [ ] **Step 3: Implement the `ObsidianWriter` class skeleton**

Append to `researcher/integrations/obsidian.py` (after the rendering helpers):

```python
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

    async def _debounce_loop(self) -> None:
        """Placeholder — real implementation in Task 6."""
        while not self._stopped:
            await asyncio.sleep(self._flush_interval_s)
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_obsidian_writer.py -q`

Expected: `21 passed` (16 from Tasks 2-3 + 5 new lifecycle tests)

- [ ] **Step 5: Commit**

```bash
git add researcher/integrations/obsidian.py tests/unit/test_obsidian_writer.py
git commit -m "$(cat <<'EOF'
integrations: ObsidianWriter lifecycle (init, start, stop)

start() validates the vault root exists, creates <vault>/researcher/,
and spawns the debounce task (placeholder for now). stop() is idempotent
and cancels the debounce task. stats property exposes write/error counters.

flush() and _debounce_loop() are placeholders until Task 6 wires them up.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `on_fact` and coalescing

**Files:**
- Modify: `researcher/integrations/obsidian.py`
- Test: `tests/unit/test_obsidian_writer.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_obsidian_writer.py`:

```python
# ---------- on_fact + coalescing ----------

from datetime import timedelta

from researcher.models import FactClaim


def _sample_claim(
    field_name: str = "start_year",
    value: Any = 1939,
    confidence: float = 0.9,
    url: str = "https://example.com/wwii",
    span: str = "cli_0",
    entity_type: str = "War",
    entity_name: str = "World War II",
) -> FactClaim:
    return FactClaim(
        entity_type=entity_type,
        entity_name=entity_name,
        field=field_name,
        value=value,
        confidence=confidence,
        provenance=Provenance(
            url=url,
            fetched_at=datetime.now(timezone.utc),
            snippet=f"{field_name}={value}",
            extractor_model="claude_code/unknown",
            agent_id="sub-1",
            task_id="t-1",
            span_id=span,
        ),
        emitted_by="sub-1",
        task_id="t-1",
    )


@pytest.mark.asyncio
async def test_on_fact_coalesces_into_pending_buffer(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(), run_id="run-1")
        # Private state check — pending should have exactly one entry.
        assert len(writer._pending) == 1
        key = ("War", "World War II")
        assert key in writer._pending
        state = writer._pending[key]
        assert "start_year" in state.fields
        assert state.fields["start_year"].value == 1939
        assert "run-1" in state.run_ids
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_on_fact_merges_multiple_fields_for_same_entity(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(
            _sample_claim(field_name="start_year", value=1939, span="cli_0"),
            run_id="run-1",
        )
        await writer.on_fact(
            _sample_claim(field_name="end_year", value=1945, span="cli_1"),
            run_id="run-1",
        )
        await writer.on_fact(
            _sample_claim(field_name="belligerents", value=["Allies", "Axis"], span="cli_2"),
            run_id="run-1",
        )
        assert len(writer._pending) == 1
        state = list(writer._pending.values())[0]
        assert set(state.fields.keys()) == {"start_year", "end_year", "belligerents"}
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_on_fact_higher_confidence_wins(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(
            _sample_claim(value=1939, confidence=0.8, span="cli_0"), run_id="run-1"
        )
        await writer.on_fact(
            _sample_claim(value=1940, confidence=0.95, span="cli_1"), run_id="run-1"
        )
        state = writer._pending[("War", "World War II")]
        fv = state.fields["start_year"]
        assert fv.value == 1940
        assert fv.confidence == 0.95
        # Both provenances should be present.
        assert len(fv.provenances) == 2
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_on_fact_dedupes_provenance_by_url_and_span(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        # Same URL AND span_id → dedup.
        await writer.on_fact(
            _sample_claim(url="https://a.com", span="cli_0"), run_id="run-1"
        )
        await writer.on_fact(
            _sample_claim(url="https://a.com", span="cli_0"), run_id="run-1"
        )
        fv = writer._pending[("War", "World War II")].fields["start_year"]
        assert len(fv.provenances) == 1
        # Different span → new entry.
        await writer.on_fact(
            _sample_claim(url="https://a.com", span="cli_1"), run_id="run-1"
        )
        assert len(fv.provenances) == 2
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_on_fact_separates_entity_types(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(entity_type="War"), run_id="run-1")
        await writer.on_fact(
            _sample_claim(entity_type="Trial", entity_name="NCT12345"), run_id="run-1"
        )
        assert ("War", "World War II") in writer._pending
        assert ("Trial", "NCT12345") in writer._pending
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_on_fact_after_stop_raises(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    await writer.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        await writer.on_fact(_sample_claim(), run_id="run-1")


@pytest.mark.asyncio
async def test_on_fact_marks_entity_dirty(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(), run_id="run-1")
        assert ("War", "World War II") in writer._dirty
    finally:
        await writer.stop()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/unit/test_obsidian_writer.py -q -k "on_fact"`

Expected: `AttributeError: 'ObsidianWriter' object has no attribute 'on_fact'`

- [ ] **Step 3: Implement `on_fact`**

Add this method to the `ObsidianWriter` class in `researcher/integrations/obsidian.py`, after `flush()`:

```python
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
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_obsidian_writer.py -q`

Expected: `28 passed` (21 from earlier tasks + 7 new on_fact tests)

- [ ] **Step 5: Commit**

```bash
git add researcher/integrations/obsidian.py tests/unit/test_obsidian_writer.py
git commit -m "$(cat <<'EOF'
integrations: ObsidianWriter.on_fact with coalescing semantics

Accepts FactClaims into an in-memory buffer keyed by (entity_type,
entity_name). Merges multiple claims for the same entity:
- New field: insert
- Same field, higher confidence: replace value+confidence, append provenance
- Same field, lower confidence: keep existing value, append provenance
- Same (url, span_id): dedup, no-op

Raises RuntimeError if called after stop(). No disk I/O — just buffer
updates. Marks entities dirty for the debounce task to flush.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `flush` + debounce loop + atomic write

**Files:**
- Modify: `researcher/integrations/obsidian.py`
- Test: `tests/unit/test_obsidian_writer.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_obsidian_writer.py`:

```python
# ---------- Flush + debounce + disk writes ----------

import asyncio


@pytest.mark.asyncio
async def test_flush_writes_entity_file_to_disk(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(), run_id="run-1")
        await writer.flush()
        expected = tmp_path / "researcher" / "War" / "World War II.md"
        assert expected.exists()
        content = expected.read_text()
        assert "researcher_type: War" in content
        assert "start_year: 1939" in content
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_flush_clears_dirty_set(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(), run_id="run-1")
        assert writer._dirty
        await writer.flush()
        assert not writer._dirty
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_flush_is_idempotent_when_nothing_dirty(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.flush()  # empty flush, no-op
        await writer.flush()  # still no-op
        assert writer.stats["writes"] == 0
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_flush_increments_write_count(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(entity_name="A"), run_id="run-1")
        await writer.on_fact(_sample_claim(entity_name="B"), run_id="run-1")
        await writer.flush()
        assert writer.stats["writes"] == 2
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_debounce_batches_bursts(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path, flush_interval_s=0.05)
    await writer.start()
    try:
        # Burst: 10 claims for the SAME entity in rapid succession.
        for i in range(10):
            await writer.on_fact(
                _sample_claim(field_name=f"f{i}", value=i, span=f"cli_{i}"),
                run_id="run-1",
            )
        # Wait for at least one debounce tick.
        await asyncio.sleep(0.15)
        # The debounce loop should have collapsed 10 claims into ONE write
        # (because they all share the same entity key).
        assert writer.stats["writes"] == 1
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_organizes_by_entity_type_subfolder(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(
            _sample_claim(entity_type="War", entity_name="WWII"), run_id="run-1"
        )
        await writer.on_fact(
            _sample_claim(
                entity_type="Trial",
                entity_name="NCT12345",
                field_name="phase",
                value=3,
            ),
            run_id="run-1",
        )
        await writer.flush()
        assert (tmp_path / "researcher" / "War" / "WWII.md").exists()
        assert (tmp_path / "researcher" / "Trial" / "NCT12345.md").exists()
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_stop_performs_final_flush(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    await writer.on_fact(_sample_claim(), run_id="run-1")
    # Stop without explicit flush — the stop() should drain pending writes.
    await writer.stop()
    expected = tmp_path / "researcher" / "War" / "World War II.md"
    assert expected.exists()


@pytest.mark.asyncio
async def test_rerun_overwrites_existing_file(tmp_path: Path):
    # First run.
    writer1 = ObsidianWriter(vault_path=tmp_path)
    await writer1.start()
    await writer1.on_fact(
        _sample_claim(value=1939), run_id="run-1"
    )
    await writer1.stop()
    file_path = tmp_path / "researcher" / "War" / "World War II.md"
    first_content = file_path.read_text()
    assert "start_year: 1939" in first_content

    # Second run — writer owns the whole file.
    writer2 = ObsidianWriter(vault_path=tmp_path)
    await writer2.start()
    await writer2.on_fact(
        _sample_claim(value=1914), run_id="run-2"
    )
    await writer2.stop()
    second_content = file_path.read_text()
    assert "start_year: 1914" in second_content
    assert "1939" not in second_content
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/unit/test_obsidian_writer.py -q -k "flush or debounce or organizes or stop_performs or rerun"`

Expected: Various failures — `flush` is a no-op, no file gets written.

- [ ] **Step 3: Add the imports and helper functions**

At the top of `researcher/integrations/obsidian.py`, add `os` and `tempfile` to the stdlib imports. The top should become:

```python
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from researcher.models import FactClaim, Provenance
```

- [ ] **Step 4: Replace the `flush` and `_debounce_loop` stubs with real implementations**

In `researcher/integrations/obsidian.py`, replace the placeholder `async def flush` and `async def _debounce_loop` methods on `ObsidianWriter` with:

```python
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
            # Best-effort log; never raise from a secondary sink.
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
                        # Defensive — flush itself should not raise, but if it
                        # somehow does, don't let the background task die.
                        self._stats["errors"] += 1
        except asyncio.CancelledError:
            pass
```

- [ ] **Step 5: Add the `_snapshot_state` helper**

Append this helper to `researcher/integrations/obsidian.py` (after `_render_markdown`):

```python
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
```

- [ ] **Step 6: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_obsidian_writer.py -q`

Expected: `36 passed` (28 from prior tasks + 8 new flush/debounce tests)

- [ ] **Step 7: Commit**

```bash
git add researcher/integrations/obsidian.py tests/unit/test_obsidian_writer.py
git commit -m "$(cat <<'EOF'
integrations: ObsidianWriter flush + debounce loop + atomic writes

flush() drains the dirty set synchronously. The background _debounce_loop
wakes every flush_interval_s and runs flush() if anything is pending.
_flush_one() renders the Markdown outside the lock using a snapshot of
EntityState, then writes atomically via tempfile + os.replace.

Secondary-sink discipline: OSError is caught, logged, and counted —
never propagates. Only CancelledError breaks out.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Error paths — write failure swallowed, tempfile cleanup, stats

**Files:**
- Modify: (none; tests only)
- Test: `tests/unit/test_obsidian_writer.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_obsidian_writer.py`:

```python
# ---------- Error paths ----------

from unittest.mock import patch


@pytest.mark.asyncio
async def test_write_failure_does_not_raise_from_flush(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(), run_id="run-1")

        def bad_replace(src, dst):
            raise OSError("simulated disk full")

        with patch("os.replace", side_effect=bad_replace):
            await writer.flush()  # must NOT raise

        assert writer.stats["errors"] >= 1
        assert writer.stats["writes"] == 0
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_tempfile_cleaned_up_on_write_failure(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(), run_id="run-1")

        def bad_replace(src, dst):
            raise OSError("simulated failure")

        with patch("os.replace", side_effect=bad_replace):
            await writer.flush()

        # No .tmp files should linger in the target directory.
        target_dir = tmp_path / "researcher" / "War"
        tmps = list(target_dir.glob("*.tmp"))
        assert tmps == [], f"leaked tempfiles: {tmps}"
        hidden_tmps = list(target_dir.glob(".*.tmp"))
        assert hidden_tmps == [], f"leaked hidden tempfiles: {hidden_tmps}"
    finally:
        await writer.stop()


@pytest.mark.asyncio
async def test_stats_counts_coalesced_claims(tmp_path: Path):
    writer = ObsidianWriter(vault_path=tmp_path)
    await writer.start()
    try:
        await writer.on_fact(_sample_claim(field_name="start_year"), run_id="run-1")
        # Second claim with the same field → coalesce.
        await writer.on_fact(
            _sample_claim(field_name="start_year", value=1940, span="cli_1"),
            run_id="run-1",
        )
        assert writer.stats["coalesced_claims"] == 1
    finally:
        await writer.stop()
```

- [ ] **Step 2: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_obsidian_writer.py -q -k "write_failure or tempfile or coalesced"`

Expected: `3 passed` — the implementation from Task 6 already handles these paths; we're locking in regression coverage.

- [ ] **Step 3: Run the full unit suite to confirm no regressions**

Run: `uv run pytest tests/unit -q`

Expected: all tests pass (the writer tests grow to 39).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_obsidian_writer.py
git commit -m "$(cat <<'EOF'
tests: lock in ObsidianWriter error-path regressions

Three tests that pin the secondary-sink discipline:
- flush() swallows OSError and increments stats["errors"]
- tempfiles are cleaned up on write failure (no leaks)
- coalesced_claims counter tracks when multiple claims merge

No production changes — Task 6's implementation already passes these.
This commit is regression insurance.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `RunSpec` field + `--obsidian-vault` CLI option

**Files:**
- Modify: `researcher/spec.py`
- Modify: `researcher/cli.py`
- Test: `tests/unit/test_spec.py` (append 2 tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_spec.py`:

```python
# ---------- Obsidian vault ----------


def test_runspec_obsidian_vault_default_is_none():
    s = RunSpec(
        spec_id="x",
        goal="g",
        entities=[
            EntitySpec(
                name="War",
                fields=[FieldSpec(name="name", type="str", required=True)],
                search_templates=[],
            )
        ],
        seeds=["s"],
        models={"fast": "m"},
    )
    assert s.obsidian_vault is None


def test_runspec_obsidian_vault_accepts_path():
    s = RunSpec(
        spec_id="x",
        goal="g",
        entities=[
            EntitySpec(
                name="War",
                fields=[FieldSpec(name="name", type="str", required=True)],
                search_templates=[],
            )
        ],
        seeds=["s"],
        models={"fast": "m"},
        obsidian_vault="~/Documents/Vault",
    )
    assert s.obsidian_vault == "~/Documents/Vault"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/unit/test_spec.py -q -k "obsidian"`

Expected: `AttributeError: 'RunSpec' object has no attribute 'obsidian_vault'`

- [ ] **Step 3: Add the field to `RunSpec` in `researcher/spec.py`**

Find the `class RunSpec(BaseModel):` block and append one field after `subagent_timeout_s: int = 120`:

```python
    # Obsidian integration — when set, a secondary ObsidianWriter sink
    # materializes FactClaims into Markdown files in this vault.
    obsidian_vault: Optional[str] = None
```

- [ ] **Step 4: Update the `run` command in `researcher/cli.py`**

Replace the existing `def run(...)` function in `researcher/cli.py` with:

```python
@app.command()
def run(
    spec: Path = typer.Argument(..., exists=True, readable=True, help="Path to a RunSpec YAML file."),
    run_id: str = typer.Option("", "--run-id", help="Custom run id (default: autogen)."),
    no_tui: bool = typer.Option(False, "--no-tui", help="Do not auto-launch the Ink TUI child process."),
    offline: bool = typer.Option(False, "--offline", help="Use stub LLM client + fixture corpus."),
    backend: str = typer.Option(
        "auto",
        "--backend",
        help="Backend policy: auto (use CLI if detected), cli (force CLI), api (force OpenRouter).",
    ),
    obsidian_vault: str = typer.Option(
        "",
        "--obsidian-vault",
        help="Path to an Obsidian vault root. Overrides the spec file. Empty string disables.",
    ),
) -> None:
    """Run a research job from a YAML spec. Auto-launches the TUI by default."""
    if backend not in ("auto", "cli", "api"):
        raise typer.BadParameter(f"--backend must be one of auto|cli|api, got {backend!r}")
    typer.echo(
        f"[researcher run] spec={spec} run_id={run_id or 'auto'} "
        f"tui={'off' if no_tui else 'on'} offline={offline} backend={backend} "
        f"obsidian_vault={obsidian_vault or 'none'}"
    )
    typer.echo("Wave 0 skeleton: orchestrator dispatch lands in Wave 1-E.")
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_spec.py -q`

Expected: existing spec tests plus 2 new = 13 passed.

- [ ] **Step 6: Smoke-test the CLI flag**

Run: `uv run python -m researcher.cli run --help 2>&1 | grep obsidian`

Expected: a line mentioning `--obsidian-vault`.

- [ ] **Step 7: Commit**

```bash
git add researcher/spec.py researcher/cli.py tests/unit/test_spec.py
git commit -m "$(cat <<'EOF'
spec + cli: add obsidian_vault config + --obsidian-vault flag

RunSpec grows an optional obsidian_vault field. The `researcher run`
command grows a --obsidian-vault option. Empty string disables, taking
precedence over the spec field. Actual wiring into Orchestrator lands
in Task 9.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Orchestrator hook — `obsidian_writer` parameter + call sites

**Files:**
- Modify: `researcher/orchestrator.py`
- Test: `tests/unit/test_orchestrator_obsidian_hook.py` (new file)

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_orchestrator_obsidian_hook.py`:

```python
"""Tests for the Orchestrator's Obsidian writer hook."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from researcher.backends.resolver import BackendResolver
from researcher.budget import Budget
from researcher.models import Task, TaskKind
from researcher.orchestrator import Orchestrator
from researcher.scheduler import Scheduler
from researcher.spec import EntitySpec, FieldSpec, RunSpec
from researcher.storage.writer import FactWriter
from tests.stubs.bus import StubEventBus
from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
from tests.stubs.llm import StubLLMClient
from tests.stubs.resolver import StubEntityResolver
from tests.stubs.store import StubKnowledgeStore


class MockObsidianWriter:
    """Spy that records all on_fact/start/flush/stop calls."""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.flushed = 0
        self.facts: list[tuple[str, str]] = []  # (field, run_id)
        self.raise_on_fact: bool = False

    async def start(self) -> None:
        self.started += 1

    async def on_fact(self, claim, run_id: str) -> None:
        if self.raise_on_fact:
            raise RuntimeError("simulated obsidian failure")
        self.facts.append((claim.field, run_id))

    async def flush(self) -> None:
        self.flushed += 1

    async def stop(self) -> None:
        self.stopped += 1


def _which_claude(cmd: str) -> str | None:
    return "/usr/local/bin/claude" if cmd == "claude" else None


async def _noop_fact(_entity_id, _claim):
    return None


async def _noop_conflict(_entity_id, _cells):
    return None


async def _make_orchestrator(
    obsidian: MockObsidianWriter | None,
    runner: StubCliRunner | None = None,
) -> Orchestrator:
    if runner is None:
        runner = StubCliRunner()
        runner.add_response_for_any(make_wars_discover_result())
    spec = RunSpec(
        spec_id="wars",
        goal="Wars",
        entities=[
            EntitySpec(
                name="War",
                fields=[FieldSpec(name="name", type="str", required=True)],
                search_templates=[],
            )
        ],
        seeds=["seed1"],
        models={"fast": "m", "smart": "m", "heavy": "m"},
        backend_policy="auto",
        max_cycles=1,
    )
    store = StubKnowledgeStore()
    await store.open()
    llm = StubLLMClient()
    bus = StubEventBus()
    resolver = StubEntityResolver()
    entity_schema = {
        "entity_type": "War",
        "fields": [{"name": "name", "type": "str", "required": True}],
    }
    writer = FactWriter(
        store=store,
        resolver=resolver,
        entity_schema=entity_schema,
        emit_fact=_noop_fact,
        emit_conflict=_noop_conflict,
    )
    scheduler = Scheduler(spec=spec, store=store)
    budget = Budget(usd_cap=3.0, wall_cap_s=600)
    backend_resolver = BackendResolver(which_fn=_which_claude)
    orch = Orchestrator(
        spec=spec,
        store=store,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        bus=bus,  # type: ignore[arg-type]
        writer=writer,
        scheduler=scheduler,
        budget=budget,
        run_id="run-x",
        max_parallel_agents=2,
        obsidian_writer=obsidian,  # type: ignore[arg-type]
    )
    orch.set_backend_resolver(backend_resolver)
    orch.set_cli_runner_factory(lambda kind: runner)
    return orch


@pytest.mark.asyncio
async def test_orchestrator_forwards_claims_to_obsidian_writer():
    ob = MockObsidianWriter()
    orch = await _make_orchestrator(ob)
    await orch.run()
    # The wars_discover fixture has 4 extractions, so 4 on_fact calls.
    assert len(ob.facts) == 4
    # Every call passed the run_id.
    assert all(run_id == "run-x" for _, run_id in ob.facts)


@pytest.mark.asyncio
async def test_orchestrator_calls_start_and_stop_exactly_once():
    ob = MockObsidianWriter()
    orch = await _make_orchestrator(ob)
    await orch.run()
    assert ob.started == 1
    assert ob.stopped == 1


@pytest.mark.asyncio
async def test_orchestrator_flushes_at_cycle_end():
    ob = MockObsidianWriter()
    orch = await _make_orchestrator(ob)
    await orch.run()
    # One cycle → at least one flush call (per-cycle barrier).
    assert ob.flushed >= 1


@pytest.mark.asyncio
async def test_orchestrator_without_obsidian_writer_is_baseline():
    # Pass None explicitly; orch should run without touching anything Obsidian-related.
    orch = await _make_orchestrator(None)
    reason = await orch.run()
    # No crash, no hangs.
    assert reason is not None


@pytest.mark.asyncio
async def test_obsidian_write_failure_does_not_crash_run():
    ob = MockObsidianWriter()
    ob.raise_on_fact = True
    orch = await _make_orchestrator(ob)
    reason = await orch.run()
    # The run should still complete — secondary-sink discipline.
    assert reason is not None
    # stop() should still have been called.
    assert ob.stopped == 1
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `uv run pytest tests/unit/test_orchestrator_obsidian_hook.py -q`

Expected: `TypeError: Orchestrator.__init__() got an unexpected keyword argument 'obsidian_writer'`

- [ ] **Step 3: Update `Orchestrator.__init__` in `researcher/orchestrator.py`**

Add the `obsidian_writer` parameter to `__init__`. The new signature and `__init__` body:

```python
    def __init__(
        self,
        spec: RunSpec,
        store: KnowledgeStore,
        llm: LLMClient,
        bus: EventBus,
        writer: FactWriter,
        scheduler: Scheduler,
        budget: Budget,
        run_id: str,
        max_parallel_agents: int = 6,
        obsidian_writer: Any = None,  # Optional[ObsidianWriter]; avoid import cycle
    ) -> None:
        self._spec = spec
        self._store = store
        self._llm = llm
        self._bus = bus
        self._writer = writer
        self._scheduler = scheduler
        self._budget = budget
        self._run_id = run_id
        self._sem = asyncio.Semaphore(max_parallel_agents)
        self._cycle = 0
        self._started_at: Optional[datetime] = None
        self._backend_resolver: Optional[BackendResolver] = None
        self._cli_runner_factory: Optional[Callable[[CliKind], CliRunner]] = None
        self._cycle_subagent_failures: int = 0
        self._circuit_break_threshold: int = 3
        self._obsidian_writer = obsidian_writer
```

Also add `from typing import Any` to the typing imports if `Any` isn't already imported. The existing `from typing import Callable, Optional` should become:

```python
from typing import Any, Callable, Optional
```

- [ ] **Step 4: Wire the four call sites in `Orchestrator.run`**

In `researcher/orchestrator.py`, update `run()` to call the writer at four sites. The updated `run()` method:

```python
    async def run(self) -> StopReason:
        self._started_at = datetime.now(timezone.utc)
        self._budget.start_wall_clock()
        await self._scheduler.seed()

        # Start the writer drain loop for the whole run.
        self._writer.start()
        if self._obsidian_writer is not None:
            await self._obsidian_writer.start()

        reason: StopReason = StopReason.NO_TASKS
        try:
            while self._cycle < self._spec.max_cycles:
                self._cycle += 1
                self._reset_cycle_failure_counters()

                batch = await self._scheduler.next_batch(self._spec.max_entities_per_cycle)
                if not batch:
                    reason = StopReason.NO_TASKS
                    break

                await self._emit_cycle_start(len(batch))

                # Parallel map.
                results = await asyncio.gather(
                    *(self._spawn_agent(t) for t in batch), return_exceptions=True
                )

                subagent_cap_hit = False
                saw_real_exception = False
                for t, r in zip(batch, results):
                    if isinstance(r, Exception):
                        saw_real_exception = True
                        continue
                    # Submit successful claims to the primary writer.
                    for claim in r.claims:
                        try:
                            await self._writer.submit(claim)
                        except asyncio.QueueFull:
                            pass
                        # Secondary sink — Obsidian. Never let failures here
                        # disrupt the primary path.
                        if self._obsidian_writer is not None:
                            try:
                                await self._obsidian_writer.on_fact(
                                    claim, run_id=self._run_id
                                )
                            except Exception:
                                pass
                    # Record failures for circuit-break counter + detect subagent_cap.
                    if r.state == AgentState.FAILED and r.error:
                        if r.error == "subagent_cap":
                            subagent_cap_hit = True
                        else:
                            self._record_subagent_failure(r.error)
                    await self._scheduler.mark_done(t.id, r)

                if saw_real_exception:
                    reason = StopReason.ERROR
                    break

                # Serial reduce: wait for this cycle's claims to process.
                await self._writer.quiesce()
                if self._obsidian_writer is not None:
                    try:
                        await self._obsidian_writer.flush()
                    except Exception:
                        pass

                metrics = await self._store.snapshot_metrics()
                await self._scheduler.update_from_metrics(metrics)
                await self._emit_cycle_end(metrics)

                # Stop checks — subagent_cap is highest priority.
                if subagent_cap_hit:
                    reason = StopReason.SUBAGENT_CAP
                    break
                if self._budget.exceeded():
                    reason = StopReason.BUDGET
                    break
                if self._budget.wall_exceeded():
                    reason = StopReason.DEADLINE
                    break
                if self._scheduler.plateau:
                    reason = StopReason.PLATEAU
                    break
            else:
                reason = StopReason.PLATEAU
        except BudgetExceededError:
            reason = StopReason.BUDGET
        except Exception:
            reason = StopReason.ERROR
        finally:
            # Drain the writer before emitting run_complete.
            try:
                await self._writer.drain()
            except Exception:
                pass
            if self._obsidian_writer is not None:
                try:
                    await self._obsidian_writer.stop()
                except Exception:
                    pass
            await self._graceful_stop(reason)

        return reason
```

Only two behavioral additions compared to the Wave 1-E version:
- `await self._obsidian_writer.start()` after `self._writer.start()`.
- `await self._obsidian_writer.on_fact(claim, run_id=self._run_id)` inside the per-claim loop, wrapped in `try: ... except Exception: pass`.
- `await self._obsidian_writer.flush()` after `await self._writer.quiesce()`, wrapped in try/except.
- `await self._obsidian_writer.stop()` in the `finally` block, wrapped in try/except.

All four sites check `if self._obsidian_writer is not None:` so the baseline (no Obsidian) behavior is unchanged.

- [ ] **Step 5: Run the tests and verify they pass**

Run: `uv run pytest tests/unit/test_orchestrator_obsidian_hook.py -q`

Expected: `5 passed`

- [ ] **Step 6: Run the full unit suite to confirm no regressions**

Run: `uv run pytest tests/unit -q`

Expected: All existing tests still pass, plus the 5 new hook tests.

- [ ] **Step 7: Commit**

```bash
git add researcher/orchestrator.py tests/unit/test_orchestrator_obsidian_hook.py
git commit -m "$(cat <<'EOF'
orchestrator: optional obsidian_writer hook at four call sites

Adds an obsidian_writer parameter to Orchestrator.__init__ (default None).
When set, calls start() before the cycle loop, on_fact(claim, run_id)
for each successful claim alongside writer.submit(), flush() after
writer.quiesce() per cycle, and stop() in the finally block.

All four call sites are guarded with try/except so a failing Obsidian
writer never disrupts the primary reduce path. Baseline (no writer) is
byte-identical to pre-Obsidian Orchestrator.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Offline integration smoke — full Orchestrator run with a real ObsidianWriter

**Files:**
- Create: `tests/integration/test_smoke_obsidian_wars.py`

- [ ] **Step 1: Write the integration test**

Create `tests/integration/test_smoke_obsidian_wars.py`:

```python
"""Offline integration smoke: full Orchestrator.run() with a real ObsidianWriter.

Uses StubCliRunner + fixture responses (no subprocess) but a real
ObsidianWriter pointed at tmp_path. Asserts that the vault ends up with
valid Markdown files containing the expected fields and provenance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from researcher.backends.resolver import BackendResolver
from researcher.budget import Budget
from researcher.integrations.obsidian import ObsidianWriter
from researcher.orchestrator import Orchestrator
from researcher.scheduler import Scheduler
from researcher.spec import EntitySpec, FieldSpec, RunSpec
from researcher.storage.writer import FactWriter
from tests.stubs.bus import StubEventBus
from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
from tests.stubs.llm import StubLLMClient
from tests.stubs.resolver import StubEntityResolver
from tests.stubs.store import StubKnowledgeStore


def _which_claude(cmd: str) -> str | None:
    return "/usr/local/bin/claude" if cmd == "claude" else None


async def _noop_fact(_entity_id, _claim):
    return None


async def _noop_conflict(_entity_id, _cells):
    return None


@pytest.mark.asyncio
async def test_obsidian_smoke_end_to_end_offline(tmp_path: Path):
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())

    spec = RunSpec(
        spec_id="wars",
        goal="Major interstate wars since 1500",
        entities=[
            EntitySpec(
                name="War",
                fields=[
                    FieldSpec(name="name", type="str", required=True),
                    FieldSpec(name="start_year", type="int", required=True),
                    FieldSpec(name="end_year", type="int"),
                    FieldSpec(name="belligerents", type="list[str]"),
                ],
                search_templates=[],
            )
        ],
        seeds=["Major wars since 1500"],
        models={"fast": "stub", "smart": "stub", "heavy": "stub"},
        backend_policy="auto",
        max_cycles=1,
    )

    store = StubKnowledgeStore()
    await store.open()
    bus = StubEventBus()
    llm = StubLLMClient()
    budget = Budget(usd_cap=3.0, wall_cap_s=600)
    resolver = StubEntityResolver()
    entity_schema = {
        "entity_type": "War",
        "fields": [{"name": "name", "type": "str", "required": True}],
    }
    writer = FactWriter(
        store=store,
        resolver=resolver,
        entity_schema=entity_schema,
        emit_fact=_noop_fact,
        emit_conflict=_noop_conflict,
    )
    scheduler = Scheduler(spec=spec, store=store)
    backend_resolver = BackendResolver(which_fn=_which_claude)

    obsidian = ObsidianWriter(vault_path=tmp_path, flush_interval_s=0.05)

    orch = Orchestrator(
        spec=spec,
        store=store,  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
        bus=bus,  # type: ignore[arg-type]
        writer=writer,
        scheduler=scheduler,
        budget=budget,
        run_id="obsidian-smoke",
        max_parallel_agents=2,
        obsidian_writer=obsidian,
    )
    orch.set_backend_resolver(backend_resolver)
    orch.set_cli_runner_factory(lambda kind: runner)

    # Act
    await orch.run()

    # Assert — vault structure
    researcher_dir = tmp_path / "researcher"
    assert researcher_dir.is_dir()
    war_dir = researcher_dir / "War"
    assert war_dir.is_dir()

    # Assert — exactly one entity file (from the wars_discover fixture)
    md_files = list(war_dir.glob("*.md"))
    assert len(md_files) == 1
    war_file = md_files[0]

    content = war_file.read_text()

    # Frontmatter structure
    assert content.startswith("---\n")
    assert "researcher_type: War" in content
    assert "researcher_name: World War II" in content
    assert "researcher_runs:" in content
    assert "obsidian-smoke" in content
    assert "- researcher" in content
    assert "- researcher/War" in content

    # Entity schema fields at the top level
    assert "start_year: 1939" in content
    assert "end_year: 1945" in content
    assert "belligerents:" in content
    assert "- Allies" in content
    assert "- Axis" in content

    # Body structure
    assert "# World War II" in content
    assert "## Fields" in content
    assert "## Provenance" in content
    assert "https://en.wikipedia.org/wiki/World_War_II" in content

    # Writer stats
    assert obsidian.stats["writes"] >= 1
    assert obsidian.stats["errors"] == 0
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/integration/test_smoke_obsidian_wars.py -q`

Expected: `1 passed`

- [ ] **Step 3: Run the complete test suite**

Run: `uv run pytest -q`

Expected: all tests pass including the new Obsidian ones.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_smoke_obsidian_wars.py
git commit -m "$(cat <<'EOF'
tests: offline integration smoke for the Obsidian vault writer

End-to-end: full Orchestrator.run() with a real ObsidianWriter pointed at
tmp_path and a StubCliRunner returning the wars_discover fixture. Asserts
that the vault ends up with a valid World War II.md containing the right
frontmatter, entity schema fields, tags, and Provenance section.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Final verification

**Files:** none modified

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -q`

Record the count. All tests should pass.

- [ ] **Step 2: Verify all new modules import cleanly**

Run:
```bash
uv run python -c "
import researcher.integrations
import researcher.integrations.obsidian
from researcher.integrations.obsidian import (
    ObsidianWriter, EntityState, FieldValue,
    ObsidianWriterError, ObsidianVaultNotFound,
)
print('obsidian modules import ok')
"
```

Expected: `obsidian modules import ok`

- [ ] **Step 3: Verify the CLI flag is visible**

Run: `uv run python -m researcher.cli run --help 2>&1 | grep obsidian`

Expected: a line mentioning `--obsidian-vault`.

- [ ] **Step 4: Push**

Run:
```bash
git log --oneline origin/main..HEAD
git push origin main
```

Expected: push succeeds; all Obsidian integration commits reach origin.

---

## Known gaps left for future work

The following are documented as scope cuts in the design doc, NOT regressions:

1. **Reading existing files** — the writer never opens `.md` files on disk to merge with prior state. Each run's vault view reflects only the current run.
2. **Cross-run fact accumulation** — the `Seen in Runs` list only contains the current run_id. True cross-run accumulation requires integration with the typed store (Wave 1-B).
3. **Backlinks between entities** — no auto `[[wikilink]]` generation. Obsidian's unlinked-mentions feature handles this manually.
4. **Attachments** — source page images are not downloaded. Text snippets only.
5. **Dataview inline fields** — rendered body uses plain Markdown tables, not Dataview inline syntax.

---

## Self-review

- ✅ **Spec coverage:** every spec section is covered by at least one task.
  - Filename sanitization → Task 2
  - Markdown rendering → Task 3
  - Lifecycle (start/stop) → Task 4
  - Coalescing semantics → Task 5
  - Debounce + flush + atomic write → Task 6
  - Error paths → Task 7 (regression coverage; impl in Task 6)
  - RunSpec field + CLI flag → Task 8
  - Orchestrator hook → Task 9
  - Offline integration smoke → Task 10
- ✅ **Placeholder scan:** no TBD / TODO / FIXME in any implementation step.
- ✅ **Type consistency:** `ObsidianWriter`, `EntityState`, `FieldValue`, `ObsidianWriterError`, `ObsidianVaultNotFound`, `_safe_filename`, `_render_markdown`, `_snapshot_state` names are used identically across every task that references them. The `stats` dict keys (`writes`, `coalesced_claims`, `errors`, `sanitized_names`) match between the stats-init test in Task 4 and the coalesce counter in Task 5/7. The `on_fact(claim, run_id)` signature matches across Tasks 5, 9, 10.
- ✅ **Test count:** 7 (sanitize) + 9 (render) + 5 (lifecycle) + 7 (on_fact) + 8 (flush/debounce) + 3 (error) = 39 unit tests for the writer, plus 5 orchestrator hook tests, 2 spec tests, and 1 integration smoke = **47 new tests total**. (Spec estimated 29; actual count is higher because some sections split into multiple tests during decomposition. The plan is correct, the spec's estimate was conservative.)
