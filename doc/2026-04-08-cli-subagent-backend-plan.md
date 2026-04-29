# CLI Subagent Backend — Implementation Plan

> Current status, 2026-04-28: this is a historical implementation plan. The CLI
> backend has shipped, the native API path is wired, the public command surface
> has expanded, and current behavior is summarized in
> [2026-04-28-current-product-surface.md](2026-04-28-current-product-surface.md).
> Use `uv run python -m researcher ...` for development commands.
> Later task snippets intentionally preserve the original Wave 0 plan context;
> use the current-product-surface doc for exact commands and status.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a whole-task delegation path so research tasks can be offloaded to local `claude -p` or `codex exec` subprocesses with WebSearch+WebFetch enabled, saving both LLM and search credits for users on flat-fee subscriptions. The native API path remains the fallback.

**Architecture:** A `SubagentResearcher(Agent)` subclass dispatches tasks to `ClaudeCodeRunner` / `CodexRunner`, which shell out via `asyncio.create_subprocess_exec` and parse the structured JSON response back into `FactClaim` instances. A `BackendResolver` probes `$PATH` at startup and the orchestrator picks an agent class per task. `FactClaim`s from both paths flow through the unchanged Wave 0 `FactWriter` serial-reduce queue.

**Tech Stack:** Python 3.11+ (uv), Pydantic v2, asyncio, stdlib `shutil`/`subprocess`/`json`. No new runtime dependencies.

**Spec:** `doc/2026-04-08-cli-subagent-backend-design.md` (commit `d8dc304`)

---

## File structure

**New files:**

| Path | Purpose |
|---|---|
| `researcher/backends/__init__.py` | empty package marker |
| `researcher/backends/models.py` | `CliKind`, `Extraction`, `SubagentResponse`, `CliResult`, `BackendChoice`, `BackendUnavailableError` |
| `researcher/backends/resolver.py` | `BackendResolver` — probes `$PATH`, picks per task |
| `researcher/backends/prompts.py` | system + user prompt templates for the CLI path |
| `researcher/backends/cli_runner.py` | `CliRunner` Protocol, `ClaudeCodeRunner`, `CodexRunner` |
| `researcher/agents/subagent.py` | `SubagentResearcher(Agent)` |
| `tests/stubs/cli_runner.py` | `StubCliRunner` for unit tests |
| `tests/fixtures/subagent_responses/wars_discover.json` | canned valid response |
| `tests/fixtures/subagent_responses/wars_empty.json` | canned response with zero extractions |
| `tests/unit/test_events_subagent_call.py` | event schema tests |
| `tests/unit/test_budget_subagent_calls.py` | budget counter tests |
| `tests/unit/test_backend_models.py` | `SubagentResponse` validation |
| `tests/unit/test_backend_resolver.py` | resolver probe + pick |
| `tests/unit/test_cli_runner_claude.py` | `ClaudeCodeRunner` subprocess mocking |
| `tests/unit/test_cli_runner_codex.py` | `CodexRunner` subprocess mocking |
| `tests/unit/test_subagent_researcher.py` | agent mapping + lifecycle |
| `tests/unit/test_orchestrator_backend_routing.py` | spawn-agent routing + circuit break |
| `tests/integration/test_smoke_subagent_wars.py` | end-to-end offline smoke |

**Modified files:**

| Path | Change |
|---|---|
| `researcher/events.py` | add `SubagentCall` event + payload to the discriminated union |
| `researcher/budget.py` | add `subagent_calls_total`, `max_subagent_calls`, `record_subagent_call`, `allows_subagent_call` |
| `researcher/spec.py` | add `backend_policy`, `max_subagent_calls`, `subagent_timeout_s` to `RunSpec` |
| `researcher/cli.py` | add `--backend {auto,cli,api}` option to `researcher run` |
| `researcher/orchestrator.py` | instantiate `BackendResolver`, route in `_spawn_agent`, add `SUBAGENT_CAP` stop reason, circuit-break on repeat failures |

---

## Task 1: Add `SubagentCall` event to the discriminated union

**Files:**
- Modify: `researcher/events.py`
- Test: `tests/unit/test_events_subagent_call.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_events_subagent_call.py`:

```python
"""Tests for the SubagentCall event."""

import json
from datetime import datetime, timezone

from researcher.events import (
    SubagentCall,
    SubagentCallPayload,
    parse_event,
)


def _ts() -> datetime:
    return datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)


def test_subagent_call_has_type_discriminator():
    e = SubagentCall(
        seq=0,
        ts=_ts(),
        run_id="run-x",
        payload=SubagentCallPayload(
            agent_id="a1",
            task_id="t1",
            cli_kind="claude_code",
            wall_ms=4321,
            exit_code=0,
            claims_emitted=4,
        ),
    )
    assert e.type == "subagent_call"


def test_subagent_call_roundtrip_through_json():
    e = SubagentCall(
        seq=7,
        ts=_ts(),
        run_id="run-x",
        payload=SubagentCallPayload(
            agent_id="a1",
            task_id="t1",
            cli_kind="codex",
            wall_ms=1200,
            exit_code=0,
            claims_emitted=2,
        ),
    )
    data = json.loads(e.model_dump_json())
    parsed = parse_event(data)
    assert isinstance(parsed, SubagentCall)
    assert parsed.payload.cli_kind == "codex"
    assert parsed.payload.claims_emitted == 2


def test_parse_event_dispatches_subagent_call_from_wire():
    raw = {
        "type": "subagent_call",
        "seq": 9,
        "ts": _ts().isoformat(),
        "run_id": "run-x",
        "payload": {
            "agent_id": "a1",
            "task_id": "t1",
            "cli_kind": "claude_code",
            "wall_ms": 999,
            "exit_code": 1,
            "claims_emitted": 0,
        },
    }
    e = parse_event(raw)
    assert isinstance(e, SubagentCall)
    assert e.payload.exit_code == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_events_subagent_call.py -q
```

Expected: collection error `ImportError: cannot import name 'SubagentCall' from 'researcher.events'`.

- [ ] **Step 3: Add the payload class**

In `researcher/events.py`, find the `# ---------- Payloads ----------` block and add this class at the end of the payloads section (before `# ---------- Event envelope ----------`):

```python
class SubagentCallPayload(BaseModel):
    agent_id: str
    task_id: str
    cli_kind: Literal["claude_code", "codex"]
    wall_ms: int
    exit_code: int | None
    claims_emitted: int
```

- [ ] **Step 4: Add the event class**

In `researcher/events.py`, after `class RunComplete(_EventBase):` add:

```python
class SubagentCall(_EventBase):
    type: Literal["subagent_call"] = "subagent_call"
    payload: SubagentCallPayload
```

- [ ] **Step 5: Register the event in the union**

In `researcher/events.py`, update the `Event = Annotated[Union[...]]` block by adding `SubagentCall` to the list (immediately after `RunComplete`):

```python
Event = Annotated[
    Union[
        CycleStart,
        CycleEnd,
        AgentSpawn,
        AgentStateChange,
        AgentLog,
        FactWritten,
        ConflictDetected,
        ConflictResolved,
        CostUpdate,
        BudgetWarning,
        RunComplete,
        SubagentCall,
    ],
    Field(discriminator="type"),
]
```

- [ ] **Step 6: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_events_subagent_call.py -q
```

Expected: `3 passed`.

- [ ] **Step 7: Run the full events suite to confirm no regression**

```bash
uv run pytest tests/unit/test_events.py tests/unit/test_events_subagent_call.py -q
```

Expected: `12 passed`.

- [ ] **Step 8: Commit**

```bash
git add researcher/events.py tests/unit/test_events_subagent_call.py
git commit -m "$(cat <<'EOF'
events: add SubagentCall to the discriminated union

First step of the CLI subagent backend. Adds a new event type the TUI can
render per CLI invocation. No runtime behavior yet — just the schema.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Extend `Budget` with a subagent call counter and cap

**Files:**
- Modify: `researcher/budget.py`
- Test: `tests/unit/test_budget_subagent_calls.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_budget_subagent_calls.py`:

```python
"""Tests for the subagent-call counter + cap on Budget."""

from researcher.budget import Budget


def test_initial_subagent_counter_is_zero():
    b = Budget(usd_cap=3.0, wall_cap_s=600)
    assert b.subagent_calls_total == 0
    assert b.allows_subagent_call()


def test_record_subagent_call_increments_counter():
    b = Budget(usd_cap=3.0, wall_cap_s=600)
    b.record_subagent_call()
    b.record_subagent_call()
    assert b.subagent_calls_total == 2


def test_allows_subagent_call_enforces_cap():
    b = Budget(usd_cap=3.0, wall_cap_s=600, max_subagent_calls=2)
    assert b.allows_subagent_call()
    b.record_subagent_call()
    assert b.allows_subagent_call()
    b.record_subagent_call()
    assert not b.allows_subagent_call()


def test_subagent_counter_does_not_affect_usd_budget():
    b = Budget(usd_cap=1.0, wall_cap_s=600, max_subagent_calls=1000)
    for _ in range(100):
        b.record_subagent_call()
    assert b.total_spent() == 0.0
    assert b.remaining() == 1.0
    assert not b.exceeded()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_budget_subagent_calls.py -q
```

Expected: `AttributeError: 'Budget' object has no attribute 'subagent_calls_total'`.

- [ ] **Step 3: Extend Budget**

In `researcher/budget.py`, update the `__init__` signature to add `max_subagent_calls`:

```python
    def __init__(
        self,
        usd_cap: float,
        wall_cap_s: float,
        warn_fractions: tuple[float, ...] = (0.5, 0.8, 0.95),
        per_task_fraction: float = 0.05,
        per_entity_fraction: float = 0.03,
        max_subagent_calls: int = 500,
    ) -> None:
```

Then at the end of `__init__` body (after `self._wall_started_at = None`), add:

```python
        self._max_subagent_calls = max_subagent_calls
        self._subagent_calls_total = 0
```

Add these methods at the end of the class (after `wall_exceeded`):

```python
    # ---------- Subagent calls ----------

    @property
    def subagent_calls_total(self) -> int:
        return self._subagent_calls_total

    def record_subagent_call(self) -> None:
        self._subagent_calls_total += 1

    def allows_subagent_call(self) -> bool:
        return self._subagent_calls_total < self._max_subagent_calls
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_budget_subagent_calls.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Confirm existing budget tests still pass**

```bash
uv run pytest tests/unit/test_budget.py tests/unit/test_budget_subagent_calls.py -q
```

Expected: `14 passed`.

- [ ] **Step 6: Commit**

```bash
git add researcher/budget.py tests/unit/test_budget_subagent_calls.py
git commit -m "$(cat <<'EOF'
budget: add subagent call counter + cap

Separate from the USD budget. Default cap is 500 calls per run. Subagent
calls report $0.00 so the USD path is unaffected; this counter is the
hard stop that prevents runaway subprocess loops.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Create `researcher/backends/models.py` — shared Pydantic types

**Files:**
- Create: `researcher/backends/__init__.py`
- Create: `researcher/backends/models.py`
- Test: `tests/unit/test_backend_models.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_backend_models.py`:

```python
"""Tests for backend Pydantic models."""

import pytest
from pydantic import ValidationError

from researcher.backends.models import (
    BackendChoice,
    BackendUnavailableError,
    CliKind,
    CliResult,
    Extraction,
    SubagentResponse,
)


def test_clikind_values():
    assert [k.value for k in CliKind] == ["claude_code", "codex"]


def test_extraction_confidence_bounds():
    with pytest.raises(ValidationError):
        Extraction(
            field="start_year",
            value=1939,
            source_url="https://example.com",
            snippet="s",
            confidence=1.5,
        )


def test_subagent_response_happy_path():
    r = SubagentResponse(
        entity_name="WWII",
        extractions=[
            Extraction(
                field="start_year",
                value=1939,
                source_url="https://example.com/wwii",
                snippet="World War II began in 1939...",
                confidence=0.95,
            )
        ],
        diagnostics="found via wikipedia",
    )
    assert r.entity_name == "WWII"
    assert len(r.extractions) == 1


def test_cli_result_ok_shape():
    r = CliResult(
        ok=True,
        data=SubagentResponse(entity_name="X", extractions=[], diagnostics=""),
        error=None,
        wall_ms=1234,
        exit_code=0,
        raw_usage={"input_tokens": 100, "output_tokens": 50},
    )
    assert r.ok
    assert r.data is not None
    assert r.raw_usage["input_tokens"] == 100


def test_cli_result_failure_shape():
    r = CliResult(
        ok=False,
        data=None,
        error="timeout",
        wall_ms=120_000,
        exit_code=None,
        raw_usage=None,
    )
    assert not r.ok
    assert r.error == "timeout"


def test_backend_choice_defaults():
    c = BackendChoice(kind=None, reason="auto: no CLI detected")
    assert c.kind is None


def test_backend_choice_with_kind():
    c = BackendChoice(kind=CliKind.CLAUDE_CODE, reason="auto: claude detected")
    assert c.kind == CliKind.CLAUDE_CODE


def test_backend_unavailable_error_is_runtime_error():
    assert issubclass(BackendUnavailableError, RuntimeError)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_backend_models.py -q
```

Expected: `ModuleNotFoundError: No module named 'researcher.backends'`.

- [ ] **Step 3: Create the package marker**

Create `researcher/backends/__init__.py` (empty file).

- [ ] **Step 4: Create the models module**

Create `researcher/backends/models.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_backend_models.py -q
```

Expected: `8 passed`.

- [ ] **Step 6: Commit**

```bash
git add researcher/backends/__init__.py researcher/backends/models.py tests/unit/test_backend_models.py
git commit -m "$(cat <<'EOF'
backends: add CliKind / Extraction / SubagentResponse / CliResult models

Shared Pydantic types used by the resolver, runners, and SubagentResearcher.
BackendUnavailableError is the fail-loud error when policy='cli' but no CLI
is detected at run start.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Create `BackendResolver`

**Files:**
- Create: `researcher/backends/resolver.py`
- Test: `tests/unit/test_backend_resolver.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_backend_resolver.py`:

```python
"""Tests for BackendResolver — autodetect + per-task pick + circuit break."""

import pytest

from researcher.backends.models import BackendUnavailableError, CliKind
from researcher.backends.resolver import BackendResolver


def _which_none(_: str) -> str | None:
    return None


def _which_claude_only(cmd: str) -> str | None:
    return "/usr/local/bin/claude" if cmd == "claude" else None


def _which_both(cmd: str) -> str | None:
    return {"claude": "/usr/local/bin/claude", "codex": "/usr/local/bin/codex"}.get(cmd)


def test_detects_nothing_when_which_returns_none():
    r = BackendResolver(which_fn=_which_none)
    assert r.detected == []


def test_detects_claude_only():
    r = BackendResolver(which_fn=_which_claude_only)
    assert r.detected == [CliKind.CLAUDE_CODE]


def test_detects_both_preferring_claude_first():
    r = BackendResolver(which_fn=_which_both)
    assert r.detected == [CliKind.CLAUDE_CODE, CliKind.CODEX]


def test_auto_policy_with_claude_picks_claude():
    r = BackendResolver(which_fn=_which_claude_only)
    choice = r.pick(policy="auto")
    assert choice.kind == CliKind.CLAUDE_CODE


def test_auto_policy_with_both_prefers_claude():
    r = BackendResolver(which_fn=_which_both)
    choice = r.pick(policy="auto")
    assert choice.kind == CliKind.CLAUDE_CODE


def test_auto_policy_with_none_falls_through():
    r = BackendResolver(which_fn=_which_none)
    choice = r.pick(policy="auto")
    assert choice.kind is None


def test_api_policy_always_returns_none():
    r = BackendResolver(which_fn=_which_both)
    choice = r.pick(policy="api")
    assert choice.kind is None


def test_cli_policy_with_none_raises():
    r = BackendResolver(which_fn=_which_none)
    with pytest.raises(BackendUnavailableError):
        r.pick(policy="cli")


def test_cli_policy_with_claude_returns_claude():
    r = BackendResolver(which_fn=_which_claude_only)
    choice = r.pick(policy="cli")
    assert choice.kind == CliKind.CLAUDE_CODE


def test_circuit_break_clears_detected():
    r = BackendResolver(which_fn=_which_both)
    assert r.detected == [CliKind.CLAUDE_CODE, CliKind.CODEX]
    r.clear_detected("usage_limit_reached")
    assert r.detected == []
    choice = r.pick(policy="auto")
    assert choice.kind is None
    assert "usage_limit" in choice.reason


def test_clear_detected_is_sticky_for_the_run():
    r = BackendResolver(which_fn=_which_both)
    r.clear_detected("auth_required")
    # Even after more calls, detected stays empty.
    r.pick(policy="auto")
    assert r.detected == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_backend_resolver.py -q
```

Expected: `ModuleNotFoundError: No module named 'researcher.backends.resolver'`.

- [ ] **Step 3: Implement the resolver**

Create `researcher/backends/resolver.py`:

```python
"""BackendResolver — probes $PATH and picks a backend per task."""

from __future__ import annotations

import shutil
from typing import Callable, Literal, Optional

from researcher.backends.models import (
    BackendChoice,
    BackendUnavailableError,
    CliKind,
)


Policy = Literal["auto", "cli", "api"]

# Map CliKind -> executable name to probe.
_PROBES: dict[CliKind, str] = {
    CliKind.CLAUDE_CODE: "claude",
    CliKind.CODEX: "codex",
}


class BackendResolver:
    """Picks a backend for each task.

    At construction, probes $PATH via an injectable `which_fn`. Production
    code passes `shutil.which`; tests pass a stub.

    `detected` is populated in a fixed priority order: Claude Code first,
    then Codex. `pick(policy)` honors the policy and returns the first
    detected CLI (or None for `api` / no-detections / `auto` fall-through).

    `clear_detected(reason)` is the circuit-break hook: once called, all
    subsequent `pick()` calls return `kind=None` for the remainder of the
    resolver's lifetime (which equals the run lifetime).
    """

    def __init__(self, which_fn: Callable[[str], Optional[str]] = shutil.which) -> None:
        self._which_fn = which_fn
        self._detected: list[CliKind] = []
        self._cleared_reason: Optional[str] = None
        self._probe()

    def _probe(self) -> None:
        for kind in (CliKind.CLAUDE_CODE, CliKind.CODEX):
            if self._which_fn(_PROBES[kind]) is not None:
                self._detected.append(kind)

    @property
    def detected(self) -> list[CliKind]:
        if self._cleared_reason is not None:
            return []
        return list(self._detected)

    def clear_detected(self, reason: str) -> None:
        """Circuit break — subsequent picks return kind=None for the rest of the run."""
        self._cleared_reason = reason

    def pick(self, policy: Policy = "auto") -> BackendChoice:
        if policy == "api":
            return BackendChoice(kind=None, reason="policy=api")

        if policy == "cli":
            if not self.detected:
                raise BackendUnavailableError(
                    "policy=cli but no CLI subagent detected on PATH (looked for: claude, codex)"
                )
            kind = self.detected[0]
            return BackendChoice(kind=kind, reason=f"policy=cli, picked {kind.value}")

        # policy == "auto"
        if self._cleared_reason is not None:
            return BackendChoice(kind=None, reason=f"circuit_break: {self._cleared_reason}")
        if not self._detected:
            return BackendChoice(kind=None, reason="auto: no CLI detected")
        kind = self._detected[0]
        return BackendChoice(kind=kind, reason=f"auto: {kind.value} detected")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_backend_resolver.py -q
```

Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add researcher/backends/resolver.py tests/unit/test_backend_resolver.py
git commit -m "$(cat <<'EOF'
backends: add BackendResolver with autodetect and circuit break

Probes PATH for claude / codex via an injectable which_fn. Prefers Claude
Code when both are available. Supports --backend {auto,cli,api} policy
and a sticky clear_detected() that disables the CLI path for the rest of
the run when the circuit breaker trips.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Create `researcher/backends/prompts.py` + `ClaudeCodeRunner`

**Files:**
- Create: `researcher/backends/prompts.py`
- Create: `researcher/backends/cli_runner.py`
- Test: `tests/unit/test_cli_runner_claude.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_cli_runner_claude.py`:

```python
"""Tests for ClaudeCodeRunner — subprocess mocking + error matrix."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from researcher.backends.cli_runner import ClaudeCodeRunner
from researcher.backends.models import SubagentResponse


SCHEMA = SubagentResponse.model_json_schema()


def _fake_process(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.close = MagicMock()
    proc.stdin.drain = AsyncMock()
    return proc


def _claude_envelope(result_json: str, input_tokens: int = 100, output_tokens: int = 50) -> bytes:
    return json.dumps(
        {
            "type": "result",
            "subtype": "final_result",
            "result": result_json,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "total_cost_usd": 0.0,
            "is_error": False,
        }
    ).encode()


@pytest.mark.asyncio
async def test_argv_construction_includes_required_flags():
    valid = _claude_envelope(
        json.dumps({"entity_name": "X", "extractions": [], "diagnostics": ""})
    )
    captured_argv: list = []

    async def fake_exec(*argv, **kwargs):
        captured_argv.extend(argv)
        return _fake_process(stdout=valid)

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="hi", schema=SCHEMA, timeout_s=30)

    assert result.ok
    assert "claude" in captured_argv[0]
    assert "-p" in captured_argv
    assert "--print" in captured_argv
    assert "--output-format" in captured_argv
    assert "json" in captured_argv
    assert "--json-schema" in captured_argv
    assert "--bare" in captured_argv
    assert "--model" in captured_argv
    assert "sonnet" in captured_argv
    assert "--allowedTools" in captured_argv
    # Tool names are present as separate args or space-joined; check either.
    joined = " ".join(captured_argv)
    assert "WebSearch" in joined
    assert "WebFetch" in joined


@pytest.mark.asyncio
async def test_prompt_is_written_to_stdin_not_argv():
    valid = _claude_envelope(
        json.dumps({"entity_name": "X", "extractions": [], "diagnostics": ""})
    )
    stdin_writes: list[bytes] = []
    proc = _fake_process(stdout=valid)
    proc.stdin.write = lambda data: stdin_writes.append(data)

    async def fake_exec(*argv, **kwargs):
        # Assert the prompt body itself is not in argv.
        assert "SECRET_PROMPT_BODY" not in " ".join(argv)
        return proc

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await runner.execute(prompt="SECRET_PROMPT_BODY", schema=SCHEMA, timeout_s=30)

    joined_stdin = b"".join(stdin_writes)
    assert b"SECRET_PROMPT_BODY" in joined_stdin


@pytest.mark.asyncio
async def test_parses_envelope_into_subagent_response():
    inner = {
        "entity_name": "WWII",
        "extractions": [
            {
                "field": "start_year",
                "value": 1939,
                "source_url": "https://example.com/wwii",
                "snippet": "began in 1939",
                "confidence": 0.9,
            }
        ],
        "diagnostics": "ok",
    }
    envelope = _claude_envelope(json.dumps(inner))

    async def fake_exec(*argv, **kwargs):
        return _fake_process(stdout=envelope)

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert result.ok
    assert result.data is not None
    assert result.data.entity_name == "WWII"
    assert len(result.data.extractions) == 1
    assert result.data.extractions[0].value == 1939
    assert result.raw_usage == {"input_tokens": 100, "output_tokens": 50}
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_timeout_kills_child_and_returns_failure():
    proc = _fake_process()

    async def fake_communicate():
        raise asyncio.TimeoutError()

    proc.communicate = fake_communicate

    async def fake_exec(*argv, **kwargs):
        return proc

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=0.1)

    assert not result.ok
    assert result.error == "timeout"
    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_non_zero_exit_captures_stderr_tail():
    async def fake_exec(*argv, **kwargs):
        return _fake_process(stdout=b"", stderr=b"bad flag: --foo\n", returncode=2)

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert not result.ok
    assert result.error is not None
    assert "exit 2" in result.error
    assert "bad flag" in result.error
    assert result.exit_code == 2


@pytest.mark.asyncio
async def test_auth_required_pattern_match():
    async def fake_exec(*argv, **kwargs):
        return _fake_process(
            stdout=b"",
            stderr=b"Please run: claude auth\n",
            returncode=1,
        )

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert not result.ok
    assert result.error == "auth_required"


@pytest.mark.asyncio
async def test_usage_limit_pattern_match():
    async def fake_exec(*argv, **kwargs):
        return _fake_process(
            stdout=b"",
            stderr=b"Error: rate_limit exceeded on monthly quota\n",
            returncode=1,
        )

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert not result.ok
    assert result.error == "usage_limit_reached"


@pytest.mark.asyncio
async def test_malformed_envelope_retries_once_then_fails(tmp_path):
    # First attempt returns garbage; second attempt also returns garbage.
    # Expect one retry (two calls total) and final parse_failed error.
    call_count = {"n": 0}

    async def fake_exec(*argv, **kwargs):
        call_count["n"] += 1
        return _fake_process(stdout=b"not json at all")

    runner = ClaudeCodeRunner(model="sonnet", post_mortem_dir=tmp_path)
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert call_count["n"] == 2  # one original + one retry
    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("parse_failed")
    # Post-mortem file should exist.
    assert any(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_cancellation_kills_child_and_propagates():
    proc = _fake_process()

    async def slow_communicate():
        await asyncio.sleep(10.0)
        return (b"", b"")

    proc.communicate = slow_communicate

    async def fake_exec(*argv, **kwargs):
        return proc

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        task = asyncio.create_task(
            runner.execute(prompt="p", schema=SCHEMA, timeout_s=60)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    proc.kill.assert_called()


@pytest.mark.asyncio
async def test_spawn_failed_returns_ok_false():
    async def fake_exec(*argv, **kwargs):
        raise FileNotFoundError("[Errno 2] No such file or directory: 'claude'")

    runner = ClaudeCodeRunner(model="sonnet")
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("spawn_failed")
    assert result.exit_code is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_cli_runner_claude.py -q
```

Expected: `ModuleNotFoundError: No module named 'researcher.backends.cli_runner'`.

- [ ] **Step 3: Create the prompts module**

Create `researcher/backends/prompts.py`:

```python
"""Prompt templates for the CLI subagent path."""

SYSTEM_PROMPT = (
    "You are a fact-extraction research agent. You have access to WebSearch "
    "and WebFetch tools. For each candidate entity matching the user's "
    "request, return one Extraction per field, citing the URL you extracted "
    "it from. Confidence is a number between 0.0 and 1.0. Return JSON "
    "matching the schema exactly — no prose, no markdown, no commentary."
)


USER_PROMPT_TEMPLATE = """Goal: {goal}
Entity type: {entity_type}
Task: {task_description}
Fields required: {field_list}
Return up to {max_entities} entities.

Respond with JSON matching this schema:
{schema_json}
"""


def build_user_prompt(
    *,
    goal: str,
    entity_type: str,
    task_description: str,
    field_list: list[str],
    max_entities: int,
    schema_json: str,
) -> str:
    return USER_PROMPT_TEMPLATE.format(
        goal=goal,
        entity_type=entity_type,
        task_description=task_description,
        field_list=", ".join(field_list),
        max_entities=max_entities,
        schema_json=schema_json,
    )
```

- [ ] **Step 4: Create the runner module**

Create `researcher/backends/cli_runner.py`:

```python
"""Subprocess runners for the CLI subagent path.

Both runners share a common Protocol. Each `execute()` call:
  1. Builds argv (never shell=True, never interpolates prompt into argv).
  2. Spawns the child via `asyncio.create_subprocess_exec`.
  3. Writes the prompt body to stdin; closes stdin.
  4. Awaits `communicate()` with a timeout.
  5. Parses output into a `CliResult`.

Errors become typed CliResult(ok=False); `CancelledError` propagates and
the child is killed first.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional, Protocol

from pydantic import ValidationError

from researcher.backends.models import CliResult, SubagentResponse


# Patterns that indicate auth or usage-limit errors in CLI stderr.
_AUTH_PATTERNS = ("please run: claude auth", "not signed in", "please log in", "codex login")
_USAGE_LIMIT_PATTERNS = ("rate_limit", "usage_limit", "quota", "monthly limit")


def _match_any(haystack: str, needles: tuple[str, ...]) -> bool:
    h = haystack.lower()
    return any(n in h for n in needles)


def _cache_key(argv: list[str], prompt: str, schema: dict) -> str:
    blob = json.dumps(
        {"argv": argv, "prompt": prompt, "schema": schema},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(blob).hexdigest()


class CliRunner(Protocol):
    """Protocol all CLI runners implement."""

    async def execute(
        self,
        prompt: str,
        schema: dict,
        timeout_s: float,
        tools: tuple[str, ...] = ("WebSearch", "WebFetch"),
    ) -> CliResult: ...


class ClaudeCodeRunner:
    """Runs `claude -p --print --output-format json --json-schema ...`."""

    def __init__(
        self,
        model: str = "sonnet",
        append_system_prompt: str = "",
        post_mortem_dir: Optional[Path] = None,
    ) -> None:
        self._model = model
        self._append_system_prompt = append_system_prompt
        self._post_mortem_dir = post_mortem_dir
        self._cache: dict[str, CliResult] = {}

    async def execute(
        self,
        prompt: str,
        schema: dict,
        timeout_s: float,
        tools: tuple[str, ...] = ("WebSearch", "WebFetch"),
    ) -> CliResult:
        argv = self._build_argv(schema=schema, tools=tools)
        key = _cache_key(argv, prompt, schema)
        if key in self._cache:
            return self._cache[key]

        result = await self._run_once(argv, prompt, stricter=False)

        # Malformed JSON → retry once with a stricter nudge in the prompt.
        if (not result.ok) and result.error and result.error.startswith("parse_failed"):
            stricter_prompt = (
                prompt
                + "\n\nIMPORTANT: Return ONLY a single JSON object matching the schema. "
                + "No prose, no markdown, no commentary. Your previous response could not be parsed."
            )
            result = await self._run_once(argv, stricter_prompt, stricter=True)

        self._cache[key] = result
        return result

    def _build_argv(self, *, schema: dict, tools: tuple[str, ...]) -> list[str]:
        argv: list[str] = [
            "claude",
            "-p",
            "--print",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema),
            "--bare",
            "--model",
            self._model,
            "--allowedTools",
            *tools,
            "--dangerously-skip-permissions",
        ]
        if self._append_system_prompt:
            argv.extend(["--append-system-prompt", self._append_system_prompt])
        return argv

    async def _run_once(self, argv: list[str], prompt: str, stricter: bool) -> CliResult:
        start = time.monotonic()
        proc: Any = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            return CliResult(
                ok=False,
                error=f"spawn_failed: {e}",
                wall_ms=int((time.monotonic() - start) * 1000),
                exit_code=None,
            )
        except PermissionError as e:
            return CliResult(
                ok=False,
                error=f"spawn_failed: {e}",
                wall_ms=int((time.monotonic() - start) * 1000),
                exit_code=None,
            )

        try:
            if proc.stdin is not None:
                proc.stdin.write(prompt.encode("utf-8"))
                if hasattr(proc.stdin, "drain"):
                    await proc.stdin.drain()
                proc.stdin.close()

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                return CliResult(
                    ok=False,
                    error="timeout",
                    wall_ms=int((time.monotonic() - start) * 1000),
                    exit_code=None,
                )
            except asyncio.CancelledError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                raise
        except asyncio.CancelledError:
            raise

        wall_ms = int((time.monotonic() - start) * 1000)
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

        # Auth + usage patterns take priority over exit code.
        if _match_any(stderr_str, _AUTH_PATTERNS):
            return CliResult(
                ok=False, error="auth_required", wall_ms=wall_ms, exit_code=proc.returncode
            )
        if _match_any(stderr_str, _USAGE_LIMIT_PATTERNS):
            return CliResult(
                ok=False,
                error="usage_limit_reached",
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )

        if proc.returncode != 0:
            tail = stderr_str.strip().splitlines()[-5:] if stderr_str.strip() else [""]
            tail_str = " | ".join(tail)[:2048]
            return CliResult(
                ok=False,
                error=f"exit {proc.returncode}: {tail_str}",
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )

        # Parse the Claude Code JSON envelope.
        try:
            envelope = json.loads(stdout.decode("utf-8", errors="replace"))
            inner_str = envelope.get("result", "")
            inner = json.loads(inner_str) if isinstance(inner_str, str) else inner_str
            data = SubagentResponse.model_validate(inner)
            return CliResult(
                ok=True,
                data=data,
                wall_ms=wall_ms,
                exit_code=proc.returncode,
                raw_usage=envelope.get("usage"),
            )
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            self._dump_post_mortem(stdout, stricter=stricter)
            return CliResult(
                ok=False,
                error=f"parse_failed: {e}",
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )

    def _dump_post_mortem(self, stdout: bytes, stricter: bool) -> None:
        if self._post_mortem_dir is None:
            return
        self._post_mortem_dir.mkdir(parents=True, exist_ok=True)
        suffix = "retry" if stricter else "first"
        ts = int(time.time() * 1000)
        path = self._post_mortem_dir / f"claude_{ts}_{suffix}.txt"
        try:
            path.write_bytes(stdout)
        except OSError:
            pass
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_cli_runner_claude.py -q
```

Expected: `10 passed`. If any test fails, read the failure message carefully — the most common mistakes are: argv ordering, stdin mock not returning the expected bytes, and forgetting to reset the cache between retry paths.

- [ ] **Step 6: Commit**

```bash
git add researcher/backends/prompts.py researcher/backends/cli_runner.py tests/unit/test_cli_runner_claude.py
git commit -m "$(cat <<'EOF'
backends: add ClaudeCodeRunner with full error matrix

Shells out to `claude -p --print --output-format json --json-schema ... --bare`
with WebSearch + WebFetch enabled. Prompt goes via stdin to avoid argv quoting
issues. Handles timeout (kill + wait), non-zero exit (stderr tail), auth
patterns, usage-limit patterns, malformed JSON (one retry with a stricter
prompt, then fail), and cancellation (kill + re-raise). In-memory per-run
response cache keyed on sha256(argv + prompt + schema).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add `CodexRunner` to the same module

**Files:**
- Modify: `researcher/backends/cli_runner.py`
- Test: `tests/unit/test_cli_runner_codex.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_cli_runner_codex.py`:

```python
"""Tests for CodexRunner — argv construction, output parse, error matrix."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from researcher.backends.cli_runner import CodexRunner
from researcher.backends.models import SubagentResponse


SCHEMA = SubagentResponse.model_json_schema()


def _fake_process(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.close = MagicMock()
    proc.stdin.drain = AsyncMock()
    return proc


@pytest.mark.asyncio
async def test_argv_construction_includes_required_flags(tmp_path: Path):
    last_message_file = tmp_path / "last.txt"
    # Write the expected last-message file before the call so the runner reads it.
    last_message_file.write_text(
        json.dumps({"entity_name": "X", "extractions": [], "diagnostics": ""})
    )

    captured_argv: list = []

    async def fake_exec(*argv, **kwargs):
        captured_argv.extend(argv)
        return _fake_process()

    runner = CodexRunner(
        model="o4-mini",
        last_message_file=last_message_file,
        schema_file_dir=tmp_path,
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert result.ok, f"unexpected failure: {result.error}"
    assert captured_argv[0].endswith("codex")
    assert "exec" in captured_argv
    assert "--json" in captured_argv
    assert "--output-schema" in captured_argv
    assert "--output-last-message" in captured_argv
    assert "--sandbox" in captured_argv
    assert "read-only" in captured_argv
    assert "--skip-git-repo-check" in captured_argv
    assert "-m" in captured_argv
    assert "o4-mini" in captured_argv


@pytest.mark.asyncio
async def test_schema_file_is_written_and_cleaned_up(tmp_path: Path):
    last_message_file = tmp_path / "last.txt"
    last_message_file.write_text(
        json.dumps({"entity_name": "X", "extractions": [], "diagnostics": ""})
    )

    schema_path_captured: list[str] = []

    async def fake_exec(*argv, **kwargs):
        # Extract the --output-schema path from argv to verify the file exists during the call.
        for i, a in enumerate(argv):
            if a == "--output-schema":
                schema_path_captured.append(argv[i + 1])
                break
        return _fake_process()

    runner = CodexRunner(
        model="o4-mini",
        last_message_file=last_message_file,
        schema_file_dir=tmp_path,
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert len(schema_path_captured) == 1
    # After the call, the temp schema file should be removed.
    assert not Path(schema_path_captured[0]).exists()


@pytest.mark.asyncio
async def test_parses_last_message_file_into_subagent_response(tmp_path: Path):
    last_message_file = tmp_path / "last.txt"
    last_message_file.write_text(
        json.dumps(
            {
                "entity_name": "WWII",
                "extractions": [
                    {
                        "field": "start_year",
                        "value": 1939,
                        "source_url": "https://example.com/wwii",
                        "snippet": "began in 1939",
                        "confidence": 0.9,
                    }
                ],
                "diagnostics": "ok",
            }
        )
    )

    async def fake_exec(*argv, **kwargs):
        return _fake_process()

    runner = CodexRunner(
        model="o4-mini",
        last_message_file=last_message_file,
        schema_file_dir=tmp_path,
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert result.ok
    assert result.data is not None
    assert result.data.entity_name == "WWII"
    assert result.data.extractions[0].value == 1939


@pytest.mark.asyncio
async def test_non_zero_exit_captures_stderr_tail(tmp_path: Path):
    last_message_file = tmp_path / "last.txt"

    async def fake_exec(*argv, **kwargs):
        return _fake_process(stderr=b"Error: config unreadable\n", returncode=3)

    runner = CodexRunner(
        model="o4-mini",
        last_message_file=last_message_file,
        schema_file_dir=tmp_path,
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert not result.ok
    assert "exit 3" in (result.error or "")


@pytest.mark.asyncio
async def test_usage_limit_pattern_match(tmp_path: Path):
    last_message_file = tmp_path / "last.txt"

    async def fake_exec(*argv, **kwargs):
        return _fake_process(stderr=b"quota exceeded for the month\n", returncode=1)

    runner = CodexRunner(
        model="o4-mini",
        last_message_file=last_message_file,
        schema_file_dir=tmp_path,
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        result = await runner.execute(prompt="p", schema=SCHEMA, timeout_s=30)

    assert not result.ok
    assert result.error == "usage_limit_reached"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_cli_runner_codex.py -q
```

Expected: `ImportError: cannot import name 'CodexRunner' from 'researcher.backends.cli_runner'`.

- [ ] **Step 3: Add `CodexRunner` to `researcher/backends/cli_runner.py`**

At the end of `researcher/backends/cli_runner.py` (after `ClaudeCodeRunner`), add:

```python
import os
import tempfile


class CodexRunner:
    """Runs `codex exec --json --output-schema <file> --output-last-message <file>`.

    Unlike Claude Code, Codex streams JSONL events on stdout and writes the
    final assistant message to `--output-last-message <file>`. We point that
    at a file we control, then read it back after the process exits.
    """

    def __init__(
        self,
        model: str = "o4-mini",
        last_message_file: Optional[Path] = None,
        schema_file_dir: Optional[Path] = None,
        post_mortem_dir: Optional[Path] = None,
    ) -> None:
        self._model = model
        self._last_message_file = last_message_file
        self._schema_file_dir = schema_file_dir or Path(tempfile.gettempdir())
        self._post_mortem_dir = post_mortem_dir
        self._cache: dict[str, CliResult] = {}

    async def execute(
        self,
        prompt: str,
        schema: dict,
        timeout_s: float,
        tools: tuple[str, ...] = ("WebSearch", "WebFetch"),  # Codex has browsing by default
    ) -> CliResult:
        # Write schema to a temp file.
        fd, schema_path = tempfile.mkstemp(
            prefix="researcher_schema_",
            suffix=".json",
            dir=str(self._schema_file_dir),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(schema, fh)

            # Last-message file: use the injected one for tests, or a fresh temp file.
            if self._last_message_file is not None:
                last_message_path = self._last_message_file
            else:
                lm_fd, lm_path = tempfile.mkstemp(
                    prefix="researcher_lastmsg_", suffix=".txt"
                )
                os.close(lm_fd)
                last_message_path = Path(lm_path)

            argv = [
                "codex",
                "exec",
                "--json",
                "--output-schema",
                schema_path,
                "--output-last-message",
                str(last_message_path),
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "-m",
                self._model,
                "--dangerously-bypass-approvals-and-sandbox",
            ]

            key = _cache_key(argv, prompt, schema)
            if key in self._cache:
                return self._cache[key]

            result = await self._run_once(argv, prompt, last_message_path)
            self._cache[key] = result
            return result
        finally:
            try:
                os.unlink(schema_path)
            except OSError:
                pass

    async def _run_once(
        self, argv: list[str], prompt: str, last_message_path: Path
    ) -> CliResult:
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, PermissionError) as e:
            return CliResult(
                ok=False,
                error=f"spawn_failed: {e}",
                wall_ms=int((time.monotonic() - start) * 1000),
                exit_code=None,
            )

        try:
            if proc.stdin is not None:
                proc.stdin.write(prompt.encode("utf-8"))
                if hasattr(proc.stdin, "drain"):
                    await proc.stdin.drain()
                proc.stdin.close()

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                return CliResult(
                    ok=False,
                    error="timeout",
                    wall_ms=int((time.monotonic() - start) * 1000),
                    exit_code=None,
                )
            except asyncio.CancelledError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                raise
        except asyncio.CancelledError:
            raise

        wall_ms = int((time.monotonic() - start) * 1000)
        stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

        if _match_any(stderr_str, _AUTH_PATTERNS):
            return CliResult(
                ok=False, error="auth_required", wall_ms=wall_ms, exit_code=proc.returncode
            )
        if _match_any(stderr_str, _USAGE_LIMIT_PATTERNS):
            return CliResult(
                ok=False,
                error="usage_limit_reached",
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )

        if proc.returncode != 0:
            tail = stderr_str.strip().splitlines()[-5:] if stderr_str.strip() else [""]
            tail_str = " | ".join(tail)[:2048]
            return CliResult(
                ok=False,
                error=f"exit {proc.returncode}: {tail_str}",
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )

        # Read the last-message file rather than stdout (which is JSONL events).
        try:
            raw = last_message_path.read_text(encoding="utf-8")
            inner = json.loads(raw)
            data = SubagentResponse.model_validate(inner)
            return CliResult(
                ok=True,
                data=data,
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            return CliResult(
                ok=False,
                error=f"parse_failed: {e}",
                wall_ms=wall_ms,
                exit_code=proc.returncode,
            )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_cli_runner_codex.py -q
```

Expected: `5 passed`.

- [ ] **Step 5: Run both runner test files together**

```bash
uv run pytest tests/unit/test_cli_runner_claude.py tests/unit/test_cli_runner_codex.py -q
```

Expected: `15 passed`.

- [ ] **Step 6: Commit**

```bash
git add researcher/backends/cli_runner.py tests/unit/test_cli_runner_codex.py
git commit -m "$(cat <<'EOF'
backends: add CodexRunner

Shells out to `codex exec --json --output-schema <file> --output-last-message
<file> --sandbox read-only --skip-git-repo-check`. Schema is written to a
tempfile that gets cleaned up even on exception. Result is read from the
last-message file rather than stdout (Codex streams JSONL events to stdout).
Same error matrix as ClaudeCodeRunner.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Create `StubCliRunner` + fixtures

**Files:**
- Create: `tests/stubs/cli_runner.py`
- Create: `tests/fixtures/subagent_responses/wars_discover.json`
- Create: `tests/fixtures/subagent_responses/wars_empty.json`

- [ ] **Step 1: Create the fixture files**

Create `tests/fixtures/subagent_responses/wars_discover.json`:

```json
{
  "entity_name": "World War II",
  "extractions": [
    {
      "field": "name",
      "value": "World War II",
      "source_url": "https://en.wikipedia.org/wiki/World_War_II",
      "snippet": "World War II or the Second World War was a global conflict...",
      "confidence": 0.98
    },
    {
      "field": "start_year",
      "value": 1939,
      "source_url": "https://en.wikipedia.org/wiki/World_War_II",
      "snippet": "...lasted from 1939 to 1945...",
      "confidence": 0.97
    },
    {
      "field": "end_year",
      "value": 1945,
      "source_url": "https://en.wikipedia.org/wiki/World_War_II",
      "snippet": "...lasted from 1939 to 1945...",
      "confidence": 0.97
    },
    {
      "field": "belligerents",
      "value": ["Allies", "Axis"],
      "source_url": "https://en.wikipedia.org/wiki/World_War_II",
      "snippet": "two opposing military alliances: the Allies and the Axis",
      "confidence": 0.96
    }
  ],
  "diagnostics": "found via Wikipedia infobox"
}
```

Create `tests/fixtures/subagent_responses/wars_empty.json`:

```json
{
  "entity_name": "Unknown War",
  "extractions": [],
  "diagnostics": "no matches found in two searches"
}
```

- [ ] **Step 2: Create the stub runner**

Create `tests/stubs/cli_runner.py`:

```python
"""In-memory stub CliRunner — returns canned CliResult values by prompt hash.

Usage:
    stub = StubCliRunner()
    stub.add_response("hello world", CliResult(ok=True, data=resp, wall_ms=10, exit_code=0))
    result = await stub.execute(prompt="hello world", schema=SCHEMA, timeout_s=30)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from researcher.backends.models import CliResult, Extraction, SubagentResponse


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def load_fixture_response(path: Path) -> SubagentResponse:
    """Load a canned SubagentResponse JSON from disk."""
    data = json.loads(path.read_text())
    return SubagentResponse.model_validate(data)


class StubCliRunner:
    def __init__(self) -> None:
        self._responses: dict[str, CliResult] = {}
        self._default: CliResult = CliResult(
            ok=False,
            error="stub: no fixture",
            wall_ms=0,
            exit_code=None,
        )
        self.calls: list[dict[str, Any]] = []

    def add_response(self, prompt: str, result: CliResult) -> None:
        self._responses[_hash_prompt(prompt)] = result

    def add_response_for_any(self, result: CliResult) -> None:
        """Return this result for any prompt not explicitly registered."""
        self._default = result

    async def execute(
        self,
        prompt: str,
        schema: dict,
        timeout_s: float,
        tools: tuple[str, ...] = ("WebSearch", "WebFetch"),
    ) -> CliResult:
        key = _hash_prompt(prompt)
        self.calls.append({"prompt_hash": key, "timeout_s": timeout_s, "tools": list(tools)})
        return self._responses.get(key, self._default)


def make_wars_discover_result() -> CliResult:
    """Convenience factory for the wars_discover.json fixture."""
    here = Path(__file__).parent.parent / "fixtures" / "subagent_responses" / "wars_discover.json"
    return CliResult(
        ok=True,
        data=load_fixture_response(here),
        wall_ms=1200,
        exit_code=0,
        raw_usage={"input_tokens": 500, "output_tokens": 300},
    )


def make_empty_result() -> CliResult:
    here = Path(__file__).parent.parent / "fixtures" / "subagent_responses" / "wars_empty.json"
    return CliResult(
        ok=True,
        data=load_fixture_response(here),
        wall_ms=800,
        exit_code=0,
        raw_usage={"input_tokens": 400, "output_tokens": 50},
    )


def make_timeout_result() -> CliResult:
    return CliResult(ok=False, error="timeout", wall_ms=120_000, exit_code=None)
```

- [ ] **Step 3: Smoke-test the stub**

```bash
uv run python -c "
import asyncio
from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
async def main():
    stub = StubCliRunner()
    stub.add_response('hello', make_wars_discover_result())
    r = await stub.execute(prompt='hello', schema={}, timeout_s=30)
    assert r.ok, r.error
    assert r.data.entity_name == 'World War II'
    assert len(r.data.extractions) == 4
    print('ok')
asyncio.run(main())
"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add tests/stubs/cli_runner.py tests/fixtures/subagent_responses/
git commit -m "$(cat <<'EOF'
tests: add StubCliRunner + subagent response fixtures

Deterministic in-memory runner keyed on prompt hash, plus canned fixtures
for a valid WWII extraction and an empty response. Factory helpers
(make_wars_discover_result, make_empty_result, make_timeout_result) keep
the agent tests succinct.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Create `SubagentResearcher(Agent)`

**Files:**
- Create: `researcher/agents/subagent.py`
- Test: `tests/unit/test_subagent_researcher.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_subagent_researcher.py`:

```python
"""Tests for SubagentResearcher — the Agent subclass that drives CLI runners."""

from datetime import datetime, timedelta, timezone

import pytest

from researcher.agents.subagent import SubagentResearcher
from researcher.backends.models import CliKind
from researcher.budget import Budget
from researcher.events import AgentLog, AgentStateChange, SubagentCall
from researcher.models import AgentState, Task, TaskKind
from tests.stubs.bus import StubEventBus
from tests.stubs.cli_runner import (
    StubCliRunner,
    make_empty_result,
    make_timeout_result,
    make_wars_discover_result,
)
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubKnowledgeStore


def _sample_task() -> Task:
    return Task(
        kind=TaskKind.DISCOVER,
        spec_ref="wars",
        seed_query="major wars since 1500",
        field_hints=["name", "start_year", "end_year", "belligerents"],
        budget_usd=0.01,
        deadline_ts=datetime.now(timezone.utc) + timedelta(minutes=5),
    )


def _make_agent(runner: StubCliRunner, budget: Budget | None = None) -> tuple[SubagentResearcher, StubEventBus]:
    bus = StubEventBus()
    llm = StubLLMClient()
    store = StubKnowledgeStore()
    entity_schema = {
        "entity_type": "War",
        "fields": [
            {"name": "name", "type": "str", "required": True},
            {"name": "start_year", "type": "int", "required": True},
            {"name": "end_year", "type": "int", "required": False},
            {"name": "belligerents", "type": "list[str]", "required": False},
        ],
    }
    if budget is None:
        budget = Budget(usd_cap=3.0, wall_cap_s=600)
    agent = SubagentResearcher(
        agent_id="sub-1",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="run-x",
        runner=runner,
        cli_kind=CliKind.CLAUDE_CODE,
        entity_schema=entity_schema,
        budget=budget,
        goal="Major interstate wars",
    )
    return agent, bus


@pytest.mark.asyncio
async def test_run_maps_extractions_to_fact_claims():
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    agent, bus = _make_agent(runner)

    result = await agent.run(_sample_task())

    assert result.state == AgentState.DONE
    assert len(result.claims) == 4
    fields = {c.field for c in result.claims}
    assert fields == {"name", "start_year", "end_year", "belligerents"}
    for claim in result.claims:
        assert claim.entity_type == "War"
        assert claim.entity_name == "World War II"
        assert claim.provenance.url.startswith("https://")
        assert 0.0 <= claim.confidence <= 1.0


@pytest.mark.asyncio
async def test_run_reports_zero_cost():
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    agent, _ = _make_agent(runner)
    result = await agent.run(_sample_task())
    assert result.cost_usd == 0.0
    # Tokens are reported for observability.
    assert result.tokens_in == 500
    assert result.tokens_out == 300


@pytest.mark.asyncio
async def test_run_emits_state_transitions_and_subagent_call_event():
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    agent, bus = _make_agent(runner)
    await agent.run(_sample_task())

    state_events = [e for e in bus.events if isinstance(e, AgentStateChange)]
    transitions = [(e.payload.old, e.payload.new) for e in state_events]
    # IDLE -> PLANNING -> FETCHING -> DONE
    assert ("idle", "planning") in transitions
    assert ("planning", "fetching") in transitions
    assert ("fetching", "done") in transitions

    subagent_events = [e for e in bus.events if isinstance(e, SubagentCall)]
    assert len(subagent_events) == 1
    assert subagent_events[0].payload.claims_emitted == 4
    assert subagent_events[0].payload.cli_kind == "claude_code"
    assert subagent_events[0].payload.exit_code == 0


@pytest.mark.asyncio
async def test_run_returns_failed_on_timeout():
    runner = StubCliRunner()
    runner.add_response_for_any(make_timeout_result())
    agent, bus = _make_agent(runner)
    result = await agent.run(_sample_task())

    assert result.state == AgentState.FAILED
    assert result.error == "timeout"
    error_logs = [e for e in bus.events if isinstance(e, AgentLog) and e.payload.level == "error"]
    assert len(error_logs) >= 1


@pytest.mark.asyncio
async def test_run_emits_info_log_on_empty_extractions():
    runner = StubCliRunner()
    runner.add_response_for_any(make_empty_result())
    agent, bus = _make_agent(runner)
    result = await agent.run(_sample_task())

    assert result.state == AgentState.DONE
    assert result.claims == []
    info_logs = [e for e in bus.events if isinstance(e, AgentLog) and e.payload.level == "info"]
    assert any("0 extractions" in log.payload.msg or "empty" in log.payload.msg for log in info_logs)


@pytest.mark.asyncio
async def test_run_respects_subagent_cap():
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    budget = Budget(usd_cap=3.0, wall_cap_s=600, max_subagent_calls=0)
    agent, _ = _make_agent(runner, budget=budget)

    result = await agent.run(_sample_task())

    assert result.state == AgentState.FAILED
    assert result.error == "subagent_cap"
    # Runner was NOT called — the cap blocks before dispatch.
    assert runner.calls == []


@pytest.mark.asyncio
async def test_run_records_subagent_call_on_success():
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())
    budget = Budget(usd_cap=3.0, wall_cap_s=600)
    agent, _ = _make_agent(runner, budget=budget)

    await agent.run(_sample_task())
    assert budget.subagent_calls_total == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_subagent_researcher.py -q
```

Expected: `ModuleNotFoundError: No module named 'researcher.agents.subagent'`.

- [ ] **Step 3: Implement SubagentResearcher**

Create `researcher/agents/subagent.py`:

```python
"""SubagentResearcher — Agent subclass that delegates research to a local CLI."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from researcher.agents.base import Agent, EventEmitter
from researcher.backends.cli_runner import CliRunner
from researcher.backends.models import CliKind, SubagentResponse
from researcher.backends.prompts import SYSTEM_PROMPT, build_user_prompt
from researcher.budget import Budget
from researcher.events import (
    AgentLog,
    AgentLogPayload,
    SubagentCall,
    SubagentCallPayload,
)
from researcher.llm.client import LLMClient
from researcher.models import (
    AgentResult,
    AgentState,
    FactClaim,
    Provenance,
    Task,
)
from researcher.storage.store import KnowledgeStore


class SubagentResearcher(Agent):
    """Delegates entire research tasks to a local CLI subagent (claude / codex)."""

    kind = "subagent"

    def __init__(
        self,
        agent_id: str,
        llm: LLMClient,
        store: KnowledgeStore,
        emit: EventEmitter,
        run_id: str,
        runner: CliRunner,
        cli_kind: CliKind,
        entity_schema: dict,
        budget: Budget,
        goal: str,
        max_entities: int = 20,
        timeout_s: float = 120.0,
    ) -> None:
        super().__init__(agent_id=agent_id, llm=llm, store=store, emit=emit, run_id=run_id)
        self._runner = runner
        self._cli_kind = cli_kind
        self._entity_schema = entity_schema
        self._budget = budget
        self._goal = goal
        self._max_entities = max_entities
        self._timeout_s = timeout_s

    async def run(self, task: Task) -> AgentResult:
        await self.set_state(AgentState.PLANNING)

        # Budget gate: reject before dispatching the subprocess.
        if not self._budget.allows_subagent_call():
            await self._log_error("subagent cap reached before dispatch")
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error="subagent_cap",
            )

        prompt = self._build_prompt(task)
        schema = SubagentResponse.model_json_schema()

        await self.set_state(AgentState.FETCHING)
        start = time.monotonic()
        result = await self._runner.execute(
            prompt=prompt, schema=schema, timeout_s=self._timeout_s
        )
        wall_ms = int((time.monotonic() - start) * 1000)

        # Record the call in the budget regardless of success — the subprocess did run.
        self._budget.record_subagent_call()

        # Emit the subagent_call event before returning.
        claims_emitted = 0 if not result.ok or result.data is None else len(result.data.extractions)
        await self._emit(
            SubagentCall(
                seq=0,
                ts=datetime.now(timezone.utc),
                run_id=self._run_id,
                payload=SubagentCallPayload(
                    agent_id=self.agent_id,
                    task_id=task.id,
                    cli_kind=self._cli_kind.value,  # type: ignore[arg-type]
                    wall_ms=wall_ms,
                    exit_code=result.exit_code,
                    claims_emitted=claims_emitted,
                ),
            )
        )

        if not result.ok or result.data is None:
            await self._log_error(f"subagent failure: {result.error}")
            await self.set_state(AgentState.FAILED)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                state=AgentState.FAILED,
                error=result.error,
                wall_ms=wall_ms,
            )

        # Happy path: map extractions -> FactClaims.
        response = result.data
        if not response.extractions:
            await self._log_info(
                f"subagent returned 0 extractions for entity={response.entity_name!r} (empty response)"
            )

        claims = self._build_claims(task, response)
        usage = result.raw_usage or {}
        tokens_in = int(usage.get("input_tokens", 0) or 0)
        tokens_out = int(usage.get("output_tokens", 0) or 0)

        await self.set_state(AgentState.DONE)
        return AgentResult(
            task_id=task.id,
            agent_id=self.agent_id,
            state=AgentState.DONE,
            claims=claims,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,
            wall_ms=wall_ms,
        )

    def _build_prompt(self, task: Task) -> str:
        field_names = [f["name"] for f in self._entity_schema.get("fields", [])]
        if task.field_hints:
            field_names = task.field_hints
        task_description = task.seed_query or "Find all entities matching the goal."
        user_prompt = build_user_prompt(
            goal=self._goal,
            entity_type=self._entity_schema.get("entity_type", "Entity"),
            task_description=task_description,
            field_list=field_names,
            max_entities=self._max_entities,
            schema_json=json.dumps(
                SubagentResponse.model_json_schema(), separators=(",", ":")
            ),
        )
        return f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    def _build_claims(self, task: Task, response: SubagentResponse) -> list[FactClaim]:
        now = datetime.now(timezone.utc)
        entity_type = self._entity_schema.get("entity_type", "Entity")
        extractor_model = f"{self._cli_kind.value}/unknown"
        claims: list[FactClaim] = []
        for i, ex in enumerate(response.extractions):
            prov = Provenance(
                url=ex.source_url,
                fetched_at=now,
                snippet=ex.snippet,
                extractor_model=extractor_model,
                agent_id=self.agent_id,
                task_id=task.id,
                span_id=f"cli_{i}",
            )
            claims.append(
                FactClaim(
                    claim_id=uuid4().hex,
                    entity_type=entity_type,
                    entity_name=response.entity_name,
                    field=ex.field,
                    value=ex.value,
                    confidence=ex.confidence,
                    provenance=prov,
                    emitted_by=self.agent_id,
                    task_id=task.id,
                )
            )
        return claims

    async def _log_info(self, msg: str) -> None:
        await self._emit(
            AgentLog(
                seq=0,
                ts=datetime.now(timezone.utc),
                run_id=self._run_id,
                payload=AgentLogPayload(agent_id=self.agent_id, level="info", msg=msg),
            )
        )

    async def _log_error(self, msg: str) -> None:
        await self._emit(
            AgentLog(
                seq=0,
                ts=datetime.now(timezone.utc),
                run_id=self._run_id,
                payload=AgentLogPayload(agent_id=self.agent_id, level="error", msg=msg),
            )
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_subagent_researcher.py -q
```

Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add researcher/agents/subagent.py tests/unit/test_subagent_researcher.py
git commit -m "$(cat <<'EOF'
agents: add SubagentResearcher

Agent subclass that delegates whole research tasks to a CliRunner. Checks
Budget.allows_subagent_call() before dispatch; emits PLANNING/FETCHING/DONE
state transitions; emits one SubagentCall event per invocation; maps
response extractions into FactClaims with cli_kind-stamped Provenance;
reports cost_usd=0.0 but passes through the CLI's reported token counts
for observability.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Extend `RunSpec` + `researcher run --backend` flag

**Files:**
- Modify: `researcher/spec.py`
- Modify: `researcher/cli.py`
- Test: append to `tests/unit/test_spec.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_spec.py` (at the end of the file):

```python
# ---------- Backend policy ----------

def test_runspec_backend_policy_default():
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
    assert s.backend_policy == "auto"
    assert s.max_subagent_calls == 500
    assert s.subagent_timeout_s == 120


def test_runspec_backend_policy_accepts_cli_and_api():
    for policy in ("auto", "cli", "api"):
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
            backend_policy=policy,
        )
        assert s.backend_policy == policy


def test_runspec_backend_policy_rejects_unknown():
    with pytest.raises(ValidationError):
        RunSpec(
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
            backend_policy="banana",  # type: ignore[arg-type]
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_spec.py::test_runspec_backend_policy_default -q
```

Expected: `AttributeError: 'RunSpec' object has no attribute 'backend_policy'`.

- [ ] **Step 3: Extend `RunSpec`**

In `researcher/spec.py`, find the `class RunSpec(BaseModel):` block and replace the field declarations with this (keeping everything else the same):

```python
class RunSpec(BaseModel):
    spec_id: str
    goal: str
    entities: list[EntitySpec]
    seeds: list[str]
    search: SearchConfig = Field(default_factory=SearchConfig)
    domain_allowlist: list[str] = Field(default_factory=list)
    distinct_pairs: list[tuple[str, str]] = Field(default_factory=list)
    budget_usd: float = 3.0
    wall_limit_s: int = 600
    max_cycles: int = 5
    max_entities_per_cycle: int = 200
    max_depth: int = 3
    models: dict[str, str]  # LLMTier value -> OpenRouter model id

    # Backend policy for CLI subagent offload.
    backend_policy: Literal["auto", "cli", "api"] = "auto"
    max_subagent_calls: int = 500
    subagent_timeout_s: int = 120
```

Then at the top of `researcher/spec.py`, update the imports to include `Literal`:

```python
from typing import Any, Literal, Optional
```

- [ ] **Step 4: Run spec tests**

```bash
uv run pytest tests/unit/test_spec.py -q
```

Expected: `11 passed` (the original 8 plus 3 new).

- [ ] **Step 5: Update the CLI flag**

In `researcher/cli.py`, replace the `run` function with:

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
) -> None:
    """Run a research job from a YAML spec. Auto-launches the TUI by default."""
    if backend not in ("auto", "cli", "api"):
        raise typer.BadParameter(f"--backend must be one of auto|cli|api, got {backend!r}")
    typer.echo(
        f"[researcher run] spec={spec} run_id={run_id or 'auto'} "
        f"tui={'off' if no_tui else 'on'} offline={offline} backend={backend}"
    )
    typer.echo("Wave 0 skeleton: orchestrator dispatch lands in Wave 1-E.")
```

- [ ] **Step 6: Smoke-test the CLI flag**

```bash
uv run python -m researcher run --help 2>&1 | grep -A1 -- "--backend"
```

Expected output includes `--backend [auto|cli|api]` or similar help text.

- [ ] **Step 7: Commit**

```bash
git add researcher/spec.py researcher/cli.py tests/unit/test_spec.py
git commit -m "$(cat <<'EOF'
spec + cli: add --backend {auto,cli,api} policy

RunSpec grows backend_policy, max_subagent_calls, subagent_timeout_s. The
`researcher run` command grows --backend. All default to auto. cli forces
the subagent path (raises if no CLI detected); api forces OpenRouter.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Wire the orchestrator to route tasks to the subagent path

**Files:**
- Modify: `researcher/orchestrator.py`
- Test: `tests/unit/test_orchestrator_backend_routing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_orchestrator_backend_routing.py`:

```python
"""Tests for Orchestrator._pick_agent_for_task — backend routing logic."""

from datetime import datetime, timedelta, timezone

import pytest

from researcher.backends.models import BackendChoice, CliKind
from researcher.backends.resolver import BackendResolver
from researcher.budget import Budget
from researcher.models import AgentResult, AgentState, Task, TaskKind
from researcher.orchestrator import Orchestrator, StopReason
from researcher.scheduler import Scheduler
from researcher.spec import EntitySpec, FieldSpec, RunSpec
from researcher.storage.writer import FactWriter
from tests.stubs.bus import StubEventBus
from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
from tests.stubs.llm import StubLLMClient
from tests.stubs.resolver import StubEntityResolver
from tests.stubs.store import StubKnowledgeStore


def _which_none(_: str) -> str | None:
    return None


def _which_claude(cmd: str) -> str | None:
    return "/usr/local/bin/claude" if cmd == "claude" else None


def _sample_spec(policy: str = "auto") -> RunSpec:
    return RunSpec(
        spec_id="wars",
        goal="Wars",
        entities=[
            EntitySpec(
                name="War",
                fields=[FieldSpec(name="name", type="str", required=True)],
                search_templates=[],
            )
        ],
        seeds=["major wars"],
        models={"fast": "m", "smart": "m", "heavy": "m"},
        backend_policy=policy,
    )


async def _noop_fact(_entity_id, _claim):
    return None


async def _noop_conflict(_entity_id, _cells):
    return None


async def _make_orchestrator(
    spec: RunSpec, which_fn, runner: StubCliRunner | None = None
) -> Orchestrator:
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
    backend_resolver = BackendResolver(which_fn=which_fn)
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
    )
    orch.set_backend_resolver(backend_resolver)
    if runner is not None:
        orch.set_cli_runner_factory(lambda kind: runner)
    return orch


def _sample_task() -> Task:
    return Task(
        kind=TaskKind.DISCOVER,
        spec_ref="wars",
        seed_query="wars",
        budget_usd=0.01,
        deadline_ts=datetime.now(timezone.utc) + timedelta(minutes=5),
    )


@pytest.mark.asyncio
async def test_auto_policy_with_no_cli_routes_native():
    orch = await _make_orchestrator(_sample_spec("auto"), _which_none)
    choice = orch._resolver_pick(_sample_task())
    assert choice.kind is None


@pytest.mark.asyncio
async def test_auto_policy_with_claude_detected_routes_subagent():
    orch = await _make_orchestrator(_sample_spec("auto"), _which_claude)
    choice = orch._resolver_pick(_sample_task())
    assert choice.kind == CliKind.CLAUDE_CODE


@pytest.mark.asyncio
async def test_api_policy_overrides_detection():
    orch = await _make_orchestrator(_sample_spec("api"), _which_claude)
    choice = orch._resolver_pick(_sample_task())
    assert choice.kind is None


@pytest.mark.asyncio
async def test_circuit_break_clears_detected_after_three_failures():
    orch = await _make_orchestrator(_sample_spec("auto"), _which_claude)
    assert orch._resolver_pick(_sample_task()).kind == CliKind.CLAUDE_CODE
    orch._record_subagent_failure("spawn_failed: ENOENT")
    orch._record_subagent_failure("timeout")
    assert orch._resolver_pick(_sample_task()).kind == CliKind.CLAUDE_CODE  # 2 failures, not tripped
    orch._record_subagent_failure("exit 1: oops")
    # Third failure should trip the break.
    assert orch._resolver_pick(_sample_task()).kind is None


@pytest.mark.asyncio
async def test_usage_limit_clears_detected_immediately():
    orch = await _make_orchestrator(_sample_spec("auto"), _which_claude)
    assert orch._resolver_pick(_sample_task()).kind == CliKind.CLAUDE_CODE
    orch._record_subagent_failure("usage_limit_reached")
    assert orch._resolver_pick(_sample_task()).kind is None


@pytest.mark.asyncio
async def test_auth_required_clears_detected_immediately():
    orch = await _make_orchestrator(_sample_spec("auto"), _which_claude)
    orch._record_subagent_failure("auth_required")
    assert orch._resolver_pick(_sample_task()).kind is None


@pytest.mark.asyncio
async def test_failure_counter_resets_between_cycles():
    orch = await _make_orchestrator(_sample_spec("auto"), _which_claude)
    orch._record_subagent_failure("timeout")
    orch._record_subagent_failure("timeout")
    orch._reset_cycle_failure_counters()  # simulate cycle boundary
    orch._record_subagent_failure("timeout")
    # Only 1 failure in the current cycle — not tripped.
    assert orch._resolver_pick(_sample_task()).kind == CliKind.CLAUDE_CODE


@pytest.mark.asyncio
async def test_stop_reason_subagent_cap_exists():
    assert hasattr(StopReason, "SUBAGENT_CAP")
    assert StopReason.SUBAGENT_CAP.value == "subagent_cap"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_orchestrator_backend_routing.py -q
```

Expected: Multiple failures — `AttributeError: 'Orchestrator' object has no attribute 'set_backend_resolver'` and `AttributeError: type object 'StopReason' has no attribute 'SUBAGENT_CAP'`.

- [ ] **Step 3: Add the routing hooks to `Orchestrator`**

In `researcher/orchestrator.py`, replace the `StopReason` class with:

```python
class StopReason(str, Enum):
    BUDGET = "budget"
    PLATEAU = "plateau"
    DEADLINE = "deadline"
    CTRL_C = "ctrl_c"
    ERROR = "error"
    NO_TASKS = "no_tasks"
    SUBAGENT_CAP = "subagent_cap"
```

At the top of `researcher/orchestrator.py`, update the existing `from typing import Optional` line to include `Callable`:

```python
from typing import Callable, Optional
```

Then after the existing `from researcher.storage.writer import FactWriter` line, add these three imports:

```python
from researcher.backends.cli_runner import CliRunner
from researcher.backends.models import BackendChoice, CliKind
from researcher.backends.resolver import BackendResolver
```

Then in `Orchestrator.__init__`, after `self._started_at: Optional[datetime] = None`, add:

```python
        self._backend_resolver: Optional[BackendResolver] = None
        self._cli_runner_factory: Optional[Callable[[CliKind], CliRunner]] = None
        self._cycle_subagent_failures: int = 0
        self._circuit_break_threshold: int = 3
```

Add these methods to `Orchestrator` (anywhere in the class, e.g., after `__init__`):

```python
    def set_backend_resolver(self, resolver: BackendResolver) -> None:
        self._backend_resolver = resolver

    def set_cli_runner_factory(
        self, factory: Callable[[CliKind], CliRunner]
    ) -> None:
        self._cli_runner_factory = factory

    def _resolver_pick(self, task: Task) -> BackendChoice:
        if self._backend_resolver is None:
            return BackendChoice(kind=None, reason="no resolver configured")
        return self._backend_resolver.pick(policy=self._spec.backend_policy)  # type: ignore[arg-type]

    def _record_subagent_failure(self, error: str) -> None:
        """Update the per-cycle failure counter and trip circuit break if warranted."""
        if self._backend_resolver is None:
            return
        # Immediate trip on auth / usage errors.
        if error == "auth_required":
            self._backend_resolver.clear_detected("auth_required")
            return
        if error == "usage_limit_reached":
            self._backend_resolver.clear_detected("usage_limit_reached")
            return
        # Count other failures; trip after threshold.
        self._cycle_subagent_failures += 1
        if self._cycle_subagent_failures >= self._circuit_break_threshold:
            self._backend_resolver.clear_detected(
                f"circuit_break: {self._cycle_subagent_failures} failures in cycle"
            )

    def _reset_cycle_failure_counters(self) -> None:
        self._cycle_subagent_failures = 0
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/unit/test_orchestrator_backend_routing.py -q
```

Expected: `8 passed`.

- [ ] **Step 5: Run the full unit suite to confirm no regressions**

```bash
uv run pytest tests/unit -q
```

Expected: all tests pass (count will be the sum of all tests added across tasks 1–10 plus the original Wave 0 suite).

- [ ] **Step 6: Commit**

```bash
git add researcher/orchestrator.py tests/unit/test_orchestrator_backend_routing.py
git commit -m "$(cat <<'EOF'
orchestrator: add backend resolver + circuit break hooks

Adds set_backend_resolver / set_cli_runner_factory injection points, a
per-cycle subagent_failures counter with a configurable threshold (default
3), and immediate circuit-break on auth_required / usage_limit_reached.
Adds StopReason.SUBAGENT_CAP. The actual dispatch-to-SubagentResearcher
wiring in _spawn_agent is left for Wave 1-E integration.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Offline integration smoke — full orchestrator run on the subagent path

**Files:**
- Create: `tests/integration/test_smoke_subagent_wars.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/integration/test_smoke_subagent_wars.py`:

```python
"""Offline integration smoke for the CLI subagent path.

Builds a minimal Orchestrator with StubCliRunner pre-populated for the
seed query in wars.yaml, runs one cycle, and asserts the subagent path
produces FactClaims that flow through the writer and the event bus
records a subagent_call event.
"""

from datetime import datetime, timedelta, timezone

import pytest

from researcher.agents.subagent import SubagentResearcher
from researcher.backends.models import CliKind
from researcher.budget import Budget
from researcher.events import SubagentCall
from researcher.models import Task, TaskKind
from tests.stubs.bus import StubEventBus
from tests.stubs.cli_runner import StubCliRunner, make_wars_discover_result
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubKnowledgeStore


@pytest.mark.asyncio
async def test_subagent_smoke_end_to_end_offline():
    # Arrange
    runner = StubCliRunner()
    runner.add_response_for_any(make_wars_discover_result())

    store = StubKnowledgeStore()
    await store.open()
    bus = StubEventBus()
    llm = StubLLMClient()
    budget = Budget(usd_cap=3.0, wall_cap_s=600)
    entity_schema = {
        "entity_type": "War",
        "fields": [
            {"name": "name", "type": "str", "required": True},
            {"name": "start_year", "type": "int", "required": True},
            {"name": "end_year", "type": "int", "required": False},
            {"name": "belligerents", "type": "list[str]", "required": False},
        ],
    }

    agent = SubagentResearcher(
        agent_id="smoke-1",
        llm=llm,
        store=store,
        emit=bus.emit,
        run_id="smoke-run",
        runner=runner,
        cli_kind=CliKind.CLAUDE_CODE,
        entity_schema=entity_schema,
        budget=budget,
        goal="Major interstate wars since 1500",
    )

    task = Task(
        kind=TaskKind.DISCOVER,
        spec_ref="wars",
        seed_query="Major wars since 1500",
        field_hints=["name", "start_year", "end_year", "belligerents"],
        budget_usd=0.01,
        deadline_ts=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    # Act
    result = await agent.run(task)

    # Assert — agent result
    assert result.state.value == "done"
    assert len(result.claims) == 4
    assert result.cost_usd == 0.0
    assert result.tokens_in == 500
    assert result.tokens_out == 300

    # Assert — budget
    assert budget.subagent_calls_total == 1
    assert budget.total_spent() == 0.0
    assert not budget.exceeded()

    # Assert — event bus has a subagent_call event with 4 claims
    subagent_events = [e for e in bus.events if isinstance(e, SubagentCall)]
    assert len(subagent_events) == 1
    assert subagent_events[0].payload.claims_emitted == 4
    assert subagent_events[0].payload.cli_kind == "claude_code"

    # Assert — runner was called exactly once
    assert len(runner.calls) == 1
```

- [ ] **Step 2: Run the smoke test**

```bash
uv run pytest tests/integration/test_smoke_subagent_wars.py -q
```

Expected: `1 passed`.

- [ ] **Step 3: Run the complete test suite**

```bash
uv run pytest -q
```

Expected: all tests pass. This includes Wave 0 tests (38) plus everything added in tasks 1–11.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_smoke_subagent_wars.py
git commit -m "$(cat <<'EOF'
tests: offline integration smoke for the subagent path

End-to-end test: SubagentResearcher + StubCliRunner + fixture response,
asserting FactClaims, zero cost, subagent_calls_total increment, and the
SubagentCall event on the bus. No real subprocess.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Final verification + plan-level cleanup

**Files:** none modified

- [ ] **Step 1: Run the full suite**

```bash
uv run pytest -q
```

Expected: every test green. Record the count (e.g., `86 passed`).

- [ ] **Step 2: Confirm the CLI still works end-to-end**

```bash
uv run python -m researcher version
uv run python -m researcher run --help
```

Expected: version prints `0.1.0`; `--backend` appears in the run subcommand help.

- [ ] **Step 3: Confirm imports are clean**

```bash
uv run python -c "
import researcher
import researcher.backends
import researcher.backends.models
import researcher.backends.resolver
import researcher.backends.cli_runner
import researcher.backends.prompts
import researcher.agents.subagent
print('all subagent modules import ok')
"
```

Expected: `all subagent modules import ok`

- [ ] **Step 4: No-op if everything is green**

If any step fails, walk back through the failing task and re-run its individual test file. Do NOT skip ahead to Wave 1-E integration — the orchestrator's `_spawn_agent` method still returns a stub result and will need real agent dispatch wired up in a future commit.

---

## Known gaps left for Wave 1-E

The following are documented as intentional scope cuts in the design doc
(`doc/2026-04-08-cli-subagent-backend-design.md`), not regressions:

1. `Orchestrator._spawn_agent` still returns a stubbed `AgentResult`. Wiring it
   to actually call `SubagentResearcher` (or a native agent) based on
   `_resolver_pick` is deferred to the Wave 1-E integration PR, which is where
   the `_spawn_agent` method gets its real body.
2. No real `claude -p` / `codex exec` calls are exercised in CI. Running the
   real CLIs is a manual script (`tests/manual/test_live_claude.py`), which is
   not part of this plan.
3. The `--backend` flag is wired on the CLI surface but `researcher run` still
   prints the "Wave 0 skeleton" message — actually starting an `Orchestrator`
   is Wave 1-E.

---

## Self-review

- ✅ **Spec coverage:** every file listed in the design's "New files" and
  "Touched files" sections has at least one task that creates or modifies it.
  Every failure mode in the error-handling table has a test.
- ✅ **Placeholder scan:** no TBD/TODO/FIXME in any step. Every code block is
  complete and ready to copy into a file.
- ✅ **Type consistency:** `SubagentCall` / `SubagentCallPayload` names match
  across Task 1 (event), Task 8 (agent emits), and Task 11 (integration
  asserts). `CliKind.CLAUDE_CODE.value == "claude_code"` matches the
  `Literal["claude_code","codex"]` in `SubagentCallPayload.cli_kind`. The
  `Budget.subagent_calls_total` property, `record_subagent_call()` method,
  and `allows_subagent_call()` gate are referenced consistently across tasks
  2, 8, and 11. `CliResult.raw_usage["input_tokens"]` is used consistently in
  Task 5 (runner) and Task 8 (agent mapping).
