# CLI Subagent Backend — Design

**Status:** draft for review
**Date:** 2026-04-08
**Author:** Aaron (+ Claude)
**Follows:** Wave 0 contracts commit `a0d3ed6`

## Context

Wave 0 shipped the interface skeleton for `researcher`. The default execution
path uses an OpenRouter-backed `LLMClient` plus our own `fetch/`, `search/`
(Tavily/Brave/Serper), and `extract/` modules. Users on a Claude Max
subscription or ChatGPT Plus/Pro plan already pay a flat monthly fee that
includes generous Claude Code CLI / Codex CLI usage. Routing research tasks
through those CLIs instead of paid APIs saves both LLM credits **and** search
credits, because the CLI does its own web search and fetching internally.

This doc specifies how to add whole-task delegation to local CLI subagents as a
parallel execution path that sits alongside the native API path, without
disturbing the Wave 0 contracts.

## Goals

1. Let users offload entire research tasks (discover, expand, verify, enrich)
   to a local `claude -p` or `codex exec` invocation with web tools enabled.
2. Save Tavily/Brave/Serper search credits on the CLI path by bypassing our
   `search/` + `fetch/` + `extract/` layers entirely.
3. Detect available CLIs automatically at startup; no manual config required in
   the common case.
4. Gracefully degrade to the native API path when the CLI hits its monthly
   quota, loses auth, or starts failing.
5. Preserve every Wave 0 invariant: typed `FactClaim`s, serial reduce in the
   `FactWriter`, event bus monotonicity, budget discipline.

## Non-goals

- A generic "LLM execution backend" abstraction above `LLMClient`. The drop-in
  raw-LLM swap was considered (approach C) and rejected because it doesn't
  save search credits.
- Cross-run persistence of subagent responses. In-memory per-run cache only.
- Prompt-injection hardening. Same risk exists on the native path; both CLIs
  run with `--bare` / `--sandbox read-only` and a tools allowlist of
  `WebSearch,WebFetch` only, so a successful injection cannot modify the host.
- Real subprocess execution in CI. Live runs are a manual script.
- Simulating API cost from CLI token counts. Subagent calls report `$0.00`.

## Decisions locked from brainstorming

| Dimension | Decision | Rationale |
|---|---|---|
| Offload granularity | Whole-task delegation | Saves both LLM + search credits |
| CLIs supported | Claude Code + Codex (autodetect) | User has both installed; Claude Code preferred |
| Activation | Auto if CLI detected; fall through to API otherwise | No config needed in common case |
| Cost accounting | $0 + separate `subagent_calls_total` counter | CLIs are on flat-fee subscriptions |
| Architecture | `SubagentResearcher(Agent)` subclass | Reuses existing Agent lifecycle; no new layers |
| Backend selection policy | `--backend {auto,cli,api}`, default `auto` | Explicit escape hatches both directions |

## Architecture

At startup the orchestrator constructs a `BackendResolver` that probes `$PATH`
for `claude` and `codex` via `shutil.which`. The resolver records a list of
available `CliKind` values and exposes `pick(task, policy) -> BackendChoice`.

The orchestrator asks the resolver for each task at spawn time. If the resolver
returns a CLI kind, the orchestrator instantiates `SubagentResearcher` with the
matching runner; otherwise it instantiates a native agent (`DiscoverAgent`,
`ExpandAgent`, etc. — Wave 1-D). `SubagentResearcher` subclasses `Agent`
(Wave 0 `researcher/agents/base.py`) and satisfies the existing ABC contract
without modification.

```
+-----------------+          +---------------------+
| Orchestrator    | spawn()  | BackendResolver     |
|  _spawn_agent() |--------->|  pick(task, policy) |
+-----------------+          +---------------------+
         |                            |
         |  CliKind or None           |
         v                            v
+-----------------+          +---------------------+
| SubagentRsrchr  | or       | DiscoverAgent (etc) |
|  (Agent)        |          |  (Agent)            |
+-----------------+          +---------------------+
         |                            |
         |  runner.execute()          |  llm.complete_structured()
         v                            v
+-----------------+          +---------------------+
| claude -p /     |          | OpenRouter + Tavily |
| codex exec      |          | + fetch + extract   |
+-----------------+          +---------------------+
         |                            |
         |  FactClaim[]               |  FactClaim[]
         +------------+---------------+
                      v
               FactWriter (serial reduce, unchanged)
```

Both paths emit identical `FactClaim` objects into the same `FactWriter` queue.
The reduce step is indistinguishable by origin.

## Components

### New files

- `researcher/backends/__init__.py`
- `researcher/backends/resolver.py` — `BackendResolver`, `BackendChoice`,
  `BackendUnavailableError`. Constructor takes an injectable `which_fn` for
  testability. `pick(task, policy)` returns a `BackendChoice(kind, reason)`
  where `kind` is a `CliKind` enum value or `None` (meaning native path).
- `researcher/backends/cli_runner.py` — `CliRunner` Protocol with `execute`
  method plus two concrete impls: `ClaudeCodeRunner`, `CodexRunner`. Both use
  `asyncio.create_subprocess_exec` (never `shell=True`, never string
  interpolation into argv). Prompts go via `stdin` to avoid argv length and
  quoting issues. Each runner holds an in-memory per-run response cache keyed
  on `sha256(prompt_body + argv_tuple + schema_json)`.
- `researcher/backends/models.py` — Pydantic models: `CliKind` enum
  (`CLAUDE_CODE`, `CODEX`), `SubagentResponse` (the JSON shape the CLI is
  required to return: `entity_name`, `extractions: list[Extraction]`,
  `diagnostics: str`), `Extraction` (`field`, `value`, `source_url`, `snippet`,
  `confidence`), `CliResult` (`ok: bool`, `data: Optional[SubagentResponse]`,
  `error: Optional[str]`, `wall_ms: int`, `exit_code: Optional[int]`,
  `raw_usage: Optional[dict]`), `BackendChoice` (`kind: Optional[CliKind]`,
  `reason: str`).
- `researcher/agents/subagent.py` — `SubagentResearcher(Agent)`. Constructor
  takes `runner: CliRunner`, `entity_schema: dict`, `budget: Budget`. Its
  `run(task)` builds the prompt, checks `budget.allows_subagent_call()`,
  dispatches to the runner, maps `SubagentResponse.extractions` to
  `FactClaim`s, emits `subagent_call` event, returns `AgentResult(cost_usd=0)`.
- `tests/stubs/cli_runner.py` — `StubCliRunner` backed by a fixture dict.
  `calls: list[dict]` records every invocation for assertions. Real subprocess
  execution is never exercised in unit or offline integration tests.
- `tests/fixtures/subagent_responses/` — JSON fixtures: `wars_discover.json`,
  `wars_empty.json`, `wars_malformed.json`.

### Touched files

- `researcher/budget.py` — add `subagent_calls_total: int = 0`,
  `max_subagent_calls: int = 500`, `record_subagent_call()`,
  `allows_subagent_call() -> bool`. No change to existing USD path; budget
  stop on USD does not fire on a pure-CLI run (cost stays at zero).
- `researcher/events.py` — add `SubagentCall` event to the discriminated union
  with payload `{agent_id, task_id, cli_kind, wall_ms, exit_code,
  claims_emitted}`. Extend `parse_event` accordingly.
- `researcher/spec.py` — add `RunSpec.backend_policy: Literal["auto","cli","api"] = "auto"`,
  `RunSpec.max_subagent_calls: int = 500`,
  `RunSpec.subagent_timeout_s: int = 120`.
- `researcher/cli.py` — add `--backend {auto,cli,api}` option to `researcher run`.
  Passes through to `RunSpec.backend_policy` at load time.
- `researcher/orchestrator.py` — instantiate `BackendResolver` once at run start,
  store as `self._resolver`. `_spawn_agent` calls `resolver.pick(task, policy)`
  and dispatches to either `SubagentResearcher` or the native agent classes.
  Add a new `StopReason.SUBAGENT_CAP`. Track a per-cycle `spawn_failed` count;
  at 3 failures, call `self._resolver.force_clear()` (a test-only hook renamed
  `_clear_detected_internal` with a documented use-at-your-own-risk contract)
  to disable the CLI path for the rest of the run.

### Prompts

Stored as a single constant in `researcher/backends/prompts.py`:

```
SYSTEM: You are a fact-extraction research agent. You have access to WebSearch
and WebFetch tools. For each candidate entity matching the user's request,
return one Extraction per field, citing the URL you extracted it from. Confidence
is 0.0 to 1.0. Return JSON matching the schema exactly — no prose.

USER: Goal: {spec_goal}
Entity type: {entity_type}
Task: {seed_query or expand_hint}
Fields required: {field_list}
Schema: {entity_schema_json}
Return up to {max_entities} entities.
```

The `--json-schema` flag on Claude Code and `--output-schema <FILE>` on Codex
enforce the response shape server-side. Our Pydantic validation is a belt +
braces second check on stdout.

## Data flow

1. **Seed.** Scheduler emits `DISCOVER` tasks from `RunSpec.seeds`.
2. **Spawn.** Orchestrator acquires the agents semaphore, asks resolver, gets
   `BackendChoice(kind=CLAUDE_CODE, reason="auto: claude detected")`,
   instantiates `SubagentResearcher`.
3. **Prompt build.** Agent emits `PLANNING`, assembles the prompt from the
   template above.
4. **Subprocess dispatch.** Agent emits `FETCHING`. Runner builds argv:
   `["claude", "-p", "--print", "--output-format", "json", "--json-schema",
   <inline_schema>, "--allowedTools", "WebSearch", "WebFetch", "--bare",
   "--model", "sonnet", "--append-system-prompt", <persona>]`. Writes prompt
   to stdin, closes stdin, `await asyncio.wait_for(proc.communicate(),
   timeout=subagent_timeout_s)`.
5. **Parse.** Runner parses the Claude Code JSON envelope
   `{"type":"result","result":"<json>","usage":{...},"total_cost_usd":0.0}`,
   then parses `result` into a `SubagentResponse`.
6. **Emit `subagent_call` event.** Orchestrator calls
   `budget.record_subagent_call()`.
7. **Map to FactClaims.** Agent converts each `Extraction` into a `FactClaim`
   with `Provenance.url=ex.source_url`, `extractor_model="claude-code/sonnet"`,
   `span_id=f"cli_{i}"`. Returns `AgentResult(claims=..., cost_usd=0.0)`.
8. **Reduce.** Claims flow into the existing `FactWriter` serial drain loop.
   Indistinguishable from native-agent claims downstream.
9. **Cycle end + stop checks.** Standard Wave 0 path. New stop: `SUBAGENT_CAP`
   when `budget.subagent_calls_total >= max_subagent_calls`.

### Concurrency

Subagent dispatch is bounded by the existing `max_parallel_agents` semaphore in
the orchestrator. No new semaphore. Our `PolitenessLimiter` does not apply —
the CLI does its own rate limiting internally. The politeness-before-LLM-sem
discipline from Wave 0 is therefore irrelevant on the CLI path (no separate
fetch step), and no deadlock risk is introduced.

### In-run cache

Each runner instance holds a `dict[str, CliResult]` keyed on
`sha256(prompt_body + argv_tuple + schema_json)`. Cache hits return the stored
result immediately and skip the subprocess call entirely. Cache lifetime is the
`CliRunner` instance — i.e., the run. No disk persistence. Cross-run caching is
deferred to v1.1.

## Error handling

All failures become typed `CliResult(ok=False, error=<code>, ...)`. The runner
never raises on expected failures; it returns. `SubagentResearcher.run` converts
a failed `CliResult` into `AgentResult(state=FAILED, error=<code>)` plus an
`agent_log(level=warn|error)` event. The orchestrator's existing
`asyncio.gather(return_exceptions=True)` never sees a raw exception on the
happy path.

**No intra-agent retries on the CLI path.** Every failure is a single shot:
`SubagentResearcher` returns `FAILED` on the first problem, full stop. There is
no per-task fallback from CLI to native within a single agent `run`. Recovery
happens at the orchestrator level via the circuit break (below), which clears
the CLI path for all subsequent tasks in the run. The only exception is
malformed JSON from Claude Code / Codex, which is handled with **one
in-runner retry** — the runner re-prompts with a stricter "return ONLY JSON
matching the schema, no prose" nudge appended before giving up. This retry is
bounded inside `ClaudeCodeRunner.execute` / `CodexRunner.execute` and counts
as a single CLI call for budget purposes.

| Failure | Detection | Response |
|---|---|---|
| CLI not installed | `shutil.which` at startup | `policy=cli` raises; `policy=auto` silently falls through |
| Spawn fails | `FileNotFoundError`/`PermissionError` from `create_subprocess_exec` | `FAILED`, per-cycle `spawn_failed` counter increments |
| Timeout | `asyncio.wait_for` raises | `proc.kill()` + `await proc.wait()`, `error="timeout"`, `FAILED` |
| Non-zero exit | `returncode != 0` | Capture stderr tail (2 KB), `error=f"exit {code}: {stderr_tail}"`, `FAILED` |
| Auth not configured | Pattern match `"Please run: claude auth"` / `"Not signed in"` | `error="auth_required"`, circuit break: globally clear `detected` |
| Usage limit reached | Pattern match `"rate_limit"` / `"quota"` / `"usage_limit"` in stderr or envelope | `error="usage_limit_reached"`, circuit break: globally clear `detected`, emit `BudgetWarning`-shaped event with `message` payload |
| Malformed JSON (1st attempt) | `JSONDecodeError` or Pydantic `ValidationError` | Dump raw stdout to `runs/<run_id>/subagent_errors/<ts>.txt`, retry **inside the runner** with stricter nudge |
| Malformed JSON (2nd attempt) | Same after retry | `error=f"parse_failed: {exc}"`, `FAILED` |
| Empty extractions | Valid response, zero `extractions` | `state=DONE`, `claims=[]`, info log |
| Ctrl-C | `CancelledError` | `proc.kill()` + re-raise |
| Subagent cap reached | `budget.allows_subagent_call() is False` | `state=FAILED`, `error="subagent_cap"`, dispatch skipped |

**Circuit break.** Either (a) 3 `spawn_failed`/`timeout`/`exit` failures within
the same cycle, OR (b) any `auth_required` event, OR (c) any
`usage_limit_reached` event clears `BackendResolver.detected` for the remainder
of the run. The per-cycle failure counter resets at each `cycle_start`; the
`detected` clear is permanent for the run once tripped. All subsequent `pick`
calls return `kind=None`; tasks route native. One `BudgetWarning`-shaped event
announces the switch so the TUI can show a banner.

**Budget interaction.** `BudgetExceededError` path untouched; subagent calls
report `$0.00`. The new `max_subagent_calls` gate is a separate hard stop with
its own `StopReason.SUBAGENT_CAP`.

## Testing

All tests are offline and deterministic. Real `claude`/`codex` subprocesses are
never spawned in CI.

### Unit tests

| File | Coverage |
|---|---|
| `test_backend_resolver.py` | prefer Claude Code; `policy=api` forces native; `policy=cli` with none detected raises; `policy=auto` with none detected falls through; `which_fn` injection |
| `test_cli_runner_claude.py` | argv construction, prompt via stdin not argv, envelope parse, timeout + kill, non-zero exit, auth pattern, usage-limit pattern, malformed JSON retries once then fails, cancellation propagates + kills child |
| `test_cli_runner_codex.py` | argv construction (`exec`, `--json`, `--output-schema`, `--output-last-message`, `--sandbox read-only`, `--skip-git-repo-check`), tempfile schema written and cleaned up, same error matrix as Claude |
| `test_subagent_researcher.py` | maps extractions to FactClaims; zero cost; state transitions; emits `subagent_call`; timeout → FAILED; empty extractions → DONE with info log; cap gate blocks dispatch |
| `test_budget_subagent_calls.py` | counter increments; cap enforced; USD budget unaffected; `exceeded()` unaffected |
| `test_events_subagent_call.py` | payload roundtrip; discriminated union dispatch |
| `test_orchestrator_backend_routing.py` | auto + detected → subagent; auto + none → native; circuit break after 3 spawn_failed; usage_limit clears globally |

### Integration test

`test_smoke_subagent_wars.py` — full orchestrator run on `specs/wars.yaml` with
`backend_policy=cli` and `StubCliRunner` pre-populated with fixture responses.
Asserts `entities_total >= 5`, `run_summary.cost_usd == 0.0`,
`subagent_calls_total > 0`, `events.jsonl` contains `subagent_call` events,
final event is `run_complete(reason="plateau")`. Gated behind
`RESEARCHER_OFFLINE=1`.

### Manual test

`tests/manual/test_live_claude.py` — exercises real `claude -p` against a
minimal spec. `@pytest.mark.manual`, excluded from CI. Run locally when we
want to verify that CLI flags haven't drifted between Claude Code releases.

### TDD order

1. `SubagentCall` event (`test_events_subagent_call.py` first)
2. Budget extensions (`test_budget_subagent_calls.py` first)
3. `BackendResolver` (`test_backend_resolver.py` first)
4. `ClaudeCodeRunner` (`test_cli_runner_claude.py` first)
5. `CodexRunner` (`test_cli_runner_codex.py` first)
6. `SubagentResearcher` (`test_subagent_researcher.py` first)
7. Orchestrator wiring (`test_orchestrator_backend_routing.py` first)
8. Offline integration smoke

Each step RED → GREEN → REFACTOR before the next.

## Scope cuts for v1

- No Thompson-sampling bandit over prompt variants; one fixed prompt per task
  kind. Re-entry: after 50+ runs with logged success rates.
- No cross-run subagent response cache. Re-entry: if the same spec is re-run
  frequently and in-run cache hit rate is measured at >20%.
- No prompt-injection defense. Sandbox + tools allowlist is the only
  mitigation.
- No live-CLI CI job. Manual script only.
- No CLI-side cost telemetry. Reported tokens are informational, not billed.
- No dynamic handoff from CLI to native mid-task. Subagent cap is a hard stop;
  cycle-level circuit break is the escape hatch.
- No multi-model routing (e.g., "use Sonnet for discover, Opus for verify").
  Single model per runner, configurable via `RunSpec.models`.

## Open risks

- **CLI flag drift.** Claude Code and Codex ship frequently. The manual live
  test and a periodic `claude --help` snapshot comparison are the only guards.
  Mitigation: fail loudly on unexpected stderr patterns rather than silently
  succeed.
- **Subprocess leak under extreme load.** If the orchestrator is killed with
  -9, running `claude`/`codex` children are orphaned. Python's `atexit` doesn't
  fire on SIGKILL. Acceptable for v1; documented.
- **Tavily still gets called on the native fall-through path.** If the CLI
  circuit-breaks mid-run, remaining tasks go native and incur Tavily cost. The
  Budget's USD cap still bounds this.
- **`RunSpec.models` semantics on the CLI path.** We honor `models["smart"]` as
  the default Claude Code `--model` on a per-runner basis. If the user set
  `models={"smart": "openrouter/hermes-3-70b"}` expecting OpenRouter, that
  value is meaningless on the CLI path. We'll document the mapping and fall
  back to `sonnet` if the model id doesn't parse as a Claude alias.

## Success criteria

1. A fresh checkout, `uv sync`, `uv run pytest tests/unit` stays green with all
   the new tests added.
2. `uv run pytest tests/integration/test_smoke_subagent_wars.py` passes with
   `RESEARCHER_OFFLINE=1`.
3. Running `uv run researcher run specs/wars.yaml` on a laptop with
   `claude` installed + authenticated spawns at least one real `subagent_call`
   event (verified by tailing `runs/<id>/events.jsonl`), produces ≥ 5 entities
   from fixture-free live data, and reports `run_summary.cost_usd == 0.0`.
4. Removing `claude` from `$PATH` and re-running the same command falls through
   to the native path without error and reports a nonzero `cost_usd`.
