# Current Product Surface

**Status:** current as of 2026-04-28.

This document is the current reference for what the project exposes today. The
older documents in this directory are historical design notes and implementation
plans; keep them for rationale, but prefer this file and the README for current
commands and behavior.

## Development Entrypoint

Use the module form in development:

```bash
uv run python -m researcher --help
```

The console script `uv run researcher` is still installed by `pyproject.toml`,
but the module form is the guaranteed command on macOS because Python can skip
editable-install `.pth` files when the file has the hidden flag.

## Command Surface

Top-level commands:

| Command | Purpose |
|---|---|
| `doctor` | Check Python, uv, Node/TUI, CLI backends, OpenRouter, search, source policy, source sets, offline fixtures, and Obsidian setup before a run. |
| `run` | Execute a YAML RunSpec, with native API, CLI subagent, or deterministic offline mode. |
| `watch` | Attach the TUI to a live run or replay a completed run. Late live attaches replay existing `events.jsonl` before tailing the socket. |
| `inspect` | Summarize a completed run: entity counts, type counts, fill rate, conflicts, cost, wall time, stop reason, DB path, and latest summary. |
| `accept` | Check a run against v1-style metrics: entity count, field fill, cost, and wall time. |
| `resume` | Continue a previously recorded run using its metadata and checkpoint state. |
| `stop` | Request graceful stop by writing `stop.requested`; the orchestrator drains and writes a run summary. |
| `export` | Export completed entities as CSV, JSON, Markdown, or Obsidian-compatible Markdown. |
| `evidence` | Emit a citation/provenance-backed evidence matrix as CSV, Markdown, or JSON. |
| `report` | Render a cited research report as Markdown, HTML, or JSON with analyst, systematic, or brief templates. |
| `graph` | Run read-only SQL against the completed run's relation tables. |
| `migrate` | List DuckDB schema migrations recorded for a run. |
| `mcp` | Start the JSON-RPC MCP server over stdin/stdout. |
| `version` | Print the package version. |
| `init spec` | Scaffold a new RunSpec and matching skill-card starter. |
| `sources build` | Ingest `source_sets` from a spec into a run store without executing a full research run. |
| `sources inspect` | Summarize source-pack records and chunks stored in a run. |
| `runs list` | List run history from `run.json` and `run_summary`. |
| `runs search` | Search prior runs by run id, goal, spec id, stop reason, entity type, or date text. |

Most human-facing summary commands default to readable table/text output. Use
`--json` for automation where supported.

## RunSpec Surface

The current YAML model supports:

| Section | Purpose |
|---|---|
| `spec_id`, `goal` | Stable identifier and plain-language research goal. |
| `backend_policy` | `auto`, `cli`, or `api`; `--offline` forces the native fixture path unless an incompatible backend is explicitly requested. |
| `max_cycles`, `max_entities_per_cycle` | Loop and per-cycle entity bounds. |
| `budget_usd`, `wall_limit_s` | Hard run caps. Native API spend feeds the budget stop/warn system; CLI subagent calls use a separate counter. |
| `max_subagent_calls`, `subagent_timeout_s` | Flat-fee CLI-subagent guardrails. |
| `entities` | Typed entity schemas, field requirements, field types, and search templates. |
| `seeds` | Initial discovery queries. |
| `distinct_pairs` | Entity names that must never merge. |
| `search` | Search provider and API-key environment variable. |
| `models` | Fast, smart, and heavy model aliases for native and verification paths. |
| `obsidian_vault` | Optional vault root for write-through Markdown notes. |
| `source_sets` | Local files, directories, URL lists, and crawl-domain seeds to ingest into DuckDB. |
| `source_policy` | Domain allowlist, denylist, trusted domains, and optional trusted-only behavior. |
| `items` | Inline wide/batch items; the CLI also accepts CSV or JSONL via `--items`. |
| `report` | Default report template and output formats. |
| `verification` | `standard` or opt-in `council` verification, with configured model tiers. |

## Run Artifacts

Each run writes artifacts under `runs/<run_id>/` by default:

| Artifact | Purpose |
|---|---|
| `store.duckdb` | Primary typed store. |
| `events.jsonl` | Durable event log for replay and debugging. |
| `events.sock` | Unix socket for live TUI attach while the run is active. |
| `run.json` | Run metadata used by history, resume, and next-command summaries. |
| `stop.requested` | Graceful-stop signal file consumed by the orchestrator. |

Current DuckDB-backed tables include entity, field, provenance, conflict,
relation, summary, migration, source, source-chunk, and verification-vote data.

## Events And TUI

The Python event contract is mirrored in `tui/src/transport/types.ts`. Current
event variants include lifecycle, planning, agent, fact, conflict, cost, budget,
coverage, interrupt, subagent, source-pack, verification-vote, and run-complete
events. Stop reasons include `budget`, `plateau`, `deadline`, `ctrl_c`, `error`,
`no_tasks`, and `subagent_cap`.

The TUI displays the plan, agent DAG, knowledge/fact stream, source-pack status,
coverage recommendations, low-confidence fields, open conflicts, budget and
acceptance progress, report/evidence readiness, and council disagreements.

## Source, Evidence, And Reports

Source packs normalize local Markdown/text/CSV/JSON/PDF-compatible inputs,
directories, URL lists, and crawl-domain seeds into `sources` and
`source_chunks`. Native agents query source chunks before web search; source
policy wraps search/fetch results and adjusts confidence based on trusted or
denied domains.

`researcher evidence` emits one row per entity-field with:

- entity type and name
- field and value
- confidence
- source URL
- supporting snippet
- provenance id
- conflict status
- last-seen run

`researcher report` renders cited Markdown, HTML, or JSON reports with inline
numeric citations, a source appendix, conflicts, coverage gaps, acceptance
metrics, next recommended seeds, and council-vote summaries when present.

## Verification Gates

Current local gates:

```bash
uv run pytest -q
uv run ruff check researcher tests
cd tui && npm test -- --run
cd tui && npm run build
```

Latest verified snapshot, 2026-04-28:

- Python tests: `533 passed, 1 skipped`
- Ruff: clean for `researcher tests`
- TUI tests: `17 passed`
- TUI build: clean

Live acceptance remains manual and should be checked against the target v1
criteria: at least 100 entities, at least 70% field fill, less than $3, less
than 600 seconds, and a working live TUI attach.

