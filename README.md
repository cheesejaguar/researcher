# 🔬 researcher

> A modular, observable, multi-agent deep-research orchestration engine with a Claude-Code-grade Ink TUI.

[![Python](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org)
[![uv](https://img.shields.io/badge/managed%20by-uv-261230?logo=python&logoColor=white)](https://docs.astral.sh/uv/)
[![TypeScript](https://img.shields.io/badge/typescript-5.5-3178C6.svg?logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Ink](https://img.shields.io/badge/tui-ink%205-blueviolet?logo=react&logoColor=white)](https://github.com/vadimdemedes/ink)
[![DuckDB](https://img.shields.io/badge/store-duckdb-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org)
[![Pydantic](https://img.shields.io/badge/pydantic-v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev)
[![Tests](https://img.shields.io/badge/tests-533%20passing-brightgreen.svg)](#-status)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-ff69b4.svg)](https://github.com/cheesejaguar/researcher/pulls)

---

## 🎯 What is this?

Give `researcher` a subject and a typed schema — *"every major war since 1500"*, *"every FDA-approved GLP-1 trial since 2020"*, *"every Iranian ballistic missile platform"* — and it spins up a swarm of LLM-powered research agents that fan out across the web, extract structured facts, reconcile them into a typed knowledge base, and keep iterating until coverage plateaus or the budget runs out.

The two things that make it interesting are not the agents themselves — those are commodity now — but:

1. 🧩 **The reduce step.** How do N parallel agents converge their findings into a single coherent, queryable, typed object store without trampling each other or producing mush?
2. 🖥️ **The hypervisor TUI.** A Claude-Code-grade Ink terminal interface that lets you watch the swarm think — agent DAG, live fact stream, cost burn-down — so the system is legible *while* it runs, not just after.

A third, smaller bet: 💸 **cost discipline**. The whole thing is designed to offload to `claude -p` / `codex exec` when those CLIs are on your PATH, so a flat-fee Claude Max or ChatGPT Plus subscription covers all the LLM calls AND all the web searches for free.

---

## ✨ Features

### 🧠 Multi-agent orchestration
- ♻️ **Plan → Map → Reduce** cycle loop with bounded parallelism
- 🎭 **Five native agent kinds**: `DiscoverAgent`, `ExpandAgent`, `VerifyAgent`, `EnrichAgent`, `CriticAgent`
- 🤝 **CLI subagent backend**: offloads whole research tasks to `claude -p` or `codex exec` with `WebSearch` + `WebFetch` enabled — saves both LLM and search credits
- 🔀 **Automatic backend detection**: probes `$PATH` at startup, prefers Claude Code, falls back to native OpenRouter agents
- 🚨 **Circuit breaker**: auth errors / usage-limit / 3 failures-per-cycle auto-disables the CLI path for the rest of the run

### 💾 Typed knowledge store
- 🦆 **DuckDB-backed** single-file persistent store with entities, fields, embeddings, provenance, conflicts, and run summaries
- 🧵 **Serial reduce step**: a single `FactWriter` drains an `asyncio.Queue` through a `validate_type → dedup → detect_conflict → score_confidence` pipeline, so map-side races can't corrupt state
- 🔗 **Entity resolution** with an injectable embedder, two-threshold (0.92 / 0.78) auto-merge / pending / distinct policy, and a name-level distinct-pairs blocklist to prevent hierarchical conflations (e.g., *Napoleonic Wars* vs *War of the Sixth Coalition*)
- 📝 **Provenance tracking**: every fact carries a URL, snippet, extractor model, agent id, task id, and span id

### 🎛️ Budget discipline
- 💵 **Per-run USD cap** with warn fractions at 50 / 80 / 95 %
- 🎯 **Per-task cap** (≤ 5% of remaining) and **per-entity cap** (≤ 3% of total)
- ⏱️ **Wall clock budget** with graceful drain on expiry
- 🚪 **Hard stop reasons**: `budget` / `plateau` / `deadline` / `ctrl_c` / `error` / `no_tasks` / `subagent_cap`
- 🪙 **Subagent counter** is separate from USD so flat-fee subscribers never trip the money gate

### 📡 Observability
- 🧾 **JSONL event bus** — every orchestrator, agent, writer, and subagent event is fsync'd to `runs/<run_id>/events.jsonl` for full replay
- 🔌 **Best-effort socket queue** for live TUI attach (bounded, drops oldest, never blocks the core)
- 🌈 **Ink TUI** with four panels (Plan, Agent DAG, Knowledge, Status bar), live socket attach, replay-then-tail, coverage hints, source-pack status, and council disagreement counts
- 📊 **Stats everywhere**: writer queue metrics, cost tracker by task id, Obsidian write counter, CLI runner cache hits

### 📚 Source packs, evidence, and reports
- 📦 **Source packs**: local files/directories, URL lists, and crawl-domain seeds are normalized into DuckDB `sources` and `source_chunks`
- 🧭 **Source policy**: `allow_domains`, `deny_domains`, `trusted_domains`, and `require_trusted` guide search/fetch and confidence scoring
- 🔎 **Source-first native agents**: native discovery/expansion searches local source chunks before falling back to web search
- 🧾 **Evidence matrix**: one row per entity-field with value, confidence, source URL, supporting snippet, provenance id, conflict status, and last-seen run
- 📝 **Cited reports**: Markdown, HTML, or JSON reports with numeric citations, source appendix, conflicts, coverage gaps, acceptance metrics, and next seed suggestions
- 🗳️ **Optional council verification**: conflicted or low-confidence fields can be checked by configured model tiers, with votes persisted and surfaced in the TUI

### 📘 Obsidian integration
- 🏠 **Write-through sink** that materializes `FactClaim`s into Markdown files in a user-provided Obsidian vault
- 🗂️ **Flat-by-type layout**: `<vault>/researcher/War/WWII.md`, merges across runs at the same canonical location
- ⚡ **Debounced live writes** (100 ms) so you can watch your vault fill up during a run
- 🪶 **YAML frontmatter** with entity-schema fields at the top level for Obsidian's Dataview plugin
- 🏷️ **Auto-tagged** with `researcher` + type-scoped tags like `researcher/War`
- 🛡️ **Secondary-sink discipline**: failures in the writer are caught, logged, and counted — they never disrupt the primary reduce path

### 🔐 Safety rails
- 🔒 **`asyncio.Lock`** serializes DuckDB writes (single-writer DB)
- ♻️ **Atomic Markdown writes** via `tempfile` + `os.replace`
- 🧹 **Graceful drain** on `Ctrl-C`: in-flight tasks get a grace window, writer drains to completion, run summary always written
- 🤝 **Politeness limiter** (`aiolimiter` per-domain) + `robots.txt` cache, acquired BEFORE the LLM semaphore to prevent deadlock
- 🪂 **`_kill_and_reap`** helper swallows `ProcessLookupError` on every cleanup site so double-kill races never mask `CancelledError`

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         researcher Orchestrator                       │
│                                                                        │
│   Scheduler → next_batch → [Agents in parallel] → FactWriter          │
│       ▲                         │                      │             │
│       │                         ▼                      ▼             │
│       │                   SubagentResearcher      reduce pipeline    │
│       │                   (claude -p / codex)     validate→dedup     │
│       │                         OR                →conflict→score    │
│       │                   DiscoverAgent /               │             │
│       │                   ExpandAgent / etc.            ▼             │
│       │                                           DuckDBKnowledge    │
│       │                                                Store         │
│       │                                                 │             │
│       └─────── snapshot_metrics + plateau detect ◄──────┘             │
│                                                                        │
│   🔊 EventBus → JSONL + bounded socket → Ink TUI                     │
│   📝 ObsidianWriter ← fact_written (secondary sink)                   │
└──────────────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Choice | Why |
|---|---|---|
| Reduce concurrency | **Serial** single-writer queue | DuckDB is single-writer; map is where parallelism lives |
| Vector store | **DuckDB + python cosine** | One store, one transaction, no two-store consistency problem |
| LLM provider | **OpenRouter** (API) + **CLI subagents** (free) | Flat-fee plans → zero marginal cost |
| Skill memory | **Hand-authored YAML** (no runtime bandit) | Avoids cold-start + skill poisoning for v1 |
| Content retries | **Schema-only** (one retry on malformed JSON) | Content-level Reflexion was cut as unmeasured |
| Lock order | **Politeness → LLM semaphore** (always) | Prevents deadlock between the two bounded resources |

---

## 🚀 Quickstart

### Prerequisites

- 🐍 **Python 3.11+**
- 📦 **[uv](https://docs.astral.sh/uv/)** (the only Python toolchain this project uses)
- 🟢 **Node.js 20+** (for the Ink TUI)
- 🤖 **[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)** *or* **[Codex CLI](https://github.com/openai/codex)** on `$PATH` (recommended — saves LLM + search credits)
  - Both are optional; the native OpenRouter path works too if you have an API key

### Install

```bash
git clone git@github.com:cheesejaguar/researcher.git
cd researcher
uv sync
```

### Build the TUI (optional)

```bash
cd tui && npm install && npm run build && cd ..
```

`researcher watch` and auto-launched TUI runs will try to build the TUI when
`tui/dist/index.js` is missing and `npm` is available. Building it once up
front makes the first live attach faster.

### Check your setup

```bash
# Validate Python, uv, Node/TUI, CLI backends, credentials, search, fixtures.
uv run python -m researcher doctor --spec specs/wars.yaml

# Machine-readable preflight for scripts/CI:
uv run python -m researcher doctor --spec specs/wars.yaml --json
```

### Scaffold a new research spec

```bash
uv run python -m researcher init spec \
  --spec-id companies \
  --goal "Public AI infrastructure companies" \
  --entity Company \
  --seed "AI infrastructure public companies" \
  --output specs/companies.yaml
```

Specs can also include source packs, source policy, batch items, report defaults,
and opt-in council verification:

```yaml
source_sets:
  - name: local-notes
    paths: ["./notes", "./sources/report.md"]
    urls: ["https://example.com/reference"]
source_policy:
  allow_domains: ["example.com"]
  deny_domains: ["lowquality.example"]
  trusted_domains: ["example.com"]
items:
  - { name: "Item to research" }
report:
  template: analyst
  formats: ["md", "json"]
verification:
  mode: council
  models: ["openrouter/hermes-3-8b", "openrouter/hermes-3-70b"]
```

### Run a research job

```bash
# Deterministic offline smoke. This forces the native fixture path and spends $0.
uv run python -m researcher run specs/wars.yaml --offline --no-tui --fast-startup

# Auto-detects claude / codex on $PATH; falls back to native OpenRouter path.
uv run python -m researcher run specs/wars.yaml

# With an Obsidian vault write-through:
uv run python -m researcher run specs/wars.yaml --obsidian-vault ~/Documents/ObsidianVault

# Force the CLI subagent path (raises if no CLI is detected):
uv run python -m researcher run specs/wars.yaml --backend cli

# Force the native OpenRouter path (requires OPENROUTER_API_KEY):
uv run python -m researcher run specs/wars.yaml --backend api

# Custom run id + artifacts directory:
uv run python -m researcher run specs/wars.yaml --run-id wars-today --runs-dir ./my-runs

# Wide/batch mode from CSV or JSONL:
uv run python -m researcher run specs/companies.yaml --items companies.csv --item-column name
```

### Watch a run in the TUI

```bash
# Attach live. If events.jsonl already exists, watch replays history first,
# then tails the Unix socket.
uv run python -m researcher watch wars-today

# Replay a completed run's events.
uv run python -m researcher watch wars-today --replay
```

### Verify and use results

```bash
# Human-readable run summary.
uv run python -m researcher inspect wars-today

# JSON summary for automation.
uv run python -m researcher inspect wars-today --json

# v1 acceptance gate: entity count, field fill, cost, wall time.
uv run python -m researcher accept wars-today

# Export for non-SQL consumers.
uv run python -m researcher export wars-today --format csv --output wars.csv
uv run python -m researcher export wars-today --format markdown --output wars.md
uv run python -m researcher export wars-today --format obsidian --output ./wars-vault

# Citation audit and cited reports.
uv run python -m researcher evidence wars-today --format csv --output evidence.csv
uv run python -m researcher report wars-today --template analyst --format md --output report.md
uv run python -m researcher report wars-today --template systematic --format html --output report.html

# Source-pack operations.
uv run python -m researcher sources build specs/wars.yaml --run-id wars-sources
uv run python -m researcher sources inspect wars-today

# Operator controls.
uv run python -m researcher stop wars-today
uv run python -m researcher resume wars-today

# Downstream query surfaces.
uv run python -m researcher runs list
uv run python -m researcher runs search wars
uv run python -m researcher graph wars-today
uv run python -m researcher mcp --db runs/wars-today/store.duckdb
```

---

## 📂 Project layout

```
researcher/
├── 🐍 researcher/                  # Python core
│   ├── orchestrator.py             # cycle loop + graceful drain
│   ├── scheduler.py                # FIFO + plateau detection
│   ├── budget.py                   # USD + wall clock + sub-caps
│   ├── spec.py                     # RunSpec loader + entity class builder
│   ├── models.py                   # Task / FactClaim / Provenance / AgentResult
│   ├── events.py                   # discriminated Event union + EventBus
│   ├── sources.py                  # source pack ingestion + source-policy wrappers
│   ├── reporting.py                # evidence matrices + cited reports
│   ├── agents/                     # 5 native agents + SubagentResearcher
│   ├── backends/                   # CLI subagent path (claude / codex)
│   ├── llm/                        # OpenRouterClient + cache + prompts + embedder
│   ├── fetch/                      # httpx + politeness + robots + readability
│   ├── search/                     # Tavily / Brave / Serper / FileSeeds
│   ├── extract/                    # schema-guided extraction + chunker
│   ├── storage/                    # DuckDBKnowledgeStore + FactWriter + resolver
│   ├── skills/                     # skill card registry
│   └── integrations/
│       └── obsidian.py             # Obsidian write-through sink
├── 🟦 tui/                          # Ink TUI
│   ├── src/
│   │   ├── App.tsx                 # four-panel layout
│   │   ├── store.ts                # pure reducer(state, event)
│   │   ├── panels/                 # Plan / AgentDag / Knowledge / StatusBar
│   │   └── transport/              # jsonl + socket transports
│   └── package.json
├── 📋 specs/                        # example run specs
│   ├── wars.yaml
│   └── glp1_trials.yaml
├── 🗒️ skills/                       # hand-authored skill cards
│   ├── wars.yaml
│   └── glp1_trials.yaml
├── 📚 doc/                          # design docs and implementation plans
│   ├── 2026-04-08-cli-subagent-backend-design.md
│   ├── 2026-04-08-cli-subagent-backend-plan.md
│   ├── 2026-04-08-obsidian-integration-design.md
│   ├── 2026-04-08-obsidian-integration-plan.md
│   └── 2026-04-28-current-product-surface.md
└── 🧪 tests/                        # 533 Python tests + 17 TUI tests as of 2026-04-28
    ├── unit/
    ├── integration/
    └── stubs/                      # deterministic test doubles
```

---

## 📋 Example run spec

```yaml
# specs/wars.yaml
spec_id: wars
goal: "Major interstate wars since 1500"
backend_policy: auto
max_cycles: 5
max_entities_per_cycle: 200
budget_usd: 3.00
wall_limit_s: 600

entities:
  - name: War
    fields:
      - { name: name,             type: str,       required: true }
      - { name: start_year,       type: int,       required: true }
      - { name: end_year,         type: int }
      - { name: belligerents,     type: "list[str]" }
      - { name: estimated_deaths, type: int }
      - { name: outcome,          type: str }
    search_templates:
      - "{q} war wikipedia"

seeds:
  - "major interstate wars since 1500"
  - "20th century wars"

distinct_pairs:
  - ["Napoleonic Wars", "War of the Sixth Coalition"]
  - ["First Battle of the Marne", "Second Battle of the Marne"]

models:
  fast:  openrouter/hermes-3-8b
  smart: openrouter/hermes-3-70b
  heavy: anthropic/claude-sonnet-4-6
```

---

## 📝 Example Obsidian output

A single entity materialized into your vault as `<vault>/researcher/War/World War II.md`:

```markdown
---
# This file is managed by researcher. Edits will be overwritten on the next run.
researcher_type: War
researcher_name: World War II
researcher_runs:
  - wars-2026-04-08
researcher_updated: 2026-04-08T14:23:00Z
tags:
  - researcher
  - researcher/War
start_year: 1939
end_year: 1945
belligerents:
  - Allies
  - Axis
outcome: Allied victory
---

# World War II

> Managed by `researcher`. Re-running the job will overwrite this file.

## Fields

| Field | Value | Confidence | Source |
|---|---|---|---|
| start_year | 1939 | 0.97 | [source](https://en.wikipedia.org/wiki/World_War_II) |
| end_year | 1945 | 0.97 | [source](https://en.wikipedia.org/wiki/World_War_II) |
| belligerents | Allies, Axis | 0.96 | [source](https://en.wikipedia.org/wiki/World_War_II) |

## Provenance

### start_year = 1939
> World War II or the Second World War was a global conflict that began in 1939...

- Source: https://en.wikipedia.org/wiki/World_War_II
- Agent: `sub-a1b2c3d4` · Extractor: `claude_code/sonnet` · Span: `cli_0`
```

---

## 🛠️ Development

### Running tests

```bash
# Python tests (uv manages everything)
uv run pytest -q                              # full suite
uv run pytest tests/unit -q                   # unit only
uv run pytest tests/unit/test_obsidian_writer.py -v   # single file

# TUI tests (vitest)
cd tui && npm test -- --run
cd tui && npm run build
```

### Linting

```bash
uv run ruff check researcher tests            # report
uv run ruff check researcher tests --fix      # auto-fix
```

### Running the manual live CLI test

```bash
# Gated behind RESEARCHER_RUN_MANUAL + claude on $PATH
RESEARCHER_RUN_MANUAL=1 uv run pytest tests/manual/ -v
```

### Adding a new entity type

1. Write a YAML spec at `specs/<name>.yaml` with the entity schema
2. *Optionally* hand-author skill cards at `skills/<name>.yaml`
3. Run: `uv run python -m researcher run specs/<name>.yaml`

---

## 📊 Status

**v0.1** — core product surface implemented; live acceptance tuning pending.

### What works today ✅

- ✅ Full orchestrator cycle loop with bounded-parallel map + serial reduce
- ✅ CLI subagent path (Claude Code + Codex) with autodetect + circuit breaker
- ✅ Real native agents (Discover / Expand / Verify / Enrich / Critic) over OpenRouter
- ✅ DuckDB knowledge store with entities / fields / provenance / conflicts
- ✅ Entity resolver with two-threshold + distinct-pairs blocklist
- ✅ FactWriter pipeline with type coercion + conflict detection
- ✅ Budget discipline (USD + wall clock + sub-caps + subagent counter)
- ✅ Obsidian write-through integration
- ✅ Ink TUI with live socket attach, replay, coverage hints, and acceptance progress
- ✅ `doctor`, `inspect`, `accept`, `export`, `evidence`, `report`, `stop`, and `resume` CLI surfaces
- ✅ Source packs, source policy, citation evidence, cited reports, run history, optional council verification, and wide/batch item mode
- ✅ Fetch / search / extract stack (httpx / Tavily / Brave / Serper / trafilatura)
- ✅ Python and TypeScript test suites, ruff clean (`533 passed, 1 skipped`; TUI `17 passed` as of 2026-04-28)
- ✅ 2 example specs: `wars.yaml`, `glp1_trials.yaml`

### Wave 2 next up 🚧

- ✅ Real `OpenRouterClient` is wired into the CLI run path when `OPENROUTER_API_KEY` is set
- ✅ Real `LocalEmbedder` is the default resolver embedder; use `--fast-startup` for hash embeddings
- ✅ `NativeAgentDeps` are wired into `run`, so `--backend api` works when API/search credentials are available
- ✅ `watch <run_id>` can replay completed runs or replay-then-tail live runs via Unix socket
- 🚧 End-to-end tuning against the §11 success criteria (≥ 100 entities, ≥ 70% field-fill, < $3, < 10 min)

### Explicit v1 scope cuts 🗃️

These are documented as intentional deferrals; see `doc/` for rationale:

- 🗃️ **Thompson-sampling bandit** for prompt variants → uniform random for now
- 🗃️ **Runtime skill-card authoring** → hand-authored YAML only
- 🗃️ **Content-level Reflexion retries** → schema-format retries only
- 🗃️ **DuckDB vss extension** → python-side cosine (fine at v1 scale)
- 🗃️ **Multilingual entity resolution** → accept known MiniLM failures on non-Latin scripts
- 🗃️ **Cross-store vector sync** → single DuckDB file is source of truth

---

## 🙏 Inspiration & prior art

- 🎓 [karpathy/autoresearch](https://github.com/karpathy) — Reflexion-style critique, lightweight skill memory, no training infrastructure
- 🧪 **ReAct / Reflexion** — Plan-Execute-Reflect as an agent primitive
- 🔗 [DSPy](https://dspy.ai/) — prompts as parameters you can search over
- 🎨 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) — the Ink TUI bar we're trying to clear

---

## 📄 License

MIT © 2026 — see [LICENSE](LICENSE)

---

## 🧭 Documentation

- 📘 **[Design docs](doc/)** — design documents and implementation plans
- 🧭 **[Current product surface](doc/2026-04-28-current-product-surface.md)** — current commands, spec fields, artifacts, and verification gates
- 🗺️ **Main v1 plan** — `~/.claude/plans/distributed-mapping-rossum.md`
- 🎯 **Example specs** — [specs/](specs/)
- 🎓 **Hand-authored skills** — [skills/](skills/)

---

<sub>🤖 Built with discipline: TDD, bite-sized tasks, subagent-driven development, and frequent commits. Every major design decision is documented in `doc/`.</sub>
