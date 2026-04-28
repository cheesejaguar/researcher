# Contributing to researcher

Thanks for your interest in contributing! This document covers everything you need to get started.

## Development Setup

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/) (the only Python toolchain — never use `pip` or `.venv/bin/python`), Node.js 20+ (for the TUI only).

```bash
# Clone and install
git clone https://github.com/cheesejaguar/researcher.git
cd researcher
uv sync                          # installs all deps + dev group
uv run python -m researcher --help # verify CLI works

# Optional extras
uv sync --extra pdf               # pymupdf for PDF source extraction
uv sync --extra otel              # OpenTelemetry trace export

# TUI (optional)
cd tui && npm install && npm run build
```

On macOS, if `uv run researcher --help` fails while `uv run python -m researcher --help`
works, check whether the editable-install `.pth` file in `.venv/site-packages`
has the hidden file flag. Python skips hidden `.pth` files, so the module form is
the guaranteed development command.

## Running Tests

```bash
uv run pytest -q tests/unit/      # full unit suite (~500 tests, <10s)
uv run ruff check .               # lint
uv run ruff format --check .      # format check
```

All tests must pass before submitting a PR. The CI gate runs the same commands.

## Coding Standards

- **Python style:** enforced by [ruff](https://docs.astral.sh/ruff/) with the config in `pyproject.toml`. Run `uv run ruff check --fix .` to auto-fix.
- **Type hints:** all public functions must be typed. `mypy` is in the dev group but not yet CI-gated.
- **Imports:** use `from __future__ import annotations` in every module. Lazy imports are encouraged for heavy deps (DuckDB, sentence-transformers) to keep `python -m researcher --help` fast.
- **No docstrings on obvious code.** Only add comments where the logic isn't self-evident.
- **Tests:** TDD discipline — write failing tests first, then implement. Each feature ships with its own test file under `tests/unit/`.

## Project Structure

```
researcher/          # core Python package
  agents/            # native research agents (discover, expand, verify, enrich, critic)
  backends/          # CLI subagent runners (claude -p, codex exec)
  fetch/             # HTTP fetcher, politeness limiter, robots.txt, PDF extraction
  integrations/      # Obsidian vault writer
  llm/               # OpenRouter client, prompt registry, embedder, cache
  mcp/               # MCP server (expose researcher as a tool)
  observability/     # OpenTelemetry adapter
  search/            # search providers (Tavily, Brave, Serper, Exa MCP, file seeds)
  storage/           # DuckDB store, entity resolver, fact writer, migrations
  skills/            # skill card registry
specs/               # example RunSpec YAML files
skills/              # hand-authored skill cards
tests/
  stubs/             # in-memory test doubles for every subsystem
  unit/              # unit tests (~500)
  integration/       # offline integration smokes
tui/                 # Ink/React TUI (TypeScript)
```

## Pull Request Process

1. **Fork and branch.** Create a feature branch from `main`.
2. **Small, focused PRs.** One logical change per PR. If a feature touches 5+ files, that's fine — but don't bundle unrelated changes.
3. **Write tests first.** Every PR that adds or changes behavior must include tests. Regressions are caught by the existing 500+ test suite.
4. **Lint clean.** `uv run ruff check .` must pass with zero errors.
5. **Descriptive commit messages.** Lead with the subsystem (`cli:`, `storage:`, `agents:`, `spec:`, etc.), then a concise summary of what changed and why. The "why" matters more than the "what."
6. **Fill out the PR template.** Summary + test plan.

## What to Work On

- Check [open issues](https://github.com/cheesejaguar/researcher/issues) for `good first issue` or `help wanted` labels.
- The plan file at the repo root describes the full v1.1+ roadmap with explicit scope cuts and re-entry criteria.
- If you want to work on something not in an issue, open one first to discuss the approach.

## Commit Message Convention

```
subsystem: short imperative summary

Longer explanation of what changed and why. Reference issue numbers
where applicable (#123). The subsystem prefix should match the
directory or module being changed:

  cli:          researcher/cli.py, researcher/backends/
  storage:      researcher/storage/
  agents:       researcher/agents/
  llm:          researcher/llm/
  fetch:        researcher/fetch/
  spec:         researcher/spec.py
  events:       researcher/events.py
  search:       researcher/search/
  observability: researcher/observability/
  mcp:          researcher/mcp/
  tui:          tui/
  tests:        tests/
```

## Reporting Bugs

Use the [bug report template](https://github.com/cheesejaguar/researcher/issues/new?template=bug_report.yml) on GitHub. Include:
- What you expected vs. what happened
- The spec YAML you used (or a minimal reproducer)
- The `events.jsonl` tail (last 20 lines) if the run produced one
- Python version (`python --version`) and OS

## Questions?

Open a [discussion](https://github.com/cheesejaguar/researcher/discussions) or file an issue. We're friendly.
