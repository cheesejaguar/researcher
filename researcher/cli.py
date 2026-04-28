"""Typer CLI for the public researcher product journey."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from researcher import __version__

if TYPE_CHECKING:
    from researcher.orchestrator import StopReason

app = typer.Typer(
    name="researcher",
    help="Multi-agent deep-research orchestration.",
    no_args_is_help=True,
    add_completion=False,
)
init_app = typer.Typer(help="Scaffold researcher project files.", add_completion=False)
app.add_typer(init_app, name="init")
sources_app = typer.Typer(help="Build and inspect source packs.", add_completion=False)
app.add_typer(sources_app, name="sources")
runs_app = typer.Typer(help="List and search recorded runs.", add_completion=False)
app.add_typer(runs_app, name="runs")

RUN_METADATA = "run.json"
STOP_REQUEST_FILE = "stop.requested"


# ---------- helpers ----------


def _default_embed_fn_sync(texts: list[str]) -> list[list[float]]:
    """Hash-based pseudo-embedder. 384-dim, deterministic, NOT semantically meaningful.

    Placeholder until Wave 1-C lands a real sentence-transformers embedder.
    Used by :class:`DefaultEntityResolver` so the reducer pipeline has a valid
    ``embed_fn`` signature without loading a ~90MB model at CLI start.
    """
    import hashlib

    out: list[list[float]] = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        vec: list[float] = []
        for i in range(384):
            b = h[i % len(h)]
            vec.append(((b / 255.0) * 2.0) - 1.0)
        out.append(vec)
    return out


async def _default_embed_fn(texts: list[str]) -> list[list[float]]:
    return _default_embed_fn_sync(texts)


def _generate_run_id(spec_path: Path) -> str:
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{spec_path.stem}-{ts}"


def _project_root() -> Path:
    return Path.cwd()


def _run_metadata_path(run_dir: Path) -> Path:
    return run_dir / RUN_METADATA


def _stop_request_path(run_dir: Path) -> Path:
    return run_dir / STOP_REQUEST_FILE


def _write_run_metadata(run_dir: Path, data: dict[str, Any]) -> None:
    payload = dict(data)
    payload["metadata_version"] = 1
    _run_metadata_path(run_dir).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_run_metadata(run_dir: Path) -> dict[str, Any]:
    path = _run_metadata_path(run_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return default


def _tui_entry_or_build(auto_build: bool = True) -> Path:
    root = _project_root()
    tui_dir = root / "tui"
    entry = tui_dir / "dist" / "index.js"
    if entry.exists():
        return entry

    package_json = tui_dir / "package.json"
    if auto_build and package_json.exists() and shutil.which("npm"):
        install_cmd = ["npm", "install"] if not (tui_dir / "node_modules").exists() else None
        if install_cmd is not None:
            subprocess.run(install_cmd, cwd=tui_dir, check=False)
        subprocess.run(["npm", "run", "build"], cwd=tui_dir, check=False)
        if entry.exists():
            return entry

    raise typer.BadParameter(
        f"TUI build not found at {entry}; run `cd tui && npm install && npm run build`"
    )


def _print_kv_table(rows: Iterable[tuple[str, Any]]) -> None:
    materialized = [(str(k), "" if v is None else str(v)) for k, v in rows]
    width = max((len(k) for k, _ in materialized), default=0)
    for key, value in materialized:
        typer.echo(f"{key.ljust(width)}  {value}")


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _json_echo(payload: Any) -> None:
    typer.echo(json.dumps(payload, separators=(",", ":"), default=str))


def _print_run_next_steps(run_id: str, runs_dir: Path, run_dir: Path) -> None:
    db_path = run_dir / "store.duckdb"
    events_path = run_dir / "events.jsonl"
    typer.echo("[researcher run] next:")
    typer.echo(f"  inspect: uv run python -m researcher inspect {run_id} --runs-dir {runs_dir}")
    typer.echo(f"  accept:  uv run python -m researcher accept {run_id} --runs-dir {runs_dir}")
    if events_path.exists():
        typer.echo(f"  replay:  uv run python -m researcher watch {run_id} --runs-dir {runs_dir} --replay")
    if db_path.exists():
        typer.echo(f"  db:      {db_path}")
        typer.echo(f"  export:  uv run python -m researcher export {run_id} --runs-dir {runs_dir} --format csv")
        typer.echo(f"  report:  uv run python -m researcher report {run_id} --runs-dir {runs_dir}")
        typer.echo(f"  evidence: uv run python -m researcher evidence {run_id} --runs-dir {runs_dir}")


def _write_text_output(text: str, output: Path | None) -> None:
    if output is None:
        typer.echo(text)
        return
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    typer.echo(str(output))


# ---------- commands ----------


@app.command()
def doctor(
    spec: Path | None = typer.Option(
        None,
        "--spec",
        exists=True,
        readable=True,
        help="Optional RunSpec YAML to validate with environment checks.",
    ),
    obsidian_vault: str = typer.Option(
        "",
        "--obsidian-vault",
        help="Optional Obsidian vault path to validate.",
    ),
    runs_dir: Path = typer.Option(
        Path("runs"),
        "--runs-dir",
        help="Runs directory to check for writability.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit compact JSON instead of a human-readable checklist.",
    ),
) -> None:
    """Check local setup before spending API money or starting a long run."""
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str = "") -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    add(
        "python",
        "pass" if sys.version_info >= (3, 11) else "fail",
        sys.version.split()[0],
    )
    add("uv", "pass" if shutil.which("uv") else "warn", shutil.which("uv") or "not on PATH")
    add("node", "pass" if shutil.which("node") else "warn", shutil.which("node") or "not on PATH")
    add("npm", "pass" if shutil.which("npm") else "warn", shutil.which("npm") or "not on PATH")

    tui_entry = _project_root() / "tui" / "dist" / "index.js"
    tui_pkg = _project_root() / "tui" / "package.json"
    add(
        "tui_build",
        "pass" if tui_entry.exists() else ("warn" if tui_pkg.exists() else "fail"),
        str(tui_entry) if tui_entry.exists() else "build with `cd tui && npm install && npm run build`",
    )

    claude = shutil.which("claude")
    codex = shutil.which("codex")
    add("claude_cli", "pass" if claude else "warn", claude or "not on PATH")
    add("codex_cli", "pass" if codex else "warn", codex or "not on PATH")
    add(
        "openrouter",
        "pass" if os.environ.get("OPENROUTER_API_KEY") else "warn",
        "OPENROUTER_API_KEY set" if os.environ.get("OPENROUTER_API_KEY") else "OPENROUTER_API_KEY missing",
    )

    try:
        if runs_dir.exists():
            writable = os.access(runs_dir, os.W_OK)
        else:
            parent = runs_dir.parent if runs_dir.parent != Path("") else Path(".")
            writable = parent.exists() and os.access(parent, os.W_OK)
        add("runs_dir", "pass" if writable else "fail", str(runs_dir))
    except Exception as exc:
        add("runs_dir", "fail", str(exc))

    if spec is not None:
        try:
            from researcher.spec import load_spec

            spec_obj = load_spec(spec)
            add("spec", "pass", f"{spec_obj.spec_id}: {spec_obj.goal}")
            provider = spec_obj.search.provider
            key_env = spec_obj.search.api_key_env or "TAVILY_API_KEY"
            if provider == "tavily":
                add(
                    "search",
                    "pass" if os.environ.get(key_env) else "warn",
                    f"{provider} via {key_env}" if os.environ.get(key_env) else f"{provider} key {key_env} missing",
                )
            else:
                add("search", "pass", provider)
            add(
                "source_policy",
                "pass",
                (
                    f"allow={len(spec_obj.source_policy.allow_domains) + len(spec_obj.domain_allowlist)} "
                    f"deny={len(spec_obj.source_policy.deny_domains)} "
                    f"trusted={len(spec_obj.source_policy.trusted_domains)}"
                ),
            )
            if spec_obj.source_sets:
                source_paths = sum(len(s.paths) for s in spec_obj.source_sets)
                source_urls = sum(len(s.urls) for s in spec_obj.source_sets)
                add("source_sets", "pass", f"sets={len(spec_obj.source_sets)} paths={source_paths} urls={source_urls}")
            offline_search = Path("tests") / "fixtures" / "offline_search" / f"{spec_obj.spec_id}.json"
            offline_pages = Path("tests") / "fixtures" / "offline_pages" / f"{spec_obj.spec_id}.json"
            add(
                "offline_fixtures",
                "pass" if offline_search.exists() and offline_pages.exists() else "warn",
                f"{offline_search}, {offline_pages}",
            )
            vault = obsidian_vault or spec_obj.obsidian_vault or ""
            if vault:
                vault_path = Path(vault).expanduser()
                add(
                    "obsidian_vault",
                    "pass" if vault_path.exists() else "fail",
                    str(vault_path),
                )
        except Exception as exc:
            add("spec", "fail", str(exc))
    elif obsidian_vault:
        vault_path = Path(obsidian_vault).expanduser()
        add(
            "obsidian_vault",
            "pass" if vault_path.exists() else "fail",
            str(vault_path),
        )

    failed = [c for c in checks if c["status"] == "fail"]
    payload = {"passed": not failed, "checks": checks}
    if json_output:
        _json_echo(payload)
    else:
        for c in checks:
            typer.echo(f"{c['status'].upper():4}  {c['name']:<18} {c['detail']}")
    if failed:
        raise typer.Exit(1)


@init_app.command("spec")
def init_spec(
    output: Path = typer.Option(
        Path("specs/new_research.yaml"),
        "--output",
        "-o",
        help="Path for the generated RunSpec YAML.",
    ),
    skill_output: Path | None = typer.Option(
        None,
        "--skill-output",
        help="Path for the generated skill card YAML. Defaults to skills/<spec_id>.yaml.",
    ),
    spec_id: str = typer.Option("new_research", "--spec-id", help="Machine-readable spec id."),
    goal: str = typer.Option("Research a typed entity set", "--goal", help="Research goal."),
    entity: str = typer.Option("Entity", "--entity", help="Primary entity type name."),
    seed: str = typer.Option("seed query", "--seed", help="Initial search seed."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files."),
) -> None:
    """Scaffold a RunSpec YAML and matching skill card."""
    output = output.expanduser()
    skill_output = (skill_output or Path("skills") / f"{spec_id}.yaml").expanduser()
    for path in (output, skill_output):
        if path.exists() and not force:
            raise typer.BadParameter(f"{path} already exists; pass --force to overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)

    spec_yaml = f"""spec_id: {spec_id}
goal: "{goal}"
backend_policy: auto
max_cycles: 5
max_entities_per_cycle: 200
budget_usd: 3.00
wall_limit_s: 600
max_subagent_calls: 200
subagent_timeout_s: 180

entities:
  - name: {entity}
    fields:
      - {{ name: name, type: str, required: true }}
      - {{ name: summary, type: str }}
      - {{ name: source_url, type: str }}
    search_templates:
      - "{{q}}"
      - "{{q}} {entity}"

seeds:
  - "{seed}"

distinct_pairs: []

search:
  provider: tavily
  api_key_env: TAVILY_API_KEY
  max_results: 10

models:
  fast: openrouter/hermes-3-8b
  smart: openrouter/hermes-3-70b
  heavy: anthropic/claude-sonnet-4-6
"""
    skill_yaml = f"""spec_id: {spec_id}
entity_type: {entity}
prompt_fragments:
  - "Prefer primary sources and preserve exact names."
validation_notes:
  - "Do not invent fields that are absent from cited sources."
"""
    output.write_text(spec_yaml, encoding="utf-8")
    skill_output.write_text(skill_yaml, encoding="utf-8")
    typer.echo(f"created spec: {output}")
    typer.echo(f"created skill: {skill_output}")


@app.command()
def run(
    spec: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to a RunSpec YAML file."
    ),
    run_id: str = typer.Option("", "--run-id", help="Custom run id (default: autogen)."),
    no_tui: bool = typer.Option(
        False, "--no-tui", help="Do not auto-launch the Ink TUI child process."
    ),
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
    runs_dir: Path = typer.Option(
        Path("runs"),
        "--runs-dir",
        help="Directory where per-run artifacts (DuckDB, events.jsonl) are written.",
    ),
    fast_startup: bool = typer.Option(
        False,
        "--fast-startup",
        help=(
            "Use the hash-based pseudo-embedder instead of LocalEmbedder "
            "(skips the MiniLM cold-load; intended for tests/CI)."
        ),
    ),
    resume: str = typer.Option(
        "",
        "--resume",
        help=(
            "Resume a previous run by id; reuses its DuckDB store and "
            "continues from the next cycle."
        ),
    ),
    mode: str = typer.Option(
        "overwrite",
        "--mode",
        help=(
            "overwrite | merge — overwrite resets entity state per run; "
            "merge accumulates fields across runs with confidence-aware "
            "conflict promotion."
        ),
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        help="Pause at declared interrupt points and prompt the user via stdin.",
    ),
    otel_endpoint: str = typer.Option(
        "",
        "--otel-endpoint",
        help=(
            "OTLP collector endpoint (e.g. localhost:4317). When set, every "
            "bus event is also exported as a structured span via the "
            "opentelemetry-sdk extra. Fail-open: missing deps or import "
            "errors are logged and the run continues without OTEL."
        ),
    ),
    items: Path | None = typer.Option(
        None,
        "--items",
        exists=True,
        readable=True,
        help="CSV or JSONL items for wide/batch research.",
    ),
    item_column: str = typer.Option(
        "name",
        "--item-column",
        help="Column/key to use as the per-item research query.",
    ),
) -> None:
    """Run a research job from a YAML spec against the CLI subagent backend."""
    if backend not in ("auto", "cli", "api"):
        raise typer.BadParameter(f"--backend must be one of auto|cli|api, got {backend!r}")
    if mode not in ("overwrite", "merge"):
        raise typer.BadParameter(f"--mode must be one of overwrite|merge, got {mode!r}")
    if offline and backend == "cli":
        raise typer.BadParameter("--offline uses local fixtures and cannot run with --backend cli")
    if offline and backend == "auto":
        typer.echo("[researcher run] --offline forces backend=api fixture path")
        backend = "api"
    from researcher.spec import load_spec as _load_run_spec

    metadata_spec = _load_run_spec(spec)

    # When resuming, the run_id is dictated by --resume; the existing run
    # directory must already exist on disk so we can reopen its DuckDB.
    actual_run_id = resume if resume else (run_id or _generate_run_id(spec))
    run_dir = runs_dir / actual_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        _stop_request_path(run_dir).unlink()
    except FileNotFoundError:
        pass
    _write_run_metadata(
        run_dir,
        {
            "run_id": actual_run_id,
            "spec_id": metadata_spec.spec_id,
            "goal": metadata_spec.goal,
            "spec_path": str(spec),
            "backend": backend,
            "offline": offline,
            "no_tui": no_tui,
            "fast_startup": fast_startup,
            "mode": mode,
            "obsidian_vault": obsidian_vault,
            "runs_dir": str(runs_dir),
            "items_path": str(items) if items else "",
            "item_column": item_column,
        },
    )

    typer.echo(
        f"[researcher run] spec={spec} run_id={actual_run_id} "
        f"backend={backend} run_dir={run_dir}" + (f" resume={resume}" if resume else "")
    )

    import asyncio as _asyncio

    reason = _asyncio.run(
        _execute_run(
            spec_path=spec,
            run_id=actual_run_id,
            run_dir=run_dir,
            backend=backend,
            obsidian_vault_override=obsidian_vault,
            offline=offline,
            no_tui=no_tui,
            fast_startup=fast_startup,
            resume=resume,
            mode=mode,
            interactive=interactive,
            otel_endpoint=otel_endpoint,
            items_path=items,
            item_column=item_column,
        )
    )
    typer.echo(f"[researcher run] done reason={reason.value}")
    _print_run_next_steps(actual_run_id, runs_dir, run_dir)


async def _execute_run(
    spec_path: Path,
    run_id: str,
    run_dir: Path,
    backend: str,
    obsidian_vault_override: str,
    offline: bool,
    no_tui: bool,
    fast_startup: bool = False,
    resume: str = "",
    mode: str = "overwrite",
    interactive: bool = False,
    otel_endpoint: str = "",
    stop_file: Path | None = None,
    items_path: Path | None = None,
    item_column: str = "name",
) -> StopReason:
    """Construct every wire and execute :meth:`Orchestrator.run` once.

    Imports are lazy so `researcher --help` doesn't pull in DuckDB, Pydantic
    model factories, or sentence-transformers on every invocation.
    """
    import asyncio as _asyncio
    from datetime import datetime

    from researcher.agents.native_deps import NativeAgentDeps
    from researcher.backends.cli_runner import ClaudeCodeRunner, CodexRunner
    from researcher.backends.models import CliKind
    from researcher.backends.resolver import BackendResolver
    from researcher.budget import Budget
    from researcher.events import (
        ConflictDetected,
        ConflictPayload,
        EventBus,
        EventSocketServer,
        FactWritten,
        FactWrittenPayload,
        SourcePackLoaded,
        SourcePackPayload,
    )
    from researcher.fetch.http import HttpFetcher
    from researcher.fetch.politeness import PolitenessLimiter
    from researcher.integrations.obsidian import ObsidianWriter
    from researcher.llm.embedder import LocalEmbedder
    from researcher.llm.offline import OfflineFixtureLLM
    from researcher.llm.openrouter import OpenRouterClient
    from researcher.llm.prompts import default_registry
    from researcher.models import LLMTier
    from researcher.orchestrator import Orchestrator
    from researcher.scheduler import Scheduler
    from researcher.search.base import SearchProvider
    from researcher.search.file_seeds import FileSeedsProvider
    from researcher.search.tavily import TavilyProvider
    from researcher.skills.registry import SkillRegistry
    from researcher.sources import (
        PolicySearchProvider,
        SourcePackSearchProvider,
        ingest_source_sets,
        load_items,
    )
    from researcher.spec import build_entity_class, load_spec
    from researcher.storage.duckdb_store import DuckDBKnowledgeStore
    from researcher.storage.resolver import DefaultEntityResolver
    from researcher.storage.writer import FactWriter
    from tests.stubs.llm import StubLLMClient

    spec_obj = load_spec(spec_path)
    if items_path is not None:
        loaded_items = load_items(items_path, item_column=item_column)
        spec_obj = spec_obj.model_copy(update={"items": loaded_items})
        typer.echo(f"[researcher run] loaded_items={len(loaded_items)} from={items_path}")

    if not spec_obj.entities:
        raise typer.BadParameter(f"spec {spec_path} declares no entity types")
    typer.echo(
        f"[researcher run] goal={spec_obj.goal!r} search={spec_obj.search.provider} "
        f"budget=${spec_obj.budget_usd:.2f} wall={spec_obj.wall_limit_s}s"
    )

    # Wave 1 supports one entity type per run — take the first.
    primary_entity = spec_obj.entities[0]
    entity_class = build_entity_class(primary_entity)
    entity_schema_dict = {
        "entity_type": primary_entity.name,
        "fields": [
            {"name": f.name, "type": f.type, "required": f.required, "enum": list(f.enum)} for f in primary_entity.fields
        ],
    }

    db_path = run_dir / "store.duckdb"
    store = DuckDBKnowledgeStore(db_path=db_path)
    await store.open()
    await store.init_schema(entity_class)
    source_records, source_chunks = ingest_source_sets(spec_obj.source_sets)
    if source_records or source_chunks:
        await store.record_sources(
            [s.__dict__ for s in source_records],
            [c.__dict__ for c in source_chunks],
        )
        typer.echo(
            f"[researcher run] source_pack sources={len(source_records)} chunks={len(source_chunks)}"
        )

    # If resuming, pull the checkpoint blob now so we can restore the
    # scheduler + budget after they're constructed below.
    checkpoint_cycle: int | None = None
    sched_data: dict = {}
    budget_data: dict = {}
    if resume:
        ckpt = await store.load_checkpoint(resume)
        if ckpt is None:
            await store.close()
            raise typer.BadParameter(f"no checkpoint found for run_id={resume!r}")
        checkpoint_cycle = int(ckpt["cycle"])
        sched_data = ckpt["data"].get("scheduler", {})
        budget_data = ckpt["data"].get("budget", {})

    try:
        bus = EventBus(jsonl_path=run_dir / "events.jsonl")
        await bus.start()
        socket_server = None
        tui_proc = None
        if sys.stdout.isatty():
            socket_server = EventSocketServer(bus=bus, socket_path=run_dir / "events.sock")
            await socket_server.start()
            if not no_tui:
                try:
                    tui_entry = _tui_entry_or_build(auto_build=True)
                    tui_proc = await _asyncio.create_subprocess_exec(
                        "node",
                        str(tui_entry),
                        "--replay",
                        str(run_dir / "events.jsonl"),
                        "--socket",
                        str(run_dir / "events.sock"),
                    )
                except OSError:
                    tui_proc = None
                except typer.BadParameter as exc:
                    typer.echo(f"[researcher run] TUI disabled: {exc}", err=True)
        if source_records or source_chunks:
            await bus.emit(
                SourcePackLoaded(
                    seq=0,
                    ts=datetime.now(UTC),
                    run_id=run_id,
                    payload=SourcePackPayload(
                        sources=len(source_records),
                        chunks=len(source_chunks),
                    ),
                )
            )

        # v1.3 #4: optional OTLP trace export. Fail-open: a missing extra
        # or a malformed endpoint MUST NOT abort the run.
        otel_adapter = None
        if otel_endpoint:
            try:
                from researcher.observability.otel import (
                    OtelEventAdapter,
                    build_otlp_exporter,
                )

                exporter = build_otlp_exporter(otel_endpoint)
                otel_adapter = OtelEventAdapter(exporter, run_id=run_id)
                otel_adapter.start_run()
                bus.add_subscriber(otel_adapter.handle)
                typer.echo(f"[researcher run] otel exporter active endpoint={otel_endpoint}")
            except Exception as exc:
                typer.echo(
                    f"[researcher run] otel disabled: {exc}",
                    err=True,
                )
                otel_adapter = None

        try:
            # Wave 1.1: prefer the real LocalEmbedder (sentence-transformers
            # MiniLM, 384-dim) over the hash-based placeholder. The
            # ``--fast-startup`` flag (and ``offline``) keep the old behavior
            # for tests/CI that can't pay the MiniLM cold-load cost.
            embed_fn = _default_embed_fn if (fast_startup or offline) else LocalEmbedder()
            resolver = DefaultEntityResolver(
                store=store,
                embed_fn=embed_fn,
                distinct_pairs=spec_obj.distinct_pairs,
            )

            async def _emit_fact_to_bus(entity_id: str, claim) -> None:
                await bus.emit(
                    FactWritten(
                        seq=0,
                        ts=datetime.now(UTC),
                        run_id=run_id,
                        payload=FactWrittenPayload(
                            entity_id=entity_id,
                            entity_type=claim.entity_type,
                            field=claim.field,
                            value=claim.value,
                            confidence=claim.confidence,
                            source_url=claim.provenance.url,
                        ),
                    )
                )

            async def _emit_conflict_to_bus(entity_id: str, cells) -> None:
                losing_value = cells[0].value if cells else None
                await bus.emit(
                    ConflictDetected(
                        seq=0,
                        ts=datetime.now(UTC),
                        run_id=run_id,
                        payload=ConflictPayload(
                            entity_id=entity_id,
                            field="",
                            winning_value=None,
                            losing_value=losing_value,
                            reason="detected by FactWriter",
                        ),
                    )
                )

            # CLI flag overrides the spec's mode policy.
            if mode != "overwrite":
                spec_obj = spec_obj.model_copy(update={"mode": mode})

            writer = FactWriter(
                store=store,
                resolver=resolver,
                entity_schema=entity_schema_dict,
                emit_fact=_emit_fact_to_bus,
                emit_conflict=_emit_conflict_to_bus,
                mode=spec_obj.mode,
                run_id=run_id,
            )

            scheduler = Scheduler(spec=spec_obj, store=store)
            budget = Budget(
                usd_cap=spec_obj.budget_usd,
                wall_cap_s=spec_obj.wall_limit_s,
                max_subagent_calls=spec_obj.max_subagent_calls,
            )

            # Restore checkpointed state before constructing the Orchestrator.
            if resume and checkpoint_cycle is not None:
                scheduler.restore(sched_data)
                budget.restore(budget_data)

            # LLM client: real OpenRouter when OPENROUTER_API_KEY is set and the
            # run isn't forced offline; otherwise the deterministic test stub.
            if offline:
                llm = OfflineFixtureLLM()
            elif not os.environ.get("OPENROUTER_API_KEY"):
                llm = StubLLMClient()
            else:
                llm = OpenRouterClient(
                    model_map={
                        LLMTier.FAST: spec_obj.models.get("fast", "anthropic/claude-haiku-4.5"),
                        LLMTier.SMART: spec_obj.models.get("smart", "anthropic/claude-sonnet-4.6"),
                        LLMTier.HEAVY: spec_obj.models.get("heavy", "anthropic/claude-opus-4.6"),
                    },
                )
            # Wire the LLM's cost tracker into the store so snapshot_metrics
            # reports real USD spend instead of hardcoded 0.0.
            if hasattr(store, "set_cost_tracker") and hasattr(llm, "cost_tracker"):
                store.set_cost_tracker(llm.cost_tracker)

            # Optional Obsidian sink. CLI flag overrides the spec file.
            obsidian = None
            vault_path = obsidian_vault_override or spec_obj.obsidian_vault
            if vault_path:
                obsidian = ObsidianWriter(vault_path=vault_path)

            skill_registry = SkillRegistry()
            skill_registry.load_dir(Path("skills"))

            backend_resolver = BackendResolver()
            detected_cli = ", ".join(k.value for k in backend_resolver.detected) or "none"
            typer.echo(f"[researcher run] detected_cli={detected_cli}")

            def _runner_factory(kind: CliKind):
                cache_dir = run_dir / "cli_cache"
                post_mortem_dir = run_dir / "subagent_errors"
                if kind == CliKind.CLAUDE_CODE:
                    return ClaudeCodeRunner(
                        model="sonnet",
                        cache_dir=cache_dir,
                        post_mortem_dir=post_mortem_dir,
                    )
                if kind == CliKind.CODEX:
                    return CodexRunner(
                        model="o4-mini",
                        cache_dir=cache_dir,
                        post_mortem_dir=post_mortem_dir,
                    )
                raise ValueError(f"unsupported CLI kind: {kind}")

            # CLI flag overrides the spec's backend_policy.
            if backend != "auto":
                spec_obj = spec_obj.model_copy(update={"backend_policy": backend})
            try:
                choice = backend_resolver.pick(spec_obj.backend_policy)
            except Exception as exc:
                raise typer.BadParameter(str(exc)) from exc
            if choice.kind is None:
                typer.echo(
                    f"[researcher run] backend=native reason={choice.reason} "
                    f"cost_mode={'fixture' if offline else 'OpenRouter/API'}"
                )
            else:
                typer.echo(
                    f"[researcher run] backend={choice.kind.value} reason={choice.reason} "
                    "cost_mode=flat-fee-cli"
                )

            orch = Orchestrator(
                spec=spec_obj,
                store=store,
                llm=llm,
                bus=bus,
                writer=writer,
                scheduler=scheduler,
                budget=budget,
                run_id=run_id,
                max_parallel_agents=6,
                obsidian_writer=obsidian,
                resume_from=checkpoint_cycle,
                db_path=str(db_path),
                stop_file=str(stop_file or _stop_request_path(run_dir)),
            )
            orch.set_backend_resolver(backend_resolver)
            orch.set_cli_runner_factory(_runner_factory)
            orch.set_skill_registry(skill_registry)

            # Wire NativeAgentDeps when the backend may route to native agents
            # (backend != "cli"). The native path needs a SearchProvider, an
            # HttpFetcher, and a PromptRegistry. Tavily is the default provider
            # when TAVILY_API_KEY is set; fall back to an empty FileSeedsProvider.
            if backend != "cli":
                search_provider: SearchProvider
                if offline:
                    fixture_path = Path("tests") / "fixtures" / "offline_search" / f"{spec_obj.spec_id}.json"
                    if fixture_path.exists():
                        search_provider = FileSeedsProvider.from_file(fixture_path)
                        typer.echo(f"[researcher run] search=offline_fixtures path={fixture_path}")
                    else:
                        search_provider = FileSeedsProvider(seeds={})
                        typer.echo(
                            f"[researcher run] search=offline_fixtures missing path={fixture_path}",
                            err=True,
                        )
                elif (
                    spec_obj.search.provider == "tavily"
                    and os.environ.get(spec_obj.search.api_key_env or "TAVILY_API_KEY")
                ):
                    search_provider = TavilyProvider()
                    typer.echo(
                        f"[researcher run] search=tavily key_env={spec_obj.search.api_key_env or 'TAVILY_API_KEY'}"
                    )
                else:
                    search_provider = FileSeedsProvider(seeds={})
                    typer.echo(
                        f"[researcher run] search={spec_obj.search.provider} unavailable; "
                        f"missing {spec_obj.search.api_key_env or 'TAVILY_API_KEY'}, native discovery may return no web results",
                        err=True,
                    )
                chunk_dicts = _source_chunks_for_search(source_records, source_chunks)
                if chunk_dicts:
                    search_provider = SourcePackSearchProvider(
                        chunks=chunk_dicts,
                        fallback=search_provider,
                        policy=spec_obj.source_policy,
                        legacy_allowlist=spec_obj.domain_allowlist,
                    )
                else:
                    search_provider = PolicySearchProvider(
                        fallback=search_provider,
                        policy=spec_obj.source_policy,
                        legacy_allowlist=spec_obj.domain_allowlist,
                    )

                http_client = (
                    _offline_fixture_http_client(spec_obj.spec_id)
                    if offline
                    else None
                )
                http_fetcher = HttpFetcher(
                    politeness=PolitenessLimiter(),
                    client=http_client,
                    max_retries=1 if offline else 3,
                )
                native_deps = NativeAgentDeps(
                    search=search_provider,
                    http=http_fetcher,
                    prompts=default_registry(),
                    max_fetch_per_task=3,
                    skill_registry=skill_registry,
                    source_policy=spec_obj.source_policy,
                )
                orch.set_native_deps(native_deps)

            if interactive:
                from researcher.interrupts import StdinInterruptHandler

                orch.set_interrupt_handler(StdinInterruptHandler())

            reason = await orch.run()
            return reason
        finally:
            if tui_proc is not None:
                try:
                    await _asyncio.wait_for(tui_proc.wait(), timeout=1.0)
                except TimeoutError:
                    tui_proc.terminate()
                    try:
                        await _asyncio.wait_for(tui_proc.wait(), timeout=2.0)
                    except TimeoutError:
                        tui_proc.kill()
            if socket_server is not None:
                await socket_server.stop()
            if otel_adapter is not None:
                try:
                    otel_adapter.flush()
                except Exception as exc:
                    typer.echo(
                        f"[researcher run] otel flush error: {exc}",
                        err=True,
                    )
            await bus.stop()
    finally:
        await store.close()


@app.command()
def watch(
    run_id: str = typer.Argument(..., help="Run id to attach to."),
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir", help="Directory where run artifacts live."
    ),
    replay: bool = typer.Option(
        False, "--replay", help="Replay JSONL from disk instead of attaching to the live socket."
    ),
) -> None:
    """Attach the TUI to a running (or completed) run."""
    run_dir = runs_dir / run_id
    tui_entry = _tui_entry_or_build(auto_build=True)

    if replay:
        events_path = run_dir / "events.jsonl"
        if not events_path.exists():
            raise typer.BadParameter(f"no events file found at {events_path}")
        args = ["node", str(tui_entry), "--replay", str(events_path)]
    else:
        socket_path = run_dir / "events.sock"
        events_path = run_dir / "events.jsonl"
        if socket_path.exists():
            args = ["node", str(tui_entry)]
            if events_path.exists():
                args.extend(["--replay", str(events_path)])
            args.extend(["--socket", str(socket_path)])
        elif events_path.exists():
            typer.echo(
                f"[researcher watch] no live socket at {socket_path}; replaying events.jsonl",
                err=True,
            )
            args = ["node", str(tui_entry), "--replay", str(events_path)]
        else:
            raise typer.BadParameter(
                f"no live socket found at {socket_path}; use --replay for completed runs"
            )

    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise typer.Exit(result.returncode)


@app.command()
def inspect(
    run_id: str = typer.Argument(..., help="Run id to inspect."),
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir", help="Directory where run artifacts live."
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit compact JSON for automation.",
    ),
) -> None:
    """Print a summary of a completed run (entity count, fill %, cost, etc.)."""
    db_path = runs_dir / run_id / "store.duckdb"
    if not db_path.exists():
        raise typer.BadParameter(f"no store found at {db_path}")
    payload = _inspect_payload(db_path=db_path, run_id=run_id)
    if json_output:
        _json_echo(payload)
        return
    _print_inspect_human(payload)


@app.command()
def accept(
    run_id: str = typer.Argument(..., help="Run id to check."),
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir", help="Directory where run artifacts live."
    ),
    min_entities: int = typer.Option(
        100, "--min-entities", help="Minimum entity count for acceptance."
    ),
    min_field_fill: float = typer.Option(
        0.70, "--min-field-fill", help="Minimum strict field-fill ratio."
    ),
    max_cost_usd: float = typer.Option(
        3.0, "--max-cost-usd", help="Maximum accepted USD spend."
    ),
    max_wall_s: float = typer.Option(
        600.0, "--max-wall-s", help="Maximum accepted wall time in seconds."
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit compact JSON for automation.",
    ),
) -> None:
    """Check a completed run against the v1 acceptance metrics."""
    db_path = runs_dir / run_id / "store.duckdb"
    if not db_path.exists():
        raise typer.BadParameter(f"no store found at {db_path}")
    metrics = _inspect_payload(db_path=db_path, run_id=run_id)
    checks = {
        "entities": metrics["entities_total"] >= min_entities,
        "field_fill": metrics["strict_fields_filled_pct"] >= min_field_fill,
        "cost": float(metrics.get("cost_usd") or 0.0) <= max_cost_usd,
        "wall": float(metrics.get("wall_s") or 0.0) <= max_wall_s,
    }
    payload = {
        "run_id": run_id,
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_entities": min_entities,
            "min_field_fill": min_field_fill,
            "max_cost_usd": max_cost_usd,
            "max_wall_s": max_wall_s,
        },
        "metrics": metrics,
    }
    if json_output:
        _json_echo(payload)
    else:
        _print_accept_human(payload)
    if not payload["passed"]:
        raise typer.Exit(1)


def _inspect_payload(db_path: Path, run_id: str) -> dict:
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        entities_total = _scalar(con, "SELECT COUNT(*) FROM entities", 0)
        by_type_rows = _rows(
            con,
            "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type ORDER BY entity_type",
        )
        by_type = {str(row[0]): int(row[1]) for row in by_type_rows}
        conflicts_open = int(
            _scalar(
                con,
                "SELECT COUNT(*) FROM conflicts WHERE status = 'open'",
                0,
            )
        )
        persisted_total = int(_scalar(con, "SELECT COUNT(*) FROM fields", 0))
        persisted_filled = int(
            _scalar(con, "SELECT COUNT(*) FROM fields WHERE value_json != 'null'", 0)
        )
        persisted_fill_pct = (
            float(persisted_filled) / float(persisted_total)
            if persisted_total
            else 0.0
        )
        schema_fields = _schema_field_names(con)
        strict_total = int(entities_total) * len(schema_fields)
        strict_fill_pct = (
            float(persisted_filled) / float(strict_total)
            if strict_total
            else persisted_fill_pct
        )
        summary = _latest_summary(con, run_id)
        return {
            "run_id": run_id,
            "db_path": str(db_path),
            "entities_total": int(entities_total),
            "entities_by_type": by_type,
            "persisted_fields_filled_pct": persisted_fill_pct,
            "strict_fields_filled_pct": strict_fill_pct,
            "schema_fields": schema_fields,
            "conflicts_open": conflicts_open,
            "cost_usd": float(summary.get("cost_usd", 0.0) or 0.0),
            "wall_s": float(summary.get("wall_s", 0.0) or 0.0),
            "reason": summary.get("reason"),
            "subagent_calls_total": int(
                summary.get("subagent_calls_total", 0) or 0
            ),
            "summary": summary,
        }
    finally:
        con.close()


def _print_inspect_human(payload: dict[str, Any]) -> None:
    summary = payload.get("summary") or {}
    obsidian = summary.get("obsidian") if isinstance(summary, dict) else None
    obsidian_stats = obsidian.get("stats", {}) if isinstance(obsidian, dict) else {}
    _print_kv_table(
        [
            ("run_id", payload.get("run_id")),
            ("reason", payload.get("reason")),
            ("entities", payload.get("entities_total")),
            ("entities_by_type", json.dumps(payload.get("entities_by_type", {}), sort_keys=True)),
            ("strict_field_fill", _pct(float(payload.get("strict_fields_filled_pct") or 0.0))),
            ("persisted_field_fill", _pct(float(payload.get("persisted_fields_filled_pct") or 0.0))),
            ("conflicts_open", payload.get("conflicts_open")),
            ("cost_usd", f"${float(payload.get('cost_usd') or 0.0):.4f}"),
            ("wall_s", f"{float(payload.get('wall_s') or 0.0):.1f}"),
            ("subagent_calls", payload.get("subagent_calls_total")),
            ("obsidian_writes", obsidian_stats.get("writes", 0)),
            ("obsidian_errors", obsidian_stats.get("errors", 0)),
            ("db_path", payload.get("db_path")),
        ]
    )


def _print_accept_human(payload: dict[str, Any]) -> None:
    checks = payload.get("checks", {})
    metrics = payload.get("metrics", {})
    status = "PASS" if payload.get("passed") else "FAIL"
    typer.echo(f"acceptance: {status}")
    _print_kv_table(
        [
            ("entities", f"{checks.get('entities')} ({metrics.get('entities_total')})"),
            (
                "field_fill",
                f"{checks.get('field_fill')} ({_pct(float(metrics.get('strict_fields_filled_pct') or 0.0))})",
            ),
            ("cost", f"{checks.get('cost')} (${float(metrics.get('cost_usd') or 0.0):.4f})"),
            ("wall", f"{checks.get('wall')} ({float(metrics.get('wall_s') or 0.0):.1f}s)"),
        ]
    )


def _rows(con, sql: str, params: list | tuple = ()) -> list:
    try:
        return con.execute(sql, params).fetchall()
    except Exception:
        return []


def _scalar(con, sql: str, default):
    rows = _rows(con, sql)
    if not rows:
        return default
    return rows[0][0]


def _latest_summary(con, run_id: str) -> dict:
    rows = _rows(
        con,
        "SELECT summary_json FROM run_summary WHERE run_id = ? "
        "ORDER BY started_at DESC LIMIT 1",
        [run_id],
    )
    if not rows:
        return {}
    try:
        return json.loads(rows[0][0])
    except Exception:
        return {}


def _schema_field_names(con) -> list[str]:
    rows = _rows(
        con,
        "SELECT json_schema FROM schema_meta ORDER BY entity_class_name LIMIT 1",
    )
    if not rows:
        return []
    try:
        schema = json.loads(rows[0][0])
    except Exception:
        return []
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        return []
    return sorted(str(k) for k in props)


def _offline_fixture_http_client(spec_id: str):
    import httpx

    path = Path("tests") / "fixtures" / "offline_pages" / f"{spec_id}.json"
    pages = json.loads(path.read_text()) if path.exists() else {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        html = pages.get(url)
        if html is None:
            return httpx.Response(404, text="not found")
        return httpx.Response(
            200,
            text=html,
            headers={"content-type": "text/html"},
            request=request,
        )

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    )


def _source_chunks_for_search(sources: list, chunks: list) -> list[dict[str, Any]]:
    by_id = {
        source.source_id: source
        for source in sources
    }
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        source = by_id.get(chunk.source_id)
        if source is None:
            continue
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "title": source.title,
                "url": source.url,
                "text": chunk.text,
            }
        )
    return rows


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run id to resume."),
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir", help="Directory where run artifacts live."
    ),
    spec: Path | None = typer.Option(
        None,
        "--spec",
        exists=True,
        readable=True,
        help="Spec path for legacy runs without run.json metadata.",
    ),
    backend: str = typer.Option(
        "",
        "--backend",
        help="Optional backend override: auto | cli | api. Defaults to run metadata.",
    ),
    no_tui: bool = typer.Option(
        False, "--no-tui", help="Do not auto-launch the Ink TUI child process."
    ),
) -> None:
    """Resume a previously-paused run from its last checkpoint."""
    if backend and backend not in ("auto", "cli", "api"):
        raise typer.BadParameter(f"--backend must be one of auto|cli|api, got {backend!r}")
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise typer.BadParameter(f"run directory not found: {run_dir}")
    metadata = _read_run_metadata(run_dir)
    stored_spec = metadata.get("spec_path")
    if spec is None and not stored_spec:
        raise typer.BadParameter("no spec found in run metadata; pass --spec")
    spec_path = spec or Path(str(stored_spec))
    if not spec_path.exists():
        raise typer.BadParameter(f"spec not found: {spec_path}")
    resolved_backend = backend or str(metadata.get("backend") or "auto")
    offline = _coerce_bool(metadata.get("offline"), default=False)
    if offline and resolved_backend == "auto":
        resolved_backend = "api"
    if offline and resolved_backend == "cli":
        raise typer.BadParameter("stored run is offline and cannot resume with backend=cli")

    typer.echo(
        f"[researcher resume] run_id={run_id} spec={spec_path} backend={resolved_backend}"
    )
    try:
        _stop_request_path(run_dir).unlink()
    except FileNotFoundError:
        pass
    import asyncio as _asyncio

    reason = _asyncio.run(
        _execute_run(
            spec_path=spec_path,
            run_id=run_id,
            run_dir=run_dir,
            backend=resolved_backend,
            obsidian_vault_override=str(metadata.get("obsidian_vault") or ""),
            offline=offline,
            no_tui=no_tui or _coerce_bool(metadata.get("no_tui"), default=False),
            fast_startup=_coerce_bool(metadata.get("fast_startup"), default=False),
            resume=run_id,
            mode=str(metadata.get("mode") or "overwrite"),
            interactive=False,
            otel_endpoint="",
            stop_file=_stop_request_path(run_dir),
            items_path=None,
            item_column=str(metadata.get("item_column") or "name"),
        )
    )
    typer.echo(f"[researcher resume] done reason={reason.value}")
    _print_run_next_steps(run_id, runs_dir, run_dir)


@app.command()
def stop(
    run_id: str = typer.Argument(..., help="Run id to stop."),
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir", help="Directory where run artifacts live."
    ),
) -> None:
    """Send a graceful stop signal to a running orchestrator."""
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise typer.BadParameter(f"run directory not found: {run_dir}")
    payload = {
        "run_id": run_id,
        "requested_at": datetime.now(UTC).isoformat(),
    }
    _stop_request_path(run_dir).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    typer.echo(
        f"[researcher stop] requested graceful stop for {run_id}; "
        "the run will stop at the next cycle boundary"
    )


@app.command("export")
def export_run(
    run_id: str = typer.Argument(..., help="Run id to export."),
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir", help="Directory where run artifacts live."
    ),
    fmt: str = typer.Option(
        "json",
        "--format",
        help="Export format: json | csv | markdown | obsidian.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file or directory. Defaults to stdout except obsidian.",
    ),
) -> None:
    """Export completed run entities for non-SQL downstream use."""
    if fmt not in {"json", "csv", "markdown", "obsidian"}:
        raise typer.BadParameter("--format must be one of json|csv|markdown|obsidian")
    db_path = runs_dir / run_id / "store.duckdb"
    if not db_path.exists():
        raise typer.BadParameter(f"no store found at {db_path}")
    entities = _export_entities(db_path)

    if fmt == "json":
        text = json.dumps({"run_id": run_id, "entities": entities}, indent=2, default=str)
        _write_or_echo(text, output)
        return
    if fmt == "csv":
        text = _entities_to_csv(entities)
        _write_or_echo(text, output)
        return
    if fmt == "markdown":
        text = _entities_to_markdown(run_id, entities)
        _write_or_echo(text, output)
        return

    target = output or (runs_dir / run_id / "export_obsidian")
    target.mkdir(parents=True, exist_ok=True)
    for entity in entities:
        entity_type = str(entity.get("type") or "Entity")
        entity_name = str(entity.get("name") or entity.get("id"))
        type_dir = target / entity_type
        type_dir.mkdir(parents=True, exist_ok=True)
        (type_dir / f"{_safe_export_filename(entity_name)}.md").write_text(
            _entity_to_markdown(entity),
            encoding="utf-8",
        )
    typer.echo(str(target))


@sources_app.command("build")
def sources_build(
    spec: Path = typer.Argument(..., exists=True, readable=True, help="RunSpec with source_sets."),
    run_id: str = typer.Option("source-pack", "--run-id", help="Run id whose DuckDB store receives sources."),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Directory where run artifacts live."),
    json_output: bool = typer.Option(False, "--json", help="Emit compact JSON."),
) -> None:
    """Ingest source_sets from a spec into a run DuckDB store."""
    import asyncio as _asyncio

    async def _run() -> dict[str, Any]:
        from researcher.sources import ingest_source_sets
        from researcher.spec import load_spec
        from researcher.storage.duckdb_store import DuckDBKnowledgeStore

        spec_obj = load_spec(spec)
        records, chunks = ingest_source_sets(spec_obj.source_sets)
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        store = DuckDBKnowledgeStore(run_dir / "store.duckdb")
        await store.open()
        try:
            await store.record_sources(
                [r.__dict__ for r in records],
                [c.__dict__ for c in chunks],
            )
        finally:
            await store.close()
        return {
            "run_id": run_id,
            "db_path": str(run_dir / "store.duckdb"),
            "sources": len(records),
            "chunks": len(chunks),
        }

    payload = _asyncio.run(_run())
    if json_output:
        _json_echo(payload)
    else:
        _print_kv_table(payload.items())


@sources_app.command("inspect")
def sources_inspect(
    run_id: str = typer.Argument(..., help="Run id to inspect."),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Directory where run artifacts live."),
    json_output: bool = typer.Option(False, "--json", help="Emit compact JSON."),
) -> None:
    """List source-pack records stored in a completed run."""
    from researcher.reporting import source_summary

    db_path = runs_dir / run_id / "store.duckdb"
    if not db_path.exists():
        raise typer.BadParameter(f"no store found at {db_path}")
    rows = source_summary(db_path)
    if json_output:
        _json_echo({"run_id": run_id, "sources": rows})
        return
    if not rows:
        typer.echo("(no sources)")
        return
    for row in rows:
        typer.echo(
            f"{row['source_id']}  {row['source_type']:<6} chunks={row['chunks']:<3} "
            f"{row['title']}  {row['url']}"
        )


@app.command()
def evidence(
    run_id: str = typer.Argument(..., help="Run id to export evidence for."),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Directory where run artifacts live."),
    fmt: str = typer.Option("csv", "--format", help="Evidence format: csv | md | json."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file; defaults to stdout."),
) -> None:
    """Emit a quote/provenance-backed evidence matrix."""
    from researcher.reporting import evidence_rows, render_evidence_csv, render_evidence_markdown

    if fmt not in {"csv", "md", "json"}:
        raise typer.BadParameter("--format must be one of csv|md|json")
    db_path = runs_dir / run_id / "store.duckdb"
    if not db_path.exists():
        raise typer.BadParameter(f"no store found at {db_path}")
    rows = evidence_rows(db_path, run_id=run_id)
    if fmt == "json":
        _write_text_output(json.dumps({"run_id": run_id, "evidence": rows}, indent=2, default=str), output)
    elif fmt == "md":
        _write_text_output(render_evidence_markdown(rows), output)
    else:
        _write_text_output(render_evidence_csv(rows), output)


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run id to render."),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Directory where run artifacts live."),
    fmt: str = typer.Option("md", "--format", help="Report format: md | html | json."),
    template: str = typer.Option("analyst", "--template", help="Report template: analyst | systematic | brief."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file; defaults to stdout."),
) -> None:
    """Render a cited research report with source appendix."""
    from researcher.reporting import (
        evidence_rows,
        render_report,
        source_summary,
        verification_votes,
    )

    if fmt not in {"md", "html", "json"}:
        raise typer.BadParameter("--format must be one of md|html|json")
    if template not in {"analyst", "systematic", "brief"}:
        raise typer.BadParameter("--template must be one of analyst|systematic|brief")
    db_path = runs_dir / run_id / "store.duckdb"
    if not db_path.exists():
        raise typer.BadParameter(f"no store found at {db_path}")
    metrics = _inspect_payload(db_path=db_path, run_id=run_id)
    rendered = render_report(
        run_id=run_id,
        metrics=metrics,
        rows=evidence_rows(db_path, run_id=run_id),
        sources=source_summary(db_path),
        votes=verification_votes(db_path, run_id=run_id),
        template=template,
        fmt=fmt,
    )
    if isinstance(rendered, dict):
        _write_text_output(json.dumps(rendered, indent=2, default=str), output)
    else:
        _write_text_output(rendered, output)


@runs_app.command("list")
def runs_list(
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Directory where run artifacts live."),
    json_output: bool = typer.Option(False, "--json", help="Emit compact JSON."),
    limit: int = typer.Option(50, "--limit", help="Maximum rows."),
) -> None:
    """List known runs from run.json and run_summary."""
    rows = _run_history_rows(runs_dir)[:limit]
    if json_output:
        _json_echo({"runs": rows})
        return
    _print_runs(rows)


@runs_app.command("search")
def runs_search(
    query: str = typer.Argument(..., help="Case-insensitive text to search."),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Directory where run artifacts live."),
    json_output: bool = typer.Option(False, "--json", help="Emit compact JSON."),
) -> None:
    """Search run history by id, goal, spec, reason, or type."""
    q = query.lower()
    rows = [
        row
        for row in _run_history_rows(runs_dir)
        if q in json.dumps(row, default=str).lower()
    ]
    if json_output:
        _json_echo({"runs": rows})
        return
    _print_runs(rows)


def _write_or_echo(text: str, output: Path | None) -> None:
    if output is None:
        typer.echo(text)
        return
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    typer.echo(str(output))


def _export_entities(db_path: Path) -> list[dict[str, Any]]:
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        entity_rows = _rows(
            con,
            "SELECT id, entity_type, name FROM entities ORDER BY entity_type, name",
        )
        field_rows = _rows(
            con,
            "SELECT entity_id, field_name, value_json, confidence FROM fields "
            "ORDER BY field_name",
        )
    finally:
        con.close()

    by_entity: dict[str, dict[str, Any]] = {}
    for entity_id, entity_type, name in entity_rows:
        by_entity[str(entity_id)] = {
            "id": str(entity_id),
            "type": str(entity_type),
            "name": str(name),
            "fields": {},
            "confidence": {},
        }
    for entity_id, field_name, value_json, confidence in field_rows:
        item = by_entity.get(str(entity_id))
        if item is None:
            continue
        try:
            value = json.loads(value_json)
        except Exception:
            value = value_json
        item["fields"][str(field_name)] = value
        item["confidence"][str(field_name)] = float(confidence)
    return list(by_entity.values())


def _entities_to_csv(entities: list[dict[str, Any]]) -> str:
    import io

    field_names = sorted(
        {
            str(field)
            for entity in entities
            for field in (entity.get("fields") or {}).keys()
            if str(field) not in {"id", "type", "name"}
        }
    )
    columns = ["id", "type", "name", *field_names]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    for entity in entities:
        row: dict[str, Any] = {
            "id": entity.get("id"),
            "type": entity.get("type"),
            "name": entity.get("name"),
        }
        for field in field_names:
            value = (entity.get("fields") or {}).get(field)
            row[field] = json.dumps(value, default=str) if isinstance(value, (list, dict)) else value
        writer.writerow(row)
    return out.getvalue()


def _entities_to_markdown(run_id: str, entities: list[dict[str, Any]]) -> str:
    parts = [f"# researcher export: {run_id}", ""]
    for entity in entities:
        parts.append(_entity_to_markdown(entity))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _entity_to_markdown(entity: dict[str, Any]) -> str:
    fields = entity.get("fields") or {}
    lines = [
        f"# {entity.get('name') or entity.get('id')}",
        "",
        f"- type: {entity.get('type')}",
        f"- id: {entity.get('id')}",
    ]
    for field, value in sorted(fields.items()):
        if isinstance(value, (list, dict)):
            rendered = json.dumps(value, default=str)
        else:
            rendered = str(value)
        lines.append(f"- {field}: {rendered}")
    return "\n".join(lines)


def _safe_export_filename(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {" ", "-", "_"} else "-" for ch in name)
    safe = "-".join(safe.strip().split())
    return safe or "entity"


def _run_history_rows(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return rows
    for run_dir in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True):
        run_id = run_dir.name
        metadata = _read_run_metadata(run_dir)
        db_path = run_dir / "store.duckdb"
        summary: dict[str, Any] = {}
        if db_path.exists():
            try:
                summary = _inspect_payload(db_path=db_path, run_id=run_id)
            except Exception:
                summary = {}
        rows.append(
            {
                "run_id": run_id,
                "spec_path": metadata.get("spec_path", ""),
                "goal": (summary.get("summary") or {}).get("goal", metadata.get("goal", "")),
                "reason": summary.get("reason"),
                "entities_total": summary.get("entities_total", 0),
                "entities_by_type": summary.get("entities_by_type", {}),
                "cost_usd": summary.get("cost_usd", 0.0),
                "wall_s": summary.get("wall_s", 0.0),
                "db_path": str(db_path) if db_path.exists() else "",
            }
        )
    return rows


def _print_runs(rows: list[dict[str, Any]]) -> None:
    if not rows:
        typer.echo("(no runs)")
        return
    for row in rows:
        by_type = json.dumps(row.get("entities_by_type") or {}, sort_keys=True)
        typer.echo(
            f"{row['run_id']}  reason={row.get('reason') or '-'}  "
            f"entities={row.get('entities_total', 0)} {by_type}  "
            f"cost=${float(row.get('cost_usd') or 0.0):.4f}"
        )


@app.command()
def graph(
    run_id: str = typer.Argument(..., help="Run id to query."),
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir", help="Directory where run artifacts live."
    ),
    sql: str = typer.Option(
        "SELECT source_id, target_id, relation_label, confidence FROM entity_relations LIMIT 50",
        "--sql",
        help="SQL query to run against the run's DuckDB store.",
    ),
) -> None:
    """Run an ad-hoc SQL query against a completed run's entity_relations table."""
    import asyncio as _asyncio

    db_path = runs_dir / run_id / "store.duckdb"
    if not db_path.exists():
        raise typer.BadParameter(f"no store found at {db_path}")

    async def _run() -> None:
        from researcher.storage.duckdb_store import DuckDBKnowledgeStore

        store = DuckDBKnowledgeStore(db_path=db_path)
        await store.open()
        try:
            rows = await store.query(sql)
            for r in rows:
                typer.echo(json.dumps(r, default=str))
        finally:
            await store.close()

    _asyncio.run(_run())


@app.command()
def migrate(
    run_id: str = typer.Argument(..., help="Run id whose store to inspect."),
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir", help="Directory where run artifacts live."
    ),
) -> None:
    """List schema migrations recorded for a run (v1.2 audit trail)."""
    import asyncio as _asyncio

    db_path = runs_dir / run_id / "store.duckdb"
    if not db_path.exists():
        raise typer.BadParameter(f"no store found at {db_path}")

    async def _list() -> None:
        from researcher.storage.duckdb_store import DuckDBKnowledgeStore

        store = DuckDBKnowledgeStore(db_path=db_path)
        await store.open()
        try:
            rows = await store.query(
                "SELECT entity_class_name, migration_type, diff_added_json, "
                "       diff_removed_json, applied_at "
                "FROM schema_migrations ORDER BY applied_at"
            )
            if not rows:
                typer.echo("(no migrations recorded)")
                return
            for r in rows:
                typer.echo(
                    f"{r['applied_at']}  {r['entity_class_name']}  "
                    f"{r['migration_type']}  added={r['diff_added_json']}  "
                    f"removed={r['diff_removed_json']}"
                )
        finally:
            await store.close()

    _asyncio.run(_list())


@app.command()
def mcp(
    db: Path = typer.Option(
        ...,
        "--db",
        exists=True,
        readable=True,
        help="Path to a researcher DuckDB store (runs/<run_id>/store.duckdb).",
    ),
    allow_start_run: bool = typer.Option(
        False,
        "--allow-start-run",
        help=(
            "Expose the start_run tool, which shells out to "
            "`researcher run <spec>` via an async subprocess launcher. "
            "Disabled by default so the server is read-only."
        ),
    ),
) -> None:
    """Start a researcher MCP server reading JSON-RPC from stdin.

    Exposes the KnowledgeStore from ``--db`` over the MCP tool
    interface so external agents (Claude Desktop, Cursor, Gemini) can
    query and optionally trigger research runs. Responses are written
    to stdout as newline-delimited JSON. Cancel with Ctrl-C.
    """
    import asyncio as _asyncio

    from researcher.mcp.server import ResearcherMcpServer
    from researcher.mcp.stdio import serve_stdio
    from researcher.storage.duckdb_store import DuckDBKnowledgeStore

    async def _serve() -> None:
        store = DuckDBKnowledgeStore(db_path=db)
        await store.open()
        try:
            launcher = _build_run_launcher() if allow_start_run else None
            server = ResearcherMcpServer(store=store, run_launcher=launcher)
            await serve_stdio(server)
        finally:
            await store.close()

    _asyncio.run(_serve())


def _build_run_launcher():
    """Return an async launcher that spawns `researcher run <spec>`.

    The launcher only waits for the subprocess to *start*, not to
    complete — exposing start_run as a fire-and-forget trigger is the
    whole point of ``--allow-start-run``. Operators who want
    synchronous run execution should continue to invoke ``researcher
    run`` directly.
    """
    import asyncio as _asyncio
    import secrets
    import sys as _sys

    async def launcher(spec_path: str, run_id: str | None) -> str:
        resolved_run_id = run_id or secrets.token_hex(6)
        args = [
            _sys.executable,
            "-m",
            "researcher",
            "run",
            spec_path,
            "--run-id",
            resolved_run_id,
        ]
        await _asyncio.create_subprocess_exec(
            *args,
            stdin=_asyncio.subprocess.DEVNULL,
            stdout=_asyncio.subprocess.DEVNULL,
            stderr=_asyncio.subprocess.DEVNULL,
        )
        return resolved_run_id

    return launcher


@app.command()
def version() -> None:
    """Print researcher version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
