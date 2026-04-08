"""Typer CLI — `researcher run | watch | inspect | resume | stop`.

Wave 1-B wires `run` into the real Orchestrator + EventBus + DuckDB store + CLI
subagent backend. The remaining subcommands (`watch`, `inspect`, `resume`,
`stop`) are still Wave 0 skeletons and land in Wave 1-E/F.
"""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

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


# ---------- commands ----------


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
) -> None:
    """Run a research job from a YAML spec against the CLI subagent backend."""
    if backend not in ("auto", "cli", "api"):
        raise typer.BadParameter(f"--backend must be one of auto|cli|api, got {backend!r}")
    if mode not in ("overwrite", "merge"):
        raise typer.BadParameter(f"--mode must be one of overwrite|merge, got {mode!r}")

    # When resuming, the run_id is dictated by --resume; the existing run
    # directory must already exist on disk so we can reopen its DuckDB.
    actual_run_id = resume if resume else (run_id or _generate_run_id(spec))
    run_dir = runs_dir / actual_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

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
            fast_startup=fast_startup,
            resume=resume,
            mode=mode,
            interactive=interactive,
            otel_endpoint=otel_endpoint,
        )
    )
    typer.echo(f"[researcher run] done reason={reason.value}")


async def _execute_run(
    spec_path: Path,
    run_id: str,
    run_dir: Path,
    backend: str,
    obsidian_vault_override: str,
    offline: bool,
    fast_startup: bool = False,
    resume: str = "",
    mode: str = "overwrite",
    interactive: bool = False,
    otel_endpoint: str = "",
) -> StopReason:
    """Construct every wire and execute :meth:`Orchestrator.run` once.

    Imports are lazy so `researcher --help` doesn't pull in DuckDB, Pydantic
    model factories, or sentence-transformers on every invocation.
    """
    from datetime import datetime

    import os

    from researcher.agents.native_deps import NativeAgentDeps
    from researcher.backends.cli_runner import ClaudeCodeRunner, CodexRunner
    from researcher.backends.models import CliKind
    from researcher.backends.resolver import BackendResolver
    from researcher.budget import Budget
    from researcher.events import (
        ConflictDetected,
        ConflictPayload,
        EventBus,
        FactWritten,
        FactWrittenPayload,
    )
    from researcher.fetch.http import HttpFetcher
    from researcher.fetch.politeness import PolitenessLimiter
    from researcher.integrations.obsidian import ObsidianWriter
    from researcher.llm.embedder import LocalEmbedder
    from researcher.llm.openrouter import OpenRouterClient
    from researcher.llm.prompts import default_registry
    from researcher.models import LLMTier
    from researcher.orchestrator import Orchestrator
    from researcher.scheduler import Scheduler
    from researcher.search.base import SearchProvider
    from researcher.search.file_seeds import FileSeedsProvider
    from researcher.search.tavily import TavilyProvider
    from researcher.spec import build_entity_class, load_spec
    from researcher.storage.duckdb_store import DuckDBKnowledgeStore
    from researcher.storage.resolver import DefaultEntityResolver
    from researcher.storage.writer import FactWriter
    from tests.stubs.llm import StubLLMClient

    spec_obj = load_spec(spec_path)

    if not spec_obj.entities:
        raise typer.BadParameter(f"spec {spec_path} declares no entity types")

    # Wave 1 supports one entity type per run — take the first.
    primary_entity = spec_obj.entities[0]
    entity_class = build_entity_class(primary_entity)
    entity_schema_dict = {
        "entity_type": primary_entity.name,
        "fields": [
            {"name": f.name, "type": f.type, "required": f.required} for f in primary_entity.fields
        ],
    }

    db_path = run_dir / "store.duckdb"
    store = DuckDBKnowledgeStore(db_path=db_path)
    await store.open()
    await store.init_schema(entity_class)

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
            if offline or not os.environ.get("OPENROUTER_API_KEY"):
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

            backend_resolver = BackendResolver()

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
            )
            orch.set_backend_resolver(backend_resolver)
            orch.set_cli_runner_factory(_runner_factory)

            # Wire NativeAgentDeps when the backend may route to native agents
            # (backend != "cli"). The native path needs a SearchProvider, an
            # HttpFetcher, and a PromptRegistry. Tavily is the default provider
            # when TAVILY_API_KEY is set; fall back to an empty FileSeedsProvider.
            if backend != "cli":
                search_provider: SearchProvider
                if (
                    spec_obj.search.provider == "tavily"
                    and os.environ.get(spec_obj.search.api_key_env or "TAVILY_API_KEY")
                ):
                    search_provider = TavilyProvider()
                else:
                    search_provider = FileSeedsProvider(seeds={})

                http_fetcher = HttpFetcher(politeness=PolitenessLimiter())
                native_deps = NativeAgentDeps(
                    search=search_provider,
                    http=http_fetcher,
                    prompts=default_registry(),
                    max_fetch_per_task=3,
                )
                orch.set_native_deps(native_deps)

            if interactive:
                from researcher.interrupts import StdinInterruptHandler

                orch.set_interrupt_handler(StdinInterruptHandler())

            reason = await orch.run()
            return reason
        finally:
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
    replay: bool = typer.Option(
        False, "--replay", help="Replay JSONL from disk instead of attaching to the live socket."
    ),
) -> None:
    """Attach the TUI to a running (or completed) run."""
    typer.echo(f"[researcher watch] run_id={run_id} replay={replay}")
    typer.echo("Wave 0 skeleton: TUI attach lands in Wave 1-F.")


@app.command()
def inspect(
    run_id: str = typer.Argument(..., help="Run id to inspect."),
) -> None:
    """Print a summary of a completed run (entity count, fill %, cost, etc.)."""
    typer.echo(f"[researcher inspect] run_id={run_id}")
    typer.echo("Wave 0 skeleton: read run_summary from runs/<run_id>/store.duckdb.")


@app.command()
def resume(
    run_id: str = typer.Argument(..., help="Run id to resume."),
) -> None:
    """Resume a previously-paused run from its last checkpoint."""
    typer.echo(f"[researcher resume] run_id={run_id}")
    typer.echo("Wave 0 skeleton: resume protocol lands in Wave 1-E.")


@app.command()
def stop(
    run_id: str = typer.Argument(..., help="Run id to stop."),
) -> None:
    """Send a graceful stop signal to a running orchestrator."""
    typer.echo(f"[researcher stop] run_id={run_id}")
    typer.echo("Wave 0 skeleton: stop protocol lands in Wave 1-E.")


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
