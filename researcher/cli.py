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
        f"backend={backend} run_dir={run_dir}"
        + (f" resume={resume}" if resume else "")
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
) -> StopReason:
    """Construct every wire and execute :meth:`Orchestrator.run` once.

    Imports are lazy so `researcher --help` doesn't pull in DuckDB, Pydantic
    model factories, or sentence-transformers on every invocation.
    """
    from datetime import datetime

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
    from researcher.integrations.obsidian import ObsidianWriter
    from researcher.llm.embedder import LocalEmbedder
    from researcher.orchestrator import Orchestrator
    from researcher.scheduler import Scheduler
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
            {"name": f.name, "type": f.type, "required": f.required}
            for f in primary_entity.fields
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
            raise typer.BadParameter(
                f"no checkpoint found for run_id={resume!r}"
            )
        checkpoint_cycle = int(ckpt["cycle"])
        sched_data = ckpt["data"].get("scheduler", {})
        budget_data = ckpt["data"].get("budget", {})

    try:
        bus = EventBus(jsonl_path=run_dir / "events.jsonl")
        await bus.start()

        try:
            # Wave 1.1: prefer the real LocalEmbedder (sentence-transformers
            # MiniLM, 384-dim) over the hash-based placeholder. The
            # ``--fast-startup`` flag (and ``offline``) keep the old behavior
            # for tests/CI that can't pay the MiniLM cold-load cost.
            embed_fn = (
                _default_embed_fn if (fast_startup or offline) else LocalEmbedder()
            )
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

            # Wave 1-B: stub LLM client. Wave 1-C swaps in an OpenRouter-backed one.
            llm = StubLLMClient()

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

            if interactive:
                from researcher.interrupts import StdinInterruptHandler

                orch.set_interrupt_handler(StdinInterruptHandler())

            reason = await orch.run()
            return reason
        finally:
            await bus.stop()
    finally:
        await store.close()


@app.command()
def watch(
    run_id: str = typer.Argument(..., help="Run id to attach to."),
    replay: bool = typer.Option(False, "--replay", help="Replay JSONL from disk instead of attaching to the live socket."),
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
def version() -> None:
    """Print researcher version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
