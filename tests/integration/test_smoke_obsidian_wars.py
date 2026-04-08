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
