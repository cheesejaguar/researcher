"""Tests for researcher.spec — RunSpec, YAML loading, dynamic entity class construction."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from researcher.spec import (
    EntitySpec,
    FieldSpec,
    RunSpec,
    SearchConfig,
    build_entity_class,
    load_spec,
)

# ---------- RunSpec defaults ----------

def test_runspec_defaults():
    s = RunSpec(
        spec_id="wars",
        goal="Major interstate wars since 1500",
        entities=[
            EntitySpec(
                name="War",
                fields=[FieldSpec(name="name", type="str", required=True)],
                search_templates=["{q} war"],
            )
        ],
        seeds=["major wars since 1500"],
        models={"fast": "openrouter/hermes-3-8b", "smart": "openrouter/hermes-3-70b", "heavy": "anthropic/claude-sonnet-4-6"},
    )
    assert s.budget_usd == 3.0
    assert s.wall_limit_s == 600
    assert s.max_cycles == 5
    assert s.max_entities_per_cycle == 200
    assert s.max_depth == 3
    assert s.domain_allowlist == []
    assert s.distinct_pairs == []
    assert s.search.provider == "tavily"
    assert s.search.max_results == 10


def test_runspec_missing_required_raises():
    with pytest.raises(ValidationError):
        RunSpec(  # type: ignore[call-arg]
            spec_id="x",
            goal="g",
            seeds=[],
            models={"fast": "m"},
        )  # missing entities


# ---------- YAML loading ----------

WARS_YAML = """
spec_id: wars
goal: Major interstate wars since 1500
budget_usd: 2.5
wall_limit_s: 300
max_cycles: 3
entities:
  - name: War
    fields:
      - { name: name, type: str, required: true }
      - { name: start_year, type: int, required: true }
      - { name: end_year, type: int }
      - { name: belligerents, type: "list[str]" }
      - { name: outcome, type: str }
    search_templates:
      - "{q} war wikipedia"
seeds:
  - "major wars since 1500"
distinct_pairs:
  - ["Napoleonic Wars", "War of the Sixth Coalition"]
search:
  provider: tavily
  api_key_env: TAVILY_API_KEY
  max_results: 8
models:
  fast: openrouter/hermes-3-8b
  smart: openrouter/hermes-3-70b
  heavy: anthropic/claude-sonnet-4-6
"""


def test_load_spec_reads_yaml(tmp_path: Path):
    p = tmp_path / "wars.yaml"
    p.write_text(WARS_YAML)
    s = load_spec(p)

    assert s.spec_id == "wars"
    assert s.budget_usd == 2.5
    assert s.wall_limit_s == 300
    assert s.max_cycles == 3
    assert len(s.entities) == 1
    war = s.entities[0]
    assert war.name == "War"
    assert len(war.fields) == 5
    assert war.fields[0].name == "name"
    assert war.fields[0].required is True
    assert war.fields[2].required is False  # end_year has no required: true
    assert s.seeds == ["major wars since 1500"]
    assert s.distinct_pairs == [("Napoleonic Wars", "War of the Sixth Coalition")]
    assert s.search.provider == "tavily"
    assert s.search.api_key_env == "TAVILY_API_KEY"
    assert s.search.max_results == 8
    assert s.models["fast"] == "openrouter/hermes-3-8b"


# ---------- build_entity_class ----------

def test_build_entity_class_required_and_optional():
    spec = EntitySpec(
        name="War",
        fields=[
            FieldSpec(name="name", type="str", required=True),
            FieldSpec(name="start_year", type="int", required=True),
            FieldSpec(name="end_year", type="int"),  # optional
            FieldSpec(name="belligerents", type="list[str]"),
            FieldSpec(name="decisive", type="bool"),
        ],
        search_templates=["{q}"],
    )
    War = build_entity_class(spec)
    assert issubclass(War, BaseModel)
    assert War.__name__ == "War"

    # Required fields enforced
    with pytest.raises(ValidationError):
        War()  # type: ignore[call-arg]

    w = War(name="WWII", start_year=1939)
    assert w.name == "WWII"
    assert w.start_year == 1939
    assert w.end_year is None  # optional defaults to None
    assert w.belligerents is None
    assert w.decisive is None


def test_build_entity_class_list_fields_accept_lists():
    spec = EntitySpec(
        name="War",
        fields=[
            FieldSpec(name="name", type="str", required=True),
            FieldSpec(name="belligerents", type="list[str]"),
            FieldSpec(name="casualty_estimates", type="list[int]"),
        ],
        search_templates=[],
    )
    War = build_entity_class(spec)
    w = War(
        name="WWII",
        belligerents=["Allies", "Axis"],
        casualty_estimates=[70_000_000, 85_000_000],
    )
    assert w.belligerents == ["Allies", "Axis"]
    assert w.casualty_estimates == [70_000_000, 85_000_000]


def test_build_entity_class_date_field():
    spec = EntitySpec(
        name="Trial",
        fields=[
            FieldSpec(name="id", type="str", required=True),
            FieldSpec(name="start_date", type="date", required=True),
        ],
        search_templates=[],
    )
    Trial = build_entity_class(spec)
    t = Trial(id="NCT12345", start_date="2024-01-15")
    assert t.start_date == date(2024, 1, 15)


def test_build_entity_class_rejects_unknown_type():
    spec = EntitySpec(
        name="Bad",
        fields=[FieldSpec(name="x", type="banana")],
        search_templates=[],
    )
    with pytest.raises(ValueError, match="banana"):
        build_entity_class(spec)


def test_search_config_defaults():
    cfg = SearchConfig()
    assert cfg.provider == "tavily"
    assert cfg.api_key_env == "TAVILY_API_KEY"
    assert cfg.max_results == 10


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
