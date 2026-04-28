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
    assert s.source_sets == []
    assert s.source_policy.deny_domains == []
    assert s.items == []
    assert s.report.template == "analyst"
    assert s.verification.mode == "standard"


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


def test_runspec_competitor_gap_sections_parse(tmp_path: Path):
    source_file = tmp_path / "source.md"
    source_file.write_text("source text")
    p = tmp_path / "with_sources.yaml"
    p.write_text(
        f"""
spec_id: sourced
goal: Source grounded run
entities:
  - name: Thing
    fields:
      - {{ name: name, type: str, required: true }}
    search_templates: []
seeds:
  - "thing"
source_sets:
  - name: local
    paths:
      - "{source_file}"
    urls:
      - "https://example.com/source"
source_policy:
  allow_domains: ["example.com"]
  deny_domains: ["bad.example"]
  trusted_domains: ["example.com"]
items:
  - {{ name: "Item A", id: "a" }}
report:
  template: systematic
  formats: ["md", "json"]
verification:
  mode: council
  models: ["fast-a", "fast-b"]
  confidence_threshold: 0.7
models:
  fast: m
"""
    )
    s = load_spec(p)
    assert s.source_sets[0].name == "local"
    assert s.source_policy.trusted_domains == ["example.com"]
    assert s.items[0]["name"] == "Item A"
    assert s.report.template == "systematic"
    assert s.verification.mode == "council"


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


# ---------- FieldSpec.enum ----------


def test_field_spec_enum_on_str_field_is_valid():
    f = FieldSpec(name="color", type="str", enum=["red", "green", "blue"])
    assert f.enum == ["red", "green", "blue"]


def test_field_spec_enum_on_list_str_field_is_valid():
    f = FieldSpec(name="tags", type="list[str]", enum=["a", "b"])
    assert f.enum == ["a", "b"]


def test_field_spec_enum_on_int_field_is_rejected():
    with pytest.raises(ValidationError) as exc:
        FieldSpec(name="count", type="int", enum=["1", "2"])
    assert "enum is only supported" in str(exc.value)


def test_field_spec_enum_on_date_field_is_rejected():
    with pytest.raises(ValidationError):
        FieldSpec(name="when", type="date", enum=["yesterday", "today"])


def test_build_entity_class_with_enum_accepts_valid_value():
    spec = EntitySpec(
        name="Thing",
        fields=[
            FieldSpec(name="name", type="str", required=True),
            FieldSpec(name="color", type="str", enum=["red", "green", "blue"]),
        ],
    )
    Cls = build_entity_class(spec)
    obj = Cls(name="widget", color="red")
    assert obj.color == "red"


def test_build_entity_class_with_enum_rejects_invalid_value():
    spec = EntitySpec(
        name="Thing",
        fields=[
            FieldSpec(name="name", type="str", required=True),
            FieldSpec(name="color", type="str", enum=["red", "green"]),
        ],
    )
    Cls = build_entity_class(spec)
    with pytest.raises(ValidationError):
        Cls(name="widget", color="purple")


def test_build_entity_class_with_list_enum_accepts_valid_list():
    spec = EntitySpec(
        name="Thing",
        fields=[
            FieldSpec(name="name", type="str", required=True),
            FieldSpec(name="tags", type="list[str]", enum=["hot", "cold"]),
        ],
    )
    Cls = build_entity_class(spec)
    obj = Cls(name="x", tags=["hot", "cold"])
    assert obj.tags == ["hot", "cold"]


def test_build_entity_class_with_list_enum_rejects_invalid_member():
    spec = EntitySpec(
        name="Thing",
        fields=[
            FieldSpec(name="name", type="str", required=True),
            FieldSpec(name="tags", type="list[str]", enum=["hot", "cold"]),
        ],
    )
    Cls = build_entity_class(spec)
    with pytest.raises(ValidationError):
        Cls(name="x", tags=["lukewarm"])


def test_build_entity_class_enum_optional_field_allows_none():
    spec = EntitySpec(
        name="Thing",
        fields=[
            FieldSpec(name="name", type="str", required=True),
            FieldSpec(name="color", type="str", enum=["red", "green"]),  # not required
        ],
    )
    Cls = build_entity_class(spec)
    obj = Cls(name="x")  # color omitted
    assert obj.color is None


def test_runspec_max_entities_per_subagent_call_default():
    s = RunSpec(
        spec_id="x",
        goal="g",
        entities=[
            EntitySpec(
                name="T",
                fields=[FieldSpec(name="name", type="str", required=True)],
                search_templates=[],
            )
        ],
        seeds=["s"],
        models={"fast": "m"},
    )
    assert s.max_entities_per_subagent_call == 10


def test_runspec_max_entities_per_subagent_call_accepts_override():
    s = RunSpec(
        spec_id="x",
        goal="g",
        entities=[
            EntitySpec(
                name="T",
                fields=[FieldSpec(name="name", type="str", required=True)],
                search_templates=[],
            )
        ],
        seeds=["s"],
        models={"fast": "m"},
        max_entities_per_subagent_call=3,
    )
    assert s.max_entities_per_subagent_call == 3


def test_expand_agent_response_model_honors_enum():
    """ExpandAgent's runtime response model must also narrow to Literal."""
    from researcher.agents.expand import _build_response_model

    field_specs = [
        {"name": "name", "type": "str", "required": True, "enum": []},
        {"name": "color", "type": "str", "required": False, "enum": ["red", "blue"]},
        {"name": "size", "type": "str", "required": False, "enum": []},
    ]
    Model = _build_response_model("Thing", field_specs, allowed=None)
    # Valid enum value → OK.
    obj = Model(color="red", size="large")
    assert obj.color == "red"
    assert obj.size == "large"
    # Invalid enum value → ValidationError.
    with pytest.raises(ValidationError):
        Model(color="purple")


def test_load_spec_supports_enum_in_yaml(tmp_path: Path):
    p = tmp_path / "s.yaml"
    p.write_text(
        """
spec_id: t
goal: t
entities:
  - name: Thing
    fields:
      - { name: name,  type: str, required: true }
      - { name: color, type: str, enum: [red, green, blue] }
seeds: ["x"]
models:
  fast: stub
  smart: stub
  heavy: stub
"""
    )
    spec = load_spec(p)
    color_field = next(f for f in spec.entities[0].fields if f.name == "color")
    assert color_field.enum == ["red", "green", "blue"]
    # And the generated class enforces the vocabulary end-to-end.
    Cls = build_entity_class(spec.entities[0])
    with pytest.raises(ValidationError):
        Cls(name="x", color="purple")
