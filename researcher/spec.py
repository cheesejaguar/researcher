"""Run-spec loading + dynamic entity class construction.

Each run is defined by a YAML file that declares the target entity schema, seed
queries, budgets, and model tier choices. At startup the orchestrator loads the
spec and uses `build_entity_class` to construct a Pydantic model for each entity
type — this is what the KnowledgeStore introspects to generate the DuckDB schema.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
    model_validator,
)

# ---------- Supported field types ----------

# Mapping from the YAML type string to a (python_type, default) pair used by
# create_model. list[...] types are handled separately below.
_SCALAR_TYPES: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "date": date,
    "datetime": datetime,
}


def _resolve_type(type_str: str) -> type:
    """Resolve a type string (e.g. 'str', 'list[int]', 'list[str]') to a Python type."""
    type_str = type_str.strip()
    if type_str.startswith("list[") and type_str.endswith("]"):
        inner = type_str[5:-1].strip()
        if inner not in _SCALAR_TYPES:
            raise ValueError(f"unsupported list inner type: {inner!r}")
        return list[_SCALAR_TYPES[inner]]  # type: ignore[valid-type]
    if type_str not in _SCALAR_TYPES:
        raise ValueError(f"unsupported field type: {type_str!r}")
    return _SCALAR_TYPES[type_str]


# ---------- Spec models ----------


class _StrictModel(BaseModel):
    """Base for all user-authored spec models.

    ``extra="forbid"`` turns typos into hard errors instead of silently
    keeping the default value (e.g. ``budget_usdd`` used to leave the real
    budget at $3.00 while looking accepted).
    """

    model_config = ConfigDict(extra="forbid")


class FieldSpec(_StrictModel):
    name: str
    type: str  # "str" | "int" | "float" | "bool" | "date" | "datetime" | "list[str]" | "list[int]"
    required: bool = False
    # Optional closed vocabulary. When set, build_entity_class constrains the
    # field to ``Literal[*enum]`` (or ``list[Literal[*enum]]`` for list types)
    # so extractors must choose one of the declared values. Only meaningful
    # for str / list[str] fields; ignored for numeric / date / bool types.
    enum: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field name must be non-empty")
        return value

    @field_validator("type")
    @classmethod
    def _type_is_supported(cls, value: str) -> str:
        _resolve_type(value)
        return value

    @model_validator(mode="after")
    def _enum_requires_str_type(self) -> FieldSpec:
        if not self.enum:
            return self
        t = self.type.strip()
        if t != "str" and t != "list[str]":
            raise ValueError(
                f"field {self.name!r}: enum is only supported for 'str' "
                f"or 'list[str]' fields, got {self.type!r}"
            )
        return self


class EntitySpec(_StrictModel):
    name: str  # e.g. "War", "ClinicalTrial"
    fields: list[FieldSpec]
    search_templates: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("entity name must be non-empty")
        return value

    @model_validator(mode="after")
    def _field_names_unique(self) -> EntitySpec:
        names = [f.name for f in self.fields]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate field name(s): {', '.join(dupes)}")
        return self


class RelationSpec(_StrictModel):
    """Declare a relation type that agents should look for between entities."""

    name: str  # e.g., "ally_of", "took_place_in"
    source_type: str
    target_type: str
    description: str = ""

    @field_validator("name", "source_type", "target_type")
    @classmethod
    def _relation_parts_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("relation name and endpoint types must be non-empty")
        return value


class SearchConfig(_StrictModel):
    provider: Literal["tavily", "brave", "serper", "file_seeds", "exa_mcp"] = "tavily"
    api_key_env: str = "TAVILY_API_KEY"
    max_results: int = Field(default=10, gt=0)
    # v1.3 #3: opt-in adaptive quality-based routing across all configured
    # providers. When True, the CLI is expected to wrap the provider list
    # in an AdaptiveSearchRouter — capability flag only; actual CLI wiring
    # is deferred to a follow-up.
    adaptive: bool = False


class SourceSet(_StrictModel):
    """User-provided source pack declared in a RunSpec."""

    name: str
    paths: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    crawl_domains: list[str] = Field(default_factory=list)


class SourcePolicy(_StrictModel):
    """Coarse source controls for search/fetch/report confidence."""

    allow_domains: list[str] = Field(default_factory=list)
    deny_domains: list[str] = Field(default_factory=list)
    trusted_domains: list[str] = Field(default_factory=list)
    require_trusted: bool = False


ReportConfigFormat = Literal["md", "html", "json"]


def _default_report_formats() -> list[ReportConfigFormat]:
    return ["md"]


class ReportConfig(_StrictModel):
    template: Literal["analyst", "systematic", "brief"] = "analyst"
    formats: list[ReportConfigFormat] = Field(default_factory=_default_report_formats)


class VerificationConfig(_StrictModel):
    mode: Literal["standard", "council"] = "standard"
    models: list[str] = Field(default_factory=list)
    confidence_threshold: float = 0.65


class RunSpec(_StrictModel):
    spec_id: str
    goal: str
    entities: list[EntitySpec]
    seeds: list[str]
    search: SearchConfig = Field(default_factory=SearchConfig)
    domain_allowlist: list[str] = Field(default_factory=list)
    source_sets: list[SourceSet] = Field(default_factory=list)
    source_policy: SourcePolicy = Field(default_factory=SourcePolicy)
    items: list[dict[str, Any]] = Field(default_factory=list)
    report: ReportConfig = Field(default_factory=ReportConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    distinct_pairs: list[tuple[str, str]] = Field(default_factory=list)
    budget_usd: float = Field(default=3.0, gt=0)
    wall_limit_s: int = Field(default=600, gt=0)
    max_cycles: int = Field(default=5, gt=0)
    max_entities_per_cycle: int = Field(default=200, gt=0)
    max_depth: int = Field(default=3, ge=0)
    # Fraction of the total budget a single task/entity may consume.
    per_task_fraction: float = Field(default=0.05, gt=0, le=1)
    per_entity_fraction: float = Field(default=0.03, gt=0, le=1)
    models: dict[str, str]  # LLMTier value -> OpenRouter model id
    # Optional per-model price overrides in USD per million input/output
    # tokens. Values not present here fall back to the client's built-in table
    # or the provider-reported cost.
    model_pricing: dict[str, tuple[float, float]] = Field(default_factory=dict)
    # Optional trajectory critic. Disabled by default; when enabled the
    # scheduler emits at most one CRITIC task per run.
    enable_critic: bool = False
    # Cap on relation-following ENRICH tasks queued per cycle.
    max_relations_per_cycle: int = Field(default=25, ge=0)
    # Backend policy for CLI subagent offload.
    backend_policy: Literal["auto", "cli", "api"] = "auto"
    max_subagent_calls: int = 500
    subagent_timeout_s: int = 120
    # Per-subagent-call workload cap: how many entities each `claude -p` /
    # `codex exec` invocation is asked to return. Lower values = shorter
    # per-call wall time, higher throughput under a fixed timeout at the
    # cost of more subagent calls total. Native path ignores this.
    max_entities_per_subagent_call: int = 10
    # Obsidian integration — when set, a secondary ObsidianWriter sink
    # materializes FactClaims into Markdown files in this vault.
    obsidian_vault: Optional[str] = None
    # Declared relation types the agents should look for between entities.
    # Populated from the YAML spec; empty list means "no graph extraction".
    relations: list[RelationSpec] = Field(default_factory=list)
    # Human-in-the-loop interrupt points. When set, the orchestrator pauses
    # at each named point and consults the registered InterruptHandler for
    # a decision. Known points: "after_initial_seed", "after_cycle_end".
    interrupt_points: list[str] = Field(default_factory=list)
    # v1.2: bounded conditional revision on objective signals.
    # When True, the orchestrator re-dispatches a task exactly once per run
    # if its AgentResult carries ``needs_revision=True`` (e.g. empty
    # extraction, schema failure, sub-threshold confidence). Strictly opt-in
    # — defaults to False to preserve existing behavior. The revision is
    # cost-bounded by construction: at most one extra dispatch per task.
    enable_conditional_revision: bool = False
    # Cross-run accumulation policy:
    #   "overwrite" — each run resets entity state (legacy behavior).
    #   "merge"     — fields accumulate across runs with conflict-aware
    #                  promotion based on confidence and temporal provenance
    #                  (first_seen_run / last_seen_run / superseded_by_run).
    mode: Literal["overwrite", "merge"] = "overwrite"

    @field_validator("spec_id")
    @classmethod
    def _spec_id_is_safe(cls, value: str) -> str:
        """Spec IDs become directory names and run-ID prefixes.

        Restrict them to a conservative character set and reject path
        traversal so a hostile spec cannot write outside ``runs/``.
        """
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
            raise ValueError(
                "spec_id must start with a letter or digit and contain only "
                "letters, digits, '.', '_' or '-'"
            )
        if ".." in value:
            raise ValueError("spec_id must not contain '..'")
        return value

    @field_validator("goal")
    @classmethod
    def _goal_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("goal must be non-empty")
        return value

    @field_validator("models")
    @classmethod
    def _models_are_valid(cls, value: dict[str, str]) -> dict[str, str]:
        allowed = {"fast", "smart", "heavy"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                f"unknown model tier(s): {', '.join(unknown)}; "
                f"expected a subset of {', '.join(sorted(allowed))}"
            )
        if not value:
            raise ValueError("models must define at least one tier")
        empty = sorted(k for k, v in value.items() if not v.strip())
        if empty:
            raise ValueError(f"empty model id for tier(s): {', '.join(empty)}")
        return value

    @field_validator("model_pricing")
    @classmethod
    def _pricing_is_non_negative(
        cls, value: dict[str, tuple[float, float]]
    ) -> dict[str, tuple[float, float]]:
        for model, pair in value.items():
            if len(pair) != 2:
                raise ValueError(
                    f"model_pricing[{model!r}] must be [input_usd_per_mtok, output_usd_per_mtok]"
                )
            if any(price < 0 for price in pair):
                raise ValueError(f"model_pricing[{model!r}] prices must be >= 0")
        return value

    @model_validator(mode="after")
    def _single_entity_type(self) -> RunSpec:
        """The store/agents are built around one entity type per run.

        Silently using only ``entities[0]`` hid the rest of a multi-entity
        spec; reject it instead so authors split the run or fix the spec.
        """
        if len(self.entities) != 1:
            raise ValueError(
                f"exactly one entity type is supported per run, got {len(self.entities)}"
            )
        return self

    @model_validator(mode="after")
    def _relation_names_unique(self) -> RunSpec:
        names = [r.name for r in self.relations]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate relation name(s): {', '.join(dupes)}")
        return self


# ---------- YAML loader ----------


def load_spec(path: Path | str) -> RunSpec:
    """Load a RunSpec from a YAML file."""
    data = yaml.safe_load(Path(path).read_text())
    return RunSpec.model_validate(data)


# ---------- Dynamic entity class construction ----------


def build_entity_class(spec: EntitySpec) -> type[BaseModel]:
    """Build a Pydantic model class from an EntitySpec.

    Required fields are declared with `...` (no default); optional fields default
    to None and are typed `Optional[T]`. Fields with a non-empty ``enum`` are
    narrowed to a ``Literal[...]`` over that vocabulary (or a list of literals
    for ``list[str]`` fields). The returned class is named after ``spec.name``
    and is suitable for introspection by the KnowledgeStore's ``init_schema``
    to derive DuckDB column types.
    """
    fields: dict[str, Any] = {}
    for f in spec.fields:
        py_type: Any
        if f.enum:
            # Literal[*enum] — closed vocabulary. Fall through to the
            # required/optional wrapping below.
            literal_type = Literal[tuple(f.enum)]  # type: ignore[valid-type]
            if f.type.strip() == "list[str]":
                py_type = list[literal_type]  # type: ignore[valid-type]
            else:
                py_type = literal_type  # type: ignore[assignment]
        else:
            py_type = _resolve_type(f.type)
        if f.required:
            fields[f.name] = (py_type, ...)
        else:
            fields[f.name] = (Optional[py_type], None)
    return create_model(spec.name, **fields)  # type: ignore[call-overload]
