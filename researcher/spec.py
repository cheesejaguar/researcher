"""Run-spec loading + dynamic entity class construction.

Each run is defined by a YAML file that declares the target entity schema, seed
queries, budgets, and model tier choices. At startup the orchestrator loads the
spec and uses `build_entity_class` to construct a Pydantic model for each entity
type — this is what the KnowledgeStore introspects to generate the DuckDB schema.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, create_model, model_validator

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


class FieldSpec(BaseModel):
    name: str
    type: str  # "str" | "int" | "float" | "bool" | "date" | "datetime" | "list[str]" | "list[int]"
    required: bool = False
    # Optional closed vocabulary. When set, build_entity_class constrains the
    # field to ``Literal[*enum]`` (or ``list[Literal[*enum]]`` for list types)
    # so extractors must choose one of the declared values. Only meaningful
    # for str / list[str] fields; ignored for numeric / date / bool types.
    enum: list[str] = Field(default_factory=list)

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


class EntitySpec(BaseModel):
    name: str  # e.g. "War", "ClinicalTrial"
    fields: list[FieldSpec]
    search_templates: list[str] = Field(default_factory=list)


class RelationSpec(BaseModel):
    """Declare a relation type that agents should look for between entities."""

    name: str  # e.g., "ally_of", "took_place_in"
    source_type: str
    target_type: str
    description: str = ""


class SearchConfig(BaseModel):
    provider: str = "tavily"  # tavily | brave | serper | file_seeds | exa_mcp
    api_key_env: str = "TAVILY_API_KEY"
    max_results: int = 10
    # v1.3 #3: opt-in adaptive quality-based routing across all configured
    # providers. When True, the CLI is expected to wrap the provider list
    # in an AdaptiveSearchRouter — capability flag only; actual CLI wiring
    # is deferred to a follow-up.
    adaptive: bool = False


class RunSpec(BaseModel):
    spec_id: str
    goal: str
    entities: list[EntitySpec]
    seeds: list[str]
    search: SearchConfig = Field(default_factory=SearchConfig)
    domain_allowlist: list[str] = Field(default_factory=list)
    distinct_pairs: list[tuple[str, str]] = Field(default_factory=list)
    budget_usd: float = 3.0
    wall_limit_s: int = 600
    max_cycles: int = 5
    max_entities_per_cycle: int = 200
    max_depth: int = 3
    models: dict[str, str]  # LLMTier value -> OpenRouter model id
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
