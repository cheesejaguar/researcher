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
from pydantic import BaseModel, Field, create_model


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


class EntitySpec(BaseModel):
    name: str  # e.g. "War", "ClinicalTrial"
    fields: list[FieldSpec]
    search_templates: list[str] = Field(default_factory=list)


class SearchConfig(BaseModel):
    provider: str = "tavily"  # tavily | brave | serper | file_seeds
    api_key_env: str = "TAVILY_API_KEY"
    max_results: int = 10


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


# ---------- YAML loader ----------


def load_spec(path: Path | str) -> RunSpec:
    """Load a RunSpec from a YAML file."""
    data = yaml.safe_load(Path(path).read_text())
    return RunSpec.model_validate(data)


# ---------- Dynamic entity class construction ----------


def build_entity_class(spec: EntitySpec) -> type[BaseModel]:
    """Build a Pydantic model class from an EntitySpec.

    Required fields are declared with `...` (no default); optional fields default
    to None and are typed `Optional[T]`. The returned class is named after
    `spec.name` and is suitable for introspection by the KnowledgeStore's
    `init_schema` to derive DuckDB column types.
    """
    fields: dict[str, Any] = {}
    for f in spec.fields:
        py_type = _resolve_type(f.type)
        if f.required:
            fields[f.name] = (py_type, ...)
        else:
            fields[f.name] = (Optional[py_type], None)
    return create_model(spec.name, **fields)  # type: ignore[call-overload]
