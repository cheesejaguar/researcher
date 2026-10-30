"""Strict validation tests for user-authored run specs.

Spec typos used to be silently ignored: ``budget_usdd`` left the real budget
at its default while looking accepted. These tests pin the strict behavior so
bad specs fail at load time, before a run directory is created.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from researcher.spec import (
    EntitySpec,
    FieldSpec,
    RelationSpec,
    RunSpec,
    SearchConfig,
)


def _entity(name: str = "War") -> EntitySpec:
    return EntitySpec(
        name=name,
        fields=[FieldSpec(name="name", type="str", required=True)],
        search_templates=[],
    )


def _spec_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "spec_id": "wars",
        "goal": "Major wars",
        "entities": [_entity()],
        "seeds": ["wars"],
        "models": {"fast": "m-fast", "smart": "m-smart"},
    }
    kwargs.update(overrides)
    return kwargs


def test_unknown_toplevel_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="budget_usdd"):
        RunSpec(**_spec_kwargs(budget_usdd=3.0))


def test_unknown_nested_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="max_result"):
        SearchConfig(provider="tavily", max_result=5)


def test_unknown_entity_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="search_template"):
        EntitySpec(
            name="War",
            fields=[FieldSpec(name="name", type="str")],
            search_template=["{q}"],
        )


def test_multiple_entity_types_are_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly one entity type"):
        RunSpec(**_spec_kwargs(entities=[_entity("War"), _entity("Battle")]))


def test_zero_entity_types_are_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly one entity type"):
        RunSpec(**_spec_kwargs(entities=[]))


def test_models_reject_unknown_tier() -> None:
    with pytest.raises(ValidationError, match="unknown model tier"):
        RunSpec(**_spec_kwargs(models={"fast": "m", "turbo": "m"}))


def test_models_require_at_least_one_tier() -> None:
    with pytest.raises(ValidationError, match="at least one tier"):
        RunSpec(**_spec_kwargs(models={}))


def test_models_reject_empty_model_id() -> None:
    with pytest.raises(ValidationError, match="empty model id"):
        RunSpec(**_spec_kwargs(models={"fast": "   "}))


@pytest.mark.parametrize("spec_id", ["../escape", "a/b", ".hidden", "a..b", ""])
def test_spec_id_must_be_path_safe(spec_id: str) -> None:
    with pytest.raises(ValidationError):
        RunSpec(**_spec_kwargs(spec_id=spec_id))


def test_goal_must_be_non_empty() -> None:
    with pytest.raises(ValidationError, match="goal must be non-empty"):
        RunSpec(**_spec_kwargs(goal="   "))


def test_model_pricing_rejects_negative_prices() -> None:
    with pytest.raises(ValidationError, match="prices must be >= 0"):
        RunSpec(**_spec_kwargs(model_pricing={"m-fast": (0.1, -1.0)}))


def test_model_pricing_rejects_wrong_shape() -> None:
    with pytest.raises(ValidationError):
        RunSpec(**_spec_kwargs(model_pricing={"m-fast": (0.1, 0.2, 0.3)}))


def test_duplicate_relation_names_are_rejected() -> None:
    relation = RelationSpec(name="ally_of", source_type="War", target_type="War")
    with pytest.raises(ValidationError, match="duplicate relation name"):
        RunSpec(**_spec_kwargs(relations=[relation, relation]))


def test_relation_parts_must_be_non_empty() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        RelationSpec(name=" ", source_type="War", target_type="War")


def test_per_task_fraction_is_bounded() -> None:
    with pytest.raises(ValidationError):
        RunSpec(**_spec_kwargs(per_task_fraction=0))
    with pytest.raises(ValidationError):
        RunSpec(**_spec_kwargs(per_task_fraction=1.5))


def test_valid_strict_spec_roundtrips() -> None:
    spec = RunSpec(
        **_spec_kwargs(
            relations=[
                RelationSpec(
                    name="ally_of",
                    source_type="War",
                    target_type="War",
                    description="co-belligerents",
                )
            ],
            enable_critic=True,
            max_relations_per_cycle=3,
        )
    )
    assert spec.spec_id == "wars"
    assert spec.relations[0].name == "ally_of"
    assert spec.enable_critic is True
    assert spec.max_relations_per_cycle == 3
