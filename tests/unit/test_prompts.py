"""Tests for the prompt registry — uniform-random variant selection."""

from __future__ import annotations

import pytest

from researcher.llm.prompts import (
    PromptRegistry,
    PromptSet,
    PromptVariant,
    default_registry,
)


def _variant(name: str = "v1") -> PromptVariant:
    return PromptVariant(
        name=name,
        system=f"system-{name}",
        user_template=f"user-{name}: {{x}}",
    )


def test_register_then_pick_returns_variant():
    reg = PromptRegistry()
    reg.register(PromptSet(kind="discover", variants=[_variant("only")]))
    picked = reg.pick("discover")
    assert picked.name == "only"
    assert picked.system == "system-only"


def test_pick_unknown_kind_raises_keyerror():
    reg = PromptRegistry()
    with pytest.raises(KeyError):
        reg.pick("nope")


def test_seeded_pick_is_deterministic():
    reg = PromptRegistry()
    reg.register(
        PromptSet(
            kind="discover",
            variants=[_variant("a"), _variant("b"), _variant("c")],
        )
    )
    reg.seed(42)
    seq_a = [reg.pick("discover").name for _ in range(20)]
    reg.seed(42)
    seq_b = [reg.pick("discover").name for _ in range(20)]
    assert seq_a == seq_b


def test_default_registry_has_expected_kinds():
    reg = default_registry()
    for kind in ("discover", "expand", "verify", "enrich", "critic"):
        v = reg.pick(kind)
        assert v.system  # non-empty
        assert v.user_template  # non-empty


def test_seeded_pick_distribution_covers_all_variants():
    reg = PromptRegistry()
    reg.register(
        PromptSet(
            kind="discover",
            variants=[_variant("a"), _variant("b"), _variant("c")],
        )
    )
    reg.seed(123)
    seen = {reg.pick("discover").name for _ in range(200)}
    # With 200 draws over 3 variants, all three should appear with overwhelming probability.
    assert seen == {"a", "b", "c"}


def test_promptset_with_no_variants_raises_on_pick():
    reg = PromptRegistry()
    reg.register(PromptSet(kind="empty", variants=[]))
    with pytest.raises(ValueError):
        reg.pick("empty")
