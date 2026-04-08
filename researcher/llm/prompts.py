"""Prompt template registry with uniform-random variant selection.

Wave 1-C uses uniform random selection over fixed prompt variants.
A later version could swap in Thompson sampling once per-variant
success metrics are tracked.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class PromptVariant:
    name: str
    system: str
    user_template: str  # .format()-style with named placeholders


@dataclass
class PromptSet:
    kind: str
    variants: list[PromptVariant] = field(default_factory=list)

    def pick(self, rng: random.Random | None = None) -> PromptVariant:
        if not self.variants:
            raise ValueError(f"prompt set '{self.kind}' has no variants")
        r = rng or random
        return r.choice(self.variants)


class PromptRegistry:
    """Holds named PromptSets and selects variants uniformly at random.

    Use `seed()` to make selection deterministic for tests.
    """

    def __init__(self) -> None:
        self._sets: dict[str, PromptSet] = {}
        self._rng = random.Random()

    def register(self, prompt_set: PromptSet) -> None:
        self._sets[prompt_set.kind] = prompt_set

    def pick(self, kind: str) -> PromptVariant:
        if kind not in self._sets:
            raise KeyError(f"no prompt set registered for kind={kind!r}")
        return self._sets[kind].pick(self._rng)

    def get_set(self, kind: str) -> PromptSet:
        """Return the full PromptSet for a kind.

        Used by multi-variant agents (e.g. BoN-MAV VerifyAgent) that need to
        fan out across every registered variant rather than sample just one.
        """
        if kind not in self._sets:
            raise KeyError(f"no prompt set registered for kind={kind!r}")
        return self._sets[kind]

    def seed(self, seed: int) -> None:
        self._rng = random.Random(seed)


def default_registry() -> PromptRegistry:
    """Ships with baseline prompt sets for discover/expand/verify/enrich/critic."""
    reg = PromptRegistry()
    reg.register(
        PromptSet(
            kind="discover",
            variants=[
                PromptVariant(
                    name="discover_v1",
                    system=(
                        "You are a research agent that discovers new entities "
                        "matching a given topic. Return a JSON list of entity names."
                    ),
                    user_template=(
                        "Goal: {goal}\n"
                        "Seed query: {seed_query}\n"
                        "Return up to {max_entities} candidate entity names "
                        "as a JSON array."
                    ),
                ),
                PromptVariant(
                    name="discover_v2",
                    system=(
                        "You are an expert researcher. Given a topic, list the most "
                        "significant entities that match. Be comprehensive but precise."
                    ),
                    user_template=(
                        "Topic: {goal}\n"
                        "Query: {seed_query}\n"
                        "Produce a JSON array of up to {max_entities} entity names."
                    ),
                ),
            ],
        )
    )
    reg.register(
        PromptSet(
            kind="expand",
            variants=[
                PromptVariant(
                    name="expand_v1",
                    system=(
                        "You are a fact-extraction agent. Given an entity name and a "
                        "list of fields, return structured values matching the schema."
                    ),
                    user_template=(
                        "Entity: {entity_name}\n"
                        "Entity type: {entity_type}\n"
                        "Fields: {fields}\n"
                        "Return JSON matching the provided schema."
                    ),
                ),
            ],
        )
    )
    reg.register(
        PromptSet(
            kind="verify",
            variants=[
                PromptVariant(
                    name="verify_factual_consistency",
                    system=(
                        "You are a verifier focused on factual_consistency. Given a "
                        "conflict between candidate values for a field, pick the value "
                        "that is most consistent with established facts. Return JSON."
                    ),
                    user_template=(
                        "Entity: {entity_name}\n"
                        "Field: {field}\n"
                        "Candidates: {candidates}\n"
                        "Pick the most factually consistent value and explain why."
                    ),
                ),
                PromptVariant(
                    name="verify_source_quality",
                    system=(
                        "You are a verifier focused on source_quality. Given a conflict, "
                        "pick the value backed by the most authoritative source(s). "
                        "Return JSON."
                    ),
                    user_template=(
                        "Entity: {entity_name}\n"
                        "Field: {field}\n"
                        "Candidates: {candidates}\n"
                        "Pick the value with the most authoritative source citation."
                    ),
                ),
                PromptVariant(
                    name="verify_entity_resolution",
                    system=(
                        "You are a verifier focused on entity_resolution. Given a "
                        "conflict, determine whether the conflicting values actually "
                        "refer to the same entity or different ones. If different, "
                        "flag it. Otherwise pick the canonical value. Return JSON."
                    ),
                    user_template=(
                        "Entity: {entity_name}\n"
                        "Field: {field}\n"
                        "Candidates: {candidates}\n"
                        "Verify these refer to the same entity, then pick the canonical value."
                    ),
                ),
            ],
        )
    )
    reg.register(
        PromptSet(
            kind="enrich",
            variants=[
                PromptVariant(
                    name="enrich_v1",
                    system=(
                        "You are a relation-following research agent. Given an entity, "
                        "identify related entities worth researching next."
                    ),
                    user_template=(
                        "Entity: {entity_name} ({entity_type})\n"
                        "Known fields: {fields}\n"
                        "Return a JSON list of related entity names worth discovering."
                    ),
                ),
            ],
        )
    )
    reg.register(
        PromptSet(
            kind="critic",
            variants=[
                PromptVariant(
                    name="critic_v1",
                    system=(
                        "You are a critic that reviews research trajectories. Suggest "
                        "improvements to the agent's approach in a short skill card."
                    ),
                    user_template=(
                        "Task: {task_summary}\n"
                        "Trajectory: {trajectory}\n"
                        "Return a short skill card with a name and a one-paragraph description."
                    ),
                ),
            ],
        )
    )
    return reg
