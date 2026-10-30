"""Shared dependencies for native research agents.

Native agents (DiscoverAgent, ExpandAgent, VerifyAgent, EnrichAgent, CriticAgent)
all need the same external tools: a search provider, an HTTP fetcher, a prompt
registry. Instead of threading these through each constructor, they live on a
single dataclass the orchestrator builds once and passes into every native
agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from researcher.fetch.http import HttpFetcher
from researcher.llm.prompts import PromptRegistry
from researcher.search.base import SearchProvider
from researcher.skills.registry import SkillRegistry
from researcher.spec import RelationSpec, SourcePolicy


@dataclass
class NativeAgentDeps:
    """Shared dependencies for all native agents."""

    search: SearchProvider
    http: HttpFetcher
    prompts: PromptRegistry
    max_fetch_per_task: int = 3  # cap HTTP fetches per single agent.run()
    skill_registry: SkillRegistry | None = None
    source_policy: SourcePolicy | None = None
    # Declared relations the EnrichAgent may record. Empty means the agent
    # still discovers related entities but emits no graph edges.
    relations: list[RelationSpec] = field(default_factory=list)
    # Maximum relation-following depth. Agents at this depth do not spawn
    # further DISCOVER tasks.
    max_depth: int = 3
