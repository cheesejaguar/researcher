"""Lexical StubEntityResolver for Wave 1 agent tests.

No embeddings, no LLM — pure string matching:
  - normalized-equal  -> AUTO_MERGE
  - Jaccard >= 0.6    -> PENDING
  - else              -> DISTINCT
"""

from __future__ import annotations

from uuid import uuid4

from researcher.storage.resolver import EntityResolver, ResolveDecision, ResolveResult


def _normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _tokens(s: str) -> set[str]:
    return set(_normalize(s).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class StubEntityResolver(EntityResolver):
    def __init__(self) -> None:
        self._names: dict[tuple[str, str], str] = {}  # (type, normalized_name) -> id
        self._blocked: set[frozenset[str]] = set()

    async def resolve(
        self, entity_type: str, name: str, context: dict | None = None
    ) -> ResolveResult:
        norm = _normalize(name)
        exact_key = (entity_type, norm)
        if exact_key in self._names:
            return ResolveResult(
                decision=ResolveDecision.AUTO_MERGE,
                entity_id=self._names[exact_key],
                similarity=1.0,
                candidate_ids=[self._names[exact_key]],
            )

        best_sim = 0.0
        best_id = ""
        candidates: list[tuple[str, float]] = []
        my_tokens = _tokens(name)
        for (etype, ename), eid in self._names.items():
            if etype != entity_type:
                continue
            sim = _jaccard(my_tokens, _tokens(ename))
            candidates.append((eid, sim))
            if sim > best_sim:
                best_sim = sim
                best_id = eid
        candidates.sort(key=lambda p: p[1], reverse=True)
        top_ids = [c[0] for c in candidates[:5]]

        if best_sim >= 0.6 and best_id:
            return ResolveResult(
                decision=ResolveDecision.PENDING,
                entity_id=best_id,
                similarity=best_sim,
                candidate_ids=top_ids,
            )

        new_id = uuid4().hex
        self._names[exact_key] = new_id
        return ResolveResult(
            decision=ResolveDecision.DISTINCT,
            entity_id=new_id,
            similarity=best_sim,
            candidate_ids=top_ids,
        )

    async def add_distinct_pair(self, id_a: str, id_b: str) -> None:
        self._blocked.add(frozenset({id_a, id_b}))

    async def is_blocked(self, id_a: str, id_b: str) -> bool:
        return frozenset({id_a, id_b}) in self._blocked
