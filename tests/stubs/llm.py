"""In-memory stub LLMClient for Wave 1 tests.

Reads deterministic cassettes keyed by content_hash. If no cassette matches,
returns a generic stub response. Embeddings are deterministic pseudo-vectors
derived from the input string so EntityResolver tests are reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from researcher.llm.client import LLMClient, LLMResponse, LLMUsage
from researcher.models import LLMTier

T = TypeVar("T", bound=BaseModel)


class StubLLMClient(LLMClient):
    def __init__(self, cassette_dir: Optional[Path] = None) -> None:
        self._cassettes: dict[str, Any] = {}
        if cassette_dir is not None and cassette_dir.exists():
            for p in cassette_dir.glob("*.json"):
                self._cassettes.update(json.loads(p.read_text()))
        self.calls: list[dict] = []

    def content_hash(self, messages: list[dict], tier: LLMTier) -> str:
        payload = json.dumps({"tier": tier.value, "messages": messages}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def complete(
        self,
        messages: list[dict],
        tier: LLMTier,
        task_id: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        key = self.content_hash(messages, tier)
        self.calls.append(
            {
                "kind": "complete",
                "tier": tier.value,
                "task_id": task_id,
                "hash": key,
                "messages": messages,
            }
        )
        if key in self._cassettes:
            entry = self._cassettes[key]
            return LLMResponse(
                text=entry["text"],
                usage=LLMUsage(
                    tokens_in=entry.get("tokens_in", 100),
                    tokens_out=entry.get("tokens_out", 50),
                    cost_usd=entry.get("cost_usd", 0.001),
                    model=f"stub/{tier.value}",
                    cache_hit=False,
                ),
            )
        return LLMResponse(
            text="STUB",
            usage=LLMUsage(
                tokens_in=100,
                tokens_out=50,
                cost_usd=0.001,
                model=f"stub/{tier.value}",
                cache_hit=False,
            ),
        )

    async def complete_structured(
        self,
        messages: list[dict],
        schema: type[T],
        tier: LLMTier,
        task_id: str,
        temperature: float = 0.0,
        schema_retry: bool = True,
    ) -> T:
        key = self.content_hash(messages, tier)
        self.calls.append(
            {
                "kind": "structured",
                "tier": tier.value,
                "task_id": task_id,
                "hash": key,
                "messages": messages,
            }
        )
        if key in self._cassettes:
            return schema.model_validate(self._cassettes[key]["data"])
        # No cassette: return the schema's defaults if possible.
        return schema()  # type: ignore[call-arg]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Deterministic pseudo-embeddings — reproducible across runs.

        Uses SHA256 byte slicing to fill a 384-dim vector in [-1, 1).
        """
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = []
            for i in range(384):
                b = h[i % len(h)]
                vec.append(((b / 255.0) * 2.0) - 1.0)
            out.append(vec)
        return out
