"""Local sentence-transformers embedder (MiniLM).

Wrapped to match the embed_fn Protocol: ``async def (list[str]) -> list[list[float]]``.
The model is lazy-loaded on the first call and cached at module level so
repeated instantiation of LocalEmbedder pays the load cost only once.
"""

from __future__ import annotations

import asyncio
from typing import Any

_MODEL: Any = None
_MODEL_LOCK = asyncio.Lock()


async def _load_model() -> Any:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    async with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        # Lazy import so the rest of the codebase doesn't pay the import cost.
        from sentence_transformers import SentenceTransformer

        def _load() -> Any:
            return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        _MODEL = await asyncio.to_thread(_load)
        return _MODEL


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with all-MiniLM-L6-v2 (384-dim)."""
    if not texts:
        return []
    model = await _load_model()

    def _encode() -> list[list[float]]:
        vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [list(map(float, vec)) for vec in vecs]

    return await asyncio.to_thread(_encode)


class LocalEmbedder:
    """Small wrapper that exposes the embed_fn Protocol as a class."""

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return await embed_texts(texts)
