"""Content-hash cache for LLM responses.

Stores completed LLM responses keyed on a deterministic hash of
(model, messages, temperature). Disk-backed JSON file for cross-run
persistence. Not thread-safe — single asyncio event loop only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from researcher.llm.client import LLMResponse, LLMUsage


class LLMResponseCache:
    """JSON-file-backed content-hash cache for LLMResponse payloads.

    Each entry stores the serialized response plus a `cached_at` and
    `expires_at` timestamp. On read, expired entries are evicted; on
    hit the returned LLMResponse has `usage.cache_hit=True`.
    """

    def __init__(
        self,
        cache_path: Path | str,
        default_ttl_s: float = 7 * 86400,  # 1 week
    ) -> None:
        self._path = Path(cache_path)
        self._ttl = default_ttl_s
        self._entries: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            if isinstance(data, dict):
                self._entries = data.get("entries", {}) or {}
        except (json.JSONDecodeError, OSError):
            self._entries = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps({"entries": self._entries}, ensure_ascii=False))
        tmp.replace(self._path)

    def get(self, key: str) -> Optional[LLMResponse]:
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        if entry.get("expires_at", 0) < time.time():
            self._misses += 1
            del self._entries[key]
            self._save()
            return None
        self._hits += 1
        # Cached usage gets its cache_hit flag flipped to True on read.
        raw = entry["response"]
        usage_data = dict(raw["usage"])
        usage_data["cache_hit"] = True
        return LLMResponse(text=raw["text"], usage=LLMUsage(**usage_data))

    def put(self, key: str, response: LLMResponse) -> None:
        now = time.time()
        self._entries[key] = {
            "response": response.model_dump(mode="json"),
            "cached_at": now,
            "expires_at": now + self._ttl,
        }
        self._save()

    @property
    def stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "entries": len(self._entries),
        }
