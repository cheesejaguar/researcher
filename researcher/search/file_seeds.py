"""Offline SearchProvider that reads fixture JSON files.

Used by tests and offline mode. Looks up results by query string from
a mapping loaded at construction time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from researcher.search.base import SearchResult


class FileSeedsProvider:
    name = "file_seeds"

    def __init__(self, seeds: dict[str, list[dict[str, Any]]]) -> None:
        self._seeds = seeds

    @classmethod
    def from_file(cls, path: Path | str) -> FileSeedsProvider:
        data = json.loads(Path(path).read_text())
        return cls(seeds=data)

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        raw = self._seeds.get(query, [])
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("snippet", ""),
                rank=i,
            )
            for i, r in enumerate(raw[:max_results])
        ]
