"""SearchProvider Protocol + shared models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""
    rank: int = 0


@runtime_checkable
class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]: ...
