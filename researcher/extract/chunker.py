"""Heading-aware paragraph chunker.

Splits a long text document into chunks at heading boundaries (lines
starting with # in Markdown or <h1>-<h6> tags in HTML-stripped text),
falling back to paragraph-level splits when no headings are found.
Chunks are capped at max_chars.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    index: int
    heading: str
    text: str
    start_char: int
    end_char: int


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def chunk_text(text: str, max_chars: int = 4000) -> list[Chunk]:
    """Split text into chunks. Prefers heading boundaries; falls back to paragraphs."""
    if not text or not text.strip():
        return []

    # Try heading-based splits first.
    headings = list(_HEADING_RE.finditer(text))
    chunks: list[Chunk] = []
    if headings:
        starts: list[tuple[int, str]] = [
            (m.start(), m.group(2).strip()) for m in headings
        ]
        # Add a synthetic intro start at 0 if the first heading isn't at position 0.
        if starts[0][0] > 0:
            prelude = text[: starts[0][0]]
            if prelude.strip():
                starts.insert(0, (0, "(intro)"))
        for i, (start, heading) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
            segment = text[start:end]
            # Strip the heading line itself from the segment body unless it's the intro.
            if heading != "(intro)":
                segment = _HEADING_RE.sub("", segment, count=1)
            segment = segment.strip()
            if not segment:
                continue
            for sub in _split_to_max(segment, max_chars):
                chunks.append(
                    Chunk(
                        index=len(chunks),
                        heading=heading,
                        text=sub,
                        start_char=start,
                        end_char=end,
                    )
                )
        return chunks

    # No headings — split by paragraphs, group into chunks of at most max_chars.
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    buffer = ""
    buffer_start = 0
    char_cursor = 0
    for para in paragraphs:
        if len(buffer) + len(para) + 2 > max_chars and buffer:
            chunks.append(
                Chunk(
                    index=len(chunks),
                    heading="",
                    text=buffer.strip(),
                    start_char=buffer_start,
                    end_char=char_cursor,
                )
            )
            buffer = ""
            buffer_start = char_cursor
        if not buffer:
            buffer_start = char_cursor
        buffer += para + "\n\n"
        char_cursor += len(para) + 2
    if buffer.strip():
        chunks.append(
            Chunk(
                index=len(chunks),
                heading="",
                text=buffer.strip(),
                start_char=buffer_start,
                end_char=char_cursor,
            )
        )
    return chunks


def _split_to_max(text: str, max_chars: int) -> list[str]:
    """Split a single segment into pieces of at most max_chars, at paragraph boundaries when possible."""
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    paragraphs = _PARAGRAPH_SPLIT.split(text)
    buffer = ""
    for para in paragraphs:
        # If a single paragraph alone is already too big, hard-split it.
        if len(para) > max_chars:
            if buffer.strip():
                out.append(buffer.strip())
                buffer = ""
            start = 0
            while start < len(para):
                out.append(para[start : start + max_chars].strip())
                start += max_chars
            continue
        if len(buffer) + len(para) + 2 > max_chars and buffer:
            out.append(buffer.strip())
            buffer = ""
        buffer += para + "\n\n"
    if buffer.strip():
        out.append(buffer.strip())
    return [o for o in out if o]
