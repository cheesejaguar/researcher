"""trafilatura wrapper — extracts main content from HTML."""

from __future__ import annotations

import trafilatura


def extract_readable(html: str, url: str | None = None) -> str | None:
    """Extract the main readable text from an HTML document.

    Returns None if extraction fails or the page has no main content.
    """
    if not html or not html.strip():
        return None
    try:
        text = trafilatura.extract(html, url=url, include_comments=False)
        return text
    except Exception:
        return None
