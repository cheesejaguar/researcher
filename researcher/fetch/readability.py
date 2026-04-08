"""trafilatura wrapper — extracts main content from HTML or PDF bytes."""

from __future__ import annotations

import trafilatura

from researcher.fetch.pdf import PdfExtractionError, extract_pdf


def _looks_like_pdf(url: str | None, content_type: str) -> bool:
    if content_type == "application/pdf":
        return True
    if url and url.lower().split("?", 1)[0].rstrip("/").endswith(".pdf"):
        return True
    return False


def extract_readable(
    html: str,
    url: str | None = None,
    raw_bytes: bytes | None = None,
    content_type: str = "",
) -> str | None:
    """Extract the main readable text from an HTML or PDF document.

    For PDF sources, callers pass the raw response bytes via ``raw_bytes``
    along with ``content_type="application/pdf"`` (or a ``url`` ending in
    ``.pdf``); the function then delegates to :func:`researcher.fetch.pdf.extract_pdf`.
    On PDF extraction failure we return ``None`` rather than raising, matching
    the existing HTML fallthrough behavior.

    Returns ``None`` if extraction fails or the page has no main content.
    """
    if raw_bytes is not None and _looks_like_pdf(url, content_type):
        try:
            doc = extract_pdf(raw_bytes)
        except PdfExtractionError:
            return None
        text = doc.full_text
        return text if text and text.strip() else None

    if not html or not html.strip():
        return None
    try:
        text = trafilatura.extract(html, url=url, include_comments=False)
        return text
    except Exception:
        return None
