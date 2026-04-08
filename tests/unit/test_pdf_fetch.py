"""Tests for pymupdf-backed PDF source handling (v1.3 #1)."""

from __future__ import annotations

import sys
from collections.abc import Callable

import httpx
import pymupdf
import pytest

from researcher.extract.chunker import chunk_text
from researcher.fetch.http import FetchResult, HttpFetcher
from researcher.fetch.pdf import PdfDoc, PdfExtractionError, extract_pdf
from researcher.fetch.politeness import PolitenessLimiter
from researcher.fetch.readability import extract_readable


def _build_pdf(pages: list[str]) -> bytes:
    """Build a minimal in-memory PDF with one page per string."""
    doc = pymupdf.open()
    try:
        for body in pages:
            page = doc.new_page()
            page.insert_text((72, 72), body)
        return doc.tobytes()
    finally:
        doc.close()


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, follow_redirects=True)


# ---------------------------------------------------------------------------
# pdf.py
# ---------------------------------------------------------------------------


def test_extract_pdf_raises_without_pymupdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Soft-import failure surfaces as PdfExtractionError."""
    # Force the `import pymupdf` inside extract_pdf to fail by mapping the
    # module name to None in sys.modules (import finds it, sees None, raises
    # ImportError).
    monkeypatch.setitem(sys.modules, "pymupdf", None)
    monkeypatch.setitem(sys.modules, "fitz", None)
    with pytest.raises(PdfExtractionError) as ei:
        extract_pdf(b"%PDF-1.4\n")
    assert "pymupdf" in str(ei.value).lower()


def test_extract_pdf_returns_doc() -> None:
    data = _build_pdf(["Hello PDF World"])
    doc = extract_pdf(data)
    assert isinstance(doc, PdfDoc)
    assert doc.page_count == 1
    assert len(doc.pages) == 1
    assert "Hello PDF World" in doc.full_text
    assert doc.pages[0].text.strip() != ""
    assert doc.pages[0].page_num == 1
    assert doc.pages[0].start_offset == 0
    assert doc.pages[0].end_offset == len(doc.full_text)


def test_extract_pdf_multi_page_offsets() -> None:
    data = _build_pdf(["Page One Alpha", "Page Two Beta", "Page Three Gamma"])
    doc = extract_pdf(data)
    assert doc.page_count == 3
    assert len(doc.pages) == 3
    # Each inserted string present in full_text in order.
    idx_a = doc.full_text.index("Alpha")
    idx_b = doc.full_text.index("Beta")
    idx_g = doc.full_text.index("Gamma")
    assert idx_a < idx_b < idx_g
    # Offsets are monotonically non-decreasing and each page's slice matches
    # the recorded text.
    prev_end = -1
    for i, page in enumerate(doc.pages):
        assert page.page_num == i + 1
        assert page.start_offset >= prev_end
        assert page.end_offset >= page.start_offset
        assert doc.full_text[page.start_offset : page.end_offset] == page.text
        prev_end = page.end_offset


def test_extract_pdf_raises_on_malformed() -> None:
    with pytest.raises(PdfExtractionError):
        extract_pdf(b"not a pdf")


# ---------------------------------------------------------------------------
# HttpFetcher — FetchResult extensions
# ---------------------------------------------------------------------------


def test_fetch_result_has_content_type_and_raw_bytes_defaults() -> None:
    fr = FetchResult(url="https://x.example/", status=200, content="", headers={}, ok=True)
    assert fr.content_type == ""
    assert fr.raw_bytes is None


async def test_http_fetcher_populates_raw_bytes_for_pdf() -> None:
    body = b"%PDF-1.4 fake body bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/pdf"},
        )

    client = _mock_client(handler)
    fetcher = HttpFetcher(
        politeness=PolitenessLimiter(default_rps=100.0),
        robots=None,
        client=client,
    )
    result = await fetcher.fetch("https://pdfs.example/paper.pdf")
    assert result.ok is True
    assert result.status == 200
    assert result.content_type == "application/pdf"
    assert result.raw_bytes == body
    assert result.content == ""
    await client.aclose()


async def test_http_fetcher_leaves_raw_bytes_none_for_html() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<html><body>hi</body></html>",
            headers={"content-type": "text/html; charset=utf-8"},
        )

    client = _mock_client(handler)
    fetcher = HttpFetcher(
        politeness=PolitenessLimiter(default_rps=100.0),
        robots=None,
        client=client,
    )
    result = await fetcher.fetch("https://example.com/page")
    assert result.ok is True
    assert result.raw_bytes is None
    assert result.content == "<html><body>hi</body></html>"
    # Params stripped, lowercased.
    assert result.content_type == "text/html"
    await client.aclose()


# ---------------------------------------------------------------------------
# extract_readable dispatch
# ---------------------------------------------------------------------------


def test_extract_readable_dispatches_to_pdf_path() -> None:
    data = _build_pdf(["Dispatch Readable Probe"])
    text = extract_readable("", raw_bytes=data, content_type="application/pdf")
    assert text is not None
    assert "Dispatch Readable Probe" in text


def test_extract_readable_passthrough_for_html() -> None:
    html = (
        "<!doctype html><html><body><article>"
        "<h1>Headline</h1>"
        "<p>Here is the main readable body of an HTML article with more than "
        "enough text to make trafilatura keep it as main content.</p>"
        "<p>Another paragraph, for good measure, that confirms this is the body.</p>"
        "</article></body></html>"
    )
    text = extract_readable(html, url="https://example.com/a")
    assert text is not None
    assert "main readable body" in text


def test_extract_readable_returns_none_on_pdf_failure() -> None:
    text = extract_readable("", raw_bytes=b"bad", content_type="application/pdf")
    assert text is None


# ---------------------------------------------------------------------------
# Chunker sanity on PDF-shaped text
# ---------------------------------------------------------------------------


def test_chunker_handles_pdf_shaped_text() -> None:
    page_a = "First page body. " * 50
    page_b = "Second page body. " * 50
    page_c = "Third page body. " * 50
    pdf_like = f"{page_a.strip()}\n\n{page_b.strip()}\n\n{page_c.strip()}"
    chunks = chunk_text(pdf_like, max_chars=400)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.text.strip() != ""
