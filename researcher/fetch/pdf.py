"""pymupdf-backed PDF text extraction.

Soft-imports pymupdf so the dependency stays optional (``uv sync --extra pdf``).
Exposes :func:`extract_pdf` which turns raw PDF bytes into a :class:`PdfDoc`
containing per-page text plus character offsets into the concatenated full
text. On import failure or malformed input we raise :class:`PdfExtractionError`
so callers can degrade gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class PdfExtractionError(RuntimeError):
    """Raised when PDF extraction cannot proceed (missing dep or malformed)."""


@dataclass
class PdfPage:
    page_num: int  # 1-indexed
    text: str
    start_offset: int  # char offset in the concatenated doc
    end_offset: int


@dataclass
class PdfDoc:
    pages: list[PdfPage]
    full_text: str
    page_count: int
    title: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


_PAGE_JOIN = "\n\n"


def extract_pdf(data: bytes) -> PdfDoc:
    """Extract readable text from PDF bytes.

    Raises :class:`PdfExtractionError` if pymupdf is not installed or the
    bytes cannot be parsed as a PDF document.
    """
    try:
        import pymupdf  # type: ignore[import-not-found]
    except ImportError as e:
        raise PdfExtractionError(
            "pymupdf not installed — install with 'uv sync --extra pdf'"
        ) from e

    doc = None
    try:
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except Exception as e:
            raise PdfExtractionError(f"malformed PDF: {e}") from e

        pages: list[PdfPage] = []
        parts: list[str] = []
        cursor = 0
        page_count = doc.page_count
        for i in range(page_count):
            try:
                page = doc.load_page(i)
                page_text = page.get_text("text") or ""
            except Exception as e:  # pragma: no cover — defensive
                raise PdfExtractionError(f"malformed PDF page {i + 1}: {e}") from e
            start = cursor
            parts.append(page_text)
            cursor += len(page_text)
            if i + 1 < page_count:
                cursor += len(_PAGE_JOIN)
            end = start + len(page_text)
            pages.append(
                PdfPage(
                    page_num=i + 1,
                    text=page_text,
                    start_offset=start,
                    end_offset=end,
                )
            )

        full_text = _PAGE_JOIN.join(parts)

        title: str | None = None
        metadata: dict[str, str] = {}
        raw_meta = getattr(doc, "metadata", None) or {}
        for k, v in raw_meta.items():
            if v is None:
                continue
            metadata[str(k)] = str(v)
        meta_title = metadata.get("title")
        if meta_title and meta_title.strip():
            title = meta_title.strip()

        return PdfDoc(
            pages=pages,
            full_text=full_text,
            page_count=page_count,
            title=title,
            metadata=metadata,
        )
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:  # pragma: no cover — defensive close
                pass
