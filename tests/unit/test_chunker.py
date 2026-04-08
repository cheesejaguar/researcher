"""Tests for the heading-aware paragraph chunker."""

from __future__ import annotations

from researcher.extract.chunker import chunk_text


def test_empty_returns_empty_list() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == [] or all(not c.text for c in chunk_text("   \n\n  "))


def test_heading_split_produces_heading_labeled_chunks() -> None:
    text = (
        "# Intro\n"
        "Paragraph about intro.\n\n"
        "# Methods\n"
        "How we did it.\n\n"
        "# Results\n"
        "What we found.\n"
    )
    chunks = chunk_text(text, max_chars=4000)
    assert len(chunks) == 3
    headings = [c.heading for c in chunks]
    assert headings == ["Intro", "Methods", "Results"]
    assert "Paragraph about intro" in chunks[0].text
    assert "How we did it" in chunks[1].text
    assert "What we found" in chunks[2].text
    # indices are sequential
    assert [c.index for c in chunks] == [0, 1, 2]


def test_no_headings_falls_back_to_paragraphs() -> None:
    text = "First paragraph here.\n\nSecond paragraph over there.\n\nThird and final one."
    chunks = chunk_text(text, max_chars=4000)
    assert len(chunks) == 1
    assert "First paragraph" in chunks[0].text
    assert "Third and final" in chunks[0].text
    assert chunks[0].heading == ""


def test_paragraph_fallback_respects_max_chars() -> None:
    paras = [f"Paragraph number {i} with some words in it." for i in range(20)]
    text = "\n\n".join(paras)
    # Force small max so chunks must split.
    chunks = chunk_text(text, max_chars=100)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 200  # allow some slack from paragraph boundary grouping


def test_long_segment_under_heading_splits() -> None:
    # One heading, one enormous section — must still be split to max_chars.
    big_para = "x" * 500
    text = f"# Big\n\n{big_para}\n\n{big_para}\n\n{big_para}\n"
    chunks = chunk_text(text, max_chars=600)
    # All chunks carry the same heading label.
    assert all(c.heading == "Big" for c in chunks)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 700  # generous upper bound


def test_intro_synthetic_label_when_text_before_first_heading() -> None:
    text = "Some prelude text.\n\n# Real Heading\n\nBody of the section.\n"
    chunks = chunk_text(text, max_chars=4000)
    assert len(chunks) == 2
    assert chunks[0].heading == "(intro)"
    assert "prelude" in chunks[0].text
    assert chunks[1].heading == "Real Heading"
    assert "Body of the section" in chunks[1].text
