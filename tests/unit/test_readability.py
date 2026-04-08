"""Tests for the trafilatura readability wrapper."""

from __future__ import annotations

from researcher.fetch.readability import extract_readable


def test_extract_from_sample_html() -> None:
    html = """
    <!doctype html>
    <html>
      <head><title>Sample</title></head>
      <body>
        <nav>nav stuff</nav>
        <article>
          <h1>A Big Headline</h1>
          <p>Here is the first paragraph of the main content. It contains several sentences
             that should be captured by the readability extractor. This paragraph is long
             enough to clearly be the main content of the page.</p>
          <p>And a second paragraph with more detail about the topic, to help the extractor
             detect that this is indeed the main article body.</p>
        </article>
        <footer>footer stuff</footer>
      </body>
    </html>
    """
    text = extract_readable(html, url="https://example.com/article")
    assert text is not None
    assert "first paragraph of the main content" in text
    # Nav / footer chrome should be stripped.
    assert "nav stuff" not in text
    assert "footer stuff" not in text


def test_extract_returns_none_on_empty() -> None:
    assert extract_readable("") is None
    assert extract_readable("   ") is None or extract_readable("   ") == ""
