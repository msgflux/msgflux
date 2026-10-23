"""Tests for msgflux.utils.html."""

import time

from msgflux.utils.html import html_to_text


def test_html_to_text_preserves_basic_structure():
    assert html_to_text("<h1>Title</h1><p>Body text.</p>") == "Title\nBody text."


def test_html_to_text_removes_balanced_script_and_style_blocks():
    html = "<p>Before</p><script>bad()</script><style>x{}</style><p>After</p>"

    assert html_to_text(html) == "Before\nAfter"


def test_html_to_text_removes_trailing_unclosed_script_tag():
    assert html_to_text("<script>marker</script><script>") == ""


def test_html_to_text_removes_many_script_blocks():
    html = "<script>marker</script>" * 1001 + "visible"

    assert html_to_text(html) == "visible"


def test_html_to_text_keeps_long_unbroken_line_content():
    text = "A" * 2000

    result = html_to_text(text)

    assert result.replace("\n", "") == text


def test_html_to_text_bounds_work_for_long_unbroken_line():
    text = "A" * 20_000

    start = time.perf_counter()
    result = html_to_text(text)
    elapsed = time.perf_counter() - start

    assert result.replace("\n", "") == text
    # The previous fallback scanned every character after the first 800 and
    # took several seconds on this input.
    assert elapsed < 1
