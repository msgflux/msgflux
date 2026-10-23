"""Boundary coverage for HTML stripping used by the WebFetch fallback."""

import subprocess
import sys

import pytest

from msgflux.utils.html import html_to_text, strip_html


@pytest.mark.parametrize("tag", ["script", "style", "noscript", "svg"])
def test_suppressed_blocks_are_removed(tag):
    assert strip_html(f"before<{tag}><b>hidden</b></{tag}>after") == "beforeafter"


def test_head_suppresses_title_until_closed():
    assert strip_html("<head><title>hidden</title></head>after") == "after"


def test_balanced_and_unbalanced_markup_keeps_visible_text_and_spacing():
    html = "before<script>hidden</script><style>hidden</style>after<script>"

    assert strip_html(html) == "beforeafter"


def test_unclosed_suppressed_block_stays_hidden_to_end_of_input():
    assert strip_html("visible<script>unfinished") == "visible"


@pytest.mark.parametrize("tag", ["script", "style", "head", "noscript", "svg"])
def test_each_unclosed_suppressed_tag_hides_until_eof(tag):
    assert strip_html(f"visible<{tag}>unfinished") == "visible"


def test_nested_suppressed_svg_remains_hidden_until_outer_close():
    html = "<svg>first<svg>second</svg>hidden</svg>visible"

    assert strip_html(html) == "visible"


def test_body_start_ends_optional_unclosed_head():
    html = "<html><head><title>title</title><body><p>Visible body</p></body></html>"

    assert strip_html(html) == "Visible body"


@pytest.mark.parametrize("flow_tag", ["p", "div"])
def test_flow_content_implicitly_ends_unclosed_head(flow_tag):
    assert strip_html(f"<head><title>T</title><{flow_tag}>Hello</{flow_tag}>") == (
        "Hello"
    )


def test_ordinary_closed_head_keeps_metadata_suppressed():
    html = (
        "<head><title>T</title><meta charset='utf-8'>"
        "<style>hidden</style><script>hidden</script></head><p>Hello</p>"
    )

    assert strip_html(html) == "Hello"


def test_block_spacing_and_entities_are_preserved():
    html = "<div>A<br/>B<p>Tom &amp; Jerry &#x1F600; &unknown;</p>C</div>"

    assert strip_html(html) == "A\nB\nTom & Jerry 😀 &unknown;\nC"


def test_raw_ampersands_and_entity_lexemes_are_preserved():
    assert strip_html("https://example.com/?a=1&b=2") == (
        "https://example.com/?a=1&b=2"
    )
    assert strip_html("AT&T and &copy") == "AT&T and &copy"
    assert strip_html("&lt;script&gt;visible&lt;/script&gt;") == (
        "<script>visible</script>"
    )


def test_multiline_entity_references_keep_legacy_decoding_with_crlf():
    html = "<p>A &amp;\r\nB &#65; &#x42;\r\nC &copy</p>"

    assert strip_html(html) == "A &\r\nB A B\r\nC &copy"


def test_long_quoted_attributes_do_not_truncate_tags():
    attribute = "x" * 25_000

    assert strip_html(f'<p title="{attribute} > still quoted">visible</p>') == "visible"


def test_deep_wrappers_do_not_require_recursive_tree_traversal():
    depth = 2_000
    html = "<div>" * depth + "visible" + "</div>" * depth

    assert strip_html(html) == "visible"


def test_repeated_unclosed_suppressed_tags_are_processed():
    assert strip_html("<script>" * 5_000) == ""


def test_repeated_malformed_brackets_remain_visible():
    brackets = "<" * 20_000

    assert strip_html(brackets) == brackets


def test_unfinished_comments_and_trailing_tokens_in_bounded_subprocess():
    script = r"""
from msgflux.utils.html import strip_html

comment = "<!--" * 25_000
assert strip_html(comment) == ""
assert strip_html("visible-prefix\r\n" + comment) == "visible-prefix"
assert strip_html('<div title="<!--">visible</div>') == "visible"
brackets = "<" * 20_000
assert strip_html(brackets) == brackets
assert strip_html("https://example.com/?a=1&b=2") == (
    "https://example.com/?a=1&b=2"
)
assert strip_html("<p>A &amp;\r\nB &#65; &#x42;\r\nC &copy</p>") == (
    "A &\r\nB A B\r\nC &copy"
)
"""

    subprocess.run(  # noqa: S603 - fixed local regression script; no shell is used.
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_webfetch_text_conversion_uses_same_stripping_behavior():
    assert html_to_text("<p>visible</p><script>hidden</script>") == "visible"
