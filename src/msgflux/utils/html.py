"""HTML text extraction using semantic chunking."""

import re
from html.parser import HTMLParser

# --- Chunking constants (ported from jina.ai tokenizer) ---
MAX_HEADING_LENGTH = 7
MAX_HEADING_CONTENT_LENGTH = 200
MAX_HEADING_UNDERLINE_LENGTH = 200
MAX_HTML_HEADING_ATTRIBUTES_LENGTH = 100
MAX_LIST_ITEM_LENGTH = 200
MAX_NESTED_LIST_ITEMS = 6
MAX_LIST_INDENT_SPACES = 7
MAX_BLOCKQUOTE_LINE_LENGTH = 200
MAX_BLOCKQUOTE_LINES = 15
MAX_CODE_BLOCK_LENGTH = 1500
MAX_CODE_LANGUAGE_LENGTH = 20
MAX_INDENTED_CODE_LINES = 20
MAX_TABLE_CELL_LENGTH = 200
MAX_TABLE_ROWS = 20
MAX_HTML_TABLE_LENGTH = 2000
MIN_HORIZONTAL_RULE_LENGTH = 3
MAX_SENTENCE_LENGTH = 400
MAX_QUOTED_TEXT_LENGTH = 300
MAX_PARENTHETICAL_CONTENT_LENGTH = 200
MAX_NESTED_PARENTHESES = 5
MAX_MATH_INLINE_LENGTH = 100
MAX_MATH_BLOCK_LENGTH = 500
MAX_PARAGRAPH_LENGTH = 1000
MAX_STANDALONE_LINE_LENGTH = 800
MAX_HTML_TAG_ATTRIBUTES_LENGTH = 100
MAX_HTML_TAG_CONTENT_LENGTH = 1000
LOOKAHEAD_RANGE = 100

# Replaces \p{Emoji_Presentation}\p{Extended_Pictographic} (requires `regex` lib)
_EMOJI = r"[\U0001F300-\U0001FFFF]"

# Sentence-ending punctuation helper
_SENT_END = rf"(?:[.!?…]|\.\.\.|[\u2026\u2047-\u2049]|{_EMOJI})"

# --- Semantic chunk regex (ported from jina.ai tokenizer, adapted for stdlib re) ---
_chunk_regex = re.compile(
    r"("
    # 1. Headings (Setext, Markdown, HTML)
    rf"(?:^(?:[#*=-]{{1,{MAX_HEADING_LENGTH}}}|\w[^\r\n]{{0,{MAX_HEADING_CONTENT_LENGTH}}}\r?\n[-=]{{2,{MAX_HEADING_UNDERLINE_LENGTH}}}|<h[1-6][^>]{{0,{MAX_HTML_HEADING_ATTRIBUTES_LENGTH}}}>)[^\r\n]{{1,{MAX_HEADING_CONTENT_LENGTH}}}(?:</h[1-6]>)?(?:\r?\n|$))"
    r"|"
    # 2. Citations
    rf"(?:\[[0-9]+\][^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}})"
    r"|"
    # 3. List items (with continuation indentation)
    rf"(?:(?:^|\r?\n)[ \t]{{0,3}}(?:[-*+•]|\d{{1,3}}\.\w\.|\[[ xX]\])[ \t]+(?:[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}})(?:\r?\n[ \t]{{2,}}(?:[^\r\n]{{1,{MAX_LIST_ITEM_LENGTH}}}))*)"
    r"|"
    # 4. Block quotes
    rf"(?:(?:^>(?:>|\s{{2,}}){{0,2}}(?:[^\r\n]{{0,{MAX_BLOCKQUOTE_LINE_LENGTH}}})(?:\r?\n[ \t]+[^\r\n]{{0,{MAX_BLOCKQUOTE_LINE_LENGTH}}})*?\r?\n?))"
    r"|"
    # 5. Code blocks (fenced, indented, HTML pre)
    rf"(?:(?:^|\r?\n)(?:```|~~~)(?:\w{{0,{MAX_CODE_LANGUAGE_LENGTH}}})?\r?\n[\s\S]{{0,{MAX_CODE_BLOCK_LENGTH}}}?(?:```|~~~)\r?\n?)"
    rf"|(?:(?:^|\r?\n)(?: {{4}}|\t)[^\r\n]{{0,{MAX_LIST_ITEM_LENGTH}}}(?:\r?\n(?: {{4}}|\t)[^\r\n]{{0,{MAX_LIST_ITEM_LENGTH}}}){{0,{MAX_INDENTED_CODE_LINES}}}\r?\n?)"
    rf"|(?:<pre>(?:<code>)?[\s\S]{{0,{MAX_CODE_BLOCK_LENGTH}}}?(?:</code>)?</pre>)"
    r"|"
    # 6. Tables (Markdown and HTML)
    rf"(?:(?:^|\r?\n)\|[^\r\n]{{0,{MAX_TABLE_CELL_LENGTH}}}\|(?:\r?\n\|[-:]{{1,{MAX_TABLE_CELL_LENGTH}}}\|)?(?:\r?\n\|[^\r\n]{{0,{MAX_TABLE_CELL_LENGTH}}}\|){{0,{MAX_TABLE_ROWS}}})"
    rf"|<table>[\s\S]{{0,{MAX_HTML_TABLE_LENGTH}}}?</table>"
    r"|"
    # 7. Horizontal rules
    rf"(?:^(?:[-*_]){{{MIN_HORIZONTAL_RULE_LENGTH},}}\s*$|<hr\s*/?>)"
    r"|"
    # 8. Standalone lines (with optional leading HTML tag)
    rf"(?:^(?:<[a-zA-Z][^>]{{0,{MAX_HTML_TAG_ATTRIBUTES_LENGTH}}}>[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}{_SENT_END}?(?:</[a-zA-Z]+>)?(?:\r?\n|$))(?:\r?\n[ \t]+[^\r\n]*)*)"
    r"|"
    # 9. Sentences (with continuation indentation)
    rf"(?:[^\r\n]{{1,{MAX_SENTENCE_LENGTH}}}{_SENT_END}?(?=\s|$)(?:\r?\n[ \t]+[^\r\n]*)*)"
    r"|"
    # 10. Quoted / parenthetical / bracketed content
    rf'(?<!\w)"""[^""]{{0,{MAX_QUOTED_TEXT_LENGTH}}}"""(?!\w)'
    rf"|(?<!\w)'[^\r\n']{{0,{MAX_QUOTED_TEXT_LENGTH}}}'(?!\w)"
    rf'|(?<!\w)"[^\r\n"]{{0,{MAX_QUOTED_TEXT_LENGTH}}}"(?!\w)'
    rf"|(?<!\w)`[^\r\n`]{{0,{MAX_QUOTED_TEXT_LENGTH}}}`(?!\w)"
    rf"|\([^\r\n()]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}(?:\([^\r\n()]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}\)[^\r\n()]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}){{0,{MAX_NESTED_PARENTHESES}}}\)"
    rf"|\[[^\r\n\[\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}(?:\[[^\r\n\[\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}\][^\r\n\[\]]{{0,{MAX_PARENTHETICAL_CONTENT_LENGTH}}}){{0,{MAX_NESTED_PARENTHESES}}}\]"
    rf"|\$[^\r\n$]{{0,{MAX_MATH_INLINE_LENGTH}}}\$"
    rf"|`[^\r\n`]{{0,{MAX_MATH_INLINE_LENGTH}}}`"
    r"|"
    # 11. Paragraphs
    rf"(?:(?:^|\r?\n\r?\n)(?:<p>)?(?:[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}{_SENT_END}?(?=\s|$)|[^\r\n]{{1,{MAX_PARAGRAPH_LENGTH}}}(?=[\r\n]|$))(?:</p>)?(?:\r?\n[ \t]+[^\r\n]*)*)"
    r"|"
    # 12. HTML tags with content
    rf"(?:<[a-zA-Z][^>]{{0,{MAX_HTML_TAG_ATTRIBUTES_LENGTH}}}(?:>[\s\S]{{0,{MAX_HTML_TAG_CONTENT_LENGTH}}}?</[a-zA-Z]+>|\s*/>))"
    r"|"
    # 13. LaTeX math
    rf"(?:(?:\$\$[\s\S]{{0,{MAX_MATH_BLOCK_LENGTH}}}?\$\$)|(?:\$[^\$\r\n]{{0,{MAX_MATH_INLINE_LENGTH}}}\$))"
    r"|"
    # 14. Fallback
    rf"(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}{_SENT_END}?(?=\s|$)|[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}}(?=[\r\n]|$))"
    r"|"
    # 15. Bounded progress for unbroken lines. Without this, a line longer
    # than the bounded alternatives makes findall retry every position and
    # silently drop the remainder after the first match.
    rf"(?:[^\r\n]{{1,{MAX_STANDALONE_LINE_LENGTH}}})"
    r")",
    re.MULTILINE | re.UNICODE,
)

# --- HTML stripping ---
_SUPPRESSED_TAGS = {"script", "style", "head", "noscript", "svg"}
_HEAD_CONTENT_TAGS = {
    "base",
    "basefont",
    "bgsound",
    "head",
    "link",
    "meta",
    "noscript",
    "script",
    "style",
    "template",
    "title",
}
_BLOCK_TAGS = {
    "div",
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "br",
    "hr",
    "li",
    "ul",
    "ol",
    "table",
    "tr",
    "td",
    "th",
    "blockquote",
    "pre",
    "section",
    "article",
    "header",
    "footer",
    "nav",
    "main",
    "aside",
    "figure",
    "figcaption",
}
_NUMERIC_ENTITY = re.compile(r"&#(\d+);|&#x([0-9a-fA-F]+);")
_NAMED_ENTITIES: dict[str, str] = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&apos;": "'",
    "&nbsp;": " ",
    "&#39;": "'",
    "&mdash;": "\u2014",
    "&ndash;": "\u2013",
    "&hellip;": "…",
    "&laquo;": "«",
    "&raquo;": "»",
    "&copy;": "©",
    "&reg;": "®",
}
_NAMED_ENTITY_RE = re.compile(
    r"&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+"
    r"|mdash|ndash|hellip|laquo|raquo|copy|reg|[a-zA-Z]+);"
)
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def _decode_entity(m: re.Match) -> str:
    entity = m.group(0)
    if entity in _NAMED_ENTITIES:
        return _NAMED_ENTITIES[entity]
    nm = _NUMERIC_ENTITY.match(entity)
    if nm:
        n = int(nm.group(1)) if nm.group(1) else int(nm.group(2), 16)
        try:
            return chr(n)
        except (ValueError, OverflowError):
            return ""
    return entity


class _HTMLTextParser(HTMLParser):
    """Collect visible text while filtering non-content HTML elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self._suppressed_tag: str | None = None
        self._suppressed_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if self._suppressed_tag == "head" and tag not in _HEAD_CONTENT_TAGS:
            self._suppressed_tag = None
            self._suppressed_depth = 0
        if self._suppressed_tag is not None:
            if tag == self._suppressed_tag:
                self._suppressed_depth += 1
            return
        if tag in _SUPPRESSED_TAGS:
            self._suppressed_tag = tag
            self._suppressed_depth = 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._suppressed_tag is not None:
            if tag == self._suppressed_tag:
                self._suppressed_depth -= 1
                if self._suppressed_depth == 0:
                    self._suppressed_tag = None
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(
        self, tag: str, _attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag not in _SUPPRESSED_TAGS:
            self.handle_starttag(tag, _attrs)

    def handle_data(self, data: str) -> None:
        if self._suppressed_tag is None:
            self.parts.append(data)

    def handle_charref(self, name: str) -> None:
        # strip_html escapes source ampersands as &#38; before parsing so the
        # legacy entity decoder sees the original reference lexemes afterward.
        if self._suppressed_tag is None:
            if name == "38":
                self.parts.append("&")
            else:
                self.parts.append(f"&#{name};")


def strip_html(html: str) -> str:
    """Remove HTML markup and decode entities, returning plain text."""
    parser = _HTMLTextParser()
    protected_source = html.replace("&", "&#38;")
    parser.feed(protected_source)

    # Older HTMLParser.close() rescans an unfinished comment for each "<".
    # Locate the pending suffix through public positions and ignore it at EOF.
    line, column = parser.getpos()
    line_start = 0
    for _ in range(line - 1):
        line_start = protected_source.index("\n", line_start) + 1
    unconsumed_start = line_start + column
    if not protected_source.startswith("<!--", unconsumed_start):
        parser.close()

    text = "".join(parser.parts)
    text = _NAMED_ENTITY_RE.sub(_decode_entity, text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def html_to_text(html: str) -> str:
    """Extract clean text from HTML using semantic chunking.

    Args:
        html: Raw HTML string.

    Returns:
        Plain text with semantic structure preserved.
    """
    text = strip_html(html)
    matches = _chunk_regex.findall(text)
    chunks = [m[0] if isinstance(m, tuple) else m for m in matches]
    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())
