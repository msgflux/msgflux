#!/usr/bin/env python3
# ruff: noqa: C901, E501, FBT001, FBT002, S310
"""Render clean SVG visuals for code comparisons and square thumbnails.

The script is intentionally simple:
- today it renders 1, 2, or 4 blocks
- the layout code auto-sizes cards based on code length

Usage examples:

    python scripts/svg_render.py \
        --left-title "Imperative" \
        --left-file imperative.py \
        --right-title "Declarative" \
        --right-file declarative.py \
        --output imperative-vs-declarative.svg

    python scripts/svg_render.py \
        --left-title "Before" --left-file before.py \
        --right-title "After" --right-file after.py

    python scripts/svg_render.py \
        --layout 4 \
        --top_left-title "Imperative" --top_left-file imperative.py \
        --top_right-title "Declarative" --top_right-file declarative.py \
        --bottom_left-title "Stepwise" --bottom_left-file stepwise.py \
        --bottom_right-title "Composed" --bottom_right-file composed.py \
        --output example-2x2.svg

    python scripts/svg_render.py \
        --layout 1 \
        --left-title "Single block" --left-file example.py \
        --output example-1x1.svg

    Inline line highlights:

        if condition:  # highlight: red
            return value  # highlight: green

    Title headers can live in the snippet itself:

        # title: Imperative
        # subtitle: Manual flow
        import msgflux as mf

    Square thumbnail:

        python scripts/svg_render.py \
            --mode thumbnail \
            --title "msgflux" \
            --subtitle "Code, prompts, and workflows" \
            --output thumbnail.svg

    Thumbnail metadata file:

        # title: msgflux
        # subtitle: Code, prompts, and workflows

    Architecture ASCII:

        python scripts/svg_render.py \
            --mode architecture \
            --input-file architecture.txt \
            --title "Open PIX Assistant" \
            --subtitle "Architecture overview" \
            --output architecture.svg

    Gallery:

        python scripts/svg_render.py \
            --mode gallery \
            --title "Open PIX Assistant" \
            --subtitle "Real photos from testing" \
            --gallery-image https://files.catbox.moe/9gwd7u.jpeg \
            --gallery-image https://files.catbox.moe/8bj5jl.jpeg \
            --gallery-image https://files.catbox.moe/otqnaa.jpeg \
            --output gallery.svg
"""

import argparse
import base64
import html
import io
import keyword
import mimetypes
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import wrap
from tokenize import COMMENT, NAME, NUMBER, OP, STRING, TokenError, generate_tokens


@dataclass(frozen=True)
class CodeBlock:
    title: str
    subtitle: str
    code: str
    green_lines: frozenset[int] = field(default_factory=frozenset)
    red_lines: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RenderLine:
    text: str
    directive: str | None
    source_line_number: int
    line_number_label: str


def read_text(path: str | None, inline_text: str | None) -> str:
    if path and inline_text:
        raise SystemExit("Use either --*-file or --*-text, not both.")
    if path:
        return Path(path).read_text(encoding="utf-8")
    if inline_text is not None:
        return (
            inline_text.replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
            .replace("\r\n", "\n")
        )
    raise SystemExit("Missing code input. Provide --*-file or --*-text.")


def _download_bytes(url_or_path: str) -> bytes:
    if re.match(r"^https?://", url_or_path, re.I):
        req = urllib.request.Request(url_or_path, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    return Path(url_or_path).read_bytes()


def image_to_data_uri(url_or_path: str) -> str:
    data = _download_bytes(url_or_path)
    mime, _ = mimetypes.guess_type(url_or_path)
    if not mime:
        mime = "image/jpeg"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def parse_gallery_spec(text: str) -> tuple[str, str, list[str]]:
    """Parse a tiny YAML-like gallery spec without external dependencies."""
    title = ""
    subtitle = ""
    images: list[str] = []
    current_item: dict[str, str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("title:"):
            title = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            continue
        if stripped.startswith("subtitle:"):
            subtitle = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            continue
        if stripped.startswith("- "):
            if current_item and current_item.get("image"):
                images.append(current_item["image"])
            current_item = {}
            remainder = stripped[2:].strip()
            if remainder.startswith("image:"):
                current_item["image"] = (
                    remainder.split(":", 1)[1].strip().strip('"').strip("'")
                )
            continue
        if stripped.startswith("image:") and current_item is not None:
            current_item["image"] = (
                stripped.split(":", 1)[1].strip().strip('"').strip("'")
            )
            continue

    if current_item and current_item.get("image"):
        images.append(current_item["image"])

    return title, subtitle, images


def split_block_headers(text: str) -> tuple[str | None, str | None, str]:
    """Extract leading title/subtitle headers from the code body.

    Supported header form:
        # title: Imperative
        # subtitle: Manual flow
    """
    lines = text.splitlines()
    title: str | None = None
    subtitle: str | None = None
    body_start = 0
    seen_header = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if seen_header:
                body_start = index + 1
            continue

        lowered = stripped.lower()
        matched = False
        for marker, kind in (
            ("# title:", "title"),
            ("#title:", "title"),
            ("# subtitle:", "subtitle"),
            ("#subtitle:", "subtitle"),
        ):
            if lowered.startswith(marker):
                value = stripped[len(marker) :].strip()
                if value:
                    if kind == "title":
                        title = value
                    else:
                        subtitle = value
                    seen_header = True
                    body_start = index + 1
                matched = True
                break
        if matched:
            continue
        break

    if not title and not subtitle:
        return None, None, text

    body = "\n".join(lines[body_start:])
    return title, subtitle, body


def split_highlight_directive(line: str) -> tuple[str, str | None]:
    """Extract a highlight directive from a trailing comment."""
    marker = "# highlight:"
    if marker not in line:
        return line, None

    code_part, directive_part = line.split(marker, 1)
    directive = directive_part.strip().lower()
    if directive not in {"green", "red"}:
        return line, None

    return code_part.rstrip(), directive


def prepare_render_lines(text: str, max_chars: int) -> list[RenderLine]:
    """Split code into wrapped lines while preserving source line numbers."""
    prepared: list[RenderLine] = []
    raw_lines = text.splitlines() or [""]
    for source_line_number, raw_line in enumerate(raw_lines, start=1):
        clean_line, directive = split_highlight_directive(raw_line)
        wrapped = wrap(
            clean_line,
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False,
            drop_whitespace=False,
            replace_whitespace=False,
        )
        wrapped_lines = wrapped or [""]
        for visual_index, wrapped_line in enumerate(wrapped_lines):
            prepared.append(
                RenderLine(
                    text=wrapped_line,
                    directive=directive,
                    source_line_number=source_line_number,
                    line_number_label=str(source_line_number)
                    if visual_index == 0
                    else "",
                )
            )
    return prepared


def compute_wrap_limit(
    available_width: float, columns: int, char_advance: float
) -> int:
    """Estimate a readable wrap width from the available code space."""
    estimated = int(available_width / char_advance)
    if columns == 1:
        return max(60, min(90, estimated))
    return max(44, min(72, estimated))


def max_source_line_count(blocks: list[CodeBlock]) -> int:
    return max(max(1, len(block.code.splitlines() or [""])) for block in blocks)


def max_clean_source_line_length(blocks: list[CodeBlock]) -> int:
    longest = 0
    for block in blocks:
        for raw_line in block.code.splitlines() or [""]:
            clean_line, _ = split_highlight_directive(raw_line)
            longest = max(longest, len(clean_line))
    return longest


def parse_line_ranges(spec: str | None) -> frozenset[int]:
    if not spec:
        return frozenset()

    lines: set[int] = set()
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start > end:
                start, end = end, start
            lines.update(range(start, end + 1))
        else:
            lines.add(int(item))

    return frozenset(n for n in lines if n > 0)


def _token_color(
    token_type: int, token_text: str, prev_name: str | None, line: str
) -> str:
    if token_type == COMMENT:
        return "#7b8794"
    if token_type == STRING:
        return "#0f766e"
    if token_type == NUMBER:
        return "#8b5cf6"
    if token_type == NAME:
        if keyword.iskeyword(token_text):
            if token_text in {"class", "def", "return", "yield"}:
                return "#b45309"
            return "#a16207"
        if prev_name in {"def", "class"}:
            return "#2563eb"
        if token_text in {"self", "cls"}:
            return "#0f766e"
        # Heuristic: highlight call targets and assignments a bit more.
        if f"{token_text}(" in line:
            return "#7c3aed"
        if "=" in line and line.lstrip().startswith(token_text):
            return "#0f766e"
        return "#334155"
    if token_type == OP:
        return "#64748b"
    return "#1f2937"


def highlight_python_line(line: str) -> list[tuple[str, str]]:
    """Return colored segments for a single line of Python-ish code."""
    try:
        tokens = list(generate_tokens(io.StringIO(line + "\n").readline))
    except TokenError:
        return [(line, "#1f2937")]

    segments: list[tuple[str, str]] = []
    prev_name: str | None = None
    cursor = 0

    for tok in tokens:
        if tok.type in {0, 4, 5, 6}:  # ENDMARKER, NEWLINE, NL, INDENT/DEDENT
            continue
        start = tok.start[1]
        end = tok.end[1]
        if start > cursor:
            segments.append((line[cursor:start], "#1f2937"))
        if tok.type == NAME and tok.string == "pass":
            color = "#b45309"
        else:
            color = _token_color(tok.type, tok.string, prev_name, line)
        segments.append((tok.string, color))
        cursor = end
        if tok.type == NAME:
            prev_name = tok.string
        elif tok.type not in {OP}:
            prev_name = None

    if cursor < len(line):
        segments.append((line[cursor:], "#1f2937"))

    return segments or [(line, "#1f2937")]


def render_svg(blocks: list[CodeBlock], show_line_numbers: bool = False) -> str:
    if not blocks:
        raise ValueError("At least one code block is required.")

    columns = len(blocks)
    if columns == 1:
        width = 0
        padding_x = 40
        padding_y = 70
    else:
        width = 1920
        padding_x = 72
        padding_y = 70
    gap_x = 28
    gap_y = 28
    header_height_compact = 78
    header_height_with_subtitle = 106
    line_height = 26
    font_size = 19
    char_advance = font_size * 0.60
    panel_inset_x = 22
    panel_content_padding_x = 12
    code_y_padding = 24
    line_number_gap = 16
    line_number_inner_padding = 8
    panel_gap_top = 12
    panel_gap_bottom = 18
    panel_padding_bottom = 22
    cols = 1 if columns == 1 else 2
    rows = (columns + cols - 1) // cols
    card_width = (
        int((width - (padding_x * 2) - gap_x * (cols - 1)) / cols) if width else 0
    )
    max_source_lines = max_source_line_count(blocks)
    max_line_number_digits = len(str(max_source_lines))
    line_number_gutter_width = 0
    if show_line_numbers:
        line_number_gutter_width = max(
            28,
            int(max_line_number_digits * char_advance + line_number_inner_padding * 2),
        )
    line_number_reserved_width = (
        line_number_gutter_width + line_number_gap if show_line_numbers else 0
    )

    bg_top = "#fff7ea"
    bg_bottom = "#f2e0be"
    card_bg = "#fffdf7"
    card_border = "#e6d3a4"
    title_color = "#11161e"
    code_color = "#1f2937"
    code_muted = "#7b8794"
    header_fill = "#f8ecd0"
    accent_line = "#ffc72c"
    code_panel_bg = "#fffdf7"
    code_panel_border = "#ead7a7"
    green_highlight = "#dcfce7"
    green_highlight_border = "#86efac"
    red_highlight = "#fee2e2"
    red_highlight_border = "#fca5a5"

    if columns == 1:
        longest_line = max_clean_source_line_length(blocks)
        estimated_panel_width = int(
            longest_line * char_advance
            + panel_content_padding_x * 2
            + line_number_reserved_width
            + 24
        )
        max_card_width = 980 if show_line_numbers else 900
        card_width = max(
            720, min(max_card_width, estimated_panel_width + panel_inset_x * 2)
        )
        width = card_width + padding_x * 2

    panel_width = card_width - panel_inset_x * 2
    code_available_width = max(
        120,
        panel_width - panel_content_padding_x * 2 - line_number_reserved_width,
    )
    max_chars = compute_wrap_limit(code_available_width, columns, char_advance)

    rows_data = []
    for row in range(rows):
        row_blocks = blocks[row * cols : (row + 1) * cols]
        row_entries = []
        row_height = 0
        for block in row_blocks:
            wrapped_lines = prepare_render_lines(block.code, max_chars)
            line_count = max(1, len(wrapped_lines))
            body_height = (
                code_y_padding + line_count * line_height + panel_padding_bottom
            )
            block_header_height = (
                header_height_with_subtitle if block.subtitle else header_height_compact
            )
            block_card_height = (
                block_header_height + panel_gap_top + body_height + panel_gap_bottom
            )
            row_height = max(row_height, block_card_height)
            row_entries.append((block, wrapped_lines, body_height, block_header_height))
        rows_data.append((row_entries, row_height))

    total_height = (
        padding_y * 2
        + sum(row_height for _, row_height in rows_data)
        + gap_y * (rows - 1)
    )
    parts: list[str] = [
        f'<svg width="{width}" height="{total_height}" viewBox="0 0 {width} {total_height}" '
        'fill="none" xmlns="http://www.w3.org/2000/svg">',
        "  <defs>",
        '    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1" gradientUnits="userSpaceOnUse">',
        f'      <stop offset="0" stop-color="{bg_top}"/>',
        f'      <stop offset="1" stop-color="{bg_bottom}"/>',
        "    </linearGradient>",
        '    <radialGradient id="glow-left" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(320 250) rotate(120) scale(620 440)">',
        '      <stop stop-color="#ffd36b" stop-opacity="0.34"/>',
        '      <stop offset="1" stop-color="#ffd36b" stop-opacity="0"/>',
        "    </radialGradient>",
        '    <radialGradient id="glow-right" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(1600 240) rotate(128) scale(700 480)">',
        '      <stop stop-color="#ffe9a8" stop-opacity="0.36"/>',
        '      <stop offset="1" stop-color="#ffe9a8" stop-opacity="0"/>',
        "    </radialGradient>",
        '    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">',
        '      <feDropShadow dx="0" dy="14" stdDeviation="18" flood-color="#0f172a" flood-opacity="0.10"/>',
        "    </filter>",
        "  </defs>",
        f'  <rect width="{width}" height="{total_height}" fill="url(#bg)"/>',
        f'  <rect width="{width}" height="{total_height}" fill="url(#glow-left)"/>',
        f'  <rect width="{width}" height="{total_height}" fill="url(#glow-right)"/>',
    ]

    current_y = padding_y
    for row_entries, row_height in rows_data:
        for col, (block, wrapped_lines, body_height, block_header_height) in enumerate(
            row_entries
        ):
            if columns == 1:
                x = padding_x
            else:
                x = padding_x + col * (card_width + gap_x)
            y = current_y
            panel_x = x + panel_inset_x
            panel_content_x = panel_x + panel_content_padding_x
            highlight_x = panel_x + 6
            highlight_width = panel_width - 12
            code_panel_y = y + block_header_height + panel_gap_top
            code_start_y = code_panel_y + code_y_padding
            panel_height = body_height
            line_highlights: list[str] = []
            line_number_texts: list[str] = []
            line_texts: list[str] = []
            gutter_x = panel_content_x
            divider_x = gutter_x + line_number_gutter_width
            line_number_x = gutter_x + line_number_inner_padding
            code_text_x = panel_content_x
            if show_line_numbers:
                code_text_x = divider_x + line_number_gap

            parts.extend(
                [
                    '  <g filter="url(#shadow)">',
                    f'    <rect x="{x}" y="{y}" width="{card_width}" height="{row_height}" rx="30" fill="{card_bg}" stroke="{card_border}"/>',
                    f'    <rect x="{x}" y="{y}" width="{card_width}" height="{block_header_height}" rx="30" fill="{header_fill}"/>',
                    f'    <rect x="{x}" y="{y + 42}" width="{card_width}" height="{max(0, block_header_height - 42)}" fill="{header_fill}"/>',
                    f'    <rect x="{x + 28}" y="{y + 24}" width="6" height="28" rx="3" fill="{accent_line}"/>',
                    f'    <text x="{x + 48}" y="{y + 50}" fill="{title_color}" font-family="DejaVu Sans, sans-serif" font-size="24" font-weight="800" letter-spacing="-0.02em">{html.escape(block.title)}</text>',
                ]
            )
            if block.subtitle:
                parts.append(
                    f'    <text x="{x + 48}" y="{y + 76}" fill="{code_muted}" font-family="DejaVu Sans, sans-serif" font-size="18" font-weight="400" letter-spacing="-0.01em">{html.escape(block.subtitle)}</text>'
                )
            parts.extend(
                [
                    f'    <line x1="{x + 30}" y1="{y + block_header_height}" x2="{x + card_width - 30}" y2="{y + block_header_height}" stroke="{card_border}" stroke-width="1"/>',
                    f'      <rect x="{x}" y="{y + block_header_height}" width="{card_width}" height="{row_height - block_header_height}" rx="0" fill="{code_panel_bg}"/>',
                    f'      <rect x="{panel_x}" y="{code_panel_y}" width="{panel_width}" height="{panel_height}" rx="18" fill="{code_panel_bg}" stroke="{code_panel_border}"/>',
                ]
            )
            if show_line_numbers:
                parts.append(
                    f'      <rect x="{gutter_x}" y="{code_panel_y + 6}" width="{line_number_gutter_width}" height="{panel_height - 12}" rx="10" fill="#fff8e9"/>'
                )
                parts.append(
                    f'      <line x1="{divider_x}" y1="{code_panel_y + 8}" x2="{divider_x}" y2="{code_panel_y + panel_height - 8}" stroke="{code_panel_border}" stroke-width="1"/>'
                )

            for line_index, render_line in enumerate(wrapped_lines):
                text_y = code_start_y + line_index * line_height
                source_line_number = render_line.source_line_number
                if (
                    render_line.directive == "green"
                    or source_line_number in block.green_lines
                ):
                    line_highlights.append(
                        f'      <rect x="{highlight_x}" y="{text_y - 18}" width="{highlight_width}" height="22" rx="7" fill="{green_highlight}" stroke="{green_highlight_border}" stroke-width="1"/>'
                    )
                elif (
                    render_line.directive == "red"
                    or source_line_number in block.red_lines
                ):
                    line_highlights.append(
                        f'      <rect x="{highlight_x}" y="{text_y - 18}" width="{highlight_width}" height="22" rx="7" fill="{red_highlight}" stroke="{red_highlight_border}" stroke-width="1"/>'
                    )
                if show_line_numbers and render_line.line_number_label:
                    padded_label = render_line.line_number_label.rjust(
                        max_line_number_digits
                    )
                    line_number_texts.append(
                        f'        <tspan x="{line_number_x}" y="{text_y}" fill="{code_muted}">{padded_label}</tspan>'
                    )
                segments = highlight_python_line(render_line.text)
                line_x = code_text_x
                if not segments:
                    segments = [(render_line.text or " ", code_color)]
                for segment_text, segment_color in segments:
                    if segment_text == "":
                        continue
                    line_texts.append(
                        f'        <tspan x="{line_x}" y="{text_y}" fill="{segment_color}">{html.escape(segment_text)}</tspan>'
                    )
                    line_x += len(segment_text.replace("\t", "    ")) * char_advance

            parts.extend(
                [
                    *line_highlights,
                    *(
                        [
                            f'      <text xml:space="preserve" font-family="JetBrains Mono, DejaVu Sans Mono, monospace" font-size="{font_size}" fill="{code_muted}">',
                            *line_number_texts,
                            "      </text>",
                        ]
                        if show_line_numbers
                        else []
                    ),
                    f'      <text xml:space="preserve" font-family="JetBrains Mono, DejaVu Sans Mono, monospace" font-size="{font_size}" fill="{code_color}">',
                    *line_texts,
                    "      </text>",
                    "  </g>",
                ]
            )
        current_y += row_height + gap_y

    parts.append("</svg>")
    return "\n".join(parts)


def render_thumbnail_svg(title: str, subtitle: str = "") -> str:
    width = 1080
    height = 1080
    bg_top = "#fbf8f1"
    bg_bottom = "#f2eadb"
    title_color = "#111111"
    subtitle_color = "#3a3a3a"
    accent = "#d8a84f"

    title_font_size = 112
    subtitle_font_size = 34
    title_lines = wrap(
        title.strip() or "Untitled",
        width=11,
        break_long_words=False,
        break_on_hyphens=False,
    ) or ["Untitled"]
    subtitle_lines = (
        wrap(
            subtitle.strip(),
            width=24,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if subtitle.strip()
        else []
    )

    line_gap = 8
    title_line_height = title_font_size + line_gap
    subtitle_line_height = subtitle_font_size + 10
    block_height = len(title_lines) * title_line_height
    if subtitle_lines:
        block_height += 38 + len(subtitle_lines) * subtitle_line_height
    start_y = (height - block_height) / 2 + title_font_size + 10

    parts: list[str] = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'fill="none" xmlns="http://www.w3.org/2000/svg">',
        "  <defs>",
        '    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1" gradientUnits="userSpaceOnUse">',
        f'      <stop offset="0" stop-color="{bg_top}"/>',
        f'      <stop offset="1" stop-color="{bg_bottom}"/>',
        "    </linearGradient>",
        '    <radialGradient id="glow" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(700 320) rotate(90) scale(360 280)">',
        f'      <stop stop-color="{accent}" stop-opacity="0.18"/>',
        '      <stop offset="1" stop-color="#d8a84f" stop-opacity="0"/>',
        "    </radialGradient>",
        "  </defs>",
        f'  <rect width="{width}" height="{height}" fill="url(#bg)"/>',
        f'  <rect width="{width}" height="{height}" fill="url(#glow)"/>',
        f'  <rect x="110" y="365" width="860" height="12" rx="6" fill="{accent}" fill-opacity="0.9"/>',
    ]

    current_y = start_y
    for index, line in enumerate(title_lines):
        line_y = current_y + index * title_line_height
        parts.append(
            f'  <text x="{width / 2}" y="{line_y}" text-anchor="middle" '
            f'fill="{title_color}" font-family="Helvetica Neue, Inter, Arial, sans-serif" '
            f'font-size="{title_font_size}" font-weight="800" letter-spacing="-0.05em">{html.escape(line)}</text>'
        )

    if subtitle_lines:
        current_y += len(title_lines) * title_line_height - 12
        for index, line in enumerate(subtitle_lines):
            line_y = current_y + index * subtitle_line_height
            parts.append(
                f'  <text x="{width / 2}" y="{line_y}" text-anchor="middle" '
                f'fill="{subtitle_color}" font-family="Helvetica Neue, Inter, Arial, sans-serif" '
                f'font-size="{subtitle_font_size}" font-weight="500" letter-spacing="-0.01em">{html.escape(line)}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def render_architecture_svg(content: str, title: str = "", subtitle: str = "") -> str:
    width = 1600
    pad_x = 88
    pad_y = 88
    title_font_size = 56
    subtitle_font_size = 28
    body_font_size = 28
    body_line_height = 36
    mono_family = "JetBrains Mono, DejaVu Sans Mono, Consolas, monospace"
    title_lines = (
        wrap(
            title.strip(),
            width=22,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if title.strip()
        else []
    )
    subtitle_lines = (
        wrap(
            subtitle.strip(),
            width=34,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if subtitle.strip()
        else []
    )
    content_lines = content.rstrip("\n").splitlines() or [""]
    max_content_len = max(len(line.expandtabs(4)) for line in content_lines)
    title_block = (
        len(title_lines) * 58
        + (24 if title_lines and subtitle_lines else 0)
        + len(subtitle_lines) * 34
    )
    body_width = max(760, min(1440, int(max_content_len * body_font_size * 0.62) + 40))
    width = max(1280, body_width + pad_x * 2)
    body_height = len(content_lines) * body_line_height + 48
    height = pad_y * 2 + title_block + 40 + body_height
    bg_top = "#fbf8f1"
    bg_bottom = "#f2eadb"
    title_color = "#111111"
    subtitle_color = "#3a3a3a"
    panel_bg = "#fffdf7"
    panel_border = "#ead7a7"
    accent = "#d8a84f"
    body_color = "#151515"

    parts: list[str] = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'fill="none" xmlns="http://www.w3.org/2000/svg">',
        "  <defs>",
        '    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1" gradientUnits="userSpaceOnUse">',
        f'      <stop offset="0" stop-color="{bg_top}"/>',
        f'      <stop offset="1" stop-color="{bg_bottom}"/>',
        "    </linearGradient>",
        '    <radialGradient id="glow" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(1220 280) rotate(90) scale(500 380)">',
        f'      <stop stop-color="{accent}" stop-opacity="0.16"/>',
        '      <stop offset="1" stop-color="#d8a84f" stop-opacity="0"/>',
        "    </radialGradient>",
        "  </defs>",
        f'  <rect width="{width}" height="{height}" fill="url(#bg)"/>',
        f'  <rect width="{width}" height="{height}" fill="url(#glow)"/>',
    ]

    cursor_y = pad_y + 64
    for index, line in enumerate(title_lines):
        parts.append(
            f'  <text x="{width / 2}" y="{cursor_y + index * 58}" text-anchor="middle" '
            f'fill="{title_color}" font-family="Helvetica Neue, Inter, Arial, sans-serif" '
            f'font-size="{title_font_size}" font-weight="800" letter-spacing="-0.04em">{html.escape(line)}</text>'
        )
    cursor_y += len(title_lines) * 58
    if subtitle_lines:
        cursor_y += 18
        for index, line in enumerate(subtitle_lines):
            parts.append(
                f'  <text x="{width / 2}" y="{cursor_y + index * 34}" text-anchor="middle" '
                f'fill="{subtitle_color}" font-family="Helvetica Neue, Inter, Arial, sans-serif" '
                f'font-size="{subtitle_font_size}" font-weight="500" letter-spacing="-0.01em">{html.escape(line)}</text>'
            )
        cursor_y += len(subtitle_lines) * 34

    bar_y = cursor_y + 0
    parts.append(
        f'  <rect x="{(width - 860) / 2:.0f}" y="{bar_y}" width="860" height="12" rx="6" fill="{accent}" fill-opacity="0.9"/>'
    )

    panel_y = bar_y + 34
    panel_x = (width - body_width) / 2
    parts.extend(
        [
            f'  <rect x="{panel_x}" y="{panel_y}" width="{body_width}" height="{body_height}" rx="34" fill="{panel_bg}" stroke="{panel_border}"/>',
            f'  <rect x="{panel_x + 28}" y="{panel_y + 28}" width="{body_width - 56}" height="{body_height - 56}" rx="22" fill="none" stroke="#111111" stroke-opacity="0.04"/>',
        ]
    )

    text_y = panel_y + 34
    for line in content_lines:
        parts.append(
            f'  <text x="{panel_x + 36}" y="{text_y}" fill="{body_color}" '
            f'font-family="{mono_family}" font-size="{body_font_size}" font-weight="500" xml:space="preserve">'
            f"{html.escape(line.expandtabs(4))}</text>"
        )
        text_y += body_line_height

    parts.append("</svg>")
    return "\n".join(parts)


def render_gallery_svg(title: str, subtitle: str, images: list[str]) -> str:
    width = 1600
    height = 1040
    bg_top = "#fbf8f1"
    bg_bottom = "#f2eadb"
    title_color = "#111111"
    subtitle_color = "#3a3a3a"
    accent = "#d8a84f"
    card_bg = "#fffdf7"
    card_border = "#ead7a7"
    panel_y = 250
    panel_h = 640
    pad_x = 80
    gap = 28
    card_w = int((width - pad_x * 2 - gap * 2) / 3)
    card_h = panel_h

    parts: list[str] = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'fill="none" xmlns="http://www.w3.org/2000/svg">',
        "  <defs>",
        '    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1" gradientUnits="userSpaceOnUse">',
        f'      <stop offset="0" stop-color="{bg_top}"/>',
        f'      <stop offset="1" stop-color="{bg_bottom}"/>',
        "    </linearGradient>",
        '    <radialGradient id="glow" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(1220 250) rotate(90) scale(520 420)">',
        f'      <stop stop-color="{accent}" stop-opacity="0.16"/>',
        '      <stop offset="1" stop-color="#d8a84f" stop-opacity="0"/>',
        "    </radialGradient>",
        '    <filter id="gallery-shadow" x="-20%" y="-20%" width="140%" height="140%">',
        '      <feDropShadow dx="0" dy="16" stdDeviation="18" flood-color="#0f172a" flood-opacity="0.10"/>',
        "    </filter>",
    ]
    parts.extend(
        [
            "  </defs>",
            f'  <rect width="{width}" height="{height}" fill="url(#bg)"/>',
            f'  <rect width="{width}" height="{height}" fill="url(#glow)"/>',
            f'  <rect x="{(width - 860) / 2:.0f}" y="178" width="860" height="12" rx="6" fill="{accent}" fill-opacity="0.9"/>',
            f'  <text x="{width / 2}" y="118" text-anchor="middle" fill="{title_color}" font-family="Helvetica Neue, Inter, Arial, sans-serif" font-size="60" font-weight="800" letter-spacing="-0.04em">{html.escape(title)}</text>',
            f'  <text x="{width / 2}" y="168" text-anchor="middle" fill="{subtitle_color}" font-family="Helvetica Neue, Inter, Arial, sans-serif" font-size="28" font-weight="500" letter-spacing="-0.01em">{html.escape(subtitle)}</text>',
        ]
    )

    x_positions = [pad_x + i * (card_w + gap) for i in range(3)]
    embedded_images = [image_to_data_uri(image) for image in images]
    for idx, (x, image_data_uri) in enumerate(zip(x_positions, embedded_images)):
        clip_id = f"clip-gallery-{idx}"
        parts.extend(
            [
                '  <g filter="url(#gallery-shadow)">',
                f'    <rect x="{x}" y="{panel_y}" width="{card_w}" height="{card_h}" rx="34" fill="{card_bg}" stroke="{card_border}"/>',
                f'    <clipPath id="{clip_id}">',
                f'      <rect x="{x + 16}" y="{panel_y + 16}" width="{card_w - 32}" height="{card_h - 32}" rx="26"/>',
                "    </clipPath>",
                f'    <image href="{html.escape(image_data_uri)}" x="{x + 16}" y="{panel_y + 16}" width="{card_w - 32}" height="{card_h - 32}" preserveAspectRatio="xMidYMid slice" clip-path="url(#{clip_id})"/>',
                "  </g>",
            ]
        )

    parts.append("</svg>")
    return "\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a clean code comparison SVG or a square thumbnail SVG."
    )

    def add_block_args(side: str) -> None:
        parser.add_argument(f"--{side}-title", default=None)
        parser.add_argument(f"--{side}-file", default=None)
        parser.add_argument(f"--{side}-text", default=None)
        parser.add_argument(f"--{side}-green-lines", default=None)
        parser.add_argument(f"--{side}-red-lines", default=None)

    for side in (
        "left",
        "right",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    ):
        add_block_args(side)

    parser.add_argument(
        "--mode",
        choices=("comparison", "thumbnail", "architecture", "gallery"),
        default="comparison",
        help="Render a code comparison, a square thumbnail, an architecture diagram, or a gallery.",
    )
    parser.add_argument(
        "--output",
        default="imperative-vs-declarative.svg",
        help="Output SVG path.",
    )
    parser.add_argument(
        "--layout",
        choices=("1", "2", "4"),
        default="2",
        help="Number of blocks to render. Use 1, 2, or 4.",
    )
    parser.add_argument(
        "--example-2x2",
        action="store_true",
        help="Render a built-in 2x2 demo instead of taking manual inputs.",
    )
    parser.add_argument(
        "--example-1x1",
        action="store_true",
        help="Render a built-in 1x1 demo instead of taking manual inputs.",
    )
    parser.add_argument(
        "--line-numbers",
        action="store_true",
        help="Show source line numbers inside the code panel.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Thumbnail title or fallback comparison title.",
    )
    parser.add_argument(
        "--subtitle",
        default=None,
        help="Thumbnail subtitle or fallback comparison subtitle.",
    )
    parser.add_argument(
        "--meta-file",
        default=None,
        help="Optional metadata file containing # title: and # subtitle: headers for thumbnail mode.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Input text file for architecture mode.",
    )
    parser.add_argument(
        "--input-text",
        default=None,
        help="Inline text for architecture mode.",
    )
    parser.add_argument(
        "--gallery-image",
        action="append",
        default=[],
        help="Image URL for gallery mode. Repeat this flag three times.",
    )
    parser.add_argument(
        "--spec-file",
        default=None,
        help="YAML-like spec file for gallery mode.",
    )
    return parser


def _clean_title(value: str | None, fallback: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def resolve_thumbnail_meta(
    cli_title: str | None, cli_subtitle: str | None, meta_file: str | None
) -> tuple[str, str]:
    title = cli_title or ""
    subtitle = cli_subtitle or ""
    if meta_file:
        header_title, header_subtitle, _ = split_block_headers(
            Path(meta_file).read_text(encoding="utf-8")
        )
        if header_title:
            title = header_title
        if header_subtitle:
            subtitle = header_subtitle
    return _clean_title(title, "Untitled"), _clean_title(subtitle, "")


def make_block(
    title: str,
    subtitle: str,
    code: str,
    green_lines: str | None,
    red_lines: str | None,
) -> CodeBlock:
    return CodeBlock(
        title=title,
        subtitle=subtitle,
        code=code,
        green_lines=parse_line_ranges(green_lines),
        red_lines=parse_line_ranges(red_lines),
    )


def resolve_block(
    cli_title: str | None,
    cli_subtitle: str | None,
    fallback_title: str,
    path: str | None,
    inline_text: str | None,
    green_lines: str | None,
    red_lines: str | None,
) -> CodeBlock:
    raw_text = read_text(path, inline_text)
    header_title, header_subtitle, body = split_block_headers(raw_text)
    return make_block(
        _clean_title(header_title or cli_title, fallback_title),
        _clean_title(header_subtitle or cli_subtitle, ""),
        body,
        green_lines,
        red_lines,
    )


def build_demo_blocks_2x2() -> list[CodeBlock]:
    """Return a built-in 2x2 example for quick previewing."""
    return [
        CodeBlock(
            title="Imperative",
            subtitle="",
            code=(
                "class SupportAgent:\n"
                "    def reply(self, message):\n"
                '        if "error" in message:  # highlight: red\n'
                '            return "check logs"  # highlight: green\n'
                '        return "ok"  # highlight: green'
            ),
        ),
        CodeBlock(
            title="Declarative",
            subtitle="",
            code=(
                "def reply(message: str) -> str:\n"
                "    match message:\n"
                '        case m if "error" in m:  # highlight: red\n'
                '            return "check logs"  # highlight: green\n'
                "        case _:\n"
                '            return "ok"  # highlight: green'
            ),
        ),
        CodeBlock(
            title="Stepwise",
            subtitle="",
            code=(
                "result = []\n"
                "for item in items:  # highlight: red\n"
                "    if item.valid:  # highlight: green\n"
                "        result.append(item.name)  # highlight: green"
            ),
        ),
        CodeBlock(
            title="Composed",
            subtitle="",
            code=(
                "result = [\n"
                "    item.name  # highlight: green\n"
                "    for item in items\n"
                "    if item.valid  # highlight: green\n"
                "]"
            ),
        ),
    ]


def build_demo_block_1x1() -> list[CodeBlock]:
    """Return a built-in 1x1 example for quick previewing."""
    return [
        CodeBlock(
            title="Single block",
            subtitle="",
            code=(
                "def summarise(items: list[str]) -> str:\n"
                "    if not items:  # highlight: red\n"
                '        return "no items"  # highlight: red\n'
                '    return ", ".join(items)  # highlight: green'
            ),
        )
    ]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "thumbnail":
        title, subtitle = resolve_thumbnail_meta(
            args.title, args.subtitle, args.meta_file
        )
        svg = render_thumbnail_svg(title, subtitle)
        Path(args.output).write_text(svg, encoding="utf-8")
        return

    if args.mode == "architecture":
        content = read_text(args.input_file, args.input_text)
        svg = render_architecture_svg(
            content,
            title=_clean_title(args.title, ""),
            subtitle=_clean_title(args.subtitle, ""),
        )
        Path(args.output).write_text(svg, encoding="utf-8")
        return

    if args.mode == "gallery":
        title = _clean_title(args.title, "Untitled")
        subtitle = _clean_title(args.subtitle, "")
        images = args.gallery_image
        if args.spec_file:
            spec_title, spec_subtitle, spec_images = parse_gallery_spec(
                Path(args.spec_file).read_text(encoding="utf-8")
            )
            title = _clean_title(spec_title or title, "Untitled")
            subtitle = _clean_title(spec_subtitle or subtitle, "")
            if spec_images:
                images = spec_images
        if len(images) != 3:
            raise SystemExit(
                "Gallery mode requires exactly 3 images, via --gallery-image or --spec-file."
            )
        svg = render_gallery_svg(title, subtitle, images)
        Path(args.output).write_text(svg, encoding="utf-8")
        return

    if getattr(args, "example_1x1", False):
        blocks = build_demo_block_1x1()
        svg = render_svg(blocks, show_line_numbers=args.line_numbers)
        Path(args.output).write_text(svg, encoding="utf-8")
        return

    if getattr(args, "example_2x2", False):
        blocks = build_demo_blocks_2x2()
        svg = render_svg(blocks, show_line_numbers=args.line_numbers)
        Path(args.output).write_text(svg, encoding="utf-8")
        return

    if args.layout == "1":
        blocks = [
            resolve_block(
                args.left_title,
                None,
                "Left block",
                args.left_file,
                args.left_text,
                args.left_green_lines,
                args.left_red_lines,
            )
        ]
    elif args.layout == "2":
        left = resolve_block(
            args.left_title,
            None,
            "Left block",
            args.left_file,
            args.left_text,
            args.left_green_lines,
            args.left_red_lines,
        )
        right = resolve_block(
            args.right_title,
            None,
            "Right block",
            args.right_file,
            args.right_text,
            args.right_green_lines,
            args.right_red_lines,
        )
        blocks = [left, right]
    else:
        blocks = [
            resolve_block(
                args.top_left_title,
                None,
                "Top left",
                args.top_left_file,
                args.top_left_text,
                args.top_left_green_lines,
                args.top_left_red_lines,
            ),
            resolve_block(
                args.top_right_title,
                None,
                "Top right",
                args.top_right_file,
                args.top_right_text,
                args.top_right_green_lines,
                args.top_right_red_lines,
            ),
            resolve_block(
                args.bottom_left_title,
                None,
                "Bottom left",
                args.bottom_left_file,
                args.bottom_left_text,
                args.bottom_left_green_lines,
                args.bottom_left_red_lines,
            ),
            resolve_block(
                args.bottom_right_title,
                None,
                "Bottom right",
                args.bottom_right_file,
                args.bottom_right_text,
                args.bottom_right_green_lines,
                args.bottom_right_red_lines,
            ),
        ]

    svg = render_svg(blocks, show_line_numbers=args.line_numbers)
    Path(args.output).write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
