"""MkDocs hook: Copy Page button.

Injects a "Copy Page" button that copies the current page's Markdown to the
clipboard. The Markdown is stored in a ``data-markdown`` HTML attribute so it
is always read from the live DOM — compatible with Material's
``navigation.instant`` SPA navigation.
"""

import html as html_lib
import re

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

BUTTON_TEXT = "Copy Page"
BUTTON_POSITION = "after-title"  # "top" | "bottom" | "after-title"

_HEADING_RE = re.compile(r"(<h[1-6][^>]*>.*?</h[1-6]>)", re.IGNORECASE | re.DOTALL)

_CSS = """\
<link rel="stylesheet" href="https://unpkg.com/@phosphor-icons/web@2.1.1/src/regular/style.css">
<style>
.copy-markdown-button {
    background: transparent;
    color: inherit;
    border: 1px solid currentColor;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 10px 0;
    opacity: 0.7;
}
.copy-markdown-button:hover { opacity: 1; }
.copy-markdown-button:active { transform: translateY(1px); }
#copy-icon { transition: opacity 0.1s ease; }
</style>"""

_JS = """\
<script>
(function () {
    let isAnimating = false;

    function copyMarkdownToClipboard(btn) {
        if (isAnimating) return;
        const markdown = btn.closest(".copy-markdown-container").getAttribute("data-markdown");
        const icon = btn.querySelector("#copy-icon");
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(markdown)
                .then(() => showCopySuccess(icon))
                .catch(() => fallback(markdown, icon));
        } else {
            fallback(markdown, icon);
        }
    }

    function fallback(text, icon) {
        const ta = document.createElement("textarea");
        ta.value = text;
        Object.assign(ta.style, { position: "fixed", top: "0", left: "0", opacity: "0" });
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        try {
            document.execCommand("copy") ? showCopySuccess(icon) : showCopyError(icon);
        } catch {
            showCopyError(icon);
        }
        document.body.removeChild(ta);
    }

    function transition(icon, newClass) {
        if (isAnimating && icon.className !== "ph ph-clipboard") {
            clearTimeout(window._copyTimeout);
            window._copyTimeout = setTimeout(() => resetIcon(icon), 2000);
            return;
        }
        isAnimating = true;
        icon.style.opacity = "0";
        setTimeout(() => {
            icon.className = newClass;
            icon.style.opacity = "1";
            window._copyTimeout = setTimeout(() => resetIcon(icon), 1800);
        }, 100);
    }

    function showCopySuccess(icon) { transition(icon, "ph ph-check"); }
    function showCopyError(icon)   { transition(icon, "ph ph-x"); }

    function resetIcon(icon) {
        icon.style.opacity = "0";
        setTimeout(() => {
            icon.className = "ph ph-clipboard";
            icon.style.opacity = "1";
            isAnimating = false;
        }, 100);
    }

    window.copyMarkdownToClipboard = copyMarkdownToClipboard;
}());
</script>"""


def _build_button(markdown: str, button_text: str) -> str:
    escaped = html_lib.escape(markdown, quote=True)
    return (
        f'{_CSS}\n'
        f'<div class="copy-markdown-container" data-markdown="{escaped}">\n'
        f'  <button class="copy-markdown-button" onclick="copyMarkdownToClipboard(this)">\n'
        f'    <i id="copy-icon" class="ph ph-clipboard" style="font-size:16px;"></i>\n'
        f'    <span>{html_lib.escape(button_text)}</span>\n'
        f'  </button>\n'
        f'</div>\n'
        f'{_JS}'
    )


_HIDE_NAV_PAGES = {"CHANGELOG.md"}


def on_page_context(
    context: dict,
    page: Page,
    config: MkDocsConfig,
    nav,
) -> dict:
    if page.file.src_path in _HIDE_NAV_PAGES:
        page.meta.setdefault("hide", [])
        for item in ("navigation", "toc"):
            if item not in page.meta["hide"]:
                page.meta["hide"].append(item)
    return context


def on_page_content(
    html: str,
    page: Page,
    config: MkDocsConfig,
    files: Files,
) -> str:
    if not page.markdown:
        return html

    button = _build_button(page.markdown, BUTTON_TEXT)

    if BUTTON_POSITION == "top":
        return button + html
    elif BUTTON_POSITION == "bottom":
        return html + button
    else:  # after-title
        match = _HEADING_RE.search(html)
        if match:
            end = match.end()
            return html[:end] + button + html[end:]
        return button + html
