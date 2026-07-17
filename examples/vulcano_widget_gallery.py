import asyncio
from collections.abc import Mapping

from rich.panel import Panel
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Static

from msgflux.vulcano import (
    BlockKind,
    CommandOptions,
    CommandResult,
    EventDraft,
    EventType,
    ToolRendererOptions,
    WorkingIndicatorOptions,
)

EXTENSION_NAME = "widget-gallery"
EXTENSION_API_VERSION = 1


def _output(text: str) -> CommandResult:
    return CommandResult(events=(EventDraft(EventType.COMMAND_OUTPUT, {"text": text}),))


def _render_card(event, _context):
    content = event.payload.get("content", {})
    if isinstance(content, Mapping):
        title = str(content.get("title", "Widget gallery"))
        body = str(content.get("body", ""))
    else:
        title = "Widget gallery"
        body = str(content)
    return Static(
        Panel(body, title=title, border_style="#ff6a1a"),
        classes="gallery-card",
    )


class GalleryOverlay(Static):
    DEFAULT_CSS = """
    GalleryOverlay {
        width: 64;
        height: auto;
        padding: 1 2;
        border: round #ff6a1a;
        background: #1b1d23;
    }

    GalleryOverlay Horizontal {
        height: auto;
        align-horizontal: right;
        margin-top: 1;
    }

    GalleryOverlay Button {
        margin-left: 1;
    }
    """

    def __init__(self, done):
        super().__init__()
        self._done = done

    def compose(self) -> ComposeResult:
        yield Static("This component is owned by a Vulcano extension.")
        with Horizontal():
            yield Button("Cancel", id="gallery-cancel")
            yield Button("Accept", id="gallery-accept", variant="primary")

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        result = "accepted" if event.button.id == "gallery-accept" else "cancelled"
        self._done(result)


def _ui_help(_args, _ctx):
    return _output(
        "\n".join(
            (
                "## Widget gallery",
                "",
                "- `/ui-card [text]` — custom Rich renderer",
                "- `/ui-markdown` — streamed Markdown and table",
                "- `/ui-lifecycle` — reasoning, diff, artifact and tool blocks",
                "- `/ui-status [text|clear]` — status contribution",
                "- `/ui-widget [above|below|clear]` — layout widget",
                "- `/ui-notify [info|warning|error]` — notification",
                "- `/ui-dialogs` — selector, confirmation, input and editor",
                "- `/ui-overlay` — focused custom component",
                "- `/ui-slots [show|clear]` — custom header and footer",
                "- `/ui-working [show|hide|reset]` — streaming indicator",
                "- `/ui-title [text|reset]` — terminal title",
                "- `/ui-theme [name]` — list or select Textual themes",
                "- `/ui-reset` — clear gallery UI contributions",
            )
        )
    )


async def _ui_card(args, ctx):
    await ctx.send_message(
        "gallery-card",
        {
            "title": "Mock agent event",
            "body": args or "A custom renderer can display structured output.",
        },
    )
    return CommandResult()


async def _ui_markdown(_args, ctx):
    chunks = (
        "## Streaming Markdown\n\n",
        "The transcript rebuilds the accumulated document after each delta.\n\n",
        "| Component | State |\n",
        "|---|---|\n",
        "| Agent output | streaming |\n",
        "| Markdown tables | rendered |\n\n",
        "```python\n",
        "async for event in agent.stream_events():\n    render(event)\n",
        "```\n",
    )
    await ctx.emit(EventDraft(EventType.ASSISTANT_STARTED))
    content = ""
    for chunk in chunks:
        await asyncio.sleep(0.06)
        content += chunk
        await ctx.emit(EventDraft(EventType.ASSISTANT_DELTA, {"delta": chunk}))
    await ctx.emit(
        EventDraft(
            EventType.ASSISTANT_COMPLETED,
            {"content": content, "status": "completed"},
        )
    )
    return CommandResult()


async def _ui_lifecycle(args, ctx):
    reasoning_id = await ctx.start_block(
        BlockKind.REASONING,
        title="Inspecting mock repository",
    )
    await ctx.update_block(reasoning_id, "Reading the parser and its tests.\n")
    await asyncio.sleep(0.08)
    await ctx.update_block(reasoning_id, "Comparing the public contracts.")
    await ctx.complete_block(reasoning_id)

    await ctx.send_block(
        BlockKind.DIFF,
        "- old_color = '#ff3344'\n+ new_color = '#ff6a1a'",
        title="Mock patch",
    )
    await ctx.send_block(
        BlockKind.ARTIFACT,
        "## Mock artifact\n\n| File | State |\n|---|---|\n| parser.py | reviewed |",
        title="Review report",
    )

    tool_call_id = await ctx.start_tool(
        "gallery-search",
        {"query": args or "ExtensionApi"},
    )
    await asyncio.sleep(0.08)
    await ctx.update_tool(
        tool_call_id,
        "gallery-search",
        {"matches": 3, "scanned": 18},
    )
    await asyncio.sleep(0.08)
    await ctx.complete_tool(
        tool_call_id,
        "gallery-search",
        ["runtime.py", "ui.py", "extensions/api.py"],
    )
    return CommandResult()


def _render_tool_call(event, _context):
    return Static(
        Panel(
            str(event.payload.get("arguments", {})),
            title="Mock search arguments",
            border_style="#ff6a1a",
        )
    )


def _render_tool_update(event, _context):
    return Static(f"Progress: {event.payload.get('update', {})}")


def _render_tool_result(event, _context):
    return Static(
        Panel(
            str(event.payload.get("result", "")),
            title="Mock search result",
            border_style="#ff6a1a",
        )
    )


def _ui_status(args, ctx):
    value = args.strip()
    ctx.ui.set_status(
        "gallery-dynamic",
        None if value in {"", "clear"} else value,
    )
    return _output("Gallery status updated.")


def _ui_widget(args, ctx):
    position = args.strip().lower() or "above"
    if position == "clear":
        ctx.ui.set_widget("gallery-dynamic", None)
        return _output("Gallery widget cleared.")
    if position not in {"above", "below"}:
        return _output("Usage: `/ui-widget [above|below|clear]`")
    placement = "below_editor" if position == "below" else "above_editor"
    ctx.ui.set_widget(
        "gallery-dynamic",
        lambda _app, _theme: Static(
            Panel(
                f"Mock widget placed {position} the editor.",
                border_style="#ff6a1a",
            )
        ),
        placement=placement,
    )
    return _output(f"Gallery widget placed {position} the editor.")


def _ui_notify(args, ctx):
    severity = args.strip().lower() or "info"
    if severity not in {"info", "warning", "error"}:
        return _output("Usage: `/ui-notify [info|warning|error]`")
    ctx.ui.notify(f"Mock {severity} notification", severity)
    return CommandResult()


async def _ui_dialogs(_args, ctx):
    choice = await ctx.ui.select("Mock selector", ["alpha", "beta", "gamma"])
    confirmed = await ctx.ui.confirm(
        "Mock confirmation",
        f"Continue with {choice or 'no selection'}?",
    )
    name = await ctx.ui.input("Mock input", "Type a name")
    notes = await ctx.ui.editor("Mock editor", "Multiline notes")
    await ctx.send_message(
        "gallery-card",
        {
            "title": "Dialog results",
            "body": (
                f"choice={choice!r}\nconfirmed={confirmed!r}\n"
                f"name={name!r}\nnotes={notes!r}"
            ),
        },
    )
    return CommandResult()


async def _ui_overlay(_args, ctx):
    result = await ctx.ui.custom(
        lambda _app, _theme, done: GalleryOverlay(done),
        overlay=True,
        overlay_options={"width": 68},
    )
    await ctx.send_message(
        "gallery-card",
        {"title": "Overlay result", "body": repr(result)},
    )
    return CommandResult()


def _ui_slots(args, ctx):
    if args.strip().lower() == "clear":
        ctx.ui.set_header(None)
        ctx.ui.set_footer(None)
        return _output("Default header and footer restored.")
    ctx.ui.set_header(
        lambda _app, _theme: Static(
            "VULCANO WIDGET GALLERY",
            id="gallery-header",
        )
    )
    ctx.ui.set_footer(
        lambda _app, _theme: Static(
            "Gallery footer • /ui-slots clear",
            id="gallery-footer",
        )
    )
    return _output("Gallery header and footer installed.")


def _ui_working(args, ctx):
    action = args.strip().lower() or "show"
    if action == "reset":
        ctx.ui.set_working_message()
        ctx.ui.set_working_visible()
        ctx.ui.set_working_indicator()
        return _output("Working indicator restored.")
    if action == "hide":
        ctx.ui.set_working_visible(False)
        return _output("Working indicator hidden.")
    if action != "show":
        return _output("Usage: `/ui-working [show|hide|reset]`")
    ctx.ui.set_working_visible(True)
    ctx.ui.set_working_message("Mock agent is processing")
    ctx.ui.set_working_indicator(
        WorkingIndicatorOptions(
            frames=("·", "•", "●", "•"),
            interval=0.12,
        )
    )
    return _output("Send a normal message to preview the indicator.")


def _ui_title(args, ctx):
    title = args.strip()
    ctx.ui.set_title(None if title in {"", "reset"} else title)
    return _output("Terminal title updated.")


def _ui_theme(args, ctx):
    name = args.strip()
    if not name:
        return _output("Available themes: " + ", ".join(ctx.ui.get_themes()))
    result = ctx.ui.set_theme(name)
    return _output("Theme updated." if result.success else str(result.error))


def _ui_reset(_args, ctx):
    ctx.ui.set_status("gallery", None)
    ctx.ui.set_status("gallery-dynamic", None)
    ctx.ui.set_status("gallery-shortcut", None)
    ctx.ui.set_widget("gallery-hint", None)
    ctx.ui.set_widget("gallery-dynamic", None)
    ctx.ui.set_header(None)
    ctx.ui.set_footer(None)
    ctx.ui.set_title(None)
    ctx.ui.set_working_message()
    ctx.ui.set_working_visible()
    ctx.ui.set_working_indicator()
    return _output("Widget gallery contributions cleared. Use `/reload` to restore.")


_COMMANDS = (
    ("ui-help", "List widget gallery commands.", _ui_help),
    ("ui-card", "Render a mock card in the transcript.", _ui_card),
    (
        "ui-lifecycle",
        "Render reasoning, diff, artifact and tool lifecycles.",
        _ui_lifecycle,
    ),
    (
        "ui-markdown",
        "Stream Markdown with a table and code block.",
        _ui_markdown,
    ),
    ("ui-status", "Set or clear a mock extension status.", _ui_status),
    ("ui-widget", "Show a mock widget above or below the editor.", _ui_widget),
    ("ui-notify", "Show a mock Textual notification.", _ui_notify),
    ("ui-dialogs", "Run every built-in UI dialog.", _ui_dialogs),
    ("ui-overlay", "Open a mock extension-owned overlay.", _ui_overlay),
    ("ui-slots", "Show or clear mock header and footer slots.", _ui_slots),
    ("ui-working", "Configure the mock streaming indicator.", _ui_working),
    ("ui-title", "Set or reset the mock terminal title.", _ui_title),
    ("ui-theme", "List or select a Textual theme.", _ui_theme),
    ("ui-reset", "Clear every widget gallery contribution.", _ui_reset),
)


def setup(api):
    api.ui.set_status("gallery", "widget gallery loaded")
    api.ui.set_widget(
        "gallery-hint",
        ["Widget gallery", "Try /ui-help"],
        placement="above_editor",
    )
    api.register_message_renderer("gallery-card", _render_card)
    api.ui.register_tool_renderer(
        "gallery-search",
        ToolRendererOptions(
            render_call=_render_tool_call,
            render_update=_render_tool_update,
            render_result=_render_tool_result,
        ),
    )
    api.ui.add_autocomplete_provider(
        lambda value: "/ui-notify warning" if value == "/ui-notify w" else None
    )
    api.register_shortcut(
        "ctrl+g",
        lambda ctx: ctx.ui.set_status("gallery-shortcut", "Ctrl+G pressed"),
    )
    for name, description, handler in _COMMANDS:
        api.register_command(
            name,
            CommandOptions(description=description, handler=handler),
        )
