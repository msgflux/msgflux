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
                "- `/ui-turn [prompt]` — grouped Agent execution and sidebar entry",
                "- `/ui-status [text|clear]` — status contribution",
                "- `/ui-widget [navbar|above|below|clear]` — layout widget",
                "- `/ui-notify [info|warning|error]` — notification",
                "- `/ui-dialogs` — selector, confirmation, input and editor",
                "- `/ui-permission [command]` — runtime-owned permission request",
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


async def _ui_turn(args, ctx):
    run_id = ctx.scope.run_id or "run_gallery"
    scope = ctx.scope.to_dict()
    user_message_id = f"msg_user_{run_id}"
    update_message_id = f"msg_update_{run_id}"
    final_message_id = f"msg_final_{run_id}"
    prompt = args.strip() or "Review the streaming adapter and show the patch."

    await ctx.emit(
        EventDraft(
            EventType.MESSAGE_USER,
            {
                "message_id": user_message_id,
                "run_id": run_id,
                "content": prompt,
                "scope": scope,
            },
        )
    )
    await ctx.emit(
        EventDraft(
            EventType.EXECUTION_STARTED,
            {
                "run_id": run_id,
                "input_message_id": user_message_id,
                "agent": "mock-code-agent",
                "scope": scope,
            },
        )
    )

    reasoning_id = await ctx.start_block(
        BlockKind.REASONING,
        title="Planning the review",
    )
    await ctx.update_block(reasoning_id, "Inspecting the event adapter.\n")
    await asyncio.sleep(0.06)
    await ctx.update_block(reasoning_id, "Selecting the smallest safe patch.")
    await ctx.complete_block(reasoning_id)

    search_call_id = await ctx.start_tool(
        "gallery-search",
        {"query": "stream_events adapter"},
    )
    await asyncio.sleep(0.06)
    await ctx.update_tool(
        search_call_id,
        "gallery-search",
        {"matches": 4, "scanned": 23},
    )
    await ctx.complete_tool(
        search_call_id,
        "gallery-search",
        ["extensions/agent.py", "events.py", "app.py"],
    )

    message_call_id = await ctx.start_tool(
        "send_user_message",
        {"content": "I found the adapter boundary; preparing the diff."},
    )
    await ctx.emit(
        EventDraft(
            EventType.ASSISTANT_USER_MESSAGE,
            {
                "message_id": update_message_id,
                "run_id": run_id,
                "tool_call_id": message_call_id,
                "content": (
                    "**Agent update**\n\n"
                    "I found the adapter boundary; preparing the diff."
                ),
                "severity": "info",
                "scope": scope,
            },
        )
    )
    await ctx.complete_tool(
        message_call_id,
        "send_user_message",
        {"delivered": True},
    )

    await ctx.send_block(
        BlockKind.DIFF,
        (
            "--- a/src/msgflux/vulcano/extensions/agent.py\n"
            "+++ b/src/msgflux/vulcano/extensions/agent.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-response = await agent.acall(message)\n"
            "+async for event in agent.stream_events(message):\n"
            "+    yield translate(event)\n"
        ),
        title="Mock adapter patch",
        details={
            "path": "src/msgflux/vulcano/extensions/agent.py",
            "operation": "modify",
            "state": "applied",
            "additions": 2,
            "deletions": 1,
            "tool_call_id": search_call_id,
        },
    )

    final_chunks = (
        "## Review complete\n\n",
        "The adapter boundary can consume native events without coupling the "
        "Agent to Textual.\n\n",
        "| Changed file | Result |\n|---|---|\n",
        "| `extensions/agent.py` | event translation added |\n",
    )
    await ctx.emit(
        EventDraft(
            EventType.ASSISTANT_STARTED,
            {
                "message_id": final_message_id,
                "run_id": run_id,
                "scope": scope,
            },
        )
    )
    final_content = ""
    for chunk in final_chunks:
        await asyncio.sleep(0.06)
        final_content += chunk
        await ctx.emit(
            EventDraft(
                EventType.ASSISTANT_DELTA,
                {
                    "message_id": final_message_id,
                    "run_id": run_id,
                    "delta": chunk,
                },
            )
        )
    await ctx.emit(
        EventDraft(
            EventType.ASSISTANT_COMPLETED,
            {
                "message_id": final_message_id,
                "run_id": run_id,
                "content": final_content,
                "status": "completed",
            },
        )
    )
    await ctx.emit(
        EventDraft(
            EventType.EXECUTION_COMPLETED,
            {
                "run_id": run_id,
                "scope": scope,
                "status": "completed",
                "final_message_id": final_message_id,
                "duration_ms": 840,
                "tool_count": 2,
                "changed_files": ["src/msgflux/vulcano/extensions/agent.py"],
            },
        )
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
    position = args.strip().lower() or "navbar"
    if position == "clear":
        ctx.ui.set_widget("gallery-dynamic", None)
        return _output("Gallery widget cleared.")
    if position not in {"navbar", "above", "below"}:
        return _output("Usage: `/ui-widget [navbar|above|below|clear]`")
    placement = {
        "navbar": "navbar",
        "above": "above_editor",
        "below": "below_editor",
    }[position]
    description = "in the navbar" if position == "navbar" else f"{position} the editor"
    ctx.ui.set_widget(
        "gallery-dynamic",
        lambda _app, _theme: Static(
            Panel(
                f"Mock widget placed {description}.",
                border_style="#ff6a1a",
            )
        ),
        placement=placement,
    )
    return _output(f"Gallery widget placed {description}.")


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


async def _ui_permission(args, ctx):
    command = args.strip() or "python -m pytest tests/vulcano"
    result = await ctx.request_permission(
        "shell",
        "Allow the mock agent to execute this shell command?",
        resource=command,
        remember_key=f"gallery-shell:{command}",
        metadata={"mock": True},
    )
    return _output(
        f"Permission decision: **{result.decision}** (source: `{result.source}`)."
    )


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
    ctx.ui.set_status("gallery-dynamic", None)
    ctx.ui.set_status("gallery-shortcut", None)
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
        "ui-turn",
        "Simulate one grouped Agent execution with a final answer.",
        _ui_turn,
    ),
    (
        "ui-markdown",
        "Stream Markdown with a table and code block.",
        _ui_markdown,
    ),
    ("ui-status", "Set or clear a mock extension status.", _ui_status),
    ("ui-widget", "Show a mock widget in a UI placement.", _ui_widget),
    ("ui-notify", "Show a mock Textual notification.", _ui_notify),
    ("ui-dialogs", "Run every built-in UI dialog.", _ui_dialogs),
    ("ui-permission", "Request permission for a mock shell command.", _ui_permission),
    ("ui-overlay", "Open a mock extension-owned overlay.", _ui_overlay),
    ("ui-slots", "Show or clear mock header and footer slots.", _ui_slots),
    ("ui-working", "Configure the mock streaming indicator.", _ui_working),
    ("ui-title", "Set or reset the mock terminal title.", _ui_title),
    ("ui-theme", "List or select a Textual theme.", _ui_theme),
    ("ui-reset", "Clear every widget gallery contribution.", _ui_reset),
)


def setup(api):
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
        "ctrl+u",
        lambda ctx: ctx.ui.set_status("gallery-shortcut", "Ctrl+U pressed"),
    )
    for name, description, handler in _COMMANDS:
        api.register_command(
            name,
            CommandOptions(description=description, handler=handler),
        )
