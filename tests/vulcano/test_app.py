import asyncio
from io import StringIO
from textwrap import dedent

import pytest

pytest.importorskip("textual")
pytest.importorskip("rich")

from rich.console import Console
from rich.markdown import Markdown
from textual.color import Color
from textual.widgets import Button, Input, OptionList, Static

from msgflux.runtime import ExecutionScope
from msgflux.vulcano import (
    BlockKind,
    BlockStatus,
    CommandOptions,
    CommandResult,
    EditorSettings,
    EventType,
    KeyBindings,
    SessionStore,
    SessionWorkspace,
    ToolRendererOptions,
    VulcanoSettings,
)
from msgflux.vulcano.app import (
    CollapsibleTranscriptBlock,
    PendingInputList,
    SessionTabBar,
    ToolExecutionBlock,
    TranscriptMessage,
    TurnNavigationItem,
    TurnSidebar,
    VulcanoApp,
)
from msgflux.vulcano.runtime import VulcanoRuntime
from msgflux.vulcano.textual_ui import VulcanoFooter, VulcanoTextArea


class _ControlledMarkdownResponder:
    def __init__(self):
        self.chunks: asyncio.Queue[str | None] = asyncio.Queue()

    async def stream(self, _prompt):
        while (chunk := await self.chunks.get()) is not None:
            yield chunk


def _write_extension(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(source), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_app_projects_streaming_runtime_events_headlessly():
    app = VulcanoApp(VulcanoRuntime(stream_delay=0, extensions_enabled=False))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        assert app.query_one("#topbar", Static).outer_size.height == 0
        sidebar = app.query_one(TurnSidebar)
        assert sidebar.is_expanded
        assert sidebar.display
        status = app.query_one("#navbar-statuses", Static)
        assert not status.display
        footer = app.query_one("#runtime-footer", VulcanoFooter)
        assert "VULCANO" in str(footer.render())
        assert "mock" in str(footer.render())
        assert app.settings.cwd.name in str(footer.render())
        welcome = next(
            message
            for message in app.query(TranscriptMessage)
            if message.kind == "welcome"
        )
        rendered_welcome = "\n".join(
            welcome.render_line(y).text for y in range(welcome.size.height)
        )
        assert "Vulcano runtime preview" in rendered_welcome

        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "hello"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.1)

        messages = list(app.query(TranscriptMessage))
        user = next(message for message in messages if message.kind == "user")
        assistant = next(message for message in messages if message.kind == "assistant")
        assert user.source_text == "hello"
        assert assistant.source_text == "Mock runtime received: hello"
        rendered_assistant = "\n".join(
            assistant.render_line(y).text for y in range(assistant.size.height)
        )
        assert "Mock runtime received: hello" in rendered_assistant
        assert "run:" not in str(footer.render())
        assert "thd:" not in str(footer.render())
        assert "Alt+S nav" in str(footer.render())
        assert "Ctrl+G sessions" in str(footer.render())
        assert "Ctrl+P" not in str(footer.render())


@pytest.mark.asyncio
async def test_session_tab_bar_switches_pins_and_closes_sessions(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.ensure("thd_one")
    store.ensure("thd_two")
    runtime = VulcanoRuntime(
        scope=ExecutionScope(thread_id="thd_one", namespace="vulcano"),
        session_store=store,
        stream_delay=0,
        extensions_enabled=False,
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(110, 40)) as pilot:
        await pilot.pause(delay=0.1)
        tabs = app.query_one(SessionTabBar)
        assert tabs.parent is app.query_one(TurnSidebar)
        assert list(tabs.entries) == ["thd_one"]
        assert tabs.entries["thd_one"].status == "active"

        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "/resume thd_two"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.2)

        assert list(tabs.entries) == ["thd_one", "thd_two"]
        assert tabs.entries["thd_one"].status == "paused"
        assert tabs.entries["thd_two"].status == "active"

        await pilot.click(tabs.entries["thd_one"].query_one(".session-tab-pin", Button))
        await pilot.pause(delay=0.1)
        assert tabs.entries["thd_one"].pinned

        await pilot.click(
            tabs.entries["thd_one"].query_one(".session-tab-select", Button)
        )
        await pilot.pause(delay=0.2)
        assert runtime.sessions.current_thread_id == "thd_one"
        assert tabs.entries["thd_one"].status == "active"
        assert tabs.entries["thd_two"].status == "paused"
        footer = app.query_one("#runtime-footer", VulcanoFooter)
        assert "thd:" not in str(footer.render())

        await pilot.click(
            tabs.entries["thd_two"].query_one(".session-tab-close", Button)
        )
        await pilot.pause(delay=0.1)
        assert list(tabs.entries) == ["thd_one"]

        await pilot.click("#session-new")
        await pilot.pause(delay=0.2)
        assert len(tabs.entries) == 2
        assert runtime.sessions.current_thread_id != "thd_one"
        transition = [
            event
            for event in runtime.history
            if event.type == EventType.SESSION_SWITCHED
        ][-1]
        assert transition.payload["kind"] == "new"

    restored = SessionWorkspace(tmp_path / "workspace.toml")
    restored.start("thd_current", ("thd_one", "thd_two", "thd_current"))
    assert [tab.thread_id for tab in restored.tabs] == ["thd_one", "thd_current"]


@pytest.mark.asyncio
async def test_session_tabs_support_tmux_style_keyboard_navigation(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    store.ensure("thd_one")
    store.ensure("thd_two")
    runtime = VulcanoRuntime(
        scope=ExecutionScope(thread_id="thd_one", namespace="vulcano"),
        session_store=store,
        stream_delay=0,
        extensions_enabled=False,
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(110, 40)) as pilot:
        await pilot.pause(delay=0.1)
        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "/resume thd_two"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.2)
        tabs = app.query_one(SessionTabBar)
        sidebar = app.query_one(TurnSidebar)

        await pilot.press("alt+s")
        assert not sidebar.is_expanded

        app.simulate_key("ctrl+g")
        app.simulate_key("p")
        await pilot.pause(delay=0.2)
        assert runtime.sessions.current_thread_id == "thd_one"
        assert prompt.text == ""
        assert not tabs.has_class("session-key-mode")

        await pilot.press("ctrl+g")
        assert tabs.has_class("session-key-mode")
        assert sidebar.is_expanded
        await pilot.press("escape")
        assert not tabs.has_class("session-key-mode")
        assert not sidebar.is_expanded

        await pilot.press("ctrl+g", "2")
        await pilot.pause(delay=0.2)
        assert runtime.sessions.current_thread_id == "thd_two"

        await pilot.press("ctrl+g", "f")
        await pilot.pause(delay=0.1)
        assert tabs.entries["thd_two"].pinned

        await pilot.press("ctrl+g", "x")
        await pilot.pause(delay=0.2)
        assert runtime.sessions.current_thread_id == "thd_one"
        assert list(tabs.entries) == ["thd_one"]

        await pilot.press("ctrl+g", "c")
        await pilot.pause(delay=0.2)
        assert len(tabs.entries) == 2
        assert runtime.sessions.current_thread_id != "thd_one"
        assert store.info(runtime.sessions.current_thread_id).parent_thread_id is None
        transition = [
            event
            for event in runtime.history
            if event.type == EventType.SESSION_SWITCHED
        ][-1]
        assert transition.payload["kind"] == "new"
        assert transition.payload["events"] == []


@pytest.mark.asyncio
async def test_session_tabs_keep_controls_visible_in_a_narrow_terminal(tmp_path):
    runtime = VulcanoRuntime(
        scope=ExecutionScope(thread_id="thd_one", namespace="vulcano"),
        session_store=SessionStore(tmp_path / "sessions"),
        stream_delay=0,
        extensions_enabled=False,
        max_session_tabs=5,
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(60, 28)) as pilot:
        await pilot.pause(delay=0.1)
        for _ in range(4):
            await pilot.press("ctrl+g", "c")
            await pilot.pause(delay=0.15)

        tabs = app.query_one(SessionTabBar)
        assert len(tabs.entries) == 5
        assert "1-5 select" in str(tabs.query_one("#session-key-hint", Static).render())
        assert tabs.size.height == tabs.virtual_size.height == 6
        assert tabs.scroll_x == 0

        active = tabs.entries[runtime.sessions.current_thread_id]
        assert active.region.x >= tabs.region.x
        assert active.region.right <= tabs.region.right
        assert active.region.y >= tabs.region.y
        assert active.region.bottom <= tabs.region.bottom
        for tab in tabs.entries.values():
            select = tab.query_one(".session-tab-select", Button)
            pin = tab.query_one(".session-tab-pin", Button)
            close = tab.query_one(".session-tab-close", Button)
            assert tab.outer_size.height == 1
            assert str(select.label).strip()
            assert select.outer_size.width == tabs.size.width - 6
            assert select.styles.text_wrap == "nowrap"
            assert pin.outer_size.width == 3
            assert close.outer_size.width == 3

        await pilot.press("ctrl+g", "1")
        await pilot.pause(delay=0.15)
        first = next(iter(tabs.entries.values()))
        assert first.region.x >= tabs.region.x
        assert first.region.right <= tabs.region.right
        assert first.region.y >= tabs.region.y
        assert first.region.bottom <= tabs.region.bottom


@pytest.mark.asyncio
async def test_session_zero_key_selects_tenth_tab(tmp_path):
    runtime = VulcanoRuntime(
        scope=ExecutionScope(thread_id="thd_one", namespace="vulcano"),
        session_store=SessionStore(tmp_path / "sessions"),
        stream_delay=0,
        extensions_enabled=False,
        max_session_tabs=10,
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(60, 28)) as pilot:
        await pilot.pause(delay=0.1)
        for _ in range(9):
            await pilot.press("ctrl+g", "c")
            await pilot.pause(delay=0.1)

        tabs = app.query_one(SessionTabBar)
        tenth_thread_id = runtime.sessions.current_thread_id
        assert len(tabs.entries) == 10

        await pilot.press("ctrl+g")
        assert "0-9 select" in str(tabs.query_one("#session-key-hint", Static).render())
        await pilot.press("1")
        await pilot.pause(delay=0.15)
        assert runtime.sessions.current_thread_id != tenth_thread_id

        await pilot.press("ctrl+g", "0")
        await pilot.pause(delay=0.15)
        assert runtime.sessions.current_thread_id == tenth_thread_id


@pytest.mark.asyncio
async def test_sidebar_toggles_with_button_and_alt_s():
    app = VulcanoApp(VulcanoRuntime(stream_delay=0, extensions_enabled=False))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        sidebar = app.query_one(TurnSidebar)
        toggle = app.query_one("#turn-sidebar-toggle", Button)
        assert sidebar.is_expanded
        assert sidebar.display
        assert str(toggle.label) == "‹"  # noqa: RUF001
        assert not toggle.disabled

        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "draft prompt"
        prompt.cursor_location = prompt.document.end
        prompt.focus()
        assert app.focused is prompt
        await pilot.press("alt+s")
        await pilot.pause()

        assert not sidebar.is_expanded
        assert not sidebar.display
        assert str(toggle.label) == "›"  # noqa: RUF001
        assert prompt.text == "draft prompt"

        await pilot.press("alt+s")
        await pilot.pause()

        assert sidebar.is_expanded
        assert sidebar.display
        assert str(toggle.label) == "‹"  # noqa: RUF001
        assert prompt.text == "draft prompt"
        assert app.query_one(SessionTabBar).parent is sidebar
        assert sidebar.region.bottom == app.query_one("#main-pane").region.bottom
        assert prompt.region.x >= sidebar.region.right

        prompt.text = "create a navigation entry"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.05)
        assert sidebar.is_expanded

        await pilot.click(toggle)
        await pilot.pause()

        assert not sidebar.is_expanded
        assert not sidebar.display
        assert str(toggle.label) == "›"  # noqa: RUF001


@pytest.mark.asyncio
async def test_sidebar_messages_use_a_fixed_scroll_region():
    app = VulcanoApp(VulcanoRuntime(stream_delay=0, extensions_enabled=False))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", VulcanoTextArea)
        for index in range(7):
            prompt.text = f"message {index} with enough text to wrap in navigation"
            prompt.cursor_location = prompt.document.end
            await pilot.press("enter")
            await pilot.pause(delay=0.05)

        messages = app.query_one("#turn-nav-messages")
        assert messages.size.height <= 10
        assert messages.virtual_size.height > messages.size.height
        assert len(app.query(TurnNavigationItem)) == 7


@pytest.mark.asyncio
async def test_app_uses_configured_editor_and_session_prefix_key(tmp_path):
    settings = VulcanoSettings(
        home=tmp_path / "home",
        cwd=tmp_path,
        editor=EditorSettings(min_height=4, max_height=9),
        keybindings=KeyBindings.from_mapping({"session_prefix": "ctrl+a"}),
    )
    app = VulcanoApp(
        VulcanoRuntime(stream_delay=0, extensions_enabled=False),
        settings=settings,
    )

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", VulcanoTextArea)
        assert prompt.editor_settings == settings.editor
        assert prompt.outer_size.height == 4

        await pilot.press("ctrl+a")
        assert app.query_one(SessionTabBar).has_class("session-key-mode")
        await pilot.press("escape")


@pytest.mark.asyncio
async def test_app_rebuilds_transcript_after_session_fork(tmp_path):
    runtime = VulcanoRuntime(
        scope=ExecutionScope(thread_id="thd_original", namespace="vulcano"),
        session_store=SessionStore(tmp_path / "sessions"),
        stream_delay=0,
        extensions_enabled=False,
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "before fork"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.05)

        prompt.text = "/fork"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.15)

        assert runtime.sessions.current_thread_id != "thd_original"
        user_messages = [
            message
            for message in app.query(TranscriptMessage)
            if message.kind == "user"
        ]
        assert [message.source_text for message in user_messages] == ["before fork"]
        footer = app.query_one("#runtime-footer", VulcanoFooter)
        assert "thd:" not in str(footer.render())
        assert app.settings.cwd.name in str(footer.render())


@pytest.mark.asyncio
async def test_streaming_markdown_rebuilds_and_renders_a_table():
    responder = _ControlledMarkdownResponder()
    runtime = VulcanoRuntime(responder=responder, extensions_enabled=False)
    app = VulcanoApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "render a table"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause()

        assistant = next(
            message
            for message in app.query(TranscriptMessage)
            if message.kind == "assistant"
        )
        responder.chunks.put_nowait("| Component | State |\n")
        await pilot.pause(delay=0.05)
        assert assistant.source_text == "| Component | State |\n"

        responder.chunks.put_nowait("|---|---|\n| Agent | streaming |\n")
        responder.chunks.put_nowait(None)
        await pilot.pause(delay=0.05)

        assert assistant.source_text == (
            "| Component | State |\n|---|---|\n| Agent | streaming |\n"
        )
        assert isinstance(assistant.content, Markdown)

        output = StringIO()
        Console(
            file=output,
            width=80,
            color_system=None,
            force_terminal=False,
        ).print(assistant.content)
        rendered = output.getvalue()
        assert "|" not in rendered
        assert any(
            "Component" in line and "State" in line for line in rendered.splitlines()
        )
        assert any(
            "Agent" in line and "streaming" in line for line in rendered.splitlines()
        )


@pytest.mark.asyncio
async def test_app_projects_pending_inputs_and_escape_cancels_execution():
    responder = _ControlledMarkdownResponder()
    runtime = VulcanoRuntime(responder=responder, extensions_enabled=False)
    app = VulcanoApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "active"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause()

        prompt.text = "steer next"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause()

        prompt.text = "follow later"
        prompt.cursor_location = prompt.document.end
        await pilot.press("alt+enter")
        await pilot.pause()

        pending = app.query_one("#pending-inputs", PendingInputList)
        assert list(pending.items.values()) == [
            ("steer", "steer next"),
            ("follow_up", "follow later"),
        ]
        assert not pending.has_class("pending-inputs-hidden")

        await pilot.press("escape")
        await pilot.pause(delay=0.05)

        assert pending.items == {}
        assert pending.has_class("pending-inputs-hidden")
        assert not runtime.is_busy


@pytest.mark.asyncio
async def test_app_projects_typed_streaming_diff_block():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)

    async def render_diff(_arguments, context):
        block_id = await context.start_block(BlockKind.DIFF, title="Parser patch")
        await context.update_block(block_id, "--- a/parser.py\n")
        await context.update_block(block_id, "+++ b/parser.py\n+fixed = True\n")
        await context.complete_block(block_id)
        return CommandResult()

    runtime.extensions.api.register_command(
        "render-diff",
        CommandOptions(description="Render a streamed diff.", handler=render_diff),
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "/render-diff"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.1)

        block = next(
            message
            for message in app.query(TranscriptMessage)
            if message.kind == f"block:{BlockKind.DIFF}"
        )
        assert block.source_text == (
            "--- a/parser.py\n+++ b/parser.py\n+fixed = True\n"
        )
        assert block.render_mode == "diff"
        assert isinstance(block.content, Markdown)


@pytest.mark.asyncio
async def test_app_projects_reasoning_and_custom_tool_lifecycle():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)
    runtime.extensions.api.ui.register_tool_renderer(
        "search",
        ToolRendererOptions(
            render_result=lambda event, _context: Static(
                f"custom result: {event.payload['result']}",
                id="custom-tool-result",
            )
        ),
    )

    async def inspect(_arguments, context):
        reasoning_id = await context.start_block(
            BlockKind.REASONING,
            title="Inspecting",
        )
        await context.update_block(reasoning_id, "Reading the parser.")
        await context.complete_block(reasoning_id)
        tool_call_id = await context.start_tool("search", {"query": "parser"})
        await context.update_tool(tool_call_id, "search", {"matches": 1})
        await context.complete_tool(tool_call_id, "search", "src/parser.py")
        return CommandResult()

    runtime.extensions.api.register_command(
        "inspect",
        CommandOptions(
            description="Inspect with reasoning and a tool.", handler=inspect
        ),
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "/inspect"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.15)

        reasoning = app.query_one(CollapsibleTranscriptBlock)
        assert not reasoning.collapsed
        assert reasoning.source_text == "Reading the parser."

        tool = app.query_one(ToolExecutionBlock)
        assert not tool.collapsed
        assert tool.status == BlockStatus.COMPLETED
        assert "search" in tool.title
        assert app.query_one("#custom-tool-result", Static).render() == (
            "custom result: src/parser.py"
        )


@pytest.mark.asyncio
async def test_failing_tool_renderer_falls_back_without_stopping_projection():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)

    def broken_renderer(_event, _context):
        raise RuntimeError("broken tool renderer")

    runtime.extensions.api.ui.register_tool_renderer(
        "broken",
        ToolRendererOptions(render_result=broken_renderer),
    )

    async def inspect(_arguments, context):
        tool_call_id = await context.start_tool("broken", {"path": "src"})
        await context.complete_tool(tool_call_id, "broken", "fallback result")
        return CommandResult()

    runtime.extensions.api.register_command(
        "broken-tool",
        CommandOptions(description="Render a broken tool.", handler=inspect),
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "/broken-tool"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.1)

        tool = app.query_one(ToolExecutionBlock)
        assert tool.status == BlockStatus.COMPLETED
        assert app.query_one(".tool-execution-body Static", Static)


@pytest.mark.asyncio
async def test_default_editor_wraps_grows_and_supports_multiline_submission():
    app = VulcanoApp(VulcanoRuntime(stream_delay=0, extensions_enabled=False))

    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", VulcanoTextArea)
        assert prompt.outer_size.height == VulcanoTextArea.MIN_HEIGHT

        prompt.text = "long input " * 30
        prompt.cursor_location = prompt.document.end
        await pilot.pause(delay=0.05)

        assert VulcanoTextArea.MIN_HEIGHT < prompt.outer_size.height
        assert prompt.outer_size.height <= VulcanoTextArea.MAX_HEIGHT

        prompt.text = "first line"
        prompt.cursor_location = prompt.document.end
        await pilot.press("shift+enter")
        prompt.insert("second line")

        assert prompt.text == "first line\nsecond line"

        await pilot.press("enter")
        await pilot.pause(delay=0.05)

        assert prompt.text == ""
        assert prompt.outer_size.height == VulcanoTextArea.MIN_HEIGHT
        assert any(
            message.source_text == "first line\nsecond line"
            for message in app.query(TranscriptMessage)
        )


@pytest.mark.asyncio
async def test_slash_command_menu_filters_and_completes_runtime_commands():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)
    app = VulcanoApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", VulcanoTextArea)
        menu = app.query_one("#command-menu", OptionList)

        assert prompt.styles.background == Color.parse("#292c33")
        assert not app.query_one("#topbar", Static).display
        assert not menu.display

        prompt.text = "/"
        prompt.cursor_location = prompt.document.end
        await pilot.pause()

        assert menu.display
        assert menu.option_count == len(runtime.commands)
        assert menu.highlighted == 0
        assert await prompt.suggester.get_suggestion("/") is None

        await pilot.press("down", "tab")
        await pilot.pause()

        assert prompt.text.startswith("/")
        assert prompt.text.endswith(" ")
        assert not menu.display

        prompt.text = "message /"
        await pilot.pause()
        assert not menu.display

        prompt.text = "/help"
        prompt.cursor_location = prompt.document.end
        await pilot.pause()
        assert menu.display

        await pilot.press("enter")
        await pilot.pause(delay=0.05)

        assert prompt.text == ""
        assert any(
            "## Commands" in message.source_text
            for message in app.query(TranscriptMessage)
        )


@pytest.mark.asyncio
async def test_clear_binding_requests_runtime_command():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)
    app = VulcanoApp(runtime)

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        app.action_request_clear()
        await pilot.pause(delay=0.05)

        assert list(app.query(TranscriptMessage)) == []
        assert any(
            event.payload.get("action") == "transcript.clear"
            for event in runtime.history
        )


@pytest.mark.asyncio
async def test_extension_customizes_textual_slots_and_event_rendering(tmp_path):
    extension = _write_extension(
        tmp_path / "custom_ui.py",
        """
        from textual.widgets import Input, Static

        from msgflux.vulcano import CommandOptions, CommandResult

        EXTENSION_NAME = "custom-ui"

        def setup(api):
            api.ui.set_status("mode", "extension UI")
            api.ui.set_widget(
                "goal",
                lambda app, theme: Static(
                    "Goal widget",
                    id="goal-widget",
                ),
                placement="below_editor",
            )
            api.ui.set_widget(
                "navigation",
                lambda app, theme: Static(
                    "Navbar widget",
                    id="navbar-widget",
                ),
            )
            api.ui.set_header(
                lambda app, theme: Static("CUSTOM HEADER", id="custom-header")
            )
            api.ui.set_footer(
                lambda app, theme: Static("CUSTOM FOOTER", id="custom-footer")
            )
            api.ui.set_editor_component(
                lambda app, theme: Input(classes="custom-editor")
            )
            api.ui.add_autocomplete_provider(
                lambda value: "/card release" if value == "/card r" else None
            )
            api.register_shortcut(
                "ctrl+g",
                lambda context: context.ui.set_status("shortcut", "conflict"),
            )
            api.register_shortcut(
                "ctrl+u",
                lambda context: context.ui.set_status("shortcut", "pressed"),
            )
            api.register_message_renderer(
                "card",
                lambda event, context: Static(
                    "Card: " + str(event.payload["content"]),
                    id="rendered-card",
                ),
            )

            async def card(arguments, context):
                context.ui.set_title("Vulcano customized")
                await context.send_message("card", arguments)
                return CommandResult()

            api.register_command(
                "card",
                CommandOptions(description="Render a custom card.", handler=card),
            )
        """,
    )
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[extension],
        discover_extensions=False,
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause(delay=0.1)

        assert app.query_one("#custom-header", Static).render() == "CUSTOM HEADER"
        assert app.query_one("#custom-footer", Static).render() == "CUSTOM FOOTER"
        assert app.query_one("#goal-widget", Static).render() == "Goal widget"
        navbar_widget = app.query_one("#navbar-widget", Static)
        assert navbar_widget.render() == "Navbar widget"
        assert navbar_widget.parent is app.query_one("#navbar-widgets")
        navbar_statuses = app.query_one("#navbar-statuses", Static)
        assert "extension UI" in str(navbar_statuses.render())

        prompt = app.query_one("#prompt", Input)
        assert prompt.has_class("custom-editor")
        prompt.value = "/ca"
        await pilot.pause()
        menu = app.query_one("#command-menu", OptionList)
        assert menu.display
        assert menu.option_count == 1
        await pilot.press("tab")
        assert prompt.value == "/card "
        assert await prompt.suggester.get_suggestion("/card r") == "/card release"
        await pilot.press("ctrl+g")
        assert app.query_one(SessionTabBar).has_class("session-key-mode")
        assert "conflict" not in str(navbar_statuses.render())
        await pilot.press("escape")
        await pilot.press("ctrl+u")
        await pilot.pause(delay=0.05)
        assert "pressed" in str(navbar_statuses.render())

        prompt.value = "/card release"
        await pilot.press("enter")
        await pilot.pause(delay=0.1)

        assert app.title == "Vulcano customized"
        assert app.query_one("#rendered-card", Static).render() == "Card: release"


@pytest.mark.asyncio
async def test_extension_dialog_and_custom_overlay_round_trip(tmp_path):
    extension = _write_extension(
        tmp_path / "dialogs.py",
        """
        from textual.widgets import Static

        from msgflux.vulcano import (
            CommandOptions,
            CommandResult,
            EventDraft,
            EventType,
        )

        EXTENSION_NAME = "dialogs"

        def setup(api):
            async def ask(arguments, context):
                del arguments
                confirmed = await context.ui.confirm(
                    "Execute?",
                    "Run the custom flow?",
                )
                return CommandResult(events=(
                    EventDraft(
                        EventType.COMMAND_OUTPUT,
                        {"text": f"confirmed={confirmed}"},
                    ),
                ))

            async def panel(arguments, context):
                del arguments
                result = await context.ui.custom(
                    lambda app, theme, done: Static(
                        "Custom panel",
                        id="custom-panel",
                    ),
                    overlay=True,
                )
                return CommandResult(events=(
                    EventDraft(
                        EventType.COMMAND_OUTPUT,
                        {"text": f"panel={result}"},
                    ),
                ))

            api.register_command(
                "ask",
                CommandOptions(description="Ask for confirmation.", handler=ask),
            )
            api.register_command(
                "panel",
                CommandOptions(description="Show a custom panel.", handler=panel),
            )
        """,
    )
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[extension],
        discover_extensions=False,
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause(delay=0.1)
        prompt = app.query_one("#prompt", VulcanoTextArea)

        prompt.text = "/ask"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.2)
        await pilot.click(app.screen.query_one("#dialog-yes", Button))
        await pilot.pause(delay=0.1)

        assert any(
            message.source_text == "confirmed=True"
            for message in app.query(TranscriptMessage)
        )

        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "/panel"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.2)
        assert app.screen.query_one("#custom-panel", Static).render() == "Custom panel"
        await pilot.press("escape")
        await pilot.pause(delay=0.1)

        assert any(
            message.source_text == "panel=None"
            for message in app.query(TranscriptMessage)
        )


@pytest.mark.asyncio
async def test_broken_renderer_is_isolated_from_event_projection(tmp_path):
    extension = _write_extension(
        tmp_path / "broken_renderer.py",
        """
        from msgflux.vulcano import CommandOptions, CommandResult

        EXTENSION_NAME = "broken-renderer"

        def setup(api):
            api.register_message_renderer(
                "broken",
                lambda event, context: lambda app, theme: 1 / 0,
            )

            async def broken(arguments, context):
                await context.send_message("broken", arguments)
                return CommandResult()

            api.register_command(
                "broken",
                CommandOptions(
                    description="Render a broken component.",
                    handler=broken,
                ),
            )
        """,
    )
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[extension],
        discover_extensions=False,
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause(delay=0.1)
        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "/broken card"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.1)

        assert any(
            "UI renderer failed: division by zero" in message.source_text
            for message in app.query(TranscriptMessage)
        )

        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "/echo projection survived"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.1)

        assert any(
            message.source_text == "projection survived"
            for message in app.query(TranscriptMessage)
        )
