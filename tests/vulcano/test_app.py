from textwrap import dedent

import pytest

pytest.importorskip("textual")
pytest.importorskip("rich")

from textual.color import Color
from textual.widgets import Button, Input, OptionList, Static

from msgflux.vulcano.app import TranscriptMessage, VulcanoApp
from msgflux.vulcano.runtime import VulcanoRuntime
from msgflux.vulcano.textual_ui import VulcanoTextArea


def _write_extension(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(source), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_app_projects_streaming_runtime_events_headlessly():
    app = VulcanoApp(VulcanoRuntime(stream_delay=0, extensions_enabled=False))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        assert app.query_one("#topbar", Static).outer_size.height == 4
        status = app.query_one("#status", Static)
        assert "mock runtime" in str(status.render())

        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "hello"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.05)

        messages = list(app.query(TranscriptMessage))
        user = next(message for message in messages if message.kind == "user")
        assistant = next(message for message in messages if message.kind == "assistant")
        assert user.source_text == "hello"
        assert assistant.source_text == "Mock runtime received: hello"


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
        assert app.query_one("#topbar", Static).styles.color == Color.parse("#ff3344")
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
        assert "extension UI" in str(app.query_one("#status", Static).render())

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
        await pilot.pause(delay=0.05)
        assert "pressed" in str(app.query_one("#status", Static).render())

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
