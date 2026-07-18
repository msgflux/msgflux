from pathlib import Path

import pytest

pytest.importorskip("textual")
pytest.importorskip("rich")

from rich.panel import Panel
from textual.containers import Container
from textual.widgets import OptionList, Static

from msgflux.vulcano import (
    EventType,
    SubmitInput,
    VulcanoRuntime,
    custom_message_type,
)
from msgflux.vulcano.app import (
    ToolExecutionBlock,
    TranscriptMessage,
    TurnActivity,
    TurnNavigationItem,
    TurnSidebar,
    VulcanoApp,
)
from msgflux.vulcano.textual_ui import VulcanoTextArea


GALLERY_EXTENSION = Path(__file__).parents[2] / "examples" / "vulcano_widget_gallery.py"


@pytest.mark.asyncio
async def test_widget_gallery_exercises_runtime_owned_ui_headlessly():
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[GALLERY_EXTENSION],
        discover_extensions=False,
    )
    await runtime.start()

    gallery_commands = {command.name for command in runtime.commands if command.owner}
    assert {
        "ui-card",
        "ui-dialogs",
        "ui-help",
        "ui-lifecycle",
        "ui-markdown",
        "ui-notify",
        "ui-overlay",
        "ui-permission",
        "ui-reset",
        "ui-slots",
        "ui-status",
        "ui-theme",
        "ui-title",
        "ui-turn",
        "ui-widget",
        "ui-working",
    } <= gallery_commands
    assert [status.text for status in runtime.ui.state.statuses] == [
        "widget gallery loaded"
    ]
    assert runtime.ui.state.widgets[0].key == "gallery-hint"

    await runtime.dispatch(SubmitInput("/ui-status indexing repository"))
    await runtime.dispatch(SubmitInput("/ui-widget below"))
    await runtime.dispatch(SubmitInput("/ui-working show"))
    await runtime.dispatch(SubmitInput("/ui-title Gallery preview"))
    await runtime.dispatch(SubmitInput("/ui-card structured result"))
    await runtime.dispatch(SubmitInput("/ui-markdown"))
    await runtime.dispatch(SubmitInput("/ui-lifecycle ExtensionApi"))
    await runtime.dispatch(SubmitInput("/ui-dialogs"))
    await runtime.dispatch(SubmitInput("/ui-overlay"))

    state = runtime.ui.state
    assert [status.text for status in state.statuses] == [
        "widget gallery loaded",
        "indexing repository",
    ]
    assert state.widgets[-1].placement == "below_editor"
    assert state.working_message == "Mock agent is processing"
    assert state.working_indicator.frames == ("·", "•", "●", "•")
    assert state.title == "Gallery preview"
    assert (
        sum(
            event.type == custom_message_type("gallery-card")
            for event in runtime.history
        )
        == 3
    )
    markdown_events = [
        event
        for event in runtime.history
        if event.type
        in {
            EventType.ASSISTANT_STARTED,
            EventType.ASSISTANT_DELTA,
            EventType.ASSISTANT_COMPLETED,
        }
    ]
    assert [event.type for event in markdown_events] == [
        EventType.ASSISTANT_STARTED,
        *(EventType.ASSISTANT_DELTA for _ in range(9)),
        EventType.ASSISTANT_COMPLETED,
    ]
    assert "| Markdown tables | rendered |" in str(
        markdown_events[-1].payload["content"]
    )
    assert any(event.type == EventType.BLOCK_STARTED for event in runtime.history)
    assert any(event.type == EventType.BLOCK_COMPLETED for event in runtime.history)
    assert any(event.type == EventType.TOOL_STARTED for event in runtime.history)
    assert any(event.type == EventType.TOOL_COMPLETED for event in runtime.history)

    await runtime.dispatch(SubmitInput("/ui-reset"))

    assert runtime.ui.state.statuses == ()
    assert runtime.ui.state.widgets == ()
    assert runtime.ui.state.title is None
    assert runtime.ui.state.working_message is None


@pytest.mark.asyncio
async def test_widget_gallery_reload_recreates_initial_contributions():
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[GALLERY_EXTENSION],
        discover_extensions=False,
    )
    await runtime.start()
    await runtime.dispatch(SubmitInput("/ui-reset"))

    await runtime.extensions.reload()

    assert [status.text for status in runtime.ui.state.statuses] == [
        "widget gallery loaded"
    ]
    assert [widget.key for widget in runtime.ui.state.widgets] == ["gallery-hint"]


@pytest.mark.asyncio
async def test_widget_gallery_materializes_cards_widgets_and_slots():
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[GALLERY_EXTENSION],
        discover_extensions=False,
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause(delay=0.1)
        prompt = app.query_one("#prompt", VulcanoTextArea)

        prompt.text = "/ui-card structured result"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.1)
        assert app.query_one(".gallery-card", Static)

        prompt.text = "/ui-widget below"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.1)
        assert len(app.query_one("#widgets-below", Container).children) == 1

        prompt.text = "/ui-slots show"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.1)
        assert app.query_one("#gallery-header", Static)
        assert app.query_one("#gallery-footer", Static)

        prompt.text = "/ui-lifecycle ExtensionApi"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.5)
        assert app.query_one(".reasoning-message")
        assert app.query_one(".diff-message")
        assert app.query_one(".artifact-message")
        tool_result = app.query_one(".tool-execution-body Static", Static)
        assert isinstance(tool_result.content, Panel)
        assert "runtime.py" in str(tool_result.content.renderable)


@pytest.mark.asyncio
async def test_widget_gallery_simulates_grouped_execution_and_sidebar_navigation():
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[GALLERY_EXTENSION],
        discover_extensions=False,
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(110, 44)) as pilot:
        await pilot.pause(delay=0.1)
        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = "/ui-turn Review the streaming adapter"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.8)

        sidebar = app.query_one(TurnSidebar)
        assert len(sidebar.entries) == 1
        navigation = next(iter(sidebar.entries.values()))
        assert isinstance(navigation, TurnNavigationItem)
        assert navigation.ordinal == 1
        assert navigation.execution_status == "completed"
        assert "Review the streaming" in str(navigation.label)

        activity = app.query_one(TurnActivity)
        assert not activity.collapsed
        assert activity.status == "completed"
        assert "2 tools" in activity.title
        assert "1 file" in activity.title
        assert "840 ms" in activity.title
        assert len(activity.query(ToolExecutionBlock)) == 2

        grouped_messages = list(activity.query(TranscriptMessage))
        assert any(
            message.kind == "assistant:user-message"
            and "preparing the diff" in message.source_text
            for message in grouped_messages
        )
        assert any(
            message.kind == "block:diff" and "stream_events" in message.source_text
            for message in grouped_messages
        )

        final = next(
            message
            for message in app.query(TranscriptMessage)
            if message.message_id and message.message_id.startswith("msg_final_")
        )
        assert "Review complete" in final.source_text
        assert final not in grouped_messages
        assert final.parent is app.query_one("#transcript")

        prompt.text = "/view compact"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.1)

        assert activity.collapsed
        assert activity.has_class("-collapsed")
        assert all(tool.collapsed for tool in activity.query(ToolExecutionBlock))
        assert all(
            tool.has_class("-collapsed") for tool in activity.query(ToolExecutionBlock)
        )

        prompt.text = "/view full"
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.1)

        assert not activity.collapsed
        assert not activity.has_class("-collapsed")
        assert all(not tool.collapsed for tool in activity.query(ToolExecutionBlock))

        await pilot.click(navigation)
        await pilot.pause()
        assert navigation.has_class("turn-nav-selected")
        assert navigation.anchor.source_text == "Review the streaming adapter"


@pytest.mark.asyncio
async def test_widget_gallery_permission_round_trip_and_session_grant():
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[GALLERY_EXTENSION],
        discover_extensions=False,
    )
    app = VulcanoApp(runtime)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause(delay=0.1)
        prompt = app.query_one("#prompt", VulcanoTextArea)
        command = "/ui-permission python -m pytest tests/vulcano"
        prompt.text = command
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.2)

        options = app.screen.query_one("#dialog-options", OptionList)
        assert options.option_count == 3
        options.highlighted = 1
        await pilot.press("enter")
        await pilot.pause(delay=0.2)

        resolved = [
            event
            for event in runtime.history
            if event.type == EventType.PERMISSION_RESOLVED
        ]
        assert resolved[-1].payload["decision"] == "allow_session"
        assert resolved[-1].payload["source"] == "user"
        assert any(
            message.kind == "permission"
            and "allowed for this session" in message.source_text
            for message in app.query(TranscriptMessage)
        )

        prompt = app.query_one("#prompt", VulcanoTextArea)
        prompt.text = command
        prompt.cursor_location = prompt.document.end
        await pilot.press("enter")
        await pilot.pause(delay=0.2)

        requests = [
            event
            for event in runtime.history
            if event.type == EventType.PERMISSION_REQUESTED
        ]
        resolved = [
            event
            for event in runtime.history
            if event.type == EventType.PERMISSION_RESOLVED
        ]
        assert requests[-1].payload["requires_confirmation"] is False
        assert resolved[-1].payload["source"] == "session"
        assert not app.screen.query("#dialog-options")
