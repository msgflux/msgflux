from pathlib import Path

import pytest

pytest.importorskip("textual")
pytest.importorskip("rich")

from textual.containers import Container
from textual.widgets import Static

from msgflux.vulcano import SubmitInput, VulcanoRuntime, custom_message_type
from msgflux.vulcano.app import VulcanoApp
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
        "ui-notify",
        "ui-overlay",
        "ui-reset",
        "ui-slots",
        "ui-status",
        "ui-theme",
        "ui-title",
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
