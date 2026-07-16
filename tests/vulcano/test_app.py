import pytest

pytest.importorskip("textual")
pytest.importorskip("rich")

from textual.widgets import Input, Static

from msgflux.vulcano.app import TranscriptMessage, VulcanoApp
from msgflux.vulcano.runtime import VulcanoRuntime


@pytest.mark.asyncio
async def test_app_projects_streaming_runtime_events_headlessly():
    app = VulcanoApp(VulcanoRuntime(stream_delay=0, extensions_enabled=False))

    async with app.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        assert app.query_one("#topbar", Static).outer_size.height == 4
        status = app.query_one("#status", Static)
        assert "mock runtime" in str(status.render())

        prompt = app.query_one("#prompt", Input)
        prompt.value = "hello"
        await pilot.press("enter")
        await pilot.pause(delay=0.05)

        messages = list(app.query(TranscriptMessage))
        user = next(message for message in messages if message.kind == "user")
        assistant = next(message for message in messages if message.kind == "assistant")
        assert user.source_text == "hello"
        assert assistant.source_text == "Mock runtime received: hello"


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
