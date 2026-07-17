import inspect
from textwrap import dedent

import pytest

from msgflux.vulcano import (
    DomainEvent,
    UiRenderContext,
    UiState,
    UiThemeResult,
    VulcanoRuntime,
    WorkingIndicatorOptions,
    custom_message_type,
)


class _FakeUiDriver:
    def __init__(self):
        self.states = []
        self.notifications = []
        self.editor_text = ""
        self.theme_name = "fake-dark"

    def apply_state(self, state):
        self.states.append(state)

    async def select(self, title, options, dialog):
        del title, dialog
        return options[0]

    async def confirm(self, title, message, dialog):
        del title, message, dialog
        return True

    async def input(self, title, placeholder, dialog):
        del title, dialog
        return placeholder

    async def editor(self, title, prefill, dialog):
        del title, dialog
        return prefill

    def notify(self, message, severity):
        self.notifications.append((message, severity))

    def set_editor_text(self, text):
        self.editor_text = text

    def get_editor_text(self):
        return self.editor_text

    def paste_to_editor(self, text):
        self.editor_text += text

    async def custom(self, factory, *, overlay, overlay_options):
        del overlay, overlay_options
        result = None

        def done(value):
            nonlocal result
            result = value

        component = factory(self, self.theme, done)
        if inspect.isawaitable(component):
            await component
        return result

    @property
    def theme(self):
        return self.theme_name

    def get_themes(self):
        return ("fake-dark", "fake-light")

    def set_theme(self, theme):
        self.theme_name = theme
        return UiThemeResult(True)

    def render_context(self):
        return UiRenderContext(
            host=self,
            theme=self.theme,
            invalidate=lambda: None,
        )


def _write_extension(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(source), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_ui_facade_binds_driver_and_tracks_owned_state():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)
    ui = runtime.extensions.api.ui

    assert ui.mode == "headless"
    assert not ui.available
    assert await ui.select("Unavailable", ["fallback"]) is None
    assert not await ui.confirm("Unavailable", "No frontend")

    driver = _FakeUiDriver()
    runtime.ui.bind(driver, mode="tui")

    ui.set_status("task", "Planning")
    ui.set_working_message("Thinking deeply")
    ui.set_working_visible(False)
    ui.set_working_indicator(WorkingIndicatorOptions(frames=(".", "o"), interval=0.2))
    ui.set_widget("goal", ["Goal", "Ship it"], placement="below_editor")
    ui.set_header("Custom header")
    ui.set_footer("Custom footer")
    ui.set_title("Vulcano test")

    state = runtime.ui.state
    assert state.statuses[0].text == "Planning"
    assert state.widgets[0].placement == "below_editor"
    assert state.widgets[0].content == ("Goal", "Ship it")
    assert state.header == "Custom header"
    assert state.footer == "Custom footer"
    assert state.title == "Vulcano test"
    assert state.working_message == "Thinking deeply"
    assert not state.working_visible
    assert state.working_indicator.frames == (".", "o")

    assert await ui.select("Pick", ["first", "second"]) == "first"
    assert await ui.confirm("Proceed", "Continue")
    assert await ui.input("Name", "Ada") == "Ada"
    assert await ui.editor("Plan", "One") == "One"
    ui.notify("Done")
    assert driver.notifications == [("Done", "info")]

    ui.set_editor_text("hello")
    ui.paste_to_editor(" world")
    assert ui.get_editor_text() == "hello world"
    assert ui.get_themes() == ("fake-dark", "fake-light")
    assert ui.set_theme("fake-light").success
    assert ui.theme == "fake-light"

    completion = ui.add_autocomplete_provider(
        lambda value: f"{value}-completed" if value == "goal" else None
    )
    assert await runtime.ui.complete("goal") == "goal-completed"
    completion.remove()
    assert await runtime.ui.complete("goal") is None

    shortcut_calls = []
    shortcut = runtime.extensions.api.register_shortcut(
        "ctrl+g",
        lambda context: shortcut_calls.append((context.mode, context.has_ui)),
    )
    assert await runtime.ui.invoke_shortcut("ctrl+g")
    assert shortcut_calls == [("tui", True)]
    shortcut.remove()
    assert not await runtime.ui.invoke_shortcut("ctrl+g")

    result = await ui.custom(
        lambda host, theme, done: done(f"{host.theme}:{theme}"),
        overlay=True,
    )
    assert result == "fake-light:fake-light"
    assert driver.states[-1] == state


@pytest.mark.asyncio
async def test_failed_extension_setup_rolls_back_ui_contributions(tmp_path):
    extension = _write_extension(
        tmp_path / "broken_ui.py",
        """
        EXTENSION_NAME = "broken-ui"

        def setup(api):
            api.ui.set_status("broken", "must disappear")
            api.ui.set_widget("broken", ["must disappear"])
            api.register_message_renderer(
                "broken",
                lambda event, context: event.payload["content"],
            )
            raise RuntimeError("UI setup exploded")
        """,
    )
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[extension],
        discover_extensions=False,
    )

    await runtime.start()

    assert runtime.ui.state == UiState()
    event = DomainEvent(
        type=custom_message_type("broken"),
        sequence=1,
        payload={"content": "hidden"},
    )
    assert await runtime.ui.render_event(event) is None
    assert runtime.extensions.records[0].state == "failed"


@pytest.mark.asyncio
async def test_renderer_registration_is_owned_and_removable():
    runtime = VulcanoRuntime(stream_delay=0, extensions_enabled=False)
    driver = _FakeUiDriver()
    runtime.ui.bind(driver)
    registration = runtime.extensions.api.register_message_renderer(
        "card",
        lambda event, context: f"{context.theme}:{event.payload['content']}",
    )
    event = DomainEvent(
        type=custom_message_type("card"),
        sequence=1,
        payload={"content": "ready"},
    )

    assert await runtime.ui.render_event(event) == "fake-dark:ready"

    registration.remove()

    assert await runtime.ui.render_event(event) is None


@pytest.mark.asyncio
async def test_reload_replaces_ui_generation_without_stale_contributions(tmp_path):
    extension = tmp_path / "versioned_ui.py"

    def write_version(version):
        _write_extension(
            extension,
            f"""
            EXTENSION_NAME = "versioned-ui"

            def setup(api):
                api.ui.set_status("version", "{version}")
                api.ui.set_widget("version", ["{version}"])
                api.register_message_renderer(
                    "version",
                    lambda event, context: "{version}",
                )
            """,
        )

    write_version("one")
    runtime = VulcanoRuntime(
        stream_delay=0,
        extension_paths=[extension],
        discover_extensions=False,
    )
    driver = _FakeUiDriver()
    runtime.ui.bind(driver)
    await runtime.start()
    event = DomainEvent(
        type=custom_message_type("version"),
        sequence=1,
    )

    assert runtime.ui.state.statuses[0].text == "one"
    assert await runtime.ui.render_event(event) == "one"

    write_version("two")
    await runtime.extensions.reload()

    assert [status.text for status in runtime.ui.state.statuses] == ["two"]
    assert [widget.content for widget in runtime.ui.state.widgets] == [("two",)]
    assert await runtime.ui.render_event(event) == "two"
