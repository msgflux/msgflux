from __future__ import annotations

import inspect
from collections.abc import Mapping as MappingCollection
from collections.abc import Sequence as SequenceCollection
from time import monotonic
from typing import Callable, Mapping, Sequence, TypeVar

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.suggester import Suggester
from textual.widget import Widget
from textual.widgets import Button, Footer, Input, Label, OptionList, Static, TextArea

from msgflux.vulcano.commands import CommandRegistry
from msgflux.vulcano.ui import (
    UiCustomFactory,
    UiDialogOptions,
    UiManager,
    UiRenderContext,
    UiSeverity,
    UiState,
    UiThemeResult,
    UiWidget,
)

__all__ = ["TextualUiDriver", "WorkingStatus"]


T = TypeVar("T")


class _VulcanoSuggester(Suggester):
    def __init__(self, ui: UiManager, commands: CommandRegistry) -> None:
        super().__init__(use_cache=False, case_sensitive=True)
        self._ui = ui
        self._commands = commands

    async def get_suggestion(self, value: str) -> str | None:
        if value.startswith("/") and " " not in value[1:]:
            return None
        custom = await self._ui.complete(value)
        if custom is not None:
            return custom if custom.startswith(value) else None
        return await self._get_command_suggestion(value)

    async def _get_command_suggestion(self, value: str) -> str | None:
        if not value.startswith("/"):
            return None

        command_text = value[1:]
        if " " not in command_text:
            return None

        name, arguments = command_text.split(" ", 1)
        try:
            command = self._commands.resolve(name)
        except LookupError:
            return None
        provider = command.get_argument_completions
        if provider is None:
            return None
        result = provider(arguments)
        if inspect.isawaitable(result):
            result = await result
        fragment = arguments.rsplit(" ", 1)[-1]
        prefix = arguments[: len(arguments) - len(fragment)]
        for completion in _completion_values(result):
            if completion.startswith(fragment):
                return f"/{name} {prefix}{completion}"
        return None


def _completion_values(result: object) -> tuple[str, ...]:
    if isinstance(result, str):
        return (result,)
    if not isinstance(result, SequenceCollection) or isinstance(result, bytes):
        return ()
    values: list[str] = []
    for item in result:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, MappingCollection):
            value = item.get("value")
            if isinstance(value, str):
                values.append(value)
    return tuple(values)


class _SlashCommandMenu(OptionList):
    """Runtime-backed slash-command selector displayed above the editor."""

    def __init__(self, commands: CommandRegistry) -> None:
        super().__init__(id="command-menu", markup=False)
        self._commands = commands
        self._visible_commands: tuple[str, ...] = ()
        self.display = False

    def update_for(self, value: str) -> None:
        if not value.startswith("/") or any(character.isspace() for character in value):
            self.dismiss_menu()
            return

        prefix = value[1:].casefold()
        commands = tuple(
            sorted(
                (
                    command
                    for command in self._commands
                    if command.name.casefold().startswith(prefix)
                    or any(
                        alias.casefold().startswith(prefix) for alias in command.aliases
                    )
                ),
                key=lambda command: command.name,
            )
        )
        if not commands:
            self.dismiss_menu()
            return

        self._visible_commands = tuple(command.name for command in commands)
        self.clear_options()
        self.add_options(
            f"/{command.name}  {command.description}" for command in commands
        )
        self.highlighted = 0
        self.display = True

    @property
    def is_open(self) -> bool:
        return bool(self.display and self._visible_commands)

    def command_at(self, index: int | None) -> str | None:
        if index is None or not 0 <= index < len(self._visible_commands):
            return None
        return self._visible_commands[index]

    def dismiss_menu(self) -> None:
        self.display = False
        self._visible_commands = ()
        self.clear_options()


class _TimedModalScreen(ModalScreen[T]):
    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    DEFAULT_CSS = """
    _TimedModalScreen {
        align: center middle;
        background: $background 70%;
    }

    _TimedModalScreen > #dialog {
        width: 64;
        max-width: 90%;
        height: auto;
        max-height: 85%;
        padding: 1 2;
        border: round $accent;
        background: $surface;
    }

    _TimedModalScreen .dialog-title {
        height: auto;
        margin-bottom: 1;
        text-style: bold;
    }

    _TimedModalScreen .dialog-message {
        height: auto;
        margin-bottom: 1;
    }

    _TimedModalScreen .dialog-actions {
        height: auto;
        align-horizontal: right;
        margin-top: 1;
    }

    _TimedModalScreen .dialog-actions Button {
        margin-left: 1;
    }

    _TimedModalScreen #dialog-options {
        height: auto;
        max-height: 16;
    }

    _TimedModalScreen #dialog-editor {
        height: 16;
    }
    """

    def __init__(
        self,
        *,
        timeout: float | None,
        cancel_result: T,
    ) -> None:
        super().__init__()
        self._timeout = timeout
        self._cancel_result = cancel_result

    def on_mount(self) -> None:
        if self._timeout is not None:
            self.set_timer(self._timeout, self._timeout_elapsed)

    def action_cancel(self) -> None:
        self.dismiss(self._cancel_result)

    def _timeout_elapsed(self) -> None:
        self.dismiss(self._cancel_result)


class _SelectScreen(_TimedModalScreen[str | None]):
    def __init__(
        self,
        title: str,
        options: Sequence[str],
        *,
        timeout: float | None,
    ) -> None:
        super().__init__(timeout=timeout, cancel_result=None)
        self._title = title
        self._options = tuple(options)

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._title, classes="dialog-title")
            yield OptionList(*self._options, id="dialog-options", markup=False)

    @on(OptionList.OptionSelected)
    def _select_option(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self._options[event.option_index])


class _ConfirmScreen(_TimedModalScreen[bool]):
    def __init__(
        self,
        title: str,
        message: str,
        *,
        timeout: float | None,
    ) -> None:
        super().__init__(timeout=timeout, cancel_result=False)
        self._title = title
        self._message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._title, classes="dialog-title")
            yield Label(self._message, classes="dialog-message")
            with Horizontal(classes="dialog-actions"):
                yield Button("No", id="dialog-no")
                yield Button("Yes", variant="primary", id="dialog-yes")

    @on(Button.Pressed)
    def _press_button(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "dialog-yes")


class _InputScreen(_TimedModalScreen[str | None]):
    def __init__(
        self,
        title: str,
        placeholder: str,
        *,
        timeout: float | None,
    ) -> None:
        super().__init__(timeout=timeout, cancel_result=None)
        self._title = title
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._title, classes="dialog-title")
            yield Input(placeholder=self._placeholder, id="dialog-input")

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#dialog-input", Input).focus()

    @on(Input.Submitted, "#dialog-input")
    def _submit_input(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)


class _EditorScreen(_TimedModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", priority=True),
        Binding("ctrl+s", "submit", "Save", priority=True),
    ]

    def __init__(
        self,
        title: str,
        prefill: str,
        *,
        timeout: float | None,
    ) -> None:
        super().__init__(timeout=timeout, cancel_result=None)
        self._title = title
        self._prefill = prefill

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._title, classes="dialog-title")
            yield TextArea(self._prefill, id="dialog-editor")
            with Horizontal(classes="dialog-actions"):
                yield Button("Cancel", id="dialog-cancel")
                yield Button("Save", variant="primary", id="dialog-save")

    def on_mount(self) -> None:
        super().on_mount()
        self.query_one("#dialog-editor", TextArea).focus()

    def action_submit(self) -> None:
        self.dismiss(self.query_one("#dialog-editor", TextArea).text)

    @on(Button.Pressed)
    def _press_button(self, event: Button.Pressed) -> None:
        if event.button.id == "dialog-save":
            self.action_submit()
        else:
            self.action_cancel()


class _CustomScreen(ModalScreen[T | None]):
    BINDINGS = [Binding("escape", "cancel", "Cancel", priority=True)]

    DEFAULT_CSS = """
    _CustomScreen {
        align: center middle;
        background: $background 55%;
    }

    _CustomScreen > #custom-component {
        width: auto;
        max-width: 95%;
        height: auto;
        max-height: 90%;
    }

    _CustomScreen.embedded {
        background: $background 85%;
    }

    _CustomScreen.embedded > #custom-component {
        width: 95%;
    }
    """

    def __init__(
        self,
        component: Widget,
        *,
        overlay: bool,
        overlay_options: Mapping[str, object] | None,
    ) -> None:
        super().__init__(classes=None if overlay else "embedded")
        self._component = component
        self._overlay_options = dict(overlay_options or {})

    def compose(self) -> ComposeResult:
        yield Container(self._component, id="custom-component")

    def on_mount(self) -> None:
        container = self.query_one("#custom-component", Container)
        for name in ("width", "height", "max_width", "max_height"):
            if name in self._overlay_options:
                setattr(container.styles, name, self._overlay_options[name])

    def action_cancel(self) -> None:
        self.dismiss(None)


class WorkingStatus(Static):
    DEFAULT_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self) -> None:
        super().__init__("", id="working")
        self._message = "working"
        self._visible = True
        self._streaming = False
        self._frames = self.DEFAULT_FRAMES
        self._interval = 0.1
        self._frame_index = 0
        self._last_frame = monotonic()

    def on_mount(self) -> None:
        self.set_interval(0.05, self._tick)
        self._refresh_content()

    def configure(
        self,
        *,
        message: str | None,
        visible: bool,
        frames: tuple[str, ...] | None,
        interval: float,
    ) -> None:
        self._message = message or "working"
        self._visible = visible
        self._frames = self.DEFAULT_FRAMES if frames is None else frames
        self._interval = interval
        self._frame_index = 0
        self._refresh_content()

    def set_streaming(self, *, streaming: bool) -> None:
        self._streaming = streaming
        self._refresh_content()

    def _tick(self) -> None:
        now = monotonic()
        if now - self._last_frame < self._interval:
            return
        self._last_frame = now
        if self._frames:
            self._frame_index = (self._frame_index + 1) % len(self._frames)
        self._refresh_content()

    def _refresh_content(self) -> None:
        self.display = self._visible and self._streaming
        indicator = self._frames[self._frame_index] if self._frames else ""
        self.update(f"{indicator} {self._message}".strip())


class TextualUiDriver:
    """Materializes runtime-owned UI contributions in one Textual App."""

    def __init__(
        self,
        app: App[object],
        ui: UiManager,
        commands: CommandRegistry,
    ) -> None:
        self.app = app
        self._commands = commands
        self._suggester = _VulcanoSuggester(ui, commands)
        self._command_menu: _SlashCommandMenu | None = None
        self._state = UiState()
        self._applied_state = UiState()
        self._apply_scheduled = False
        self._base_status = "starting runtime..."
        self._streaming = False

    def create_default_header(self) -> Widget:
        return Static("VULCANO  /  MSGFLUX", id="topbar")

    def create_default_footer(self) -> Widget:
        return Footer()

    def create_working_status(self) -> WorkingStatus:
        return WorkingStatus()

    def create_command_menu(self) -> OptionList:
        self._command_menu = _SlashCommandMenu(self._commands)
        return self._command_menu

    def create_default_editor(self) -> Input:
        return Input(
            placeholder="Message Vulcano or enter /help",
            id="prompt",
            suggester=self._suggester,
        )

    @property
    def command_menu_visible(self) -> bool:
        return self._command_menu is not None and self._command_menu.is_open

    def update_command_menu(self, value: str) -> None:
        if self._command_menu is not None:
            self._command_menu.update_for(value)

    def dismiss_command_menu(self) -> None:
        if self._command_menu is not None:
            self._command_menu.dismiss_menu()

    def move_command_selection(self, direction: int) -> None:
        if self._command_menu is None:
            return
        if direction > 0:
            self._command_menu.action_cursor_down()
        else:
            self._command_menu.action_cursor_up()

    def accept_command_selection(self, index: int | None = None) -> bool:
        if self._command_menu is None:
            return False
        selected_index = self._command_menu.highlighted if index is None else index
        command = self._command_menu.command_at(selected_index)
        if command is None:
            return False
        editor = self.app.query_one("#prompt", Input)
        editor.value = f"/{command} "
        editor.cursor_position = len(editor.value)
        self._command_menu.dismiss_menu()
        editor.focus()
        return True

    def is_exact_command(self, value: str) -> bool:
        return (
            value.startswith("/")
            and not any(character.isspace() for character in value)
            and value[1:] in self._commands
        )

    def apply_state(self, state: UiState) -> None:
        self._state = state
        if self._apply_scheduled or not self.app.is_running:
            return
        self._apply_scheduled = True
        self.app.call_later(self._apply_pending_state)

    async def _apply_pending_state(self) -> None:
        self._apply_scheduled = False
        state = self._state
        previous = self._applied_state
        self._applied_state = state

        self.app.title = state.title or "Vulcano"
        self._refresh_status()
        working = self.app.query_one("#working", WorkingStatus)
        working.configure(
            message=state.working_message,
            visible=state.working_visible,
            frames=state.working_indicator.frames,
            interval=state.working_indicator.interval,
        )
        working.set_streaming(streaming=self._streaming)

        if state.widgets != previous.widgets:
            await self._sync_widgets(state.widgets)
        if state.header is not previous.header:
            await self._sync_slot(
                "#header-slot",
                state.header,
                self.create_default_header,
            )
        if state.footer is not previous.footer:
            await self._sync_slot(
                "#footer-slot",
                state.footer,
                self.create_default_footer,
            )
        if state.editor_component is not previous.editor_component:
            await self._sync_editor(state.editor_component)

        if self._state != state:
            self.apply_state(self._state)

    def set_base_status(self, text: str) -> None:
        self._base_status = text
        if self.app.is_running:
            self._refresh_status()

    def set_streaming(self, *, streaming: bool) -> None:
        self._streaming = streaming
        if self.app.is_running:
            self.app.query_one("#working", WorkingStatus).set_streaming(
                streaming=streaming
            )

    def _refresh_status(self) -> None:
        statuses = [status.text for status in self._state.statuses]
        text = "  •  ".join((self._base_status, *statuses))
        self.app.query_one("#status", Static).update(text)

    async def _sync_widgets(self, widgets: tuple[UiWidget, ...]) -> None:
        above = self.app.query_one("#widgets-above", Container)
        below = self.app.query_one("#widgets-below", Container)
        await above.remove_children()
        await below.remove_children()
        for contribution in widgets:
            target = below if contribution.placement == "below_editor" else above
            try:
                component = await self.materialize(contribution.content)
            except Exception as error:
                self.notify(
                    f"Widget {contribution.key!r} failed: {error}",
                    "error",
                )
                component = Static(
                    f"Widget {contribution.key!r} failed: {error}",
                    classes="error-message",
                )
            component.add_class("extension-widget")
            await target.mount(component)

    async def _sync_slot(
        self,
        selector: str,
        content: object | None,
        default_factory: Callable[[], Widget],
    ) -> None:
        container = self.app.query_one(selector, Container)
        await container.remove_children()
        if content is None:
            component = default_factory()
        else:
            try:
                component = await self.materialize(content)
            except Exception as error:
                self.notify(str(error), "error")
                component = default_factory()
        await container.mount(component)

    async def _sync_editor(self, content: object | None) -> None:
        container = self.app.query_one("#editor-slot", Container)
        current = self.app.query_one("#prompt", Input)
        value = current.value
        await container.remove_children()
        if content is None:
            editor = self.create_default_editor()
        else:
            try:
                component = await self.materialize(content)
                if not isinstance(component, Input):
                    raise TypeError("Custom editor factories must return textual Input")
                editor = component
            except Exception as error:
                self.notify(str(error), "error")
                editor = self.create_default_editor()
        editor.id = "prompt"
        editor.value = value
        if editor.suggester is None:
            editor.suggester = self._suggester
        await container.mount(editor)
        self.update_command_menu(editor.value)
        editor.focus()

    async def materialize(self, content: object) -> Widget:
        value = content
        if callable(value):
            value = value(self.app, self.theme)
        if inspect.isawaitable(value):
            value = await value
        if isinstance(value, Widget):
            return value
        if isinstance(value, (tuple, list)) and all(
            isinstance(item, str) for item in value
        ):
            value = "\n".join(value)
        return Static(value)

    async def select(
        self,
        title: str,
        options: Sequence[str],
        dialog: UiDialogOptions,
    ) -> str | None:
        if not options:
            return None
        return await self.app.push_screen_wait(
            _SelectScreen(title, options, timeout=dialog.timeout)
        )

    async def confirm(
        self,
        title: str,
        message: str,
        dialog: UiDialogOptions,
    ) -> bool:
        return await self.app.push_screen_wait(
            _ConfirmScreen(title, message, timeout=dialog.timeout)
        )

    async def input(
        self,
        title: str,
        placeholder: str,
        dialog: UiDialogOptions,
    ) -> str | None:
        return await self.app.push_screen_wait(
            _InputScreen(title, placeholder, timeout=dialog.timeout)
        )

    async def editor(
        self,
        title: str,
        prefill: str,
        dialog: UiDialogOptions,
    ) -> str | None:
        return await self.app.push_screen_wait(
            _EditorScreen(title, prefill, timeout=dialog.timeout)
        )

    def notify(
        self,
        message: str,
        severity: UiSeverity,
    ) -> None:
        resolved = "information" if severity == "info" else severity
        self.app.notify(message, severity=resolved)

    def set_editor_text(self, text: str) -> None:
        self.app.query_one("#prompt", Input).value = text

    def get_editor_text(self) -> str:
        return self.app.query_one("#prompt", Input).value

    def paste_to_editor(self, text: str) -> None:
        self.app.query_one("#prompt", Input).insert_text_at_cursor(text)

    async def custom(
        self,
        factory: UiCustomFactory[T],
        *,
        overlay: bool,
        overlay_options: Mapping[str, object] | None,
    ) -> T | None:
        screen: _CustomScreen[T] | None = None
        completed = False
        early_result: T | None = None

        def done(result: T) -> None:
            nonlocal completed, early_result
            if completed:
                return
            completed = True
            if screen is None:
                early_result = result
                return
            self.app.call_later(screen.dismiss, result)

        component = factory(self.app, self.theme, done)
        if inspect.isawaitable(component):
            component = await component
        if completed:
            return early_result
        widget = await self.materialize(component)
        screen = _CustomScreen(
            widget,
            overlay=overlay,
            overlay_options=overlay_options,
        )
        result = await self.app.push_screen_wait(screen)
        completed = True
        return result

    @property
    def theme(self) -> object:
        return self.app.get_theme(self.app.theme) or self.app.theme

    def get_themes(self) -> tuple[str, ...]:
        return tuple(sorted(self.app.available_themes))

    def set_theme(self, theme: str) -> UiThemeResult:
        if self.app.get_theme(theme) is None:
            return UiThemeResult(False, f"Unknown Textual theme: {theme}")
        self.app.theme = theme
        return UiThemeResult(True)

    def render_context(self) -> UiRenderContext:
        return UiRenderContext(
            host=self.app,
            theme=self.theme,
            invalidate=lambda: self.app.call_later(self.app.refresh, layout=True),
        )
