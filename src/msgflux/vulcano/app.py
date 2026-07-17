from __future__ import annotations

from rich.markdown import Markdown
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.widget import Widget
from textual.widgets import Input, OptionList, Static

from msgflux.vulcano.actions import SubmitInput
from msgflux.vulcano.events import DomainEvent, EventType
from msgflux.vulcano.runtime import RuntimeProtocol
from msgflux.vulcano.textual_ui import TextualUiDriver

__all__ = ["TranscriptMessage", "VulcanoApp"]


class TranscriptMessage(Static):
    """A transcript item that keeps its source text for incremental updates."""

    def __init__(
        self,
        content: str,
        *,
        kind: str,
        markdown: bool = False,
        classes: str | None = None,
        message_id: str | None = None,
    ) -> None:
        super().__init__(classes=classes, id=message_id)
        self.source_text = ""
        self.kind = kind
        self.markdown = markdown
        self.set_content(content)

    def set_content(self, content: str) -> None:
        self.source_text = content
        if self.markdown:
            self.update(Markdown(content or " "))
            return
        self.update(Text(content))


class VulcanoApp(App[None]):
    """Thin Textual projection of a runtime action/event stream."""

    TITLE = "Vulcano"
    SUB_TITLE = "msgflux code agent"

    CSS = """
    Screen {
        background: #0c0e12;
        color: #d7dae0;
        layout: vertical;
    }

    #header-slot, #footer-slot, #widgets-above, #widgets-below, #editor-slot {
        width: 100%;
        height: auto;
    }

    #topbar {
        height: 4;
        padding: 1 2;
        background: #171014;
        color: #ff3344;
        text-style: bold;
        border-bottom: solid #5e202a;
    }

    #transcript {
        height: 1fr;
        padding: 1 2;
        scrollbar-color: #3c4352;
        scrollbar-color-hover: #5b6578;
        scrollbar-size: 1 1;
    }

    TranscriptMessage {
        width: 100%;
        height: auto;
        margin-bottom: 1;
        padding: 1 2;
    }

    .welcome-message {
        color: #9da5b4;
        border: round #272c36;
        background: #10131a;
    }

    .user-message {
        margin-left: 8;
        color: #e4e5e9;
        background: #292c33;
    }

    .assistant-message {
        margin-right: 8;
        background: #12151c;
    }

    .command-input {
        color: #c8cad1;
        background: #22252b;
        padding-top: 0;
        padding-bottom: 0;
        border-left: thick #ff3344;
    }

    .command-output {
        background: #10131a;
        border-left: thick #68b684;
    }

    .error-message {
        color: #ff9b9b;
        background: #241416;
        border-left: thick #d96c75;
    }

    #status {
        height: 1;
        padding: 0 2;
        color: #778091;
        background: #10131a;
    }

    #working {
        height: 1;
        padding: 0 2;
        color: #ff3344;
        background: #10131a;
    }

    .extension-widget {
        width: 100%;
        height: auto;
        padding: 0 2;
    }

    #command-menu {
        display: none;
        width: auto;
        height: auto;
        max-height: 12;
        margin: 0 2;
        padding: 0 1;
        color: #d5d7dc;
        background: #1b1d23;
        border: round #ff3344;
        scrollbar-color: #ff3344;
        scrollbar-size: 1 1;
    }

    #command-menu > .option-list--option-highlighted {
        color: #ffffff;
        background: #54202a;
        text-style: bold;
    }

    #prompt {
        height: 3;
        margin: 0 2;
        padding: 0 1;
        color: #e4e5e9;
        border: round #60646e;
        background: #292c33;
    }

    #prompt:focus {
        border: round #ff3344;
    }

    Footer {
        height: 1;
        background: #12151c;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "request_clear", "Clear"),
        Binding("ctrl+c", "request_quit", "Quit", priority=True),
    ]

    def __init__(self, runtime: RuntimeProtocol) -> None:
        super().__init__()
        self.runtime = runtime
        self._ui_driver = TextualUiDriver(self, runtime.ui, runtime.commands)
        self._assistant_views: dict[str, TranscriptMessage] = {}
        self._assistant_content: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        with Container(id="header-slot"):
            yield self._ui_driver.create_default_header()
        with VerticalScroll(id="transcript"):
            yield TranscriptMessage(
                (
                    "**Vulcano runtime preview**\n\n"
                    "The runtime is currently mocked. Use `/help` to inspect "
                    "runtime-owned commands."
                ),
                kind="welcome",
                markdown=True,
                classes="welcome-message",
            )
        yield Static("starting runtime...", id="status")
        yield self._ui_driver.create_working_status()
        yield Container(id="widgets-above")
        yield self._ui_driver.create_command_menu()
        with Container(id="editor-slot"):
            yield self._ui_driver.create_default_editor()
        yield Container(id="widgets-below")
        with Container(id="footer-slot"):
            yield self._ui_driver.create_default_footer()

    def on_mount(self) -> None:
        self.runtime.ui.bind(self._ui_driver, mode="tui")
        self.query_one("#prompt", Input).focus()
        self._consume_events()

    def on_unmount(self) -> None:
        self.runtime.ui.unbind(self._ui_driver)

    async def on_key(self, event: events.Key) -> None:
        if self._handle_command_menu_key(event):
            return
        try:
            handled = await self.runtime.ui.invoke_shortcut(event.key)
        except Exception as error:
            self.notify(f"Shortcut failed: {error}", severity="error")
            return
        if handled:
            event.prevent_default()
            event.stop()

    def _handle_command_menu_key(self, event: events.Key) -> bool:
        if not self._ui_driver.command_menu_visible:
            return False

        if event.key in {"down", "up"}:
            direction = 1 if event.key == "down" else -1
            self._ui_driver.move_command_selection(direction)
        elif event.key == "tab":
            self._ui_driver.accept_command_selection()
        elif event.key == "escape":
            self._ui_driver.dismiss_command_menu()
        elif event.key == "enter":
            editor = self.query_one("#prompt", Input)
            if self._ui_driver.is_exact_command(editor.value):
                self._ui_driver.dismiss_command_menu()
                return False
            self._ui_driver.accept_command_selection()
        else:
            return False

        event.prevent_default()
        event.stop()
        return True

    @on(Input.Changed, "#prompt")
    def _on_prompt_changed(self, event: Input.Changed) -> None:
        self._ui_driver.update_command_menu(event.value)

    @on(OptionList.OptionSelected, "#command-menu")
    def _on_command_selected(self, event: OptionList.OptionSelected) -> None:
        self._ui_driver.accept_command_selection(event.option_index)

    @on(Input.Submitted, "#prompt")
    def _on_prompt_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        self._ui_driver.dismiss_command_menu()
        if text:
            self._dispatch_text(text)

    @work(group="runtime-dispatch")
    async def _dispatch_text(self, text: str) -> None:
        await self.runtime.dispatch(SubmitInput(text))

    @work(exclusive=True, group="runtime-events")
    async def _consume_events(self) -> None:
        subscription = self.runtime.subscribe()
        try:
            await self.runtime.start()
            async for event in subscription:
                await self._project_event(event)
        finally:
            await subscription.aclose()

    async def _project_event(self, event: DomainEvent) -> None:
        if self._project_lifecycle_event(event):
            return

        if event.type == EventType.ASSISTANT_STARTED:
            self._ui_driver.set_streaming(streaming=True)
        elif event.type == EventType.ASSISTANT_COMPLETED:
            self._ui_driver.set_streaming(streaming=False)

        if await self._project_registered_renderer(event):
            return
        if await self._project_message_event(event):
            return
        await self._project_command_event(event)

    def _project_lifecycle_event(self, event: DomainEvent) -> bool:
        if event.type == EventType.RUNTIME_STARTED:
            command_count = event.payload.get("commands", 0)
            runtime_kind = event.payload.get("runtime", "runtime")
            self._ui_driver.set_base_status(
                f"{runtime_kind} runtime  •  {command_count} commands  •  /help"
            )
            return True
        if event.type == EventType.RUNTIME_STOPPED:
            self._ui_driver.set_streaming(streaming=False)
            self._ui_driver.set_base_status("runtime stopped")
            self.exit()
            return True
        return False

    async def _project_registered_renderer(self, event: DomainEvent) -> bool:
        try:
            rendered = await self.runtime.ui.render_event(event)
            if rendered is None:
                return False
            await self._append_rendered(rendered)
        except Exception as error:
            await self._append_message(
                f"UI renderer failed: {error}",
                kind="error",
                classes="error-message",
            )
            return True
        return True

    async def _project_message_event(self, event: DomainEvent) -> bool:
        if event.type == EventType.MESSAGE_USER:
            await self._append_message(
                str(event.payload.get("content", "")),
                kind="user",
                classes="user-message",
            )
            return True

        if event.type in {
            EventType.ASSISTANT_STARTED,
            EventType.ASSISTANT_DELTA,
            EventType.ASSISTANT_COMPLETED,
        }:
            await self._project_assistant_event(event)
            return True
        return False

    async def _project_command_event(self, event: DomainEvent) -> None:
        if event.type == EventType.COMMAND_STARTED:
            await self._append_message(
                str(event.payload.get("raw", "")),
                kind="command",
                classes="command-input",
            )
            return

        if event.type == EventType.COMMAND_OUTPUT:
            await self._append_message(
                str(event.payload.get("text", "")),
                kind="command-output",
                markdown=True,
                classes="command-output",
            )
            return

        if event.type in {
            EventType.COMMAND_ERROR,
            EventType.EXTENSION_FAILED,
            EventType.RUNTIME_ERROR,
        }:
            await self._append_message(
                str(
                    event.payload.get("message")
                    or event.payload.get("error")
                    or "Unknown runtime error"
                ),
                kind="error",
                classes="error-message",
            )
            return

        if event.type == EventType.CLIENT_ACTION:
            if event.payload.get("action") == "transcript.clear":
                await self.query_one("#transcript", VerticalScroll).remove_children()
                self._assistant_views.clear()
                self._assistant_content.clear()
            return

    async def _project_assistant_event(self, event: DomainEvent) -> None:
        key, view = await self._ensure_assistant(event)
        if event.type == EventType.ASSISTANT_STARTED:
            return
        if event.type == EventType.ASSISTANT_DELTA:
            content = self._assistant_content.get(key, "") + str(
                event.payload.get("delta", "")
            )
            self._assistant_content[key] = content
            view.set_content(content)
            self._scroll_to_end()
            return

        content = str(
            event.payload.get("content", self._assistant_content.get(key, ""))
        )
        self._assistant_content[key] = content
        view.set_content(content)

    async def _ensure_assistant(
        self,
        event: DomainEvent,
    ) -> tuple[str, TranscriptMessage]:
        key = event.correlation_id or f"event-{event.sequence}"
        view = self._assistant_views.get(key)
        if view is None:
            view = await self._append_message(
                "",
                kind="assistant",
                markdown=True,
                classes="assistant-message",
            )
            self._assistant_views[key] = view
            self._assistant_content[key] = ""
        return key, view

    async def _append_message(
        self,
        content: str,
        *,
        kind: str,
        markdown: bool = False,
        classes: str,
    ) -> TranscriptMessage:
        view = TranscriptMessage(
            content,
            kind=kind,
            markdown=markdown,
            classes=classes,
        )
        await self.query_one("#transcript", VerticalScroll).mount(view)
        self._scroll_to_end()
        return view

    async def _append_rendered(self, content: object) -> Widget:
        view = await self._ui_driver.materialize(content)
        await self.query_one("#transcript", VerticalScroll).mount(view)
        self._scroll_to_end()
        return view

    def _scroll_to_end(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_end(
            animate=False,
            immediate=True,
        )

    def action_request_clear(self) -> None:
        self._dispatch_text("/clear")

    def action_request_quit(self) -> None:
        self._dispatch_text("/quit")
