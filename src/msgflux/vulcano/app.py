from __future__ import annotations

import json

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widget import Widget
from textual.widgets import Collapsible, Input, OptionList, Static

from msgflux.vulcano.actions import CancelExecution, InputMode, SubmitInput
from msgflux.vulcano.blocks import BlockKind, BlockStatus
from msgflux.vulcano.config import VulcanoSettings
from msgflux.vulcano.events import DomainEvent, EventType
from msgflux.vulcano.runtime import RuntimeProtocol
from msgflux.vulcano.textual_ui import TextualUiDriver, VulcanoTextArea
from msgflux.vulcano.ui import UiManager

__all__ = [
    "CollapsibleTranscriptBlock",
    "PendingInputList",
    "ToolExecutionBlock",
    "TranscriptMessage",
    "VulcanoApp",
]


class PendingInputList(Static):
    """Compact projection of runtime-owned steering and follow-up queues."""

    def __init__(self, *, widget_id: str | None = None) -> None:
        super().__init__("", id=widget_id, classes="pending-inputs-hidden")
        self.items: dict[str, tuple[str, str]] = {}

    def queued(self, event: DomainEvent) -> None:
        key = event.correlation_id or f"event-{event.sequence}"
        self.items[key] = (
            str(event.payload.get("mode", "steer")),
            str(event.payload.get("content", "")),
        )
        self._refresh_content()

    def dequeued(self, event: DomainEvent) -> None:
        if event.correlation_id is not None:
            self.items.pop(event.correlation_id, None)
        self._refresh_content()

    def clear(self) -> None:
        self.items.clear()
        self._refresh_content()

    def _refresh_content(self) -> None:
        if not self.items:
            self.add_class("pending-inputs-hidden")
            self.update("")
            return
        self.remove_class("pending-inputs-hidden")
        lines = ["Pending"]
        for mode, content in self.items.values():
            label = "follow-up" if mode == "follow_up" else "steer"
            summary = " ".join(content.split())
            if len(summary) > 96:
                summary = f"{summary[:93]}..."
            lines.append(f"  {label}  {summary}")
        self.update(Text("\n".join(lines)))


class TranscriptMessage(Static):
    """A transcript item that keeps its source text for incremental updates."""

    def __init__(
        self,
        content: str,
        *,
        kind: str,
        markdown: bool = False,
        render_mode: str | None = None,
        classes: str | None = None,
        message_id: str | None = None,
    ) -> None:
        super().__init__(classes=classes, id=message_id)
        self.source_text = ""
        self.kind = kind
        self.render_mode = render_mode or ("markdown" if markdown else "text")
        self.markdown = self.render_mode in {"markdown", "diff"}
        self._render_timer = None
        self._rendered_text = ""
        self.set_content(content)

    def set_content(self, content: str) -> None:
        if self._render_timer is not None:
            self._render_timer.stop()
            self._render_timer = None
        self.source_text = content
        self._refresh_renderable()

    def _refresh_renderable(self) -> None:
        content = self.source_text
        if self.render_mode == "diff":
            self.update(Markdown(f"```diff\n{content}\n```" if content else " "))
        elif self.markdown:
            self.update(Markdown(content or " "))
        else:
            self.update(Text(content))
        self._rendered_text = content

    def append_content(self, content: str) -> None:
        """Append a streamed delta and rebuild the accumulated projection."""
        if not content:
            return
        self.source_text += content
        if not self.is_mounted:
            self._refresh_renderable()
        elif self._render_timer is None:
            self._render_timer = self.set_timer(1 / 30, self.flush_content)

    def flush_content(self) -> None:
        """Render pending streamed content immediately."""
        self._render_timer = None
        if self.source_text != self._rendered_text:
            self._refresh_renderable()

    def on_unmount(self) -> None:
        if self._render_timer is not None:
            self._render_timer.stop()
            self._render_timer = None


class CollapsibleTranscriptBlock(Collapsible):
    """Collapsible shell around a streamed transcript block."""

    def __init__(
        self,
        content: str,
        *,
        title: str,
        kind: str,
        collapsed: bool = True,
        classes: str | None = None,
    ) -> None:
        self.message = TranscriptMessage(
            content,
            kind=kind,
            markdown=True,
            classes=classes,
        )
        super().__init__(
            self.message,
            title=title,
            collapsed=collapsed,
            classes="collapsible-transcript-block",
        )

    @property
    def source_text(self) -> str:
        return self.message.source_text

    def set_content(self, content: str) -> None:
        self.message.set_content(content)

    def append_content(self, content: str) -> None:
        self.message.append_content(content)

    def flush_content(self) -> None:
        self.message.flush_content()


class ToolExecutionBlock(Collapsible):
    """Incrementally updated tool call/result with extension renderers."""

    def __init__(
        self,
        ui: UiManager,
        driver: TextualUiDriver,
        event: DomainEvent,
    ) -> None:
        self._ui = ui
        self._driver = driver
        self._body = Container(classes="tool-execution-body")
        self._event = event
        self.tool_call_id = str(event.payload.get("tool_call_id", ""))
        self.tool_name = str(event.payload.get("name", "tool"))
        self.status = BlockStatus.PENDING
        super().__init__(
            self._body,
            title=f"○ {self.tool_name}",
            collapsed=False,
            classes="tool-execution",
        )

    async def apply_event(self, event: DomainEvent) -> None:
        self._event = event
        status = str(event.payload.get("status", BlockStatus.STREAMING))
        self.status = status
        marker = {
            BlockStatus.COMPLETED: "✓",
            BlockStatus.FAILED: "✗",
            BlockStatus.ABORTED: "!",
        }.get(status, "○")
        self.title = f"{marker} {self.tool_name}"
        self.set_classes("tool-execution")
        if status == BlockStatus.COMPLETED:
            self.add_class("tool-completed")
        elif status in {BlockStatus.FAILED, BlockStatus.ABORTED}:
            self.add_class("tool-failed")
        else:
            self.add_class("tool-pending")

        try:
            rendered = await self._ui.render_tool_event(
                event,
                expanded=not self.collapsed,
            )
        except Exception as error:
            self._driver.notify(
                f"Tool renderer for {self.tool_name!r} failed: {error}",
                "error",
            )
            rendered = None
        component = await self._driver.materialize(
            rendered if rendered is not None else self._default_renderable(event)
        )
        await self._body.remove_children()
        await self._body.mount(component)

    async def abort(self, *, sequence: int, correlation_id: str) -> None:
        payload = dict(self._event.payload)
        payload.update(
            {
                "status": BlockStatus.ABORTED,
                "result": "Cancelled",
                "is_error": True,
            }
        )
        await self.apply_event(
            DomainEvent(
                type=EventType.TOOL_COMPLETED,
                sequence=sequence,
                payload=payload,
                correlation_id=correlation_id,
            )
        )

    @staticmethod
    def _default_renderable(event: DomainEvent) -> object:
        if event.type == EventType.TOOL_STARTED:
            value = event.payload.get("arguments", {})
            label = "Arguments"
        elif event.type == EventType.TOOL_UPDATED:
            value = event.payload.get("update", "")
            label = "Update"
        else:
            value = event.payload.get("result", "")
            label = "Error" if event.payload.get("is_error") else "Result"

        if isinstance(value, str):
            body = value or " "
        else:
            body = f"```json\n{json.dumps(value, indent=2, default=str)}\n```"
        border_style = "red" if event.payload.get("is_error") else "bright_black"
        return Panel(Markdown(body), title=label, border_style=border_style)

    @on(Collapsible.Toggled)
    def _on_toggled(self, event: Collapsible.Toggled) -> None:
        if event.collapsible is self:
            self.run_worker(
                self.apply_event(self._event),
                group=f"tool-render-{self.tool_call_id}",
                exclusive=True,
            )


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
        color: #ff6a1a;
        text-style: bold;
        border-bottom: solid #6e3519;
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
        color: #e4e5e9;
        background: #292c33;
    }

    .assistant-message {
        margin-right: 8;
        background: #12151c;
    }

    .reasoning-message {
        margin-right: 12;
        color: #9da5b4;
        text-style: italic;
        background: #10131a;
        border-left: solid #3c4352;
    }

    .diff-message, .artifact-message {
        margin-right: 4;
        background: #10131a;
        border: round #303744;
    }

    CollapsibleTranscriptBlock, ToolExecutionBlock {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    .collapsible-transcript-block > CollapsibleTitle {
        color: #9da5b4;
        background: #10131a;
    }

    .tool-execution > CollapsibleTitle {
        color: #c8cad1;
        background: #151820;
        text-style: bold;
    }

    .tool-execution-body {
        height: auto;
        padding: 0 1;
        background: #10131a;
    }

    .tool-failed > CollapsibleTitle {
        color: #ff9b9b;
    }

    .command-input {
        color: #c8cad1;
        background: #22252b;
        padding-top: 0;
        padding-bottom: 0;
    }

    .command-output {
        background: #10131a;
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
        color: #ff6a1a;
        background: #10131a;
    }

    #pending-inputs {
        width: auto;
        height: auto;
        max-height: 6;
        margin: 0 2;
        padding: 0 1;
        color: #9da5b4;
        background: #171a21;
        border-left: solid #ff6a1a;
    }

    .pending-inputs-hidden {
        display: none;
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
        border: round #ff6a1a;
        scrollbar-color: #ff6a1a;
        scrollbar-size: 1 1;
    }

    #command-menu > .option-list--option-highlighted {
        color: #ffffff;
        background: #5a2a16;
        text-style: bold;
    }

    #prompt {
        height: 3;
        max-height: 15;
        margin: 0 2;
        padding: 0 1;
        color: #e4e5e9;
        border: round #60646e;
        background: #292c33;
    }

    #prompt:focus {
        border: round #ff6a1a;
    }

    VulcanoFooter {
        height: 1;
        padding: 0 2;
        color: #9da5b4;
        background: #12151c;
    }
    """

    BINDINGS = []

    def __init__(
        self,
        runtime: RuntimeProtocol,
        settings: VulcanoSettings | None = None,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.settings = settings or VulcanoSettings.defaults()
        self._ui_driver = TextualUiDriver(
            self,
            runtime.ui,
            runtime.commands,
            self.settings,
        )
        self._assistant_views: dict[str, TranscriptMessage] = {}
        self._block_views: dict[
            tuple[str, str],
            TranscriptMessage | CollapsibleTranscriptBlock,
        ] = {}
        self._tool_views: dict[tuple[str, str], ToolExecutionBlock] = {}
        self._active_assistant_streams: set[str] = set()
        self._active_block_streams: set[tuple[str, str]] = set()
        self._active_tool_streams: set[tuple[str, str]] = set()

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
        yield PendingInputList(widget_id="pending-inputs")
        yield self._ui_driver.create_command_menu()
        with Container(id="editor-slot"):
            yield self._ui_driver.create_default_editor()
        yield Container(id="widgets-below")
        with Container(id="footer-slot"):
            yield self._ui_driver.create_default_footer()

    def on_mount(self) -> None:
        self.runtime.ui.bind(self._ui_driver, mode="tui")
        self._ui_driver.focus_editor()
        self._consume_events()

    def on_unmount(self) -> None:
        self.runtime.ui.unbind(self._ui_driver)

    async def on_key(self, event: events.Key) -> None:
        if self._handle_command_menu_key(event):
            return
        keybindings = self.settings.keybindings
        if keybindings.matches("cancel", event.key):
            self.action_cancel_execution()
            event.prevent_default()
            event.stop()
            return
        if keybindings.matches("clear", event.key):
            self.action_request_clear()
            event.prevent_default()
            event.stop()
            return
        if keybindings.matches("quit", event.key):
            self.action_request_quit()
            event.prevent_default()
            event.stop()
            return
        if keybindings.matches("command_palette", event.key):
            self.action_command_palette()
            event.prevent_default()
            event.stop()
            return
        focused = self.focused
        if (
            isinstance(focused, Input)
            and focused.id == "prompt"
            and (
                keybindings.matches("submit", event.key)
                or keybindings.matches("follow_up", event.key)
            )
        ):
            mode: InputMode = (
                "follow_up" if keybindings.matches("follow_up", event.key) else "auto"
            )
            self._submit_prompt(focused.value, mode=mode)
            event.prevent_default()
            event.stop()
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
            if self._ui_driver.is_exact_command(self._ui_driver.get_editor_text()):
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

    @on(VulcanoTextArea.Changed, "#prompt")
    def _on_prompt_area_changed(self, event: VulcanoTextArea.Changed) -> None:
        self._ui_driver.update_command_menu(event.text_area.text)

    @on(OptionList.OptionSelected, "#command-menu")
    def _on_command_selected(self, event: OptionList.OptionSelected) -> None:
        self._ui_driver.accept_command_selection(event.option_index)

    @on(Input.Submitted, "#prompt")
    def _on_prompt_submitted(self, event: Input.Submitted) -> None:
        self._submit_prompt(event.value)

    @on(VulcanoTextArea.Submitted, "#prompt")
    def _on_prompt_area_submitted(self, event: VulcanoTextArea.Submitted) -> None:
        self._submit_prompt(event.value, mode=event.mode)

    def _submit_prompt(self, value: str, *, mode: InputMode = "auto") -> None:
        text = value.strip()
        if (
            self._ui_driver.command_menu_visible
            and not self._ui_driver.is_exact_command(value)
        ):
            self._ui_driver.accept_command_selection()
            return
        self._ui_driver.set_editor_text("")
        self._ui_driver.dismiss_command_menu()
        if text:
            self._dispatch_text(text, mode)

    @work(group="runtime-dispatch")
    async def _dispatch_text(self, text: str, mode: InputMode = "auto") -> None:
        await self.runtime.dispatch(SubmitInput(text, mode=mode))

    @work(exclusive=True, group="runtime-cancel")
    async def _dispatch_cancel(self) -> None:
        await self.runtime.dispatch(CancelExecution(reason="escape"))

    @work(exclusive=True, group="command-palette")
    async def _open_command_palette(self) -> None:
        await self._ui_driver.open_command_palette()

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
        scope = event.payload.get("scope")
        if isinstance(scope, dict):
            self._ui_driver.set_execution_scope(scope)
        if await self._project_execution_event(event):
            return

        stream_key = event.correlation_id or f"event-{event.sequence}"
        if event.type == EventType.ASSISTANT_STARTED:
            self._active_assistant_streams.add(stream_key)
            self._refresh_streaming_state()
        elif event.type == EventType.ASSISTANT_COMPLETED:
            self._active_assistant_streams.discard(stream_key)
            self._refresh_streaming_state()

        if await self._project_registered_renderer(event):
            return
        if await self._project_message_event(event):
            return
        await self._project_command_event(event)

    async def _project_execution_event(self, event: DomainEvent) -> bool:
        if event.type == EventType.SESSION_SWITCHED:
            await self._project_session_switch(event)
            return True
        if event.type == EventType.INPUT_QUEUED:
            pending = self.query_one("#pending-inputs", PendingInputList)
            pending.queued(event)
            self._ui_driver.set_queue_count(len(pending.items))
            return True
        if event.type == EventType.INPUT_DEQUEUED:
            pending = self.query_one("#pending-inputs", PendingInputList)
            pending.dequeued(event)
            self._ui_driver.set_queue_count(len(pending.items))
            return True
        if event.type == EventType.INPUT_QUEUE_CLEARED:
            self.query_one("#pending-inputs", PendingInputList).clear()
            self._ui_driver.set_queue_count(0)
            return True
        if event.type != EventType.EXECUTION_CANCELLED:
            return False

        await self._project_cancellation(event)
        return True

    async def _project_cancellation(self, event: DomainEvent) -> None:
        target = event.payload.get("target_correlation_id")
        if not isinstance(target, str):
            return
        self._active_assistant_streams.discard(target)
        active_blocks = [key for key in self._active_block_streams if key[0] == target]
        for key in active_blocks:
            view = self._block_views.get(key)
            if view is not None:
                view.flush_content()
                view.add_class("error-message")
            self._active_block_streams.discard(key)
        active_tools = [key for key in self._active_tool_streams if key[0] == target]
        for key in active_tools:
            view = self._tool_views.get(key)
            if view is not None:
                await view.abort(sequence=event.sequence, correlation_id=target)
            self._active_tool_streams.discard(key)
        self._refresh_streaming_state()

    async def _project_session_switch(self, event: DomainEvent) -> None:
        await self._clear_transcript()
        thread_id = event.payload.get("thread_id")
        if isinstance(thread_id, str):
            self._ui_driver.set_execution_scope(
                {"thread_id": thread_id, "run_id": None}
            )
        raw_events = event.payload.get("events", ())
        if not isinstance(raw_events, list):
            return
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            await self._project_event(DomainEvent.from_dict(raw_event))

    def _project_lifecycle_event(self, event: DomainEvent) -> bool:
        if event.type == EventType.RUNTIME_STARTED:
            command_count = event.payload.get("commands", 0)
            runtime_kind = event.payload.get("runtime", "runtime")
            self._ui_driver.set_runtime_metadata(
                str(runtime_kind),
                agent=(
                    str(event.payload["agent"])
                    if event.payload.get("agent") is not None
                    else None
                ),
                thread_id=(
                    str(event.payload["thread_id"])
                    if event.payload.get("thread_id") is not None
                    else None
                ),
            )
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
        if event.type in {
            EventType.BLOCK_STARTED,
            EventType.BLOCK_DELTA,
            EventType.BLOCK_COMPLETED,
        }:
            await self._project_block_event(event)
            return True
        if event.type in {
            EventType.TOOL_STARTED,
            EventType.TOOL_UPDATED,
            EventType.TOOL_COMPLETED,
        }:
            await self._project_tool_event(event)
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
                await self._clear_transcript()
            return

    async def _clear_transcript(self) -> None:
        await self.query_one("#transcript", VerticalScroll).remove_children()
        self._assistant_views.clear()
        self._block_views.clear()
        self._tool_views.clear()
        self.runtime.ui.clear_tool_render_state()
        self._active_assistant_streams.clear()
        self._active_block_streams.clear()
        self._active_tool_streams.clear()
        self.query_one("#pending-inputs", PendingInputList).clear()
        self._ui_driver.set_queue_count(0)
        self._refresh_streaming_state()

    async def _project_tool_event(self, event: DomainEvent) -> None:
        correlation_id = event.correlation_id or "uncorrelated"
        tool_call_id = str(event.payload.get("tool_call_id", f"event-{event.sequence}"))
        key = (correlation_id, tool_call_id)
        view = self._tool_views.get(key)
        if view is None:
            view = ToolExecutionBlock(self.runtime.ui, self._ui_driver, event)
            await self.query_one("#transcript", VerticalScroll).mount(view)
            self._tool_views[key] = view
        await view.apply_event(event)

        if event.type == EventType.TOOL_STARTED:
            self._active_tool_streams.add(key)
        elif event.type == EventType.TOOL_COMPLETED:
            self._active_tool_streams.discard(key)
        self._refresh_streaming_state()
        self._scroll_to_end()

    async def _project_block_event(self, event: DomainEvent) -> None:
        correlation_id = event.correlation_id or "uncorrelated"
        block_id = str(event.payload.get("block_id", f"event-{event.sequence}"))
        key = (correlation_id, block_id)
        view = await self._ensure_block(event)
        if event.type == EventType.BLOCK_STARTED:
            self._active_block_streams.add(key)
            self._refresh_streaming_state()
            return
        if event.type == EventType.BLOCK_DELTA:
            view.append_content(str(event.payload.get("delta", "")))
            self._scroll_to_end()
            return

        content = event.payload.get("content")
        if content is not None:
            view.set_content(str(content))
        else:
            view.flush_content()
        status = str(event.payload.get("status", BlockStatus.COMPLETED))
        if status in {BlockStatus.FAILED, BlockStatus.ABORTED}:
            view.add_class("error-message")
        self._active_block_streams.discard(key)
        self._refresh_streaming_state()
        self._scroll_to_end()

    async def _ensure_block(
        self,
        event: DomainEvent,
    ) -> TranscriptMessage | CollapsibleTranscriptBlock:
        correlation_id = event.correlation_id or "uncorrelated"
        block_id = str(event.payload.get("block_id", f"event-{event.sequence}"))
        key = (correlation_id, block_id)
        view = self._block_views.get(key)
        if view is not None:
            return view

        kind = str(event.payload.get("kind", BlockKind.TEXT))
        content = str(event.payload.get("content", ""))
        classes = {
            BlockKind.REASONING: "reasoning-message",
            BlockKind.DIFF: "diff-message",
            BlockKind.ARTIFACT: "artifact-message",
            BlockKind.ERROR: "error-message",
        }.get(kind, "assistant-message")
        if kind == BlockKind.REASONING:
            view = CollapsibleTranscriptBlock(
                content,
                title=str(event.payload.get("title", "Thinking")),
                kind=f"block:{kind}",
                classes=classes,
            )
            await self.query_one("#transcript", VerticalScroll).mount(view)
            self._scroll_to_end()
        else:
            view = await self._append_message(
                content,
                kind=f"block:{kind}",
                markdown=kind != BlockKind.ERROR,
                render_mode="diff" if kind == BlockKind.DIFF else None,
                classes=classes,
            )
        self._block_views[key] = view
        return view

    async def _project_assistant_event(self, event: DomainEvent) -> None:
        view = await self._ensure_assistant(event)
        if event.type == EventType.ASSISTANT_STARTED:
            return
        if event.type == EventType.ASSISTANT_DELTA:
            view.append_content(str(event.payload.get("delta", "")))
            self._scroll_to_end()
            return

        content = event.payload.get("content")
        if content is not None:
            view.set_content(str(content))
        else:
            view.flush_content()
        self._scroll_to_end()

    async def _ensure_assistant(
        self,
        event: DomainEvent,
    ) -> TranscriptMessage:
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
        return view

    async def _append_message(
        self,
        content: str,
        *,
        kind: str,
        markdown: bool = False,
        render_mode: str | None = None,
        classes: str,
    ) -> TranscriptMessage:
        view = TranscriptMessage(
            content,
            kind=kind,
            markdown=markdown,
            render_mode=render_mode,
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

    def _refresh_streaming_state(self) -> None:
        self._ui_driver.set_streaming(
            streaming=bool(
                self._active_assistant_streams
                or self._active_block_streams
                or self._active_tool_streams
            )
        )

    def action_request_clear(self) -> None:
        self._dispatch_text("/clear")

    def action_cancel_execution(self) -> None:
        self._dispatch_cancel()

    def action_command_palette(self) -> None:
        self._open_command_palette()

    def action_request_quit(self) -> None:
        self._dispatch_text("/quit")
