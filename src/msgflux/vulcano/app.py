from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, HorizontalScroll, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Collapsible, Input, OptionList, Static

from msgflux.vulcano.actions import (
    ActivateSessionTab,
    CancelExecution,
    CloseSessionTab,
    InputMode,
    ResolvePermission,
    SubmitInput,
    ToggleSessionPin,
)
from msgflux.vulcano.blocks import BlockKind, BlockStatus
from msgflux.vulcano.config import TranscriptMode, VulcanoSettings
from msgflux.vulcano.events import DomainEvent, EventType
from msgflux.vulcano.permissions import PermissionActionDecision
from msgflux.vulcano.runtime import RuntimeProtocol
from msgflux.vulcano.textual_ui import TextualUiDriver, VulcanoTextArea
from msgflux.vulcano.ui import UiDialogOptions, UiManager

__all__ = [
    "CollapsibleTranscriptBlock",
    "PendingInputList",
    "SessionTab",
    "SessionTabBar",
    "ToolExecutionBlock",
    "TranscriptMessage",
    "TurnActivity",
    "TurnNavigationItem",
    "TurnSidebar",
    "VulcanoApp",
]


_PERMISSION_LABELS: dict[PermissionActionDecision, str] = {
    "allow_once": "Allow once",
    "allow_session": "Allow for this session",
    "deny": "Deny",
}


def _permission_decision(value: object) -> PermissionActionDecision | None:
    if value == "allow_once":
        return "allow_once"
    if value == "allow_session":
        return "allow_session"
    if value == "deny":
        return "deny"
    return None


def _permission_prompt(event: DomainEvent) -> str:
    description = str(event.payload.get("description", "Authorize this operation?"))
    owner = str(event.payload.get("owner", "runtime"))
    operation = str(event.payload.get("operation", "operation"))
    lines = [
        "Permission required",
        "",
        description,
        "",
        f"Owner: {owner}",
        f"Operation: {operation}",
    ]
    resource = event.payload.get("resource")
    if resource is not None:
        lines.append(f"Target: {resource}")
    return "\n".join(lines)


class SessionTab(Horizontal):
    """One runtime-owned durable session projected as an interactive tab."""

    def __init__(
        self,
        thread_id: str,
        *,
        pinned: bool,
        status: str,
        persistence: bool,
    ) -> None:
        self.thread_id = thread_id
        self.pinned = pinned
        self.status = status
        self.persistence = persistence
        super().__init__(classes="session-tab")
        self.add_class(f"session-tab-{status}")

    def compose(self) -> ComposeResult:
        yield Button(
            self._select_label(),
            classes="session-tab-select",
            tooltip=f"Activate {self.thread_id} ({self.status})",
        )
        yield Button(
            "◆" if self.pinned else "◇",
            classes="session-tab-pin",
            disabled=not self.persistence,
            tooltip=(
                "Session persistence disabled"
                if not self.persistence
                else "Unpin session"
                if self.pinned
                else "Pin session"
            ),
        )
        yield Button(
            "×",  # noqa: RUF001
            classes="session-tab-close",
            tooltip="Close session tab",
        )

    def update_state(self, *, pinned: bool, status: str, persistence: bool) -> None:
        self.pinned = pinned
        self.status = status
        self.persistence = persistence
        self.remove_class(
            "session-tab-active",
            "session-tab-idle",
            "session-tab-paused",
            "session-tab-terminated",
        )
        self.add_class(f"session-tab-{status}")
        if not self.is_mounted:
            return
        select = self.query_one(".session-tab-select", Button)
        select.label = self._select_label()
        select.tooltip = f"Activate {self.thread_id} ({status})"
        pin = self.query_one(".session-tab-pin", Button)
        pin.label = "◆" if pinned else "◇"
        pin.disabled = not persistence
        pin.tooltip = (
            "Session persistence disabled"
            if not persistence
            else "Unpin session"
            if pinned
            else "Pin session"
        )

    def _select_label(self) -> str:
        marker = {
            "active": "●",
            "idle": "○",
            "paused": "‖",
            "terminated": "×",  # noqa: RUF001
        }.get(self.status, "·")
        summary = self.thread_id
        if summary.startswith("thd_"):
            summary = summary[4:12]
        elif len(summary) > 12:
            summary = summary[:12]
        return f"{marker} {summary}"


class SessionTabBar(HorizontalScroll):
    """Reconciled projection of the runtime session workspace."""

    def __init__(self) -> None:
        super().__init__(id="session-tabs", classes="session-tabs-hidden")
        self.entries: dict[str, SessionTab] = {}
        self.active_thread_id: str | None = None
        self.persistence = False
        self.max_tabs = 5
        self._ordered_thread_ids: tuple[str, ...] = ()
        self._key_mode = False

    def compose(self) -> ComposeResult:
        yield Static(
            self._key_hint(),
            id="session-key-hint",
        )

    def set_key_mode(self, active: bool) -> None:  # noqa: FBT001
        self._key_mode = active
        self.set_class(active, "session-key-mode")
        self.set_class(not self.entries and not active, "session-tabs-hidden")

    def adjacent_thread_id(self, offset: int) -> str | None:
        if not self._ordered_thread_ids:
            return None
        if self.active_thread_id not in self._ordered_thread_ids:
            return self._ordered_thread_ids[0]
        active_index = self._ordered_thread_ids.index(self.active_thread_id)
        return self._ordered_thread_ids[
            (active_index + offset) % len(self._ordered_thread_ids)
        ]

    def thread_id_at(self, ordinal: int) -> str | None:
        index = ordinal - 1
        if not 0 <= index < len(self._ordered_thread_ids):
            return None
        return self._ordered_thread_ids[index]

    def _key_hint(self) -> str:
        if self.max_tabs == 1:
            selection = "1"
        elif self.max_tabs == 10:
            selection = "0-9"
        else:
            selection = f"1-{self.max_tabs}"
        return (
            f"SESSION  n next  p previous  {selection} select  f pin  "
            "x close  c new  Esc cancel"
        )

    async def apply_event(self, event: DomainEvent) -> None:
        raw_tabs = event.payload.get("tabs", ())
        self.persistence = bool(event.payload.get("persistence", False))
        raw_max_tabs = event.payload.get("max_tabs")
        if isinstance(raw_max_tabs, int) and not isinstance(raw_max_tabs, bool):
            self.max_tabs = raw_max_tabs
            self.query_one("#session-key-hint", Static).update(self._key_hint())
        raw_active_thread_id = event.payload.get("active_thread_id")
        self.active_thread_id = (
            str(raw_active_thread_id) if raw_active_thread_id is not None else None
        )
        tabs = (
            tuple(item for item in raw_tabs if isinstance(item, Mapping))
            if isinstance(raw_tabs, Sequence) and not isinstance(raw_tabs, (str, bytes))
            else ()
        )
        ordered_thread_ids = tuple(
            dict.fromkeys(
                thread_id
                for item in tabs
                for thread_id in (str(item.get("thread_id", "")),)
                if thread_id
            )
        )
        self._ordered_thread_ids = ordered_thread_ids
        incoming = set(ordered_thread_ids)
        for thread_id in tuple(self.entries):
            if thread_id in incoming:
                continue
            view = self.entries.pop(thread_id)
            await view.remove()
        for item in tabs:
            thread_id = str(item.get("thread_id", ""))
            if not thread_id:
                continue
            pinned = bool(item.get("pinned", False))
            status = str(item.get("status", "idle"))
            view = self.entries.get(thread_id)
            if view is None:
                view = SessionTab(
                    thread_id,
                    pinned=pinned,
                    status=status,
                    persistence=self.persistence,
                )
                self.entries[thread_id] = view
                await self.mount(view)
            else:
                view.update_state(
                    pinned=pinned,
                    status=status,
                    persistence=self.persistence,
                )
        self.set_class(
            not self.entries and not self._key_mode,
            "session-tabs-hidden",
        )
        active_view = self.entries.get(self.active_thread_id or "")
        if active_view is not None:
            active_view.scroll_visible(animate=False)


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
        super().__init__(classes=classes)
        self.source_text = ""
        self.message_id = message_id
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
        self.status = BlockStatus.STREAMING
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

    def set_status(self, status: str, mode: TranscriptMode) -> None:
        self.status = status
        self.collapsed = (
            mode == "compact"
            if status not in {BlockStatus.FAILED, BlockStatus.ABORTED}
            else False
        )

    def set_transcript_mode(self, mode: TranscriptMode) -> None:
        self.set_status(self.status, mode)


class TurnNavigationItem(Button):
    """Focusable sidebar entry linked to one user-message widget."""

    def __init__(
        self,
        *,
        ordinal: int,
        message_id: str,
        content: str,
        anchor: TranscriptMessage,
    ) -> None:
        self.ordinal = ordinal
        self.message_id = message_id
        self.summary = " ".join(content.split()) or "Empty message"
        if len(self.summary) > 34:
            self.summary = f"{self.summary[:31]}..."
        self.anchor = anchor
        self.execution_status: str | None = None
        super().__init__(classes="turn-nav-item")
        self._update_label()

    def set_execution_status(self, status: str) -> None:
        self.execution_status = status
        self.set_classes("turn-nav-item")
        self.add_class(f"turn-nav-{status}")
        self._update_label()

    def _update_label(self) -> None:
        marker = {
            "running": "●",
            BlockStatus.COMPLETED: "✓",
            BlockStatus.FAILED: "✗",
            BlockStatus.ABORTED: "!",
            "cancelled": "!",
        }.get(self.execution_status, "·")
        self.label = f"{marker} {self.ordinal}. {self.summary}"


class TurnSidebar(VerticalScroll):
    """Derived user-message index used to navigate the transcript."""

    def __init__(self) -> None:
        super().__init__(id="turn-sidebar", classes="turn-sidebar-hidden")
        self.entries: dict[str, TurnNavigationItem] = {}
        self._user_collapsed = False

    def compose(self) -> ComposeResult:
        yield Static("MESSAGES", classes="turn-nav-heading")

    @property
    def is_expanded(self) -> bool:
        return bool(self.entries) and not self._user_collapsed

    def toggle(self) -> bool:
        if not self.entries:
            return False
        self._user_collapsed = not self._user_collapsed
        self._sync_visibility()
        return self.is_expanded

    async def add_message(
        self,
        *,
        message_id: str,
        content: str,
        anchor: TranscriptMessage,
    ) -> TurnNavigationItem:
        existing = self.entries.get(message_id)
        if existing is not None:
            return existing
        item = TurnNavigationItem(
            ordinal=len(self.entries) + 1,
            message_id=message_id,
            content=content,
            anchor=anchor,
        )
        self.entries[message_id] = item
        self._sync_visibility()
        await self.mount(item)
        return item

    def set_execution_status(self, message_id: str, status: str) -> None:
        item = self.entries.get(message_id)
        if item is not None:
            item.set_execution_status(status)

    def select(self, message_id: str) -> None:
        for key, item in self.entries.items():
            item.set_class(key == message_id, "turn-nav-selected")

    async def clear_entries(self) -> None:
        self.entries.clear()
        await self.query(TurnNavigationItem).remove()
        self._sync_visibility()

    def _sync_visibility(self) -> None:
        self.set_class(not self.is_expanded, "turn-sidebar-hidden")


class TurnActivity(Collapsible):
    """Collapsible projection of non-final activity for one execution."""

    def __init__(
        self,
        run_id: str,
        transcript_mode: TranscriptMode = "full",
    ) -> None:
        self.run_id = run_id
        self.transcript_mode = transcript_mode
        self.status = "running"
        self.duration_ms: int | None = None
        self.reported_tool_count = 0
        self.tool_call_ids: set[str] = set()
        self.changed_files: set[str] = set()
        self.body = Container(classes="turn-activity-body")
        super().__init__(
            self.body,
            title="● Activity · running",
            collapsed=False,
            classes="turn-activity turn-activity-running",
        )

    def observe(self, event: DomainEvent) -> None:
        if event.type == EventType.TOOL_STARTED:
            tool_call_id = event.payload.get("tool_call_id")
            if tool_call_id is not None:
                self.tool_call_ids.add(str(tool_call_id))
        elif (
            event.type == EventType.BLOCK_STARTED
            and event.payload.get("kind") == BlockKind.DIFF
        ):
            details = event.payload.get("details", {})
            if isinstance(details, Mapping) and details.get("path") is not None:
                self.changed_files.add(str(details["path"]))
        elif event.type == EventType.EXECUTION_COMPLETED:
            self.status = str(event.payload.get("status", BlockStatus.COMPLETED))
            tool_count = event.payload.get("tool_count")
            if isinstance(tool_count, int) and not isinstance(tool_count, bool):
                self.reported_tool_count = tool_count
            changed_files = event.payload.get("changed_files", ())
            if isinstance(changed_files, Sequence) and not isinstance(
                changed_files, (str, bytes)
            ):
                self.changed_files.update(str(path) for path in changed_files)
            duration = event.payload.get("duration_ms")
            if isinstance(duration, int) and not isinstance(duration, bool):
                self.duration_ms = duration
            self._apply_mode()

        self._update_title()

    def set_transcript_mode(self, mode: TranscriptMode) -> None:
        self.transcript_mode = mode
        self._apply_mode()

    def _apply_mode(self) -> None:
        if self.status == BlockStatus.COMPLETED:
            self.collapsed = self.transcript_mode == "compact"
        elif self.status in {
            "running",
            BlockStatus.FAILED,
            BlockStatus.ABORTED,
            "cancelled",
        }:
            self.collapsed = False

    def _update_title(self) -> None:
        marker = {
            "running": "●",
            BlockStatus.COMPLETED: "✓",
            BlockStatus.FAILED: "✗",
            BlockStatus.ABORTED: "!",
            "cancelled": "!",
        }.get(self.status, "○")
        details: list[str] = []
        tool_count = max(len(self.tool_call_ids), self.reported_tool_count)
        if tool_count:
            count = tool_count
            details.append(f"{count} tool" + ("s" if count != 1 else ""))
        if self.changed_files:
            count = len(self.changed_files)
            details.append(f"{count} file" + ("s" if count != 1 else ""))
        if self.duration_ms is not None:
            if self.duration_ms < 1000:
                details.append(f"{self.duration_ms} ms")
            else:
                details.append(f"{self.duration_ms / 1000:.1f} s")
        suffix = f" · {' · '.join(details)}" if details else ""
        state = "running" if self.status == "running" else self.status
        self.title = f"{marker} Activity · {state}{suffix}"
        self.remove_class(
            "turn-activity-running",
            "turn-activity-completed",
            "turn-activity-failed",
            "turn-activity-aborted",
            "turn-activity-cancelled",
        )
        self.add_class(f"turn-activity-{self.status}")


class ToolExecutionBlock(Collapsible):
    """Incrementally updated tool call/result with extension renderers."""

    def __init__(
        self,
        ui: UiManager,
        driver: TextualUiDriver,
        event: DomainEvent,
        transcript_mode: TranscriptMode = "full",
    ) -> None:
        self._ui = ui
        self._driver = driver
        self._body = Container(classes="tool-execution-body")
        self._event = event
        self.tool_call_id = str(event.payload.get("tool_call_id", ""))
        self.tool_name = str(event.payload.get("name", "tool"))
        self.status = BlockStatus.PENDING
        self.transcript_mode = transcript_mode
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
        self.remove_class("tool-completed", "tool-failed", "tool-pending")
        if status == BlockStatus.COMPLETED:
            self.add_class("tool-completed")
        elif status in {BlockStatus.FAILED, BlockStatus.ABORTED}:
            self.add_class("tool-failed")
        else:
            self.add_class("tool-pending")
        if status == BlockStatus.COMPLETED:
            self.collapsed = self.transcript_mode == "compact"
        elif status in {BlockStatus.FAILED, BlockStatus.ABORTED}:
            self.collapsed = False

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

    def set_transcript_mode(self, mode: TranscriptMode) -> None:
        self.transcript_mode = mode
        if self.status == BlockStatus.COMPLETED:
            self.collapsed = mode == "compact"
        elif self.status in {BlockStatus.FAILED, BlockStatus.ABORTED}:
            self.collapsed = False

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

    #conversation-shell {
        width: 100%;
        height: 1fr;
    }

    #session-tabs {
        width: 100%;
        height: 4;
        padding: 0 1;
        background: #10131a;
        border-bottom: solid #272c36;
        scrollbar-size: 1 1;
        scrollbar-color: #3c4352;
    }

    .session-tabs-hidden {
        display: none;
    }

    #session-key-hint {
        display: none;
        width: 100%;
        height: 3;
        padding: 0 1;
        color: #ffb27d;
        background: #24170f;
        text-style: bold;
        content-align: left middle;
    }

    #session-tabs.session-key-mode .session-tab {
        display: none;
    }

    #session-tabs.session-key-mode #session-key-hint {
        display: block;
    }

    .session-tab {
        width: 19;
        min-width: 19;
        max-width: 19;
        height: 3;
        background: #171a21;
    }

    .session-tab Button {
        height: 3;
        min-height: 3;
        padding: 0 1;
        color: #9da5b4;
        background: #171a21;
        border: none;
        text-wrap: nowrap;
    }

    .session-tab .session-tab-select {
        width: 13;
        min-width: 13;
        max-width: 13;
    }

    .session-tab .session-tab-pin, .session-tab .session-tab-close {
        width: 3;
        min-width: 3;
        padding: 0;
    }

    .session-tab-active Button,
    .session-tab Button:hover,
    .session-tab Button:focus {
        color: #ffffff;
        background: #5a2a16;
    }

    .session-tab-active .session-tab-select {
        color: #ff8a3d;
        text-style: bold;
    }

    .session-tab-paused .session-tab-select {
        color: #c8cad1;
    }

    .session-tab-terminated .session-tab-select {
        color: #ff9b9b;
    }

    #turn-sidebar {
        width: 24;
        height: 100%;
        padding: 1;
        background: #10131a;
        border-right: solid #272c36;
        scrollbar-size: 1 1;
        scrollbar-color: #3c4352;
    }

    .turn-sidebar-hidden {
        display: none;
    }

    #turn-sidebar-toggle {
        width: 3;
        min-width: 3;
        height: 3;
        min-height: 3;
        margin-top: 1;
        padding: 0;
        color: #ff8a3d;
        background: #171a21;
        border: none;
    }

    #turn-sidebar-toggle:hover, #turn-sidebar-toggle:focus {
        color: #ffffff;
        background: #5a2a16;
        text-style: bold;
    }

    #turn-sidebar-toggle:disabled {
        color: #3c4352;
        background: #10131a;
    }

    .turn-nav-heading {
        width: 100%;
        height: 2;
        color: #778091;
        text-style: bold;
        content-align: left middle;
    }

    .turn-nav-item {
        width: 100%;
        min-width: 0;
        height: auto;
        min-height: 3;
        margin-bottom: 1;
        padding: 0 1;
        color: #9da5b4;
        background: #171a21;
        border: none;
        text-align: left;
    }

    .turn-nav-item:hover, .turn-nav-item:focus, .turn-nav-selected {
        color: #ffffff;
        background: #352218;
        text-style: bold;
    }

    .turn-nav-running {
        color: #ff8a3d;
    }

    .turn-nav-completed {
        color: #c8cad1;
    }

    .turn-nav-failed, .turn-nav-aborted, .turn-nav-cancelled {
        color: #ff9b9b;
    }

    #transcript {
        width: 1fr;
        height: 100%;
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

    .user-facing-message {
        margin-right: 6;
        color: #d7dae0;
        background: #181b22;
        border: round #3c4352;
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

    TurnActivity {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    .turn-activity > CollapsibleTitle {
        color: #c8cad1;
        background: #151820;
        text-style: bold;
    }

    .turn-activity-running > CollapsibleTitle {
        color: #ff8a3d;
    }

    .turn-activity-failed > CollapsibleTitle,
    .turn-activity-aborted > CollapsibleTitle,
    .turn-activity-cancelled > CollapsibleTitle {
        color: #ff9b9b;
    }

    .turn-activity-body {
        width: 100%;
        height: auto;
        padding: 1;
        background: #0f1218;
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

    .permission-message {
        color: #d7dae0;
        background: #181b22;
        border-left: solid #ff6a1a;
    }

    .permission-denied, .permission-cancelled {
        color: #ff9b9b;
        background: #241416;
        border-left: solid #d96c75;
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
        for key in self.settings.keybindings.keys("session_prefix"):
            self._bindings.bind(
                key,
                "session_prefix",
                show=False,
                priority=True,
            )
        self.transcript_mode = self.settings.transcript.mode
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
        self._active_execution_streams: set[str] = set()
        self._activity_views: dict[str, TurnActivity] = {}
        self._correlation_runs: dict[str, str] = {}
        self._run_user_messages: dict[str, str] = {}
        self._user_message_views: dict[str, TranscriptMessage] = {}
        self._session_key_mode = False

    def compose(self) -> ComposeResult:
        with Container(id="header-slot"):
            yield self._ui_driver.create_default_header()
        yield SessionTabBar()
        with Horizontal(id="conversation-shell"):
            yield TurnSidebar()
            yield Button("›", id="turn-sidebar-toggle", disabled=True)  # noqa: RUF001
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
        self._sync_sidebar_toggle()
        self._ui_driver.focus_editor()
        self._consume_events()

    def on_unmount(self) -> None:
        self.runtime.ui.unbind(self._ui_driver)

    async def on_key(self, event: events.Key) -> None:
        if self._handle_command_menu_key(event):
            return
        if self._handle_configured_app_key(event):
            return
        keybindings = self.settings.keybindings
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

    def _handle_configured_app_key(self, event: events.Key) -> bool:
        if self._session_key_mode:
            self._handle_session_key(event)
        else:
            keybindings = self.settings.keybindings
            for action, handler in (
                ("toggle_sidebar", self.action_toggle_sidebar),
                ("cancel", self.action_cancel_execution),
                ("clear", self.action_request_clear),
                ("quit", self.action_request_quit),
                ("command_palette", self.action_command_palette),
            ):
                if not keybindings.matches(action, event.key):
                    continue
                handler()
                break
            else:
                return False
        event.prevent_default()
        event.stop()
        return True

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

    @on(Button.Pressed, ".turn-nav-item")
    def _on_turn_navigation(self, event: Button.Pressed) -> None:
        item = event.button
        if not isinstance(item, TurnNavigationItem):
            return
        self.query_one(TurnSidebar).select(item.message_id)
        item.anchor.scroll_visible(
            animate=False,
            immediate=True,
            top=True,
            force=True,
        )
        event.stop()

    @on(Button.Pressed, "#turn-sidebar-toggle")
    def _on_turn_sidebar_toggle(self, event: Button.Pressed) -> None:
        self.action_toggle_sidebar()
        event.stop()

    @on(Button.Pressed, ".session-tab-select")
    def _on_session_tab_selected(self, event: Button.Pressed) -> None:
        tab = event.button.parent
        if isinstance(tab, SessionTab):
            self._dispatch_session_action(ActivateSessionTab(tab.thread_id))
        event.stop()

    @on(Button.Pressed, ".session-tab-pin")
    def _on_session_tab_pin(self, event: Button.Pressed) -> None:
        tab = event.button.parent
        if isinstance(tab, SessionTab):
            self._dispatch_session_action(ToggleSessionPin(tab.thread_id))
        event.stop()

    @on(Button.Pressed, ".session-tab-close")
    def _on_session_tab_close(self, event: Button.Pressed) -> None:
        tab = event.button.parent
        if isinstance(tab, SessionTab):
            self._dispatch_session_action(CloseSessionTab(tab.thread_id))
        event.stop()

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

    @work(group="session-tab-actions")
    async def _dispatch_session_action(
        self,
        action: ActivateSessionTab | CloseSessionTab | ToggleSessionPin,
    ) -> None:
        try:
            await self.runtime.dispatch(action)
        except Exception as error:
            self.notify(str(error), severity="error")

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
        if await self._project_permission_event(event):
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

    async def _project_permission_event(self, event: DomainEvent) -> bool:
        if event.type == EventType.PERMISSION_REQUESTED:
            if not event.payload.get("requires_confirmation", True):
                return True
            request_id = str(event.payload.get("request_id", ""))
            raw_options = event.payload.get("options", ())
            options = (
                tuple(
                    decision
                    for value in raw_options
                    for decision in (_permission_decision(value),)
                    if decision is not None
                )
                if isinstance(raw_options, Sequence)
                and not isinstance(raw_options, (str, bytes))
                else ()
            )
            if not request_id or not options:
                return True
            selected = await self.runtime.ui.select(
                _permission_prompt(event),
                tuple(_PERMISSION_LABELS[decision] for decision in options),
                UiDialogOptions(),
            )
            labels = {_PERMISSION_LABELS[decision]: decision for decision in options}
            decision = labels.get(selected, "deny")
            await self.runtime.dispatch(
                ResolvePermission(request_id=request_id, decision=decision)
            )
            return True

        if event.type != EventType.PERMISSION_RESOLVED:
            return False
        decision = str(event.payload.get("decision", "deny"))
        operation = str(event.payload.get("operation", "operation"))
        resource = event.payload.get("resource")
        label = {
            "allow_once": "allowed once",
            "allow_session": "allowed for this session",
            "cancelled": "cancelled",
            "deny": "denied",
        }.get(decision, decision)
        content = f"Permission {operation}: {label}"
        if resource is not None:
            content = f"{content}\n{resource}"
        await self._append_message(
            content,
            kind="permission",
            classes=f"permission-message permission-{decision}",
            target=self._event_mount_target(event),
        )
        return True

    async def _project_execution_event(self, event: DomainEvent) -> bool:
        if event.type == EventType.EXECUTION_STARTED:
            await self._project_execution_started(event)
            return True
        if event.type == EventType.EXECUTION_COMPLETED:
            await self._project_execution_completed(event)
            return True
        if event.type == EventType.SESSION_SWITCHED:
            await self._project_session_switch(event)
            return True
        if event.type == EventType.SESSION_TABS_UPDATED:
            await self.query_one(SessionTabBar).apply_event(event)
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

    async def _project_execution_started(self, event: DomainEvent) -> None:
        run_id = self._event_run_id(event)
        if run_id is None:
            return
        if event.correlation_id is not None:
            self._correlation_runs[event.correlation_id] = run_id
        input_message_id = event.payload.get("input_message_id")
        if input_message_id is not None:
            resolved_message_id = str(input_message_id)
            self._run_user_messages[run_id] = resolved_message_id
            self.query_one(TurnSidebar).set_execution_status(
                resolved_message_id,
                "running",
            )
        activity = self._activity_views.get(run_id)
        if activity is None:
            activity = TurnActivity(
                run_id,
                transcript_mode=self.transcript_mode,
            )
            self._activity_views[run_id] = activity
            await self.query_one("#transcript", VerticalScroll).mount(activity)
        activity.observe(event)
        self._active_execution_streams.add(run_id)
        self._refresh_streaming_state()
        self._scroll_to_end()

    async def _project_execution_completed(self, event: DomainEvent) -> None:
        run_id = self._event_run_id(event)
        if run_id is None:
            return
        activity = self._activity_views.get(run_id)
        final_message_id = event.payload.get("final_message_id")
        if final_message_id is not None and activity is not None:
            final_view = self._assistant_views.get(str(final_message_id))
            if final_view is not None and final_view.parent is activity.body:
                await final_view.remove()
                await self.query_one("#transcript", VerticalScroll).mount(
                    final_view,
                    after=activity,
                )
        if activity is not None:
            activity.observe(event)
        status = str(event.payload.get("status", BlockStatus.COMPLETED))
        input_message_id = self._run_user_messages.get(run_id)
        if input_message_id is not None:
            self.query_one(TurnSidebar).set_execution_status(
                input_message_id,
                status,
            )
        self._active_execution_streams.discard(run_id)
        self._refresh_streaming_state()
        self._scroll_to_end()

    async def _project_cancellation(self, event: DomainEvent) -> None:
        target = event.payload.get("target_correlation_id")
        if not isinstance(target, str):
            return
        run_id = self._correlation_runs.get(target)
        if run_id is not None:
            activity = self._activity_views.get(run_id)
            if activity is not None:
                activity.observe(
                    DomainEvent(
                        type=EventType.EXECUTION_COMPLETED,
                        sequence=event.sequence,
                        payload={"run_id": run_id, "status": BlockStatus.ABORTED},
                        correlation_id=target,
                    )
                )
            input_message_id = self._run_user_messages.get(run_id)
            if input_message_id is not None:
                self.query_one(TurnSidebar).set_execution_status(
                    input_message_id,
                    BlockStatus.ABORTED,
                )
            self._active_execution_streams.discard(run_id)
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
        if isinstance(thread_id, str):
            self._ui_driver.set_execution_scope(
                {"thread_id": thread_id, "run_id": None}
            )

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
            await self._append_rendered(rendered, event=event)
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
            message_id = str(
                event.payload.get("message_id", f"message-{event.sequence}")
            )
            view = await self._append_message(
                str(event.payload.get("content", "")),
                kind="user",
                classes="user-message",
                message_id=message_id,
            )
            self._user_message_views[message_id] = view
            await self.query_one(TurnSidebar).add_message(
                message_id=message_id,
                content=view.source_text,
                anchor=view,
            )
            self._sync_sidebar_toggle()
            run_id = self._event_run_id(event)
            if run_id is not None:
                self._run_user_messages[run_id] = message_id
            return True

        if event.type == EventType.ASSISTANT_USER_MESSAGE:
            await self._append_message(
                str(event.payload.get("content", "")),
                kind="assistant:user-message",
                markdown=True,
                classes="user-facing-message",
                message_id=(
                    str(event.payload["message_id"])
                    if event.payload.get("message_id") is not None
                    else None
                ),
                target=self._event_mount_target(event),
            )
            self._scroll_to_end()
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
            action = event.payload.get("action")
            if action == "transcript.clear":
                await self._clear_transcript()
            elif action == "transcript.view":
                try:
                    self.set_transcript_mode(str(event.payload.get("mode", "")))
                except ValueError as error:
                    await self._append_message(
                        str(error),
                        kind="error",
                        classes="error-message",
                    )
            return

    async def _clear_transcript(self) -> None:
        await self.query_one("#transcript", VerticalScroll).remove_children()
        await self.query_one(TurnSidebar).clear_entries()
        self._sync_sidebar_toggle()
        self._assistant_views.clear()
        self._block_views.clear()
        self._tool_views.clear()
        self._activity_views.clear()
        self._correlation_runs.clear()
        self._run_user_messages.clear()
        self._user_message_views.clear()
        self.runtime.ui.clear_tool_render_state()
        self._active_assistant_streams.clear()
        self._active_block_streams.clear()
        self._active_tool_streams.clear()
        self._active_execution_streams.clear()
        self.query_one("#pending-inputs", PendingInputList).clear()
        self._ui_driver.set_queue_count(0)
        self._refresh_streaming_state()

    async def _project_tool_event(self, event: DomainEvent) -> None:
        correlation_id = event.correlation_id or "uncorrelated"
        tool_call_id = str(event.payload.get("tool_call_id", f"event-{event.sequence}"))
        key = (correlation_id, tool_call_id)
        view = self._tool_views.get(key)
        if view is None:
            view = ToolExecutionBlock(
                self.runtime.ui,
                self._ui_driver,
                event,
                transcript_mode=self.transcript_mode,
            )
            await self._event_mount_target(event).mount(view)
            self._tool_views[key] = view
        activity = self._activity_for_event(event)
        if activity is not None:
            activity.observe(event)
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
        if isinstance(view, CollapsibleTranscriptBlock):
            view.set_status(status, self.transcript_mode)
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
                collapsed=self.transcript_mode == "compact",
                classes=classes,
            )
            await self._event_mount_target(event).mount(view)
            self._scroll_to_end()
        else:
            view = await self._append_message(
                content,
                kind=f"block:{kind}",
                markdown=kind != BlockKind.ERROR,
                render_mode="diff" if kind == BlockKind.DIFF else None,
                classes=classes,
                target=self._event_mount_target(event),
            )
        activity = self._activity_for_event(event)
        if activity is not None:
            activity.observe(event)
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
        if event.payload.get("message_id") is not None:
            key = str(event.payload["message_id"])
        view = self._assistant_views.get(key)
        if view is None:
            target = None
            if not event.payload.get("is_final"):
                target = self._event_mount_target(event)
            view = await self._append_message(
                "",
                kind="assistant",
                markdown=True,
                classes="assistant-message",
                message_id=(
                    str(event.payload["message_id"])
                    if event.payload.get("message_id") is not None
                    else None
                ),
                target=target,
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
        message_id: str | None = None,
        target: Widget | None = None,
    ) -> TranscriptMessage:
        view = TranscriptMessage(
            content,
            kind=kind,
            markdown=markdown,
            render_mode=render_mode,
            classes=classes,
            message_id=message_id,
        )
        mount_target = target or self.query_one("#transcript", VerticalScroll)
        await mount_target.mount(view)
        self._scroll_to_end()
        return view

    async def _append_rendered(
        self,
        content: object,
        *,
        event: DomainEvent | None = None,
    ) -> Widget:
        view = await self._ui_driver.materialize(content)
        target = (
            self._event_mount_target(event)
            if event is not None
            else self.query_one("#transcript", VerticalScroll)
        )
        await target.mount(view)
        self._scroll_to_end()
        return view

    def _activity_for_event(self, event: DomainEvent) -> TurnActivity | None:
        run_id = self._event_run_id(event)
        return self._activity_views.get(run_id) if run_id is not None else None

    def _event_mount_target(self, event: DomainEvent) -> Widget:
        activity = self._activity_for_event(event)
        if activity is not None:
            return activity.body
        return self.query_one("#transcript", VerticalScroll)

    def _event_run_id(self, event: DomainEvent) -> str | None:
        run_id = event.payload.get("run_id")
        if run_id is not None:
            return str(run_id)
        scope = event.payload.get("scope")
        if isinstance(scope, Mapping) and scope.get("run_id") is not None:
            return str(scope["run_id"])
        if event.correlation_id is not None:
            return self._correlation_runs.get(event.correlation_id)
        return None

    def _sync_sidebar_toggle(self) -> None:
        sidebar = self.query_one(TurnSidebar)
        toggle = self.query_one("#turn-sidebar-toggle", Button)
        toggle.disabled = not sidebar.entries
        toggle.label = "‹" if sidebar.is_expanded else "›"  # noqa: RUF001
        key = self.settings.keybindings.primary("toggle_sidebar")
        action = "Collapse" if sidebar.is_expanded else "Expand"
        toggle.tooltip = f"{action} message navigation" + (
            f" ({key})" if key is not None else ""
        )

    def set_transcript_mode(self, mode: str) -> None:
        if mode not in {"full", "compact"}:
            raise ValueError("Transcript mode must be 'full' or 'compact'")
        resolved_mode: TranscriptMode = "compact" if mode == "compact" else "full"
        self.transcript_mode = resolved_mode
        for activity in self._activity_views.values():
            activity.set_transcript_mode(resolved_mode)
        for tool in self._tool_views.values():
            tool.set_transcript_mode(resolved_mode)
        for block in self._block_views.values():
            if isinstance(block, CollapsibleTranscriptBlock):
                block.set_transcript_mode(resolved_mode)
        self._scroll_to_end()

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
                or self._active_execution_streams
            )
        )

    def action_request_clear(self) -> None:
        self._dispatch_text("/clear")

    def action_cancel_execution(self) -> None:
        self._dispatch_cancel()

    def action_command_palette(self) -> None:
        self._open_command_palette()

    def action_session_prefix(self) -> None:
        self._session_key_mode = True
        self._ui_driver.dismiss_command_menu()
        self.set_focus(None)
        self.query_one(SessionTabBar).set_key_mode(True)

    def action_toggle_sidebar(self) -> None:
        self.query_one(TurnSidebar).toggle()
        self._sync_sidebar_toggle()

    def _handle_session_key(self, event: events.Key) -> None:
        bar = self.query_one(SessionTabBar)
        key = event.key.lower()
        self._finish_session_key_mode()
        if key == "escape":
            return
        offset = {"n": 1, "right": 1, "p": -1, "left": -1}.get(key)
        if offset is not None:
            self._activate_session_from_keyboard(bar.adjacent_thread_id(offset))
            return
        if len(key) == 1 and key in "0123456789":
            ordinal = 10 if key == "0" else int(key)
            self._activate_session_from_keyboard(
                bar.thread_id_at(ordinal),
                ordinal=ordinal,
            )
            return
        active_thread_id = bar.active_thread_id
        if key == "f":
            if not bar.persistence:
                self.notify("Session persistence is disabled", severity="warning")
            elif active_thread_id is not None:
                self._dispatch_session_action(ToggleSessionPin(active_thread_id))
            return
        if key == "x":
            if active_thread_id is not None:
                self._dispatch_session_action(CloseSessionTab(active_thread_id))
            return
        if key == "c":
            self._dispatch_text("/new")
            return
        self.notify(
            f"Unknown session key: {event.key}",
            severity="warning",
        )

    def _finish_session_key_mode(self) -> None:
        self._session_key_mode = False
        self.query_one(SessionTabBar).set_key_mode(False)
        self._ui_driver.focus_editor()

    def _activate_session_from_keyboard(
        self,
        thread_id: str | None,
        *,
        ordinal: int | None = None,
    ) -> None:
        if thread_id is None:
            message = (
                f"Session tab {ordinal} is not open"
                if ordinal is not None
                else "No session tab is available"
            )
            self.notify(message, severity="warning")
            return
        self._dispatch_session_action(ActivateSessionTab(thread_id))

    def action_request_quit(self) -> None:
        self._dispatch_text("/quit")
