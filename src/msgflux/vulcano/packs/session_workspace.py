from __future__ import annotations

from typing import TYPE_CHECKING, cast

from msgflux.vulcano.commands import CommandContext, CommandOptions, CommandResult
from msgflux.vulcano.events import EventDraft, EventType
from msgflux.vulcano.sessions import SessionController

if TYPE_CHECKING:
    from msgflux.vulcano.extensions.api import ExtensionApi

__all__ = ["SessionWorkspacePack"]


class SessionWorkspacePack:
    """Runtime-owned slash commands for durable session workspaces."""

    name = "session-workspace"

    def setup(self, api: ExtensionApi) -> None:
        sessions = api.services.get("sessions")
        if not isinstance(sessions, SessionController):
            raise RuntimeError(
                "SessionWorkspacePack requires the 'sessions' runtime service"
            )
        commands = (
            (
                "session",
                CommandOptions(
                    description="Show the active durable session.",
                    usage="/session",
                    handler=_session_command,
                    category="session",
                ),
            ),
            (
                "sessions",
                CommandOptions(
                    description="List durable sessions.",
                    usage="/sessions",
                    handler=_sessions_command,
                    category="session",
                ),
            ),
            (
                "resume",
                CommandOptions(
                    description="Resume another durable session.",
                    usage="/resume <thread-id>",
                    handler=_resume_command,
                    get_argument_completions=lambda _value: (
                        tuple(info.thread_id for info in sessions.list())
                        if sessions.enabled
                        else ()
                    ),
                    category="session",
                ),
            ),
            (
                "new",
                CommandOptions(
                    description="Start a new empty durable session.",
                    usage="/new",
                    handler=_new_session_command,
                    category="session",
                ),
            ),
            (
                "close",
                CommandOptions(
                    description="Close an open session tab.",
                    usage="/close [thread-id]",
                    handler=_close_session_command,
                    get_argument_completions=lambda _value: tuple(
                        tab.thread_id for tab in sessions.tabs
                    ),
                    category="session",
                ),
            ),
            (
                "pin",
                CommandOptions(
                    description="Pin or unpin an open session tab.",
                    usage="/pin [thread-id]",
                    handler=_pin_session_command,
                    get_argument_completions=lambda _value: tuple(
                        tab.thread_id for tab in sessions.tabs
                    ),
                    category="session",
                ),
            ),
            (
                "fork",
                CommandOptions(
                    description="Fork this session and continue on the new thread.",
                    usage="/fork [event-sequence]",
                    handler=_fork_command,
                    category="session",
                ),
            ),
            (
                "export",
                CommandOptions(
                    description="Export this session as Markdown.",
                    usage="/export [path]",
                    handler=_export_command,
                    category="session",
                ),
            ),
        )
        for name, options in commands:
            api.register_command(name, options)


def _session_control(context: CommandContext) -> SessionController:
    return cast(SessionController, context.services["sessions"])


def _session_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    del arguments
    sessions = _session_control(context)
    state = "enabled" if sessions.enabled else "disabled"
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {
                    "text": (
                        "## Session\n\n"
                        f"- Thread: `{sessions.current_thread_id}`\n"
                        f"- Persistence: **{state}**"
                    )
                },
            ),
        )
    )


def _sessions_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    del arguments
    sessions = _session_control(context)
    records = sessions.list()
    lines = ["## Sessions", "", "| Thread | Events | Parent |", "|---|---:|---|"]
    lines.extend(
        (
            f"| `{record.thread_id}` | {record.event_count} | "
            f"`{record.parent_thread_id or '-'}` |"
        )
        for record in records
    )
    return CommandResult(
        events=(EventDraft(EventType.COMMAND_OUTPUT, {"text": "\n".join(lines)}),)
    )


def _resume_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    thread_id = arguments.strip()
    if not thread_id:
        raise ValueError("Usage: /resume <thread-id>")
    info = _session_control(context).request_resume(thread_id)
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {"text": f"Resuming session `{info.thread_id}`."},
            ),
        )
    )


def _new_session_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    if arguments.strip():
        raise ValueError("Usage: /new")
    info = _session_control(context).request_new()
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {"text": f"Started new session `{info.thread_id}`."},
            ),
        )
    )


def _optional_tab_argument(arguments: str, usage: str) -> str | None:
    values = arguments.split()
    if len(values) > 1:
        raise ValueError(f"Usage: {usage}")
    return values[0] if values else None


def _close_session_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    thread_id = _optional_tab_argument(arguments, "/close [thread-id]")
    tab = _session_control(context).request_close_tab(thread_id)
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {"text": f"Closing session tab `{tab.thread_id}`."},
            ),
        )
    )


def _pin_session_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    thread_id = _optional_tab_argument(arguments, "/pin [thread-id]")
    tab = _session_control(context).request_toggle_tab_pin(thread_id)
    action = "Unpinning" if tab.pinned else "Pinning"
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {"text": f"{action} session tab `{tab.thread_id}`."},
            ),
        )
    )


def _fork_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    value = arguments.strip()
    try:
        sequence = int(value) if value else None
    except ValueError as error:
        raise ValueError("Usage: /fork [event-sequence]") from error
    if sequence is not None and sequence < 1:
        raise ValueError("Fork event sequence must be greater than zero")
    info = _session_control(context).request_fork(sequence)
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {"text": f"Forked session as `{info.thread_id}`."},
            ),
        )
    )


def _export_command(
    arguments: str,
    context: CommandContext,
) -> CommandResult:
    destination = arguments.strip() or None
    path = _session_control(context).export(destination)
    return CommandResult(
        events=(
            EventDraft(
                EventType.COMMAND_OUTPUT,
                {"text": f"Exported session to `{path}`."},
            ),
        )
    )
