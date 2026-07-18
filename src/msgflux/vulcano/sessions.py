from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Mapping

from msgflux.runtime.context import new_thread_id
from msgflux.vulcano.events import DomainEvent, EventType

__all__ = [
    "SessionController",
    "SessionInfo",
    "SessionStore",
    "SessionTransition",
]


_VALID_THREAD_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_REPLAY_EVENT_TYPES = {
    EventType.EXECUTION_STARTED,
    EventType.EXECUTION_COMPLETED,
    EventType.MESSAGE_USER,
    EventType.ASSISTANT_USER_MESSAGE,
    EventType.ASSISTANT_STARTED,
    EventType.ASSISTANT_DELTA,
    EventType.ASSISTANT_COMPLETED,
    EventType.BLOCK_STARTED,
    EventType.BLOCK_DELTA,
    EventType.BLOCK_COMPLETED,
    EventType.TOOL_STARTED,
    EventType.TOOL_UPDATED,
    EventType.TOOL_COMPLETED,
    EventType.COMMAND_STARTED,
    EventType.COMMAND_OUTPUT,
    EventType.COMMAND_ERROR,
    EventType.RUNTIME_ERROR,
    EventType.CLIENT_ACTION,
}


@dataclass(frozen=True)
class SessionInfo:
    thread_id: str
    created_at: str
    updated_at: str
    parent_thread_id: str | None = None
    forked_from_sequence: int | None = None
    event_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "thread_id": self.thread_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parent_thread_id": self.parent_thread_id,
            "forked_from_sequence": self.forked_from_sequence,
            "event_count": self.event_count,
        }


@dataclass(frozen=True)
class SessionTransition:
    kind: Literal["resume", "fork"]
    thread_id: str


class SessionStore:
    """Append-only JSONL transcript storage keyed by ExecutionScope thread id."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()

    def ensure(
        self,
        thread_id: str,
        *,
        parent_thread_id: str | None = None,
        forked_from_sequence: int | None = None,
    ) -> SessionInfo:
        path = self._path(thread_id)
        if path.is_file():
            return self.info(thread_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        now = _now()
        metadata = {
            "record": "session",
            "version": 1,
            "thread_id": thread_id,
            "created_at": now,
            "parent_thread_id": parent_thread_id,
            "forked_from_sequence": forked_from_sequence,
        }
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        return self.info(thread_id)

    def append(self, thread_id: str, event: DomainEvent) -> None:
        self.ensure(thread_id)
        record = {"record": "event", "event": event.to_dict()}
        with self._path(thread_id).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def load(self, thread_id: str) -> tuple[DomainEvent, ...]:
        path = self._path(thread_id)
        if not path.is_file():
            raise LookupError(f"Unknown Vulcano session: {thread_id}")
        events: list[DomainEvent] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid session JSON at {path}:{line_number}: {error}"
                    ) from error
                if record.get("record") != "event":
                    continue
                raw_event = record.get("event")
                if not isinstance(raw_event, Mapping):
                    raise ValueError(f"Invalid event record at {path}:{line_number}")
                events.append(DomainEvent.from_dict(raw_event))
        return tuple(events)

    def replay(self, thread_id: str) -> tuple[DomainEvent, ...]:
        replayable = tuple(
            event for event in self.load(thread_id) if _is_replayable_event(event)
        )
        return _complete_interrupted_streams(replayable)

    def list(self) -> tuple[SessionInfo, ...]:
        if not self.directory.is_dir():
            return ()
        sessions = [self.info(path.stem) for path in self.directory.glob("*.jsonl")]
        return tuple(sorted(sessions, key=lambda item: item.updated_at, reverse=True))

    def info(self, thread_id: str) -> SessionInfo:
        path = self._path(thread_id)
        if not path.is_file():
            raise LookupError(f"Unknown Vulcano session: {thread_id}")
        metadata: Mapping[str, object] | None = None
        event_count = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("record") == "session":
                    metadata = record
                elif record.get("record") == "event":
                    event_count += 1
        if metadata is None:
            raise ValueError(f"Session metadata is missing: {path}")
        updated_at = datetime.fromtimestamp(
            path.stat().st_mtime,
            UTC,
        ).isoformat()
        return SessionInfo(
            thread_id=thread_id,
            created_at=str(metadata.get("created_at", updated_at)),
            updated_at=updated_at,
            parent_thread_id=_optional_string(metadata.get("parent_thread_id")),
            forked_from_sequence=_optional_integer(
                metadata.get("forked_from_sequence")
            ),
            event_count=event_count,
        )

    def fork(
        self,
        thread_id: str,
        *,
        through_sequence: int | None = None,
        target_thread_id: str | None = None,
    ) -> SessionInfo:
        source_events = self.load(thread_id)
        resolved_sequence = through_sequence
        if resolved_sequence is None and source_events:
            resolved_sequence = source_events[-1].sequence
        target = target_thread_id or new_thread_id()
        self.ensure(
            target,
            parent_thread_id=thread_id,
            forked_from_sequence=resolved_sequence,
        )
        for event in source_events:
            if resolved_sequence is None or event.sequence <= resolved_sequence:
                self.append(target, event)
        return self.info(target)

    def export_markdown(
        self,
        thread_id: str,
        destination: str | Path,
    ) -> Path:
        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _events_to_markdown(thread_id, self.replay(thread_id)),
            encoding="utf-8",
        )
        return path

    def _path(self, thread_id: str) -> Path:
        if not _VALID_THREAD_ID.fullmatch(thread_id):
            raise ValueError(f"Invalid Vulcano thread id: {thread_id!r}")
        return self.directory / f"{thread_id}.jsonl"


class SessionController:
    """Extension-facing session facade with deferred runtime transitions."""

    def __init__(
        self,
        store: SessionStore | None,
        thread_id: str,
        *,
        export_directory: str | Path,
    ) -> None:
        self.store = store
        self.current_thread_id = thread_id
        self.export_directory = Path(export_directory).expanduser().resolve()
        self._pending: SessionTransition | None = None

    @property
    def enabled(self) -> bool:
        return self.store is not None

    def list(self) -> tuple[SessionInfo, ...]:
        return self._require_store().list()

    def request_resume(self, thread_id: str) -> SessionInfo:
        info = self._require_store().info(thread_id)
        self._pending = SessionTransition("resume", thread_id)
        return info

    def request_fork(self, through_sequence: int | None = None) -> SessionInfo:
        info = self._require_store().fork(
            self.current_thread_id,
            through_sequence=through_sequence,
        )
        self._pending = SessionTransition("fork", info.thread_id)
        return info

    def export(
        self,
        destination: str | Path | None = None,
    ) -> Path:
        target = destination or (
            self.export_directory / f"vulcano-{self.current_thread_id}.md"
        )
        return self._require_store().export_markdown(
            self.current_thread_id,
            target,
        )

    def consume_transition(self) -> SessionTransition | None:
        transition = self._pending
        self._pending = None
        return transition

    def activate(self, thread_id: str) -> None:
        self.current_thread_id = thread_id

    def _require_store(self) -> SessionStore:
        if self.store is None:
            raise RuntimeError("Vulcano session persistence is disabled")
        return self.store


def _events_to_markdown(thread_id: str, events: tuple[DomainEvent, ...]) -> str:
    lines = [f"# Vulcano session `{thread_id}`", ""]
    block_content: dict[str, str] = {}
    for event in events:
        if event.type == EventType.MESSAGE_USER:
            lines.extend(("## User", "", str(event.payload.get("content", "")), ""))
        elif event.type == EventType.ASSISTANT_COMPLETED:
            _append_assistant_markdown(lines, event)
        elif event.type == EventType.ASSISTANT_USER_MESSAGE:
            lines.extend(
                (
                    "### Agent update",
                    "",
                    str(event.payload.get("content", "")),
                    "",
                )
            )
        elif event.type in {
            EventType.BLOCK_STARTED,
            EventType.BLOCK_DELTA,
            EventType.BLOCK_COMPLETED,
        }:
            _append_block_markdown(lines, block_content, event)
        elif event.type == EventType.TOOL_COMPLETED:
            _append_tool_markdown(lines, event)
        elif event.type == EventType.COMMAND_STARTED:
            lines.extend(("## Command", "", f"`{event.payload.get('raw', '')}`", ""))
        elif event.type == EventType.COMMAND_OUTPUT:
            lines.extend((str(event.payload.get("text", "")), ""))
    return "\n".join(lines).rstrip() + "\n"


def _is_replayable_event(event: DomainEvent) -> bool:
    if event.type not in _REPLAY_EVENT_TYPES:
        return False
    return not (
        event.type == EventType.CLIENT_ACTION
        and event.payload.get("action") == "transcript.view"
    )


def _complete_interrupted_streams(
    events: tuple[DomainEvent, ...],
) -> tuple[DomainEvent, ...]:
    state = _ReplayState({}, {}, {}, {})
    for event in events:
        _track_replay_event(state, event)

    completed = list(events)
    sequence = max((event.sequence for event in events), default=0)

    def append(event_type: str, payload: Mapping[str, object], correlation: str | None):
        nonlocal sequence
        sequence += 1
        completed.append(
            DomainEvent(
                type=event_type,
                sequence=sequence,
                payload=payload,
                correlation_id=correlation,
            )
        )

    for correlation_id, content in state.assistants.items():
        append(
            EventType.ASSISTANT_COMPLETED,
            {"content": content, "status": "aborted"},
            correlation_id,
        )
    for (correlation_id, block_id), payload in state.blocks.items():
        append(
            EventType.BLOCK_COMPLETED,
            {
                "block_id": block_id,
                "content": str(payload.get("content", "")),
                "status": "aborted",
            },
            correlation_id,
        )
    for (correlation_id, tool_call_id), payload in state.tools.items():
        append(
            EventType.TOOL_COMPLETED,
            {
                "tool_call_id": tool_call_id,
                "name": str(payload.get("name", "tool")),
                "result": "Interrupted before completion",
                "is_error": True,
                "status": "aborted",
            },
            correlation_id,
        )
    for (correlation_id, run_id), payload in state.executions.items():
        append(
            EventType.EXECUTION_COMPLETED,
            {
                "run_id": run_id,
                "scope": payload.get("scope", {}),
                "status": "aborted",
                "final_message_id": None,
            },
            correlation_id,
        )
    return tuple(completed)


@dataclass
class _ReplayState:
    assistants: dict[str | None, str]
    blocks: dict[tuple[str | None, str], dict[str, object]]
    tools: dict[tuple[str | None, str], dict[str, object]]
    executions: dict[tuple[str | None, str], dict[str, object]]


def _track_replay_event(state: _ReplayState, event: DomainEvent) -> None:
    if event.type in {
        EventType.EXECUTION_STARTED,
        EventType.EXECUTION_COMPLETED,
    }:
        _track_execution_event(state, event)
    elif event.type in {
        EventType.ASSISTANT_STARTED,
        EventType.ASSISTANT_DELTA,
        EventType.ASSISTANT_COMPLETED,
    }:
        _track_assistant_event(state, event)
    elif event.type in {
        EventType.BLOCK_STARTED,
        EventType.BLOCK_DELTA,
        EventType.BLOCK_COMPLETED,
    }:
        _track_block_event(state, event)
    elif event.type in {
        EventType.TOOL_STARTED,
        EventType.TOOL_UPDATED,
        EventType.TOOL_COMPLETED,
    }:
        _track_tool_event(state, event)


def _track_execution_event(state: _ReplayState, event: DomainEvent) -> None:
    run_id = event.payload.get("run_id")
    if run_id is None:
        scope = event.payload.get("scope", {})
        if isinstance(scope, Mapping):
            run_id = scope.get("run_id")
    if run_id is None:
        return
    key = (event.correlation_id, str(run_id))
    if event.type == EventType.EXECUTION_STARTED:
        state.executions[key] = dict(event.payload)
    else:
        state.executions.pop(key, None)


def _track_assistant_event(state: _ReplayState, event: DomainEvent) -> None:
    correlation_id = event.correlation_id
    if event.type == EventType.ASSISTANT_STARTED:
        state.assistants[correlation_id] = ""
    elif event.type == EventType.ASSISTANT_DELTA:
        state.assistants[correlation_id] = state.assistants.get(
            correlation_id, ""
        ) + str(event.payload.get("delta", ""))
    else:
        state.assistants.pop(correlation_id, None)


def _track_block_event(state: _ReplayState, event: DomainEvent) -> None:
    block_id = str(event.payload.get("block_id", ""))
    key = (event.correlation_id, block_id)
    if event.type == EventType.BLOCK_STARTED:
        state.blocks[key] = dict(event.payload)
    elif event.type == EventType.BLOCK_DELTA:
        payload = state.blocks.setdefault(key, {"block_id": block_id})
        payload["content"] = str(payload.get("content", "")) + str(
            event.payload.get("delta", "")
        )
    else:
        state.blocks.pop(key, None)


def _track_tool_event(state: _ReplayState, event: DomainEvent) -> None:
    tool_call_id = str(event.payload.get("tool_call_id", ""))
    key = (event.correlation_id, tool_call_id)
    if event.type == EventType.TOOL_COMPLETED:
        state.tools.pop(key, None)
    else:
        state.tools[key] = dict(event.payload)


def _append_assistant_markdown(lines: list[str], event: DomainEvent) -> None:
    content = str(event.payload.get("content", ""))
    if content:
        lines.extend(("## Assistant", "", content, ""))


def _append_block_markdown(
    lines: list[str],
    block_content: dict[str, str],
    event: DomainEvent,
) -> None:
    block_id = str(event.payload.get("block_id", ""))
    if event.type == EventType.BLOCK_STARTED:
        block_content[block_id] = str(event.payload.get("content", ""))
    elif event.type == EventType.BLOCK_DELTA:
        block_content[block_id] = block_content.get(block_id, "") + str(
            event.payload.get("delta", "")
        )
    else:
        content = str(event.payload.get("content", block_content.get(block_id, "")))
        if content:
            lines.extend(("### Block", "", content, ""))


def _append_tool_markdown(lines: list[str], event: DomainEvent) -> None:
    name = str(event.payload.get("name", "tool"))
    result = event.payload.get("result", "")
    lines.extend((f"### Tool: `{name}`", "", f"```text\n{result}\n```", ""))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_integer(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("forked_from_sequence must be an integer or null")
    return value
