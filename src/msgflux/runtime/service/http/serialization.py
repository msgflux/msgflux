"""Convert runtime presentation state to portable JSON wire records."""

from typing import Any

import msgspec

from msgflux.chat_messages import ChatMessages
from msgflux.runtime.event_hub import ThreadSnapshot
from msgflux.runtime.events import ExecutionEvent
from msgflux.runtime.service.http.records import EventRecord, SnapshotRecord


def _encode_custom(value: Any) -> Any:
    if isinstance(value, ChatMessages):
        return value.to_chatml()
    raise TypeError(f"Unsupported service wire type: {type(value).__name__}")


def encode_json(value: Any) -> bytes:
    return msgspec.json.encode(value, enc_hook=_encode_custom)


def _portable(value: Any) -> Any:
    return msgspec.json.decode(encode_json(value))


def snapshot_record(snapshot: ThreadSnapshot) -> SnapshotRecord:
    messages = _portable(snapshot.messages)
    return SnapshotRecord(
        thread_id=snapshot.thread_id,
        namespace=snapshot.namespace,
        messages=tuple(messages) if messages is not None else None,
        active_runs=tuple(_portable(snapshot.active_runs)),
        running_tools=tuple(_portable(snapshot.running_tools)),
        background_tasks=tuple(_portable(snapshot.background_tasks)),
        approvals=tuple(_portable(snapshot.approvals)),
    )


def event_record(event: ExecutionEvent) -> EventRecord:
    return EventRecord(
        type=event.type,
        timestamp=event.timestamp,
        data=_portable(event.data),
        run_id=event.run_id,
        source_path=event.source_path,
    )
