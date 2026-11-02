"""The snapshot encoding path preserves the portable observation contract."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from uuid import UUID

import msgspec
import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.runtime.event_hub import (
    BackgroundTaskSnapshot,
    LiveRunSnapshot,
    RunningToolSnapshot,
    ThreadSnapshot,
)
from msgflux.runtime.service.records import SnapshotRecord
from msgflux.runtime.service.serialization import (
    encode_json,
    encode_snapshot,
    snapshot_record,
)


class _Mode(Enum):
    ACTIVE = "active"


class _Tagged(msgspec.Struct, tag="example", rename="camel"):
    field_name: str


@pytest.mark.parametrize("history", [None, "", "olá\n世界", "history " * 150_000])
def test_snapshot_encoding_preserves_empty_unicode_and_large_histories(history):
    messages = None
    if history is not None:
        messages = ChatMessages()
        messages.add_user(history)
    snapshot = ThreadSnapshot("thread", namespace="agent", messages=messages)

    encoded = encode_snapshot(snapshot)
    assert encoded == encode_json(snapshot_record(snapshot))
    restored = msgspec.json.decode(encoded, type=SnapshotRecord)
    assert restored.thread_id == "thread" and restored.version == 1
    assert (restored.messages is None) == (messages is None)
    if messages is not None:
        assert restored.messages[0]["content"] == history


@pytest.mark.parametrize(
    "value",
    [
        (1, [2, (3, None)]),
        b"\x00\xff",
        bytearray(b"binary"),
        {1: "first", "1": "last"},
        {"values": [float("inf"), float("-inf"), float("nan")]},
        datetime(2026, 11, 1, tzinfo=timezone.utc),
        Decimal("1.25"),
        UUID(int=42),
        _Mode.ACTIVE,
        _Tagged("tagged value"),
    ],
)
def test_snapshot_encoding_preserves_nested_wire_representations(value):
    snapshot = ThreadSnapshot(
        "thread",
        active_runs=(
            LiveRunSnapshot("run", ("child",), streaming_message="in progress"),
        ),
        running_tools=(
            RunningToolSnapshot("run", (), "call", "tool", {"value": value}),
        ),
        background_tasks=(
            BackgroundTaskSnapshot("task", "tool", "running", {"value": value}),
        ),
    )

    encoded = encode_snapshot(snapshot)
    assert encoded == encode_json(snapshot_record(snapshot))
    restored = msgspec.json.decode(encoded, type=SnapshotRecord)
    assert restored.active_runs[0]["source_path"] == ["child"]
    assert (
        restored.running_tools[0]["arguments"]
        == restored.background_tasks[0]["progress"]
    )


def test_snapshot_encoding_keeps_source_containers_unchanged():
    arguments = {"values": [1, {"nested": (2, 3)}]}
    snapshot = ThreadSnapshot(
        "thread",
        running_tools=(RunningToolSnapshot("run", (), "call", "tool", arguments),),
    )
    first = encode_snapshot(snapshot)
    assert arguments == {"values": [1, {"nested": (2, 3)}]}
    arguments["values"].append(4)
    assert first == encode_json(
        snapshot_record(
            ThreadSnapshot(
                "thread",
                running_tools=(
                    RunningToolSnapshot(
                        "run", (), "call", "tool", {"values": [1, {"nested": (2, 3)}]}
                    ),
                ),
            )
        )
    )
    assert encode_snapshot(snapshot) != first


def test_snapshot_encoding_rejects_unsupported_values():
    snapshot = ThreadSnapshot(
        "thread",
        running_tools=(
            RunningToolSnapshot("run", (), "call", "tool", {"value": object()}),
        ),
    )
    with pytest.raises(TypeError, match="Unsupported service wire type"):
        snapshot_record(snapshot)
    with pytest.raises(TypeError, match="Unsupported service wire type"):
        encode_snapshot(snapshot)
