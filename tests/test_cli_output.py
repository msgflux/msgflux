import json

import pytest

from msgflux._private.cli_output import format_observation, write_observation
from msgflux.runtime.service.records import EventRecord, SnapshotRecord
from msgflux.utils.msgspec import msgspec_dumps


def event(event_type, data, source_path=()):
    return EventRecord(
        type=event_type,
        timestamp="2026-10-21T12:00:00Z",
        data=data,
        run_id="run-example",
        source_path=source_path,
    )


def test_human_tool_output_keeps_command_and_multiline_result(capsys):
    record = event(
        "tool.end",
        {
            "tool_name": "bash",
            "arguments": {"command": "printf 'alpha\\nbeta\\n'"},
            "result": {"returncode": 0, "stdout": "alpha\nbeta", "stderr": ""},
            "error": None,
        },
        source_path=("agent:main", "tool:bash"),
    )

    write_observation("event", record, as_json=False)

    output = capsys.readouterr().out
    assert "[tool.end agent:main/tool:bash]" in output
    assert "command: printf 'alpha\\nbeta\\n'" in output
    assert "returncode: 0" in output
    assert "stdout:\n      alpha\n      beta" in output
    assert "error: null" in output
    assert '"tool_name"' not in output


def test_human_error_and_list_payloads_keep_false_and_empty_values():
    record = event(
        "tool.permission_denied",
        {
            "error": "Write permission required",
            "resources": [{"path": "hello.txt", "allowed": False}],
            "details": {},
            "alternatives": [],
        },
    )

    output = format_observation("event", record, as_json=False)

    assert "error: Write permission required" in output
    assert "- path: hello.txt\n      allowed: false" in output
    assert "details:\n    {}" in output
    assert "alternatives:\n    []" in output


@pytest.mark.parametrize("event_type", ["message.delta", "commentary.delta"])
def test_human_deltas_keep_their_existing_event_label(event_type):
    record = event(event_type, {"delta": "Working on it."})

    assert format_observation("event", record, as_json=False) == (
        f"[{event_type}] Working on it."
    )


@pytest.mark.parametrize(
    "kind,record",
    [
        ("snapshot", SnapshotRecord(thread_id="thread-example")),
        (
            "event",
            event("tool.end", {"result": "alpha\nbeta", "error": None}),
        ),
    ],
)
def test_machine_output_retains_the_versioned_ndjson_envelope(kind, record, capsys):
    expected = msgspec_dumps({"type": kind, kind: record, "version": 1})

    write_observation(kind, record, as_json=True)

    output = capsys.readouterr().out
    assert output == expected + "\n"
    assert len(output.splitlines()) == 1
    assert json.loads(output)["version"] == 1


def test_human_snapshot_and_empty_unknown_event_are_readable():
    snapshot = SnapshotRecord(thread_id="thread-example")

    assert format_observation("snapshot", snapshot, as_json=False) == (
        "[snapshot] thread_id=thread-example"
    )
    assert format_observation("event", event("custom.event", {}), as_json=False) == (
        "[custom.event]"
    )
