"""Shared rendering for command-line service observations."""

import sys

from msgflux.utils.msgspec import msgspec_dumps


def write_observation(kind: str, record, *, as_json: bool) -> None:
    if as_json:
        line = msgspec_dumps({"type": kind, kind: record, "version": 1})
    elif kind == "event":
        content = record.data.get("delta")
        if not isinstance(content, str):
            content = msgspec_dumps(record.data)
        source = "/".join(record.source_path)
        label = f"{record.type} {source}" if source else record.type
        line = f"[{label}] {content}"
    else:
        line = f"[snapshot] thread_id={record.thread_id}"
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
