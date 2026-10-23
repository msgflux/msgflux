"""Shared rendering for command-line service observations."""

import sys
from collections.abc import Mapping

from msgflux.utils.msgspec import msgspec_dumps

_CONTROL_ESCAPES = {"\n": r"\n", "\r": r"\r", "\t": r"\t"}


def escape_human_text(value: str) -> str:
    """Render terminal controls literally so human output stays inert."""
    return "".join(
        _CONTROL_ESCAPES.get(character)
        or (
            f"\\x{ord(character):02x}"
            if ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F
            else character
        )
        for character in value
    )


def _human_lines(value):
    if isinstance(value, Mapping) and value:
        for key, item in value.items():
            if isinstance(item, (Mapping, list, tuple)) or (
                isinstance(item, str) and "\n" in item
            ):
                yield f"{key}:"
                yield from (f"  {line}" for line in _human_lines(item))
            else:
                content = item if isinstance(item, str) else msgspec_dumps(item)
                yield f"{key}: {content}"
    elif isinstance(value, (list, tuple)) and value:
        for item in value:
            lines = iter(_human_lines(item))
            yield f"- {next(lines)}"
            yield from (f"  {line}" for line in lines)
    elif isinstance(value, str):
        yield from value.split("\n")
    else:
        yield msgspec_dumps(value)


def format_observation(kind: str, record, *, as_json: bool) -> str:
    """Format one observation for people or the versioned NDJSON contract."""
    if as_json:
        return msgspec_dumps({"type": kind, kind: record, "version": 1})
    if kind == "event":
        content = record.data.get("delta")
        source = "/".join(record.source_path)
        label = f"{record.type} {source}" if source else record.type
        if isinstance(content, str):
            return f"[{label}] {content}"
        details = "\n".join(f"  {line}" for line in _human_lines(record.data))
        return f"[{label}]\n{details}" if record.data else f"[{label}]"
    return f"[snapshot] thread_id={record.thread_id}"


def write_observation(kind: str, record, *, as_json: bool) -> None:
    line = format_observation(kind, record, as_json=as_json)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
