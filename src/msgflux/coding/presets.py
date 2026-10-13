"""Builtin instructions and defaults for configurable coding agents."""

from __future__ import annotations

import msgspec


class CodingPreset(msgspec.Struct, frozen=True):
    description: str
    instructions: str
    permissions: tuple[str, ...] = ("read", "write", "execute")
    active: tuple[str, ...] = ("workspace",)
    deferred: tuple[str, ...] = ()


_PRESETS = {
    "explorer": CodingPreset(
        description=(
            "Locate relevant code, trace dependencies, and report file references."
        ),
        permissions=("read",),
        instructions=(
            "Investigate the requested question without changing files. Start with "
            "targeted searches, then read the relevant definitions and their callers. "
            "Follow repository instructions. Distinguish observed behavior from "
            "inference; identify missing evidence. Return a concise explanation "
            "with file paths and line references, relevant dependencies, and "
            "concrete findings useful to the caller. Avoid copying large files "
            "or suggesting unrelated changes."
        ),
    ),
}


def get_coding_preset(name: str) -> CodingPreset:
    """Resolve a builtin preset without constructing models or tools."""
    try:
        return _PRESETS[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unknown coding preset {name!r}") from error
