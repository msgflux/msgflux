"""Builtin instructions and defaults for configurable coding agents."""

from __future__ import annotations

import msgspec


class CodingPreset(msgspec.Struct, frozen=True):
    description: str
    instructions: str
    delegation_guidance: str = ""
    permissions: tuple[str, ...] = ("read", "write", "execute")
    active: tuple[str, ...] = ("workspace",)
    deferred: tuple[str, ...] = ()


_PRESETS = {
    "explorer": CodingPreset(
        description=(
            "Locate relevant code, trace dependencies, research the web, and report "
            "concise findings with file references and source URLs."
        ),
        permissions=("read",),
        active=("workspace", "web_fetch", "web_search"),
        delegation_guidance=(
            "Delegate repository searches, dependency tracing, and web research "
            "to this subagent. Give it a focused question and use its concise "
            "report to preserve your context instead of collecting raw search "
            "results yourself."
        ),
        instructions=(
            "Investigate the requested question without changing files. Start with "
            "targeted searches, then read the relevant definitions and their callers. "
            "Follow repository instructions. Distinguish observed behavior from "
            "inference; identify missing evidence. Return a concise explanation "
            "with file paths and line references, relevant dependencies, and "
            "concrete findings useful to the caller. Avoid copying large files "
            "or suggesting unrelated changes. Use web_fetch for a known URL and "
            "web_search for broader or current external information. "
            "Include source URLs for web findings. Return the useful conclusions "
            "and evidence to the caller; keep raw pages and search output in your "
            "own context. If a source cannot be accessed, report that limitation "
            "without presenting guesses as verified facts."
        ),
    ),
    "builder": CodingPreset(
        description=(
            "Implement focused code changes and verify them with local checks."
        ),
        delegation_guidance=(
            "Delegate focused implementation and verification tasks to this "
            "subagent when useful. Supply the requirements, relevant research "
            "findings, and constraints; use its changed-file and validation report."
        ),
        instructions=(
            "Implement the caller's requested change within the agreed scope. "
            "Follow repository instructions and inspect the relevant definitions "
            "before editing. Use the editing tools supplied for the active model. "
            "Use Bash for focused local searches and verification when available. "
            "Ask the caller for broader exploration or web research when needed. "
            "Respect effective workspace permissions: if writing or execution is "
            "unavailable, report the limitation instead of claiming edits or checks. "
            "Return a concise report of changed files, checks run, and remaining "
            "limitations."
        ),
    ),
}


def get_coding_preset(name: str) -> CodingPreset:
    """Resolve a builtin preset without constructing models or tools."""
    try:
        return _PRESETS[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unknown coding preset {name!r}") from error
