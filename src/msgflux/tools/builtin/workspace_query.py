"""Bounded, authorization-aware workspace navigation tools."""

from __future__ import annotations

import asyncio
import math
import time
from fnmatch import fnmatchcase
from functools import cache
from io import StringIO
from typing import Any

import msgspec

from msgflux.runtime.workspace import WorkspaceFilesystem, workspace_path
from msgflux.tools.builtin.workspace import _tool_path
from msgflux.tools.config import tool_config
from msgflux.tools.types import Hidden

_IGNORE_LIMIT = 64 * 1024
_GIT_DIR = ".git"


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class _IgnoreRule(msgspec.Struct, frozen=True, kw_only=True):
    base: str
    spec: Any


def _load_ignore(filesystem: WorkspaceFilesystem, directory: str) -> _IgnoreRule | None:
    try:
        prefix = filesystem.read_prefix(
            f"{directory.rstrip('/')}/.gitignore", max_bytes=_IGNORE_LIMIT + 1
        )
    except FileNotFoundError:
        return None
    if len(prefix) > _IGNORE_LIMIT:
        raise ValueError(".gitignore exceeds the bounded ignore-file limit")
    try:
        from pathspec.gitignore import GitIgnoreSpec  # noqa: PLC0415
    except ImportError:
        raise RuntimeError(
            "Workspace navigation requires the optional `pathspec` package"
        ) from None
    return _IgnoreRule(
        base=directory,
        spec=GitIgnoreSpec.from_lines(prefix.decode("utf-8").splitlines()),
    )


def _ignored(path: str, rules: tuple[_IgnoreRule, ...], *, directory: bool) -> bool:
    ignored = False
    for rule in rules:
        relative = path[len(rule.base) :].lstrip("/")
        result = rule.spec.check_file(relative + ("/" if directory else ""))
        if result.include is not None:
            ignored = result.include
    return ignored


def _walk(  # noqa: C901
    filesystem: WorkspaceFilesystem,
    root: str,
    *,
    max_depth: int,
    max_nodes: int,
    deadline: float | None = None,
):
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer")
    _positive(max_nodes, "max_nodes")
    seen = 0

    def visit(  # noqa: C901
        directory: str, depth: int, rules: tuple[_IgnoreRule, ...]
    ):
        nonlocal seen
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("workspace traversal exceeded its time limit")
        local_rules = rules
        entries = filesystem.scandir(directory, max_entries=max(1, max_nodes - seen))
        ignore_entries = [item for item in entries if item.name == ".gitignore"]
        if ignore_entries:
            if ignore_entries[0].kind != "file":
                raise PermissionError("Workspace .gitignore is not a regular file")
            rule = _load_ignore(filesystem, directory)
            if rule is not None:
                local_rules = (*rules, rule)
        for entry in entries:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("workspace traversal exceeded its time limit")
            child = f"{directory.rstrip('/')}/{entry.name}"
            if entry.name == _GIT_DIR:
                continue
            if _ignored(child, local_rules, directory=entry.kind == "directory"):
                continue
            seen += 1
            if seen > max_nodes:
                raise ValueError("Workspace traversal exceeds max_nodes")
            yield child, entry
            if entry.kind == "directory":
                if depth >= max_depth:
                    raise ValueError("Workspace traversal exceeds max_depth")
                yield from visit(child, depth + 1, local_rules)

    yield from visit(root, 0, ())


def _glob_match(pattern: str, value: str) -> bool:
    if (
        not isinstance(pattern, str)
        or not pattern
        or "\x00" in pattern
        or "\\" in pattern
    ):
        raise ValueError("pattern must be non-empty virtual POSIX text")
    absolute = pattern.startswith("/")
    if value.startswith("/") != absolute:
        return False
    parts = tuple(part for part in pattern.strip("/").split("/") if part)
    values = tuple(part for part in value.strip("/").split("/") if part)

    @cache
    def match(pattern_index: int, value_index: int) -> bool:
        if pattern_index == len(parts):
            return value_index == len(values)
        part = parts[pattern_index]
        if part == "**":
            return match(pattern_index + 1, value_index) or (
                value_index < len(values) and match(pattern_index, value_index + 1)
            )
        return (
            value_index < len(values)
            and fnmatchcase(values[value_index], part)
            and match(pattern_index + 1, value_index + 1)
        )

    return match(0, 0)


def _relative(root: str, path: str) -> str:
    value = path[len(root) :].lstrip("/")
    return value or "."


class _WorkspaceQuery:
    def __init__(
        self,
        *,
        cwd: str,
        max_depth: int,
        max_nodes: int,
        max_results: int,
        max_seconds: float = 5.0,
    ):
        self.cwd = workspace_path(cwd)
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        _positive(max_nodes, "max_nodes")
        _positive(max_results, "max_results")
        if (
            type(max_seconds) not in (int, float)
            or not math.isfinite(max_seconds)
            or max_seconds <= 0
        ):
            raise ValueError("max_seconds must be positive")
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.max_results = max_results
        self.max_seconds = float(max_seconds)


@tool_config(runtime_inputs=["filesystem"], retry=False)
class LsTool(_WorkspaceQuery):
    """List one authorized directory without exposing host paths.

    Args:
        path: Virtual directory path, relative to the configured working directory.
    """

    name = "ls"
    display_name = "List workspace"
    annotations = {"path": str, "return": dict[str, Any]}

    def __init__(self, *, cwd: str = "/", max_entries: int = 10_000):
        self.cwd = workspace_path(cwd)
        _positive(max_entries, "max_entries")
        self.max_entries = max_entries

    def __call__(self, path: str = ".", *, filesystem: Hidden[WorkspaceFilesystem]):
        target = _tool_path(path, self.cwd)
        entries = filesystem.scandir(target, max_entries=self.max_entries)
        return {
            "path": target,
            "entries": [{"name": item.name, "kind": item.kind} for item in entries],
        }

    async def acall(self, path: str = ".", *, filesystem: Hidden[WorkspaceFilesystem]):
        return await asyncio.to_thread(self, path, filesystem=filesystem)


@tool_config(runtime_inputs=["filesystem"], retry=False)
class GlobTool(_WorkspaceQuery):
    """Find bounded virtual paths using shell-independent glob matching.

    Args:
        pattern: Virtual glob pattern; ``*`` does not cross a path separator.
        path: Search root, relative to the configured working directory.
    """

    name = "glob"
    display_name = "Find workspace paths"
    annotations = {"pattern": str, "path": str, "return": dict[str, Any]}

    def __init__(
        self,
        *,
        cwd: str = "/",
        max_depth: int = 32,
        max_nodes: int = 10_000,
        max_results: int = 1_000,
        max_seconds: float = 5.0,
    ):
        super().__init__(
            cwd=cwd,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_results=max_results,
            max_seconds=max_seconds,
        )

    def __call__(
        self,
        pattern: str,
        path: str = ".",
        *,
        filesystem: Hidden[WorkspaceFilesystem],
    ):
        if (
            not isinstance(pattern, str)
            or not pattern
            or "\x00" in pattern
            or "\\" in pattern
            or len(pattern) > 4096
            or pattern.count("/") > 256
        ):
            raise ValueError("pattern exceeds bounded complexity")
        root = _tool_path(path, self.cwd)
        matches = []
        deadline = time.monotonic() + self.max_seconds
        for candidate, entry in _walk(
            filesystem,
            root,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
            deadline=deadline,
        ):
            if time.monotonic() > deadline:
                raise TimeoutError("glob exceeded its time limit")
            value = candidate if pattern.startswith("/") else _relative(root, candidate)
            if _glob_match(pattern, value):
                if entry.kind == "other":
                    continue
                matches.append({"path": candidate, "kind": entry.kind})
                if len(matches) >= self.max_results:
                    return {"matches": matches, "truncated": True}
        return {"matches": matches, "truncated": False}

    async def acall(self, pattern, path=".", *, filesystem):
        return await asyncio.to_thread(self, pattern, path, filesystem=filesystem)


@tool_config(runtime_inputs=["filesystem"], retry=False)
class GrepTool(_WorkspaceQuery):
    """Search bounded UTF-8 workspace files with a timed regex engine.

    Args:
        pattern: Regular expression to search for.
        path: Search root, relative to the configured working directory.
    """

    name = "grep"
    display_name = "Search workspace text"
    annotations = {"pattern": str, "path": str, "return": dict[str, Any]}

    def __init__(
        self,
        *,
        cwd: str = "/",
        max_depth: int = 32,
        max_nodes: int = 10_000,
        max_results: int = 1_000,
        max_file_bytes: int = 1_000_000,
        max_output_bytes: int = 1_000_000,
        max_seconds: float = 5.0,
    ):
        super().__init__(
            cwd=cwd,
            max_depth=max_depth,
            max_nodes=max_nodes,
            max_results=max_results,
            max_seconds=max_seconds,
        )
        _positive(max_file_bytes, "max_file_bytes")
        _positive(max_output_bytes, "max_output_bytes")
        self.max_file_bytes = max_file_bytes
        self.max_output_bytes = max_output_bytes

    def __call__(  # noqa: C901
        self,
        pattern: str,
        path: str = ".",
        *,
        filesystem: Hidden[WorkspaceFilesystem],
    ):
        if not isinstance(pattern, str) or "\x00" in pattern or len(pattern) > 4096:
            raise ValueError("pattern exceeds bounded complexity")
        try:
            import regex  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "GrepTool requires the optional `regex` package"
            ) from exc
        try:
            compiled = regex.compile(pattern)
        except Exception as exc:
            raise ValueError("Invalid grep pattern") from exc
        root = _tool_path(path, self.cwd)
        matches, skipped = [], []
        output_bytes = 0

        def append_record(records, record):
            nonlocal output_bytes
            size = len(msgspec.json.encode(record))
            if output_bytes + size > self.max_output_bytes:
                return False
            records.append(record)
            output_bytes += size
            return True

        deadline = time.monotonic() + self.max_seconds
        for candidate, entry in _walk(
            filesystem,
            root,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
            deadline=deadline,
        ):
            if time.monotonic() > deadline:
                raise TimeoutError("grep exceeded its time limit")
            if entry.kind != "file":
                continue
            data = filesystem.read_prefix(candidate, max_bytes=self.max_file_bytes + 1)
            reason = None
            if len(data) > self.max_file_bytes:
                reason = "oversized"
            elif b"\x00" in data:
                reason = "binary"
            else:
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    reason = "binary"
            if reason is not None:
                if not append_record(skipped, {"path": candidate, "reason": reason}):
                    return {"matches": matches, "skipped": skipped, "truncated": True}
                continue
            for line_number, raw_line in enumerate(StringIO(text, newline=None), 1):
                if time.monotonic() > deadline:
                    raise TimeoutError("grep exceeded its time limit")
                line = raw_line.rstrip("\r\n")
                try:
                    found = compiled.search(line, timeout=0.05)
                except TimeoutError as exc:
                    raise ValueError("grep pattern exceeded its time limit") from exc
                if found:
                    item = {
                        "path": candidate,
                        "line": line_number,
                        "text": line[:4096],
                        "truncated": len(line) > 4096,
                    }
                    if not append_record(matches, item):
                        return {
                            "matches": matches,
                            "skipped": skipped,
                            "truncated": True,
                        }
                    if len(matches) >= self.max_results:
                        return {
                            "matches": matches,
                            "skipped": skipped,
                            "truncated": True,
                        }
        return {"matches": matches, "skipped": skipped, "truncated": False}

    async def acall(self, pattern, path=".", *, filesystem):
        return await asyncio.to_thread(self, pattern, path, filesystem=filesystem)


__all__ = ["GlobTool", "GrepTool", "LsTool"]
