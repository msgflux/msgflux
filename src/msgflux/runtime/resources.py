"""Explicit local application layout; never grants model filesystem access."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from msgflux.runtime.tool_results import LocalToolResultStore

_THREAD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class RuntimeResources:
    """Configure shared plans/results and separate per-thread checkpoints.

    Construction creates no files. Applications explicitly initialize storage;
    Agent creation and execution do not implicitly write into the user's home.
    """

    def __init__(self, root: str | os.PathLike[str] = "~/.msgflux") -> None:
        self.root = Path(root).expanduser().absolute()

    @property
    def plans_path(self) -> Path:
        return self.root / "plans"

    def initialize(self, *, extra_dirs: Iterable[str] = ()) -> RuntimeResources:
        """Create required directories and optional safe top-level directories."""
        if isinstance(extra_dirs, (str, bytes)):
            raise TypeError("extra_dirs must be an iterable of directory names")
        names = tuple(extra_dirs)
        for name in names:
            if not isinstance(name, str) or not _THREAD_ID.fullmatch(name):
                raise ValueError("Extra directory names must be safe single components")
        for name in dict.fromkeys(("tool-results", "threads", *names)):
            (self.root / name).mkdir(mode=0o700, parents=True, exist_ok=True)
        return self

    def tool_result_store(
        self,
        *,
        max_result_bytes: int = 64 * 1024 * 1024,
        max_store_bytes: int = 1024 * 1024 * 1024,
    ) -> LocalToolResultStore:
        return LocalToolResultStore(
            self.root / "tool-results",
            max_result_bytes=max_result_bytes,
            max_store_bytes=max_store_bytes,
        )

    def checkpoint_store(self, thread_id: str):
        """Open an existing SQLite adapter; the caller owns its close lifecycle.

        This path helper does not restrict that adapter to a single thread or
        inject execution scope. Pass the same thread ID when invoking the Agent.
        """
        from msgflux.data.stores import SQLiteCheckpointStore  # noqa: PLC0415

        if not isinstance(thread_id, str) or not _THREAD_ID.fullmatch(thread_id):
            raise ValueError("thread_id must be a safe non-empty directory identifier")
        path = self.root / "threads" / thread_id / "checkpoint.sqlite"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        return SQLiteCheckpointStore(path=str(path))
