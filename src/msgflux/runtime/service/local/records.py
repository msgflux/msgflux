"""Private on-disk identity contract for one local AgentService daemon."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

import msgspec


class LocalServiceRecord(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    instance_id: str
    factory: str
    cwd: str
    url: str
    pid: int
    token: str
    version: Literal[1] = 1

    def __post_init__(self) -> None:
        if not self.instance_id or not self.factory or not self.cwd or not self.token:
            raise ValueError("Local service record has an empty required field")
        if self.pid < 1:
            raise ValueError("Local service record PID must be positive")
        if ":" not in self.factory:
            raise ValueError("Factory must use MODULE:CALLABLE syntax")
        module, callable_name = self.factory.split(":", 1)
        if not module or not callable_name or any(c.isspace() for c in self.factory):
            raise ValueError("Factory must use MODULE:CALLABLE syntax")
        if not self.cwd.startswith("/"):
            raise ValueError("Local service cwd must be absolute")
        parsed = urlsplit(self.url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Local service URL has an invalid port") from exc
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or port is None
            or not 1 <= port <= 65535
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Local service URL must use a valid 127.0.0.1 address")

    def __repr__(self) -> str:
        return (
            "LocalServiceRecord("
            f"instance_id={self.instance_id!r}, factory={self.factory!r}, "
            f"cwd={self.cwd!r}, url={self.url!r}, pid={self.pid!r}, "
            "token=<redacted>, version=1)"
        )
