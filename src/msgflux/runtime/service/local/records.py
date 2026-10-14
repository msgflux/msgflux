"""Private on-disk identity contract for one local AgentService daemon."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

import msgspec

from msgflux.runtime.service.local.manifest import validate_factory_path


class LocalServiceStatus(msgspec.Struct, frozen=True):
    """Verified daemon identity without authentication material."""

    status: Literal["running", "stopped", "unreachable"]
    runtime_dir: str
    url: str | None = None
    instance_id: str | None = None
    pid: int | None = None


class LocalServiceRecord(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    instance_id: str
    cwd: str
    url: str
    pid: int
    token: str
    version: Literal[1] = 1
    factory: str | None = None
    config_path: str | None = None
    config_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.instance_id or not self.cwd or not self.token:
            raise ValueError("Local service record has an empty required field")
        if self.pid < 1:
            raise ValueError("Local service record PID must be positive")
        factory_source = self.factory is not None
        config_source = self.config_path is not None or self.config_digest is not None
        if factory_source == config_source:
            raise ValueError("Local service record must have exactly one source")
        if factory_source:
            validate_factory_path(self.factory)
        elif (
            not self.config_path
            or not self.config_path.startswith("/")
            or not self.config_digest
            or len(self.config_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.config_digest)
        ):
            raise ValueError(
                "Config source requires an absolute path and SHA-256 digest"
            )
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
            f"config_path={self.config_path!r}, config_digest={self.config_digest!r}, "
            f"cwd={self.cwd!r}, url={self.url!r}, pid={self.pid!r}, "
            "token=<redacted>, version=1)"
        )
