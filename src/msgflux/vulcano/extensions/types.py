from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Mapping, Protocol

from msgflux.vulcano.events import DomainEvent

if TYPE_CHECKING:
    from msgflux.vulcano.permissions import ExtensionPermissionApi
    from msgflux.vulcano.ui import ExtensionUiApi

__all__ = [
    "EXTENSION_API_VERSION",
    "EXTENSION_ENTRY_POINT_GROUP",
    "ExtensionCleanup",
    "ExtensionContext",
    "ExtensionControl",
    "ExtensionDiagnostic",
    "ExtensionInfo",
    "ExtensionLoadReport",
    "ExtensionObserver",
    "ExtensionReloadReport",
    "ExtensionSettings",
    "ExtensionSource",
]


EXTENSION_API_VERSION = 1
EXTENSION_ENTRY_POINT_GROUP = "msgflux.vulcano.extensions"


@dataclass(frozen=True)
class ExtensionSource:
    kind: str
    identifier: str
    priority: int
    trusted: bool
    path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "identifier": self.identifier,
            "priority": self.priority,
            "trusted": self.trusted,
            "path": str(self.path) if self.path is not None else None,
        }


@dataclass(frozen=True)
class ExtensionSettings:
    cwd: Path
    explicit_paths: tuple[Path, ...] = ()
    enabled: bool = True
    auto_discover: bool = True
    trust_project: bool = False
    user_directory: Path | None = None

    @property
    def resolved_user_directory(self) -> Path:
        return self.user_directory or Path.home() / ".vulcano" / "extensions"


@dataclass(frozen=True)
class ExtensionContext:
    cwd: Path
    generation: int
    source: ExtensionSource
    ui: ExtensionUiApi
    permissions: ExtensionPermissionApi
    services: Mapping[str, object] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return self.ui.mode

    @property
    def has_ui(self) -> bool:
        return self.ui.available


@dataclass(frozen=True)
class ExtensionInfo:
    name: str
    api_version: int
    source: ExtensionSource
    generation: int
    state: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "api_version": self.api_version,
            "source": self.source.to_dict(),
            "generation": self.generation,
            "state": self.state,
            "error": self.error,
        }


@dataclass(frozen=True)
class ExtensionDiagnostic:
    extension: str
    phase: str
    message: str
    source: ExtensionSource

    def to_dict(self) -> dict[str, object]:
        return {
            "extension": self.extension,
            "phase": self.phase,
            "message": self.message,
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True)
class ExtensionLoadReport:
    loaded: tuple[ExtensionInfo, ...] = ()
    failed: tuple[ExtensionInfo, ...] = ()
    diagnostics: tuple[ExtensionDiagnostic, ...] = ()


@dataclass(frozen=True)
class ExtensionReloadReport:
    unloaded: tuple[ExtensionInfo, ...] = ()
    loaded: tuple[ExtensionInfo, ...] = ()
    failed: tuple[ExtensionInfo, ...] = ()
    diagnostics: tuple[ExtensionDiagnostic, ...] = ()


ExtensionCleanup = Callable[[], None | Awaitable[None]]
ExtensionObserver = Callable[
    [DomainEvent, ExtensionContext],
    None | Awaitable[None],
]


class ExtensionControl(Protocol):
    @property
    def enabled(self) -> bool: ...

    @property
    def records(self) -> tuple[ExtensionInfo, ...]: ...

    @property
    def diagnostics(self) -> tuple[ExtensionDiagnostic, ...]: ...

    async def reload(self) -> ExtensionReloadReport: ...
