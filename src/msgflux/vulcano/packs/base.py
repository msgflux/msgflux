from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from msgflux.vulcano.extensions.api import ExtensionApi

__all__ = [
    "ExtensionPack",
    "ExtensionPackRegistration",
    "validate_pack_name",
]


_VALID_PACK_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


class _Registration(Protocol):
    def remove(self) -> None: ...


class ExtensionPack(Protocol):
    """Reusable group of capabilities installed through one ExtensionApi."""

    name: str

    def setup(self, api: ExtensionApi) -> None: ...


class ExtensionPackRegistration:
    """Atomic removal handle for one installed extension pack."""

    def __init__(
        self,
        name: str,
        registrations: tuple[_Registration, ...],
        on_remove: Callable[[ExtensionPackRegistration], None],
    ) -> None:
        self.name = name
        self._registrations = registrations
        self._on_remove: Callable[[ExtensionPackRegistration], None] | None = on_remove

    @property
    def active(self) -> bool:
        return self._on_remove is not None

    def remove(self) -> None:
        callback = self._on_remove
        if callback is None:
            return
        self._on_remove = None
        errors: list[str] = []
        for registration in reversed(self._registrations):
            try:
                registration.remove()
            except Exception as error:
                errors.append(str(error))
        self._registrations = ()
        callback(self)
        if errors:
            raise RuntimeError("; ".join(errors))

    def __enter__(self) -> ExtensionPackRegistration:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.remove()


def validate_pack_name(name: object) -> str:
    if not isinstance(name, str) or not _VALID_PACK_NAME.fullmatch(name):
        raise ValueError(
            "Extension pack names must be lowercase identifiers containing "
            "letters, numbers, underscores, or hyphens"
        )
    return name
