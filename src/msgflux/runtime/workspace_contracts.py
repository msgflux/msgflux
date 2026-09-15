"""Serializable resource identity and declared single-file write guarantees."""

from typing import Literal

import msgspec

WriteGuarantee = Literal["cooperative_compare", "atomic_compare"]


class WorkspacePromptInfo(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Trusted model-facing description, never authority or discovered file data.

    Backends may override the default. Do not include credentials, private host
    paths or claims of isolation not enforced by the implementation.
    """

    storage: str = "unspecified"
    guidance: str = ""

    def __post_init__(self):
        if not isinstance(self.storage, str) or not self.storage.strip():
            raise ValueError("storage must be non-empty text")
        if not isinstance(self.guidance, str):
            raise TypeError("guidance must be text")


class WorkspaceIdentity(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Host-resolved identity, not authority or a recipe for opening a resource.

    A backend must change generation when replacing a resource, and config_revision
    when its mount or security configuration changes. Never put credentials here.
    """

    backend: str
    resource_id: str
    generation: str
    config_revision: str = "1"

    def __post_init__(self):
        for value in (
            self.backend,
            self.resource_id,
            self.generation,
            self.config_revision,
        ):
            if (
                not isinstance(value, str)
                or not value
                or any(ord(char) < 32 for char in value)
            ):
                raise ValueError("Workspace identity fields must be non-empty strings")


class WorkspaceWriteCapabilities(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Trusted declarations, not proof of OS isolation or crash durability.

    atomic_replace alone does not imply compare-and-write safety. Cooperative
    comparison coordinates backend participants, not arbitrary external writers.
    Atomic comparison covers all writers admitted by the backend's resource model.
    None of these declarations promises a multi-file transaction.
    """

    atomic_replace: bool = False
    cooperative_compare: bool = False
    atomic_compare: bool = False

    def __post_init__(self):
        if any(
            type(value) is not bool
            for value in (
                self.atomic_replace,
                self.cooperative_compare,
                self.atomic_compare,
            )
        ):
            raise TypeError("Write capabilities must be booleans")

    def require(self, guarantee: WriteGuarantee) -> None:
        if guarantee not in ("cooperative_compare", "atomic_compare"):
            raise ValueError("Unknown workspace write guarantee")
        supported = self.atomic_compare or (
            guarantee == "cooperative_compare" and self.cooperative_compare
        )
        if not supported:
            raise NotImplementedError(f"Workspace does not support {guarantee}")
