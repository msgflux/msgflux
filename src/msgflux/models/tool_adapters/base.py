"""Minimum contract for provider-owned native tool transports."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar


class ToolTransportAdapter(ABC):
    """Translate tool protocols without executing tools or granting authority.

    Implementations declare their provider/API, versioned codec identity, logical
    tool kind and native call/output item types as class attributes. Instances
    must remain stateless: logical tool routes belong to individual requests.
    Checkpoints persist codec metadata, never adapter instances or import paths.
    """

    provider: ClassVar[str]
    api_mode: ClassVar[str]
    codec: ClassVar[str]
    version: ClassVar[int]
    kind: ClassVar[str]
    item_type: ClassVar[str]
    output_type: ClassVar[str]

    @abstractmethod
    def declaration(self) -> dict[str, Any]:
        """Return the provider's native tool declaration."""

    @abstractmethod
    def supports(self, entry: Any) -> bool:
        """Whether native transport can preserve this catalog entry's inputs."""

    @abstractmethod
    def validate_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Validate codec-specific continuation data; raise on invalid values."""

    @abstractmethod
    def decode(
        self, item: Mapping[str, Any], name: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Decode a complete native call into arguments and serializable metadata."""

    @abstractmethod
    def render(
        self,
        call_id: str,
        result: Any,
        metadata: Mapping[str, Any],
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Encode a canonical result or execution error for provider continuation."""

    @abstractmethod
    def project_history(self, item: Mapping[str, Any]) -> dict[str, Any]:
        """Project a native call/output to portable function-call history."""

    @abstractmethod
    def interrupted(self, item: Mapping[str, Any], reason: str) -> dict[str, Any]:
        """Close an interrupted native call without implying successful execution."""
