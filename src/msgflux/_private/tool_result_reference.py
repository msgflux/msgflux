"""Dependency-light schema shared by runtime storage and typed tool results."""

import re

import msgspec

_RESULT_ID = re.compile(r"res_[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def validate_result_id(result_id: str) -> None:
    if not isinstance(result_id, str) or not _RESULT_ID.fullmatch(result_id):
        raise ValueError("Invalid tool result ID")


class ToolResultRef(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Serializable reference; intentionally contains no filesystem location."""

    result_id: str
    size_bytes: int
    sha256: str
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        validate_result_id(self.result_id)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        if not isinstance(self.sha256, str) or not _DIGEST.fullmatch(self.sha256):
            raise ValueError("Invalid SHA256 digest")
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise ValueError("media_type must be a non-empty string")
        if len(self.media_type) > 256:
            raise ValueError("media_type must not exceed 256 characters")

    @property
    def uri(self) -> str:
        return f"tool-result://{self.result_id}"

    def to_dict(self) -> dict:
        """Return plain data suitable for existing JSON checkpoint stores."""
        return msgspec.to_builtins(self)
