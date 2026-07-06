from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, TypeVar, get_args, get_origin

T = TypeVar("T")


class Hidden(Generic[T]):
    """Type marker for parameters hidden from the model-facing tool schema."""


def is_hidden_annotation(annotation: Any) -> bool:
    """Return whether an annotation is a `Hidden[...]` marker."""
    return annotation is Hidden or get_origin(annotation) is Hidden


def unwrap_hidden_annotation(annotation: Any) -> Any | None:
    """Return the wrapped type from `Hidden[T]`, or Any for bare `Hidden`."""
    if not is_hidden_annotation(annotation):
        return None
    if annotation is Hidden:
        return Any
    args = get_args(annotation)
    return args[0] if args else Any


@dataclass
class ToolMetadata:
    """Normalized metadata extracted from a Python callable tool."""

    name: str
    description: str
    annotations: Dict[str, Any]
    tool_config: Dict[str, Any]
    impl: Callable
    display_name: str | None = None
    usage_guidance: str | None = None
    source_tool: Any | None = None


class ToolBucket:
    """Base class for tools that absorb other tools by kind."""

    tool_kind = "bucket"
    capture_kind: str

    def add(self, tool: ToolMetadata) -> None:
        raise NotImplementedError


class ToolLibraryOperator:
    """Base class for runtime tools that operate through ToolLibraryHandle."""

    tool_kind = "runtime"
