from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Container,
    Dict,
    Generic,
    Iterator,
    Mapping,
    MutableMapping,
    TypeVar,
    get_args,
    get_origin,
)

from msgflux.tools.helpers import BACKGROUND_TASK_TOOL_KIND, is_background_capable

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

    @classmethod
    def find_bucket_for_metadata(
        cls,
        metadata: ToolMetadata,
        bucket_names_by_capture_kind: Mapping[str, str],
        *,
        reserved_tool_kinds: Container[str],
    ) -> str | None:
        tool_kind = metadata.tool_config.get("tool_kind", "tool")
        if tool_kind == cls.tool_kind or tool_kind in reserved_tool_kinds:
            return None
        return bucket_names_by_capture_kind.get(tool_kind)

    @classmethod
    def validate_registration(
        cls,
        metadata: ToolMetadata,
        bucket_names_by_capture_kind: Mapping[str, str],
    ) -> str | None:
        if metadata.tool_config.get("tool_kind") != cls.tool_kind:
            return None
        capture_kind = cls.require_capture_kind(metadata.name, metadata.impl)
        existing = bucket_names_by_capture_kind.get(capture_kind)
        if existing is not None and existing != metadata.name:
            raise ValueError(
                f"The bucket capture kind `{capture_kind}` is already handled by "
                f"`{existing}`."
            )
        return capture_kind

    @staticmethod
    def require_capture_kind(bucket_name: str, bucket_impl: Any) -> str:
        capture_kind = getattr(bucket_impl, "capture_kind", None)
        if not isinstance(capture_kind, str) or not capture_kind:
            raise ValueError(
                f"The bucket tool `{bucket_name}` must define capture_kind."
            )
        return capture_kind

    @staticmethod
    def validate_capture(bucket_name: str, metadata: ToolMetadata) -> None:
        if is_background_capable(metadata.tool_config):
            raise ValueError(
                "Bucket-captured tools cannot use `background=True` or "
                f"`allow_background=True`. Tool `{metadata.name}` would be captured "
                f"by bucket `{bucket_name}`."
            )

    @classmethod
    def add_to_bucket(
        cls,
        bucket_tool: Any,
        bucket_name: str,
        metadata: ToolMetadata,
    ) -> None:
        cls.validate_capture(bucket_name, metadata)
        bucket_impl = getattr(bucket_tool, "impl", None)
        if bucket_impl is None or not hasattr(bucket_impl, "add"):
            raise ValueError(f"The bucket tool `{bucket_name}` cannot capture tools.")
        bucket_impl.add(metadata)
        cls.refresh_tool(bucket_tool)

    @staticmethod
    def refresh_tool(bucket_tool: Any) -> None:
        bucket_impl = getattr(bucket_tool, "impl", None)
        if bucket_impl is None:
            return
        description = getattr(bucket_impl, "description", None)
        if isinstance(description, str):
            bucket_tool.set_description(description)
        bucket_tool.register_buffer(
            "usage_guidance",
            getattr(bucket_impl, "usage_guidance", None),
        )

    @classmethod
    def iter_capture_candidates(
        cls,
        *,
        bucket_name: str,
        capture_kind: str,
        tools: Mapping[str, Any],
        tool_configs: Mapping[str, Mapping[str, Any]],
        is_reserved_tool: Callable[[str, Any, Mapping[str, Any]], bool],
    ) -> Iterator[tuple[str, Any]]:
        for tool_name, tool in tools.items():
            config = tool_configs.get(tool_name, {})
            if tool_name == bucket_name or is_reserved_tool(tool_name, tool, config):
                continue
            if config.get("tool_kind") != capture_kind:
                continue
            yield tool_name, tool

    @classmethod
    def validate_existing_captures(
        cls,
        *,
        bucket_name: str,
        capture_kind: str,
        tools: Mapping[str, Any],
        tool_configs: Mapping[str, Mapping[str, Any]],
        metadata_factory: Callable[[Any], ToolMetadata],
        is_reserved_tool: Callable[[str, Any, Mapping[str, Any]], bool],
    ) -> None:
        for _, tool in cls.iter_capture_candidates(
            bucket_name=bucket_name,
            capture_kind=capture_kind,
            tools=tools,
            tool_configs=tool_configs,
            is_reserved_tool=is_reserved_tool,
        ):
            cls.validate_capture(bucket_name, metadata_factory(tool))

    @classmethod
    def pop_existing_captures(
        cls,
        *,
        bucket_name: str,
        capture_kind: str,
        tools: MutableMapping[str, Any],
        tool_configs: MutableMapping[str, Any],
        metadata_factory: Callable[[Any], ToolMetadata],
        is_reserved_tool: Callable[[str, Any, Mapping[str, Any]], bool],
    ) -> list[ToolMetadata]:
        captured = []
        candidates = list(
            cls.iter_capture_candidates(
                bucket_name=bucket_name,
                capture_kind=capture_kind,
                tools=tools,
                tool_configs=tool_configs,
                is_reserved_tool=is_reserved_tool,
            )
        )
        for tool_name, tool in candidates:
            metadata = metadata_factory(tool)
            tools.pop(tool_name)
            tool_configs.pop(tool_name, None)
            captured.append(metadata)
        return captured


class ToolLibraryOperator:
    """Base class for runtime tools that operate through ToolLibraryHandle."""

    tool_config = {"inject_handle": True}

    @classmethod
    def is_runtime_tool(cls, tool: Any | None) -> bool:
        if tool is None:
            return False
        impl = getattr(tool, "impl", tool)
        return isinstance(impl, cls)

    @classmethod
    def is_runtime_metadata(cls, metadata: ToolMetadata) -> bool:
        return isinstance(metadata.impl, cls)


class ToolBackground(ToolLibraryOperator):
    """Base class for builtin tools that manage background tasks."""

    tool_kind = BACKGROUND_TASK_TOOL_KIND

    @classmethod
    def record_registered_tool(
        cls,
        tool_name: str,
        config: Mapping[str, Any],
        state: Any,
    ) -> None:
        if config.get("tool_kind") == cls.tool_kind:
            state.disabled_background_task_tool_names.discard(tool_name)

    @staticmethod
    def is_agent_capable(tool: Any | None, config: Mapping[str, Any]) -> bool:
        impl = getattr(tool, "impl", None) if tool is not None else None
        return config.get("tool_kind") == "agent" or bool(
            getattr(impl, "supports_task_message", False)
        )

    @staticmethod
    def is_installed_tool(tool_name: str, state: Any) -> bool:
        return (
            tool_name in state.background_task_tool_names
            or tool_name in state.agent_task_tool_names
        )

    @classmethod
    def sync_task_tools(
        cls,
        *,
        library: Any,
        state: Any,
        base_tools: Iterator[Callable],
        agent_tools: Iterator[Callable],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> None:
        if state.background_tool_names:
            cls._ensure_task_tools(
                library=library,
                state=state,
                tools=base_tools,
                installed_names=state.background_task_tool_names,
                metadata_factory=metadata_factory,
            )
        else:
            cls._remove_task_tools(
                library=library,
                installed_names=state.background_task_tool_names,
            )
            cls._remove_task_tools(
                library=library,
                installed_names=state.agent_task_tool_names,
            )
            state.disabled_background_task_tool_names.clear()
            return

        if state.background_agent_tool_names:
            cls._ensure_task_tools(
                library=library,
                state=state,
                tools=agent_tools,
                installed_names=state.agent_task_tool_names,
                metadata_factory=metadata_factory,
            )
        else:
            cls._remove_task_tools(
                library=library,
                installed_names=state.agent_task_tool_names,
            )

    @classmethod
    def _ensure_task_tools(
        cls,
        *,
        library: Any,
        state: Any,
        tools: Iterator[Callable],
        installed_names: set[str],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> None:
        for tool in tools:
            metadata = metadata_factory(tool)
            tool_name = metadata.name
            if tool_name in state.disabled_background_task_tool_names:
                continue
            if tool_name in library.library:
                existing_config = library.tool_configs.get(tool_name, {})
                if existing_config.get("tool_kind") != cls.tool_kind:
                    raise ValueError(
                        f"The background task tool `{tool_name}` conflicts with "
                        "an existing tool."
                    )
                installed_names.add(tool_name)
                continue
            library.add(metadata)
            installed_names.add(tool_name)

    @staticmethod
    def _remove_task_tools(*, library: Any, installed_names: set[str]) -> None:
        for tool_name in list(installed_names):
            if tool_name in library.library:
                library._remove_registered_tool(tool_name)
            installed_names.discard(tool_name)
