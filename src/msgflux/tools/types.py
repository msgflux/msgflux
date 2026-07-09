from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    Mapping,
    TypeVar,
    get_args,
    get_origin,
)

from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.helpers import (
    BACKGROUND_TASK_TOOL_KIND,
    TOOL_BUCKET_KIND,
    is_background_capable,
)

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


class ToolBucket:
    """Base class for tools that absorb other tools by kind."""

    tool_kind = TOOL_BUCKET_KIND
    capture_kind: str

    def add(self, tool: ToolMetadata) -> None:
        """Store a captured tool and refresh metadata derived from its contents."""
        if tool.tool_config.get("on_demand", False):
            raise ValueError(
                "On-demand tools must be registered through `ToolLibrary.add(...)`."
            )
        self.validate_capture(tool)
        if tool.name in self.tools:
            raise ValueError(f"Duplicate tool name `{tool.name}` in bucket.")
        self.tools[tool.name] = tool
        self.refresh()

    @property
    def tools(self) -> Dict[str, ToolMetadata]:
        if not hasattr(self, "_tools"):
            self._tools = {}
        return self._tools

    def refresh(self) -> None:
        """Refresh presentation metadata after the library captures a tool."""

    @property
    def capture_kinds(self) -> tuple[str, ...]:
        capture_kind = getattr(self, "capture_kind", None)
        if not isinstance(capture_kind, str) or not capture_kind.strip():
            raise ValueError("A bucket tool must define a non-empty capture_kind.")

        kinds = tuple(kind.strip() for kind in capture_kind.split("|"))
        if not all(kinds):
            raise ValueError("Bucket capture_kind values cannot be empty.")
        if len(set(kinds)) != len(kinds):
            raise ValueError("Bucket capture_kind values must be unique.")
        return kinds

    @staticmethod
    def validate_capture(metadata: ToolMetadata) -> None:
        if is_background_capable(metadata.tool_config):
            raise ValueError(
                "Bucket-captured tools cannot use `background=True` or "
                f"`allow_background=True`. Tool `{metadata.name}` cannot be captured."
            )

    @classmethod
    def has_kind(
        cls,
        tool: Any | None,
        config: Mapping[str, Any],
        kind: str,
    ) -> bool:
        tool_kind = config.get("tool_kind", "tool")
        if tool_kind == kind:
            return True
        if tool_kind != cls.tool_kind or tool is None:
            return False
        bucket = getattr(tool, "impl", tool)
        return isinstance(bucket, cls) and kind in bucket.capture_kinds

    @classmethod
    def find_bucket(
        cls,
        metadata: ToolMetadata,
        tools: Mapping[str, Any],
        tool_configs: Mapping[str, Mapping[str, Any]],
    ) -> str | None:
        tool_kind = metadata.tool_config.get("tool_kind", "tool")
        if tool_kind == cls.tool_kind:
            return None
        return cls.find_bucket_for_kind(tool_kind, tools, tool_configs)

    @classmethod
    def find_bucket_for_kind(
        cls,
        tool_kind: str,
        tools: Mapping[str, Any],
        tool_configs: Mapping[str, Mapping[str, Any]],
    ) -> str | None:
        for bucket_name, tool in tools.items():
            config = tool_configs.get(bucket_name, {})
            if config.get("tool_kind") != cls.tool_kind:
                continue
            if cls.has_kind(tool, config, tool_kind):
                return bucket_name
        return None

    @classmethod
    def find_capture_candidates(
        cls,
        bucket: ToolBucket,
        tools: Mapping[str, Any],
        tool_configs: Mapping[str, Mapping[str, Any]],
    ) -> list[tuple[str, Any]]:
        candidates = []
        for tool_name, tool in tools.items():
            config = tool_configs.get(tool_name, {})
            if config.get(
                "tool_kind"
            ) == cls.tool_kind or ToolLibraryOperator.is_runtime_tool(tool):
                continue
            if config.get("tool_kind", "tool") in bucket.capture_kinds:
                candidates.append((tool_name, tool))
        return candidates

    @classmethod
    def validate_registration(
        cls,
        metadata: ToolMetadata,
        tools: Mapping[str, Any],
        tool_configs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        bucket = metadata.impl
        if not isinstance(bucket, cls):
            raise ValueError(
                f"The bucket tool `{metadata.name}` must inherit ToolBucket."
            )
        for capture_kind in bucket.capture_kinds:
            existing_bucket = cls.find_bucket_for_kind(
                capture_kind,
                tools,
                tool_configs,
            )
            if existing_bucket is not None:
                raise ValueError(
                    f"The bucket capture kind `{capture_kind}` is already handled by "
                    f"`{existing_bucket}`."
                )


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
        return ToolBucket.has_kind(tool, config, "agent")

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
