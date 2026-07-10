from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    TypeVar,
    get_args,
    get_origin,
)

from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.helpers import (
    BACKGROUND_TASK_TOOL_KIND,
    DEFAULT_AGENT_BACKGROUND_CAPABILITIES,
    TOOL_BUCKET_KIND,
    is_background_capable,
    is_reserved_tool_kind,
    normalize_background_capabilities,
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
        try:
            self.refresh()
        except Exception:
            self.tools.pop(tool.name, None)
            raise

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
    def find_captured_metadata(
        cls,
        tool_name: str,
        tools: Mapping[str, Any],
        tool_configs: Mapping[str, Mapping[str, Any]],
    ) -> ToolMetadata | None:
        """Find metadata that is owned by a registered bucket."""
        for bucket_name, tool in tools.items():
            config = tool_configs.get(bucket_name, {})
            if config.get("tool_kind") != cls.tool_kind:
                continue
            bucket = getattr(tool, "impl", tool)
            if not isinstance(bucket, cls):
                continue
            metadata = bucket.tools.get(tool_name)
            if metadata is not None:
                return metadata
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
            ) == cls.tool_kind or ToolLibraryOperator.is_operator_tool(tool):
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
    """Base class for tools that operate through ToolLibraryHandle."""

    tool_config = {"inject_handle": True}

    @classmethod
    def is_operator_tool(cls, tool: Any | None) -> bool:
        if tool is None:
            return False
        impl = getattr(tool, "impl", tool)
        return isinstance(impl, cls)

    @classmethod
    def is_operator_metadata(cls, metadata: ToolMetadata) -> bool:
        return isinstance(metadata.impl, cls)


class ToolBackground(ToolLibraryOperator):
    """Base class for builtin tools that manage background tasks."""

    tool_kind = BACKGROUND_TASK_TOOL_KIND

    @classmethod
    def is_active_task_tool(
        cls,
        *,
        library: Any,
        tool_name: str,
        config: Mapping[str, Any],
        base_tools: Iterable[Callable],
        capability_tools: Mapping[str, Iterable[Callable]],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> bool:
        if not is_reserved_tool_kind(config):
            return False
        background_tools = list(cls._iter_background_tools(library))
        if not background_tools:
            return False

        capabilities = {
            capability
            for tool, source_config in background_tools
            for capability in cls.get_background_capabilities(tool, source_config)
        }
        task_tools = cls._task_tools_for_capabilities(
            base_tools=base_tools,
            capability_tools=capability_tools,
            capabilities=capabilities,
            metadata_factory=metadata_factory,
        )
        return tool_name in {
            metadata_factory(task_tool).name for task_tool in task_tools
        }

    @staticmethod
    def is_agent_source(tool: Any | None, config: Mapping[str, Any]) -> bool:
        return ToolBucket.has_kind(tool, config, "agent")

    @classmethod
    def get_background_capabilities(
        cls,
        tool: Any | None,
        config: Mapping[str, Any],
    ) -> tuple[str, ...]:
        declared_capabilities = config.get("background_capabilities")
        if declared_capabilities is not None and not is_background_capable(config):
            raise ValueError(
                "`background_capabilities` requires `background=True` or "
                "`allow_background=True`."
            )
        if not is_background_capable(config):
            return ()
        if declared_capabilities is None:
            if cls.is_agent_source(tool, config):
                return DEFAULT_AGENT_BACKGROUND_CAPABILITIES
            return ()
        capabilities = normalize_background_capabilities(declared_capabilities)
        agent_capabilities = {"message"}
        if agent_capabilities.intersection(capabilities) and not cls.is_agent_source(
            tool, config
        ):
            raise ValueError(
                "`message` background capability is currently only supported by "
                "agent sources."
            )
        return capabilities

    @classmethod
    def validate_background_capabilities(
        cls,
        tool: Any | None,
        config: Mapping[str, Any],
    ) -> None:
        cls.get_background_capabilities(tool, config)

    @classmethod
    def sync_task_tools(
        cls,
        *,
        library: Any,
        disabled_tool_names: set[str],
        base_tools: Iterable[Callable],
        capability_tools: Mapping[str, Iterable[Callable]],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> None:
        background_tools = list(cls._iter_background_tools(library))
        all_task_tools = cls._all_task_tools(
            base_tools=base_tools,
            capability_tools=capability_tools,
            metadata_factory=metadata_factory,
        )
        if background_tools:
            capabilities = {
                capability
                for tool, config in background_tools
                for capability in cls.get_background_capabilities(tool, config)
            }
            required_task_tools = cls._task_tools_for_capabilities(
                base_tools=base_tools,
                capability_tools=capability_tools,
                capabilities=capabilities,
                metadata_factory=metadata_factory,
            )
            cls._ensure_task_tools(
                library=library,
                disabled_tool_names=disabled_tool_names,
                tools=required_task_tools,
                metadata_factory=metadata_factory,
            )
            required_names = {
                metadata_factory(task_tool).name for task_tool in required_task_tools
            }
            cls._remove_task_tools(
                library=library,
                tools=(
                    task_tool
                    for task_tool in all_task_tools
                    if metadata_factory(task_tool).name not in required_names
                ),
                metadata_factory=metadata_factory,
            )
            return

        cls._remove_task_tools(
            library=library,
            tools=all_task_tools,
            metadata_factory=metadata_factory,
        )
        disabled_tool_names.clear()

    @classmethod
    def _iter_background_tools(
        cls,
        library: Any,
    ) -> Iterator[tuple[Any, Mapping[str, Any]]]:
        for tool_name, tool in library.library.items():
            config = library.tool_configs.get(tool_name, {})
            if is_reserved_tool_kind(config):
                continue
            if is_background_capable(config):
                yield tool, config

    @classmethod
    def _all_task_tools(
        cls,
        *,
        base_tools: Iterable[Callable],
        capability_tools: Mapping[str, Iterable[Callable]],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> tuple[Callable, ...]:
        return cls._task_tools_for_capabilities(
            base_tools=base_tools,
            capability_tools=capability_tools,
            capabilities=capability_tools.keys(),
            metadata_factory=metadata_factory,
        )

    @classmethod
    def _task_tools_for_capabilities(
        cls,
        *,
        base_tools: Iterable[Callable],
        capability_tools: Mapping[str, Iterable[Callable]],
        capabilities: Iterable[str],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> tuple[Callable, ...]:
        selected_tools = list(base_tools)
        capability_names = set(capabilities)
        for capability, tools in capability_tools.items():
            if capability in capability_names:
                selected_tools.extend(tools)

        unique_tools: Dict[str, Callable] = {}
        for task_tool in selected_tools:
            metadata = metadata_factory(task_tool)
            unique_tools.setdefault(metadata.name, task_tool)
        return tuple(unique_tools.values())

    @classmethod
    def _ensure_task_tools(
        cls,
        *,
        library: Any,
        disabled_tool_names: set[str],
        tools: Iterable[Callable],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> None:
        for tool in tools:
            metadata = metadata_factory(tool)
            tool_name = metadata.name
            if tool_name in disabled_tool_names:
                continue
            if tool_name in library.library:
                existing_config = library.tool_configs.get(tool_name, {})
                if not is_reserved_tool_kind(existing_config):
                    raise ValueError(
                        f"The background task tool `{tool_name}` conflicts with "
                        "an existing tool."
                    )
                continue
            library.add(metadata)

    @classmethod
    def _remove_task_tools(
        cls,
        *,
        library: Any,
        tools: Iterable[Callable],
        metadata_factory: Callable[[Callable], ToolMetadata],
    ) -> None:
        for tool in tools:
            tool_name = metadata_factory(tool).name
            config = library.tool_configs.get(tool_name, {})
            if tool_name in library.library and is_reserved_tool_kind(config):
                library._remove_registered_tool(tool_name)
