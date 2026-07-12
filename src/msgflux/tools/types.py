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
    is_agent_tool_impl,
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
    """Base class for tools that capture tools matching a configuration."""

    tool_kind = TOOL_BUCKET_KIND
    capture: Mapping[str, Any] | None = None
    expose_captured_names = False

    def add(self, tool: ToolMetadata) -> None:
        """Store a captured tool and refresh metadata derived from its contents."""
        self.validate_capture(tool)
        if tool.name in self.tools:
            raise ValueError(f"Duplicate tool name `{tool.name}` in bucket.")
        self.tools[tool.name] = tool
        try:
            self.refresh()
        except Exception:
            self.tools.pop(tool.name, None)
            raise

    def remove(self, tool_name: str) -> ToolMetadata:
        """Release a captured tool and refresh metadata derived from its contents."""
        try:
            tool = self.tools.pop(tool_name)
        except KeyError as exc:
            raise ValueError(
                f"Tool `{tool_name}` is not captured by this bucket."
            ) from exc
        try:
            self.refresh()
        except Exception:
            self.tools[tool_name] = tool
            raise
        return tool

    @property
    def tools(self) -> Dict[str, ToolMetadata]:
        if not hasattr(self, "_tools"):
            self._tools = {}
        return self._tools

    def refresh(self) -> None:
        """Refresh presentation metadata after the library captures a tool."""

    @property
    def capture_rules(self) -> Mapping[str, Any]:
        """Return the validated configuration predicates for this bucket."""
        capture = getattr(self, "capture", None)
        if not isinstance(capture, Mapping) or not capture:
            raise ValueError("A bucket tool must define a non-empty `capture` mapping.")
        if not all(isinstance(key, str) and key for key in capture):
            raise ValueError("Bucket capture keys must be non-empty strings.")
        for key, value in capture.items():
            self._capture_values(key, value)
        return capture

    @staticmethod
    def _capture_values(key: str, value: Any) -> tuple[Any, ...]:
        """Normalize a capture value for matching and overlap validation."""
        if key != "tool_kind":
            return (value,)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                "Bucket `capture['tool_kind']` must be a non-empty string."
            )
        values = tuple(part.strip() for part in value.split("|"))
        if not all(values):
            raise ValueError("Bucket `capture['tool_kind']` values cannot be empty.")
        if len(set(values)) != len(values):
            raise ValueError("Bucket `capture['tool_kind']` values must be unique.")
        return values

    def captures_config(self, tool_config: Mapping[str, Any]) -> bool:
        return all(
            tool_config.get(key) in self._capture_values(key, value)
            for key, value in self.capture_rules.items()
        )

    def captures(self, metadata: ToolMetadata) -> bool:
        return self.captures_config(metadata.tool_config)

    def validate_capture(self, metadata: ToolMetadata) -> None:
        if is_background_capable(metadata.tool_config):
            raise ValueError(
                "Bucket-captured tools cannot use `background=True` or "
                f"`allow_background=True`. Tool `{metadata.name}` cannot be captured."
            )
        if not self.captures(metadata):
            raise ValueError(
                f"Tool `{metadata.name}` does not match this bucket's capture rule."
            )

    @classmethod
    def find_bucket(
        cls,
        metadata: ToolMetadata,
        tools: Mapping[str, Any],
        tool_configs: Mapping[str, Mapping[str, Any]],
    ) -> str | None:
        if metadata.tool_config.get("tool_kind") == cls.tool_kind:
            return None
        for bucket_name, tool in tools.items():
            config = tool_configs.get(bucket_name, {})
            if config.get("tool_kind") != cls.tool_kind:
                continue
            bucket = getattr(tool, "impl", tool)
            if isinstance(bucket, cls) and bucket.captures(metadata):
                return bucket_name
        return None

    @classmethod
    def find_capturing_bucket(
        cls,
        tool_name: str,
        tools: Mapping[str, Any],
        tool_configs: Mapping[str, Mapping[str, Any]],
    ) -> str | None:
        for bucket_name, tool in tools.items():
            config = tool_configs.get(bucket_name, {})
            if config.get("tool_kind") != cls.tool_kind:
                continue
            bucket = getattr(tool, "impl", tool)
            if isinstance(bucket, cls) and tool_name in bucket.tools:
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
            if config.get("tool_kind") == cls.tool_kind:
                continue
            if bucket.captures_config(config):
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
        capture = bucket.capture_rules
        for bucket_name, tool in tools.items():
            config = tool_configs.get(bucket_name, {})
            if config.get("tool_kind") != cls.tool_kind:
                continue
            registered_bucket = getattr(tool, "impl", tool)
            if not isinstance(registered_bucket, cls):
                continue
            if cls._captures_overlap(capture, registered_bucket.capture_rules):
                raise ValueError(
                    f"The bucket capture for `{metadata.name}` overlaps with "
                    f"`{bucket_name}`."
                )

    @classmethod
    def _captures_overlap(
        cls,
        first: Mapping[str, Any],
        second: Mapping[str, Any],
    ) -> bool:
        """Return whether two capture rules can match the same configuration."""
        for key in first.keys() & second.keys():
            first_values = cls._capture_values(key, first[key])
            second_values = cls._capture_values(key, second[key])
            if not any(
                first_value == second_value
                for first_value in first_values
                for second_value in second_values
            ):
                return False
        return True


class ToolLibraryOperator:
    """Base class for tools that operate through ToolLibraryHandle."""

    tool_config = {"inject_handle": True}

    @classmethod
    def is_operator_tool(cls, tool: Any | None) -> bool:
        if tool is None:
            return False
        impl = getattr(tool, "impl", tool)
        return isinstance(impl, cls)


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
    def is_agent_source(tool: Any | None) -> bool:
        return is_agent_tool_impl(getattr(tool, "impl", tool))

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
            if cls.is_agent_source(tool):
                return DEFAULT_AGENT_BACKGROUND_CAPABILITIES
            return ()
        capabilities = normalize_background_capabilities(declared_capabilities)
        agent_capabilities = {"message"}
        if agent_capabilities.intersection(capabilities) and not cls.is_agent_source(
            tool
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
