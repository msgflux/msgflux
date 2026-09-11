from __future__ import annotations

from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import msgspec

from msgflux.tools.catalog import ToolRef
from msgflux.tools.helpers import (
    BACKGROUND_TASK_TOOL_KIND,
    DEFAULT_AGENT_BACKGROUND_CAPABILITIES,
    RESERVED_TOOL_KINDS,
    TOOL_BUCKET_KIND,
    is_agent_tool_impl,
    normalize_background_capabilities,
)
from msgflux.tools.runtime import _copy_mapping

T = TypeVar("T")

if TYPE_CHECKING:
    from msgflux.nn.modules.tool.definitions import ToolDefinition


class ToolBucketEntry(msgspec.Struct, frozen=True, kw_only=True):
    """Execution-free captured-tool projection retained by bucket interfaces."""

    ref: ToolRef
    description: str | None = None
    display_name: str | None = None
    usage_guidance: str | None = None
    kind: str = "tool"
    namespace: str | None = None
    metadata: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ToolRef):
            raise TypeError("ref must be a ToolRef")
        msgspec.structs.force_setattr(
            self,
            "metadata",
            _copy_mapping(self.metadata, "bucket_entry.metadata"),
        )

    @property
    def name(self) -> str:
        return self.ref.tool_id

    @classmethod
    def from_definition(
        cls,
        definition: Any,
        *,
        ref: ToolRef,
    ) -> ToolBucketEntry:
        return cls(
            ref=ref,
            description=definition.description,
            display_name=definition.display_name,
            usage_guidance=definition.usage_guidance,
            kind=definition.kind,
            namespace=definition.metadata.get("execution_namespace"),
            metadata=definition.metadata.get("bucket", {}),
        )


class Hidden(Generic[T]):
    """Type marker for parameters hidden from the model-facing tool schema."""


def _normalize_hidden_annotation(annotation: Any) -> Any:
    # Python 3.10 get_type_hints adds Optional for parameters defaulting to None.
    if get_origin(annotation) in (Union, UnionType):
        args = get_args(annotation)
        non_none = tuple(arg for arg in args if arg is not type(None))
        if len(args) == 2 and len(non_none) == 1:
            return non_none[0]
    return annotation


def is_hidden_annotation(annotation: Any) -> bool:
    """Return whether an annotation is a `Hidden[...]` marker."""
    annotation = _normalize_hidden_annotation(annotation)
    return annotation is Hidden or get_origin(annotation) is Hidden


def unwrap_hidden_annotation(annotation: Any) -> Any | None:
    """Return the wrapped type from `Hidden[T]`, or Any for bare `Hidden`."""
    annotation = _normalize_hidden_annotation(annotation)
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

    def patch_schema_annotations(
        self,
        annotations: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Patch the normalized public annotations after bucket changes."""
        return annotations

    def compose_usage_guidance(self, declared: str | None) -> str | None:
        """Combine stable declaration guidance with dynamic bucket guidance."""
        return getattr(self, "usage_guidance", None) or declared

    def add(self, ref: ToolRef) -> None:
        """Retain one captured tool reference without owning its executor."""
        if not isinstance(ref, ToolRef):
            raise TypeError("Buckets can only retain ToolRef values.")
        if ref.tool_id in self.tools:
            raise ValueError(f"Duplicate tool name `{ref.tool_id}` in bucket.")
        self.tools[ref.tool_id] = ref

    def remove(self, tool_name: str) -> ToolRef:
        """Release and return one captured tool reference."""
        try:
            return self.tools.pop(tool_name)
        except KeyError as exc:
            raise ValueError(
                f"Tool `{tool_name}` is not captured by this bucket."
            ) from exc

    @property
    def tools(self) -> Dict[str, ToolRef]:
        if not hasattr(self, "_tools"):
            self._tools = {}
        return self._tools

    def get_ref(self, tool_name: str) -> ToolRef:
        """Return a captured tool reference without exposing its implementation."""
        try:
            return self.tools[tool_name]
        except KeyError as exc:
            raise ValueError(
                f"Tool `{tool_name}` is not captured by this bucket."
            ) from exc

    def refresh(self, entries: tuple[ToolBucketEntry, ...] = ()) -> None:
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

    def captures(self, definition: Any) -> bool:
        return self.captures_config(definition.declaration)

    def validate_capture(self, definition: Any) -> None:
        declaration = definition.declaration
        loop_options = {
            option
            for option in (
                "dispatch",
                "background",
                "allow_background",
                "detached",
                "call_as_response",
                "return_direct",
                "handoff",
            )
            if declaration.get(option, False)
        }
        if loop_options:
            formatted_options = ", ".join(
                f"`{option}=True`" for option in sorted(loop_options)
            )
            raise ValueError(
                "Bucket-captured tools cannot define model-loop behavior "
                f"({formatted_options}). Configure that behavior on the public "
                f"bucket instead. Tool `{definition.name}` cannot be captured."
            )
        if not self.captures(definition):
            raise ValueError(
                f"Tool `{definition.name}` does not match this bucket's capture rule."
            )

    @classmethod
    def find_bucket(
        cls,
        definition: Any,
        tools: Mapping[str, Any],
        get_definition: Callable[[str], Any],
    ) -> str | None:
        if definition.kind == cls.tool_kind:
            return None
        for bucket_name, tool in tools.items():
            bucket_definition = get_definition(bucket_name)
            if bucket_definition.kind != cls.tool_kind:
                continue
            bucket = getattr(tool, "impl", tool)
            if isinstance(bucket, cls) and bucket.captures(definition):
                return bucket_name
        return None

    @classmethod
    def find_capturing_bucket(
        cls,
        tool_name: str,
        tools: Mapping[str, Any],
        get_definition: Callable[[str], Any],
    ) -> str | None:
        for bucket_name, tool in tools.items():
            if get_definition(bucket_name).kind != cls.tool_kind:
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
        get_definition: Callable[[str], Any],
    ) -> list[tuple[str, Any]]:
        candidates = []
        for tool_name, tool in tools.items():
            definition = get_definition(tool_name)
            if definition.kind == cls.tool_kind:
                continue
            if bucket.captures_config(definition.declaration):
                candidates.append((tool_name, tool))
        return candidates

    @classmethod
    def validate_registration(
        cls,
        name: str,
        bucket: ToolBucket,
        tools: Mapping[str, Any],
        get_definition: Callable[[str], Any],
    ) -> None:
        if not isinstance(bucket, cls):
            raise ValueError(f"The bucket tool `{name}` must inherit ToolBucket.")
        capture = bucket.capture_rules
        for bucket_name, tool in tools.items():
            if get_definition(bucket_name).kind != cls.tool_kind:
                continue
            registered_bucket = getattr(tool, "impl", tool)
            if not isinstance(registered_bucket, cls):
                continue
            if cls._captures_overlap(capture, registered_bucket.capture_rules):
                raise ValueError(
                    f"The bucket capture for `{name}` overlaps with `{bucket_name}`."
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

    tool_config = {"runtime_inputs": ("handle",)}

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
        definition: Any,
        base_tools: Iterable[Callable],
        capability_tools: Mapping[str, Iterable[Callable]],
        definition_factory: Callable[[Callable], ToolDefinition],
    ) -> bool:
        if not cls.is_reserved_definition(definition):
            return False
        background_tools = list(cls._iter_background_tools(library))
        if not background_tools:
            return False

        capabilities = {
            capability
            for source_definition in background_tools
            for capability in cls.get_background_capabilities(source_definition)
        }
        task_tools = cls._task_tools_for_capabilities(
            base_tools=base_tools,
            capability_tools=capability_tools,
            capabilities=capabilities,
            definition_factory=definition_factory,
        )
        return tool_name in {
            definition_factory(task_tool).name for task_tool in task_tools
        }

    @staticmethod
    def is_agent_source(tool: Any | None) -> bool:
        return is_agent_tool_impl(getattr(tool, "impl", tool))

    @staticmethod
    def is_reserved_definition(definition: Any) -> bool:
        return definition.kind in RESERVED_TOOL_KINDS

    @staticmethod
    def is_background_definition(definition: Any) -> bool:
        return definition.dispatch.name in {"background", "optional_background"}

    @classmethod
    def get_background_capabilities(
        cls,
        definition: Any,
    ) -> tuple[str, ...]:
        if not cls.is_background_definition(definition):
            return ()
        declared_capabilities = definition.dispatch.options.get("capabilities")
        if declared_capabilities is None:
            if cls.is_agent_source(definition.executor):
                return DEFAULT_AGENT_BACKGROUND_CAPABILITIES
            return ()
        capabilities = normalize_background_capabilities(declared_capabilities)
        agent_capabilities = {"message"}
        if agent_capabilities.intersection(capabilities) and not cls.is_agent_source(
            definition.executor
        ):
            raise ValueError(
                "`message` background capability is currently only supported by "
                "agent sources."
            )
        return capabilities

    @classmethod
    def validate_background_capabilities(
        cls,
        definition: Any,
    ) -> None:
        cls.get_background_capabilities(definition)

    @classmethod
    def sync_task_tools(
        cls,
        *,
        library: Any,
        disabled_tool_names: set[str],
        base_tools: Iterable[Callable],
        capability_tools: Mapping[str, Iterable[Callable]],
        definition_factory: Callable[[Callable], ToolDefinition],
    ) -> None:
        background_tools = list(cls._iter_background_tools(library))
        all_task_tools = cls._all_task_tools(
            base_tools=base_tools,
            capability_tools=capability_tools,
            definition_factory=definition_factory,
        )
        if background_tools:
            capabilities = {
                capability
                for definition in background_tools
                for capability in cls.get_background_capabilities(definition)
            }
            required_task_tools = cls._task_tools_for_capabilities(
                base_tools=base_tools,
                capability_tools=capability_tools,
                capabilities=capabilities,
                definition_factory=definition_factory,
            )
            cls._ensure_task_tools(
                library=library,
                disabled_tool_names=disabled_tool_names,
                tools=required_task_tools,
                definition_factory=definition_factory,
            )
            required_names = {
                definition_factory(task_tool).name for task_tool in required_task_tools
            }
            cls._remove_task_tools(
                library=library,
                tools=(
                    task_tool
                    for task_tool in all_task_tools
                    if definition_factory(task_tool).name not in required_names
                ),
                definition_factory=definition_factory,
            )
            return

        cls._remove_task_tools(
            library=library,
            tools=all_task_tools,
            definition_factory=definition_factory,
        )
        disabled_tool_names.clear()

    @classmethod
    def _iter_background_tools(
        cls,
        library: Any,
    ) -> Iterator[Any]:
        for tool_name in library.library:
            definition = library.get_tool_definition(tool_name)
            if cls.is_reserved_definition(definition):
                continue
            if cls.is_background_definition(definition):
                yield definition

    @classmethod
    def _all_task_tools(
        cls,
        *,
        base_tools: Iterable[Callable],
        capability_tools: Mapping[str, Iterable[Callable]],
        definition_factory: Callable[[Callable], ToolDefinition],
    ) -> tuple[Callable, ...]:
        return cls._task_tools_for_capabilities(
            base_tools=base_tools,
            capability_tools=capability_tools,
            capabilities=capability_tools.keys(),
            definition_factory=definition_factory,
        )

    @classmethod
    def _task_tools_for_capabilities(
        cls,
        *,
        base_tools: Iterable[Callable],
        capability_tools: Mapping[str, Iterable[Callable]],
        capabilities: Iterable[str],
        definition_factory: Callable[[Callable], ToolDefinition],
    ) -> tuple[Callable, ...]:
        selected_tools = list(base_tools)
        capability_names = set(capabilities)
        for capability, tools in capability_tools.items():
            if capability in capability_names:
                selected_tools.extend(tools)

        unique_tools: Dict[str, Callable] = {}
        for task_tool in selected_tools:
            definition = definition_factory(task_tool)
            unique_tools.setdefault(definition.name, task_tool)
        return tuple(unique_tools.values())

    @classmethod
    def _ensure_task_tools(
        cls,
        *,
        library: Any,
        disabled_tool_names: set[str],
        tools: Iterable[Callable],
        definition_factory: Callable[[Callable], ToolDefinition],
    ) -> None:
        for tool in tools:
            definition = definition_factory(tool)
            tool_name = definition.name
            if tool_name in disabled_tool_names:
                continue
            if tool_name in library.library:
                existing = library.get_tool_definition(tool_name)
                if not cls.is_reserved_definition(existing):
                    raise ValueError(
                        f"The background task tool `{tool_name}` conflicts with "
                        "an existing tool."
                    )
                continue
            library.add(definition)

    @classmethod
    def _remove_task_tools(
        cls,
        *,
        library: Any,
        tools: Iterable[Callable],
        definition_factory: Callable[[Callable], ToolDefinition],
    ) -> None:
        for tool in tools:
            tool_name = definition_factory(tool).name
            if tool_name in library.library and cls.is_reserved_definition(
                library.get_tool_definition(tool_name)
            ):
                library._remove_registered_tool(tool_name)
