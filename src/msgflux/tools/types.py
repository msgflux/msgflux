from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
    get_args,
    get_origin,
)

from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.handles import normalize_handle_access
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


class _ToolMetadataView(Mapping[str, ToolMetadata]):
    """Read-only live view of one bucket's children in a tool library."""

    def __init__(self, loader: Callable[[], Mapping[str, ToolMetadata]]) -> None:
        self._loader = loader

    def __getitem__(self, key: str) -> ToolMetadata:
        return self._loader()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._loader())

    def __len__(self) -> int:
        return len(self._loader())

    def __contains__(self, key: object) -> bool:
        return key in self._loader()

    def get(self, key: str, default: Any = None) -> Any:
        return self._loader().get(key, default)

    def items(self):
        return self._loader().items()

    def keys(self):
        return self._loader().keys()

    def values(self):
        return self._loader().values()


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


def split_hidden_annotations(
    annotations: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Separate model-facing annotations from runtime-injected parameters."""
    public_annotations: Dict[str, Any] = {}
    hidden_params: Dict[str, Any] = {}
    for name, annotation in annotations.items():
        hidden_type = unwrap_hidden_annotation(annotation)
        if hidden_type is None:
            public_annotations[name] = annotation
        elif name == "return":
            raise ValueError("`Hidden[...]` cannot be used as a return type.")
        else:
            hidden_params[name] = hidden_type
    return public_annotations, hidden_params


class ToolBucket:
    """Base class for tools that exclusively own matching library tools.

    ``capture`` supports exact ``tool_config`` predicates plus the structural
    selectors ``source``, ``name``, ``capabilities``, and ``match.any``. A
    bucket may capture another bucket when ``source`` is ``"bucket"`` or
    ``"any"``. ``ToolLibrary`` retains nested buckets in its ownership tree so
    they continue receiving late registrations and propagating presentation
    updates to their parents.
    """

    tool_kind = TOOL_BUCKET_KIND
    capture: Mapping[str, Any] | None = None
    expose_captured_names = False
    _CAPTURE_SOURCES = {"tool", "bucket", "any"}

    def add(self, tool: ToolMetadata) -> None:
        """Stage a tool before this bucket is registered in a library."""
        if hasattr(self, "_tools_view"):
            raise RuntimeError(
                "Registered buckets are mutated through ToolLibrary, not bucket.add()."
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

    def remove(self, tool_name: str) -> ToolMetadata:
        """Remove a staged tool before this bucket is registered."""
        if hasattr(self, "_tools_view"):
            raise RuntimeError(
                "Registered buckets are mutated through ToolLibrary, not "
                "bucket.remove()."
            )
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
    def tools(self) -> Mapping[str, ToolMetadata]:
        view = getattr(self, "_tools_view", None)
        if view is not None:
            return view
        if not hasattr(self, "_tools"):
            self._tools = {}
        return self._tools

    def _bind_tools(
        self,
        loader: Callable[[], Mapping[str, ToolMetadata]],
    ) -> list[ToolMetadata]:
        """Bind the live library view and return previously staged tools."""
        pending = list(getattr(self, "_tools", {}).values())
        self._tools = {}
        self._tools_view = _ToolMetadataView(loader)
        return pending

    def _unbind_tools(self, pending: Iterable[ToolMetadata] = ()) -> None:
        """Restore an unregistered bucket with optional staged tools."""
        if hasattr(self, "_tools_view"):
            del self._tools_view
        self._tools = {metadata.name: metadata for metadata in pending}

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
        rules = {
            key: value
            for key, value in capture.items()
            if key not in {"policy", "source", "match"}
        }
        if not rules:
            match = capture.get("match")
            if match is None:
                raise ValueError("A bucket must define at least one capture predicate.")
        for key, value in rules.items():
            self._capture_values(key, value)
        _ = self.capture_source
        _ = self.capture_alternatives
        _ = self.capture_policy
        return rules

    @property
    def capture_source(self) -> str:
        """Return which candidate type this bucket can capture."""
        capture = getattr(self, "capture", None)
        source = (
            capture.get("source", "tool") if isinstance(capture, Mapping) else "tool"
        )
        if not isinstance(source, str) or source not in self._CAPTURE_SOURCES:
            expected = ", ".join(sorted(self._CAPTURE_SOURCES))
            raise ValueError(
                f"Unknown bucket capture source `{source}`. "
                f"Expected one of: {expected}."
            )
        return source

    @property
    def capture_alternatives(self) -> tuple[Mapping[str, Any], ...]:
        """Return validated OR alternatives declared through `match.any`."""
        capture = getattr(self, "capture", None)
        match = capture.get("match") if isinstance(capture, Mapping) else None
        if match is None:
            return ({},)
        if not isinstance(match, Mapping) or set(match) != {"any"}:
            raise ValueError("Bucket capture `match` must contain only `any`.")
        alternatives = match["any"]
        if isinstance(alternatives, (str, bytes, Mapping)) or not isinstance(
            alternatives, Sequence
        ):
            raise ValueError("Bucket capture `match.any` must be a list of mappings.")
        normalized = tuple(alternatives)
        if not normalized:
            raise ValueError("Bucket capture `match.any` cannot be empty.")
        for alternative in normalized:
            self._validate_capture_alternative(alternative, capture)
        return normalized

    def _validate_capture_alternative(
        self,
        alternative: Any,
        capture: Mapping[str, Any],
    ) -> None:
        if not isinstance(alternative, Mapping) or not alternative:
            raise ValueError(
                "Each bucket capture `match.any` entry must be a non-empty mapping."
            )
        if not all(isinstance(key, str) and key for key in alternative):
            raise ValueError(
                "Bucket capture `match.any` keys must be non-empty strings."
            )
        unknown = set(alternative) & {"policy", "source", "match"}
        if unknown:
            raise ValueError(
                f"Bucket capture `{sorted(unknown)[0]}` is only valid at the top level."
            )
        duplicate = set(alternative) & (set(capture) - {"policy", "source", "match"})
        if duplicate:
            raise ValueError(
                f"Bucket capture predicate `{sorted(duplicate)[0]}` cannot be "
                "declared both at the top level and in `match.any`."
            )
        for key, value in alternative.items():
            self._capture_values(key, value)

    @property
    def capture_policy(self) -> Mapping[str, Any]:
        """Return optional restrictions applied to captured tool metadata."""
        capture = getattr(self, "capture", None)
        if not isinstance(capture, Mapping):
            return {}
        policy = capture.get("policy")
        if policy is None:
            return {}
        if not isinstance(policy, Mapping) or not policy:
            raise ValueError("Bucket capture `policy` must be a non-empty mapping.")
        supported = {"handle"}
        unknown = sorted(set(policy) - supported)
        if unknown:
            names = ", ".join(sorted(supported))
            raise ValueError(
                f"Unknown bucket capture policy `{unknown[0]}`. "
                f"Supported policies: {names}."
            )
        handle = normalize_handle_access(policy.get("handle"))
        if handle is None:
            raise ValueError(
                "Bucket capture policy `handle` must declare allowed access."
            )
        return {"handle": handle}

    @staticmethod
    def _capture_names(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            values = (value,)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = tuple(value)
        else:
            values = ()
        if not values or not all(
            isinstance(item, str) and item.strip() for item in values
        ):
            raise ValueError(
                "Bucket `capture['name']` must be a non-empty string or list "
                "of strings."
            )
        if len(set(values)) != len(values):
            raise ValueError("Bucket `capture['name']` values must be unique.")
        return values

    @staticmethod
    def _capture_capabilities(value: Any) -> tuple[Any, ...]:
        if not isinstance(value, Mapping) or len(value) != 1:
            raise ValueError(
                "Bucket `capture['capabilities']` must contain `all` or `any`."
            )
        mode, capabilities = next(iter(value.items()))
        if mode not in {"all", "any"}:
            raise ValueError(
                "Bucket `capture['capabilities']` must contain `all` or `any`."
            )
        if isinstance(capabilities, (str, bytes, Mapping)) or not isinstance(
            capabilities, Sequence
        ):
            raise ValueError(f"Bucket capability `{mode}` must be a list of strings.")
        values = tuple(capabilities)
        if not values or not all(
            isinstance(item, str) and item.strip() for item in values
        ):
            raise ValueError(
                f"Bucket capability `{mode}` must be a non-empty list of strings."
            )
        if len(set(values)) != len(values):
            raise ValueError("Bucket capability values must be unique.")
        return ((mode, values),)

    @classmethod
    def _capture_values(cls, key: str, value: Any) -> tuple[Any, ...]:
        """Normalize a capture value for matching and overlap validation."""
        if key == "name":
            return cls._capture_names(value)
        if key == "capabilities":
            return cls._capture_capabilities(value)
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

    @classmethod
    def _matches_rule(
        cls,
        metadata: ToolMetadata,
        key: str,
        value: Any,
    ) -> bool:
        if key == "name":
            return metadata.name in cls._capture_values(key, value)
        if key == "capabilities":
            mode, expected = cls._capture_values(key, value)[0]
            declared = set(metadata.tool_config.get("capabilities") or ())
            expected_set = set(expected)
            return (
                expected_set <= declared
                if mode == "all"
                else bool(expected_set & declared)
            )
        return metadata.tool_config.get(key) in cls._capture_values(key, value)

    def captures(self, metadata: ToolMetadata) -> bool:
        is_bucket = metadata.tool_config.get("tool_kind") == self.tool_kind
        if self.capture_source == "tool" and is_bucket:
            return False
        if self.capture_source == "bucket" and not is_bucket:
            return False
        base = self.capture_rules
        if not all(
            self._matches_rule(metadata, key, value) for key, value in base.items()
        ):
            return False
        return any(
            all(
                self._matches_rule(metadata, key, value)
                for key, value in alternative.items()
            )
            for alternative in self.capture_alternatives
        )

    def validate_capture(self, metadata: ToolMetadata) -> None:
        if metadata.tool_config.get(
            "tool_kind"
        ) != self.tool_kind and is_background_capable(metadata.tool_config):
            raise ValueError(
                "Bucket-captured tools cannot use `background=True` or "
                f"`allow_background=True`. Tool `{metadata.name}` cannot be captured."
            )
        if not self.captures(metadata):
            raise ValueError(
                f"Tool `{metadata.name}` does not match this bucket's capture rule."
            )
        allowed_handle = self.capture_policy.get("handle")
        required_handle = metadata.tool_config.get("handle")
        if allowed_handle is not None and required_handle is not None:
            for domain, actions in required_handle.items():
                unsupported = sorted(set(actions) - set(allowed_handle.get(domain, ())))
                if unsupported:
                    raise ValueError(
                        f"Tool `{metadata.name}` requires handle access "
                        f"`{domain}.{unsupported[0]}` outside this bucket's "
                        "capture policy."
                    )

    @classmethod
    def _captures_overlap(
        cls,
        first: Mapping[str, Any],
        second: Mapping[str, Any],
    ) -> bool:
        """Return whether two capture rules can match the same configuration."""
        for key in first.keys() & second.keys():
            # Capability sets are open-ended: distinct requirements can coexist
            # on one tool and therefore never prove two selectors disjoint.
            if key == "capabilities":
                continue
            first_values = cls._capture_values(key, first[key])
            second_values = cls._capture_values(key, second[key])
            if not any(
                first_value == second_value
                for first_value in first_values
                for second_value in second_values
            ):
                return False
        return True

    @classmethod
    def capture_overlaps(
        cls,
        first: ToolBucket,
        second: ToolBucket,
    ) -> bool:
        """Return whether two complete bucket selectors may own one candidate."""
        first_sources = (
            {"tool", "bucket"}
            if first.capture_source == "any"
            else {first.capture_source}
        )
        second_sources = (
            {"tool", "bucket"}
            if second.capture_source == "any"
            else {second.capture_source}
        )
        if not first_sources & second_sources:
            return False

        first_patterns = [
            dict(first.capture_rules, **alternative)
            for alternative in first.capture_alternatives
        ]
        second_patterns = [
            dict(second.capture_rules, **alternative)
            for alternative in second.capture_alternatives
        ]
        return any(
            cls._captures_overlap(left, right)
            for left in first_patterns
            for right in second_patterns
        )


class ToolLibraryOperator:
    """Base class for tools that operate through ToolLibraryHandle."""

    tool_config = {"handle": {"tools": ["list"]}}

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
        transaction: Any = None,
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
                transaction=transaction,
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
                transaction=transaction,
            )
            return

        cls._remove_task_tools(
            library=library,
            tools=all_task_tools,
            metadata_factory=metadata_factory,
            transaction=transaction,
        )
        disabled_tool_names.clear()

    @classmethod
    def _iter_background_tools(
        cls,
        library: Any,
    ) -> Iterator[tuple[Any, Mapping[str, Any]]]:
        for node in library._bucket_graph.iter_nodes():
            if is_reserved_tool_kind(node.config):
                continue
            if is_background_capable(node.config):
                yield node.tool, node.config

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
        transaction: Any = None,
    ) -> None:
        for tool in tools:
            metadata = metadata_factory(tool)
            tool_name = metadata.name
            if tool_name in disabled_tool_names:
                continue
            capturing_bucket = library._bucket_graph.find_owner(tool_name)
            if capturing_bucket is not None:
                continue
            if tool_name in library.library:
                existing_config = library.tool_configs.get(tool_name, {})
                if not is_reserved_tool_kind(existing_config):
                    raise ValueError(
                        f"The background task tool `{tool_name}` conflicts with "
                        "an existing tool."
                    )
                continue
            if transaction is None:
                library.add(metadata)
            else:
                library._add(metadata, transaction=transaction)

    @classmethod
    def _remove_task_tools(
        cls,
        *,
        library: Any,
        tools: Iterable[Callable],
        metadata_factory: Callable[[Callable], ToolMetadata],
        transaction: Any = None,
    ) -> None:
        for tool in tools:
            tool_name = metadata_factory(tool).name
            config = library.tool_configs.get(tool_name, {})
            if tool_name in library.library and is_reserved_tool_kind(config):
                library._remove_reconciled_tool(
                    tool_name,
                    transaction=transaction,
                )
