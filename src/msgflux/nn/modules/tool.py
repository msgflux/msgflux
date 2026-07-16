import asyncio
import inspect
from copy import deepcopy
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    get_type_hints,
)

import msgflux.nn.functional as F
from msgflux.auto import AutoParams
from msgflux.core.dotdict import dotdict
from msgflux.exceptions import TaskError, TaskInterruptRequestedError
from msgflux.logger import logger
from msgflux.nn.modules.container import ModuleDict
from msgflux.nn.modules.module import Module
from msgflux.protocols.mcp import (
    MCPClient,
    convert_mcp_schema_to_tool_schema,
    extract_tool_result_text,
    filter_tools,
)
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    InMemoryAgentInboxStore,
)
from msgflux.runtime.background import BackgroundTaskDispatcher
from msgflux.runtime.context import get_execution_context
from msgflux.tasks import InMemoryTaskStore
from msgflux.telemetry.span import (
    aset_tool_attributes,
    set_tool_attributes,
)
from msgflux.tools.bucket_graph import ToolBucketGraph
from msgflux.tools.bucket_manager import ToolBucketManager
from msgflux.tools.builtin.task_tool import (
    BACKGROUND_CAPABILITY_TOOLS,
    BASE_TASK_TOOLS,
)
from msgflux.tools.builtin.tool_search import ToolSearchTool
from msgflux.tools.dataclasses import PreparedToolExecution, ToolMetadata
from msgflux.tools.exceptions import ToolNotAvailableError
from msgflux.tools.handles import (
    ToolBucketHandle,
    ToolLibraryHandle,
    normalize_handle_access,
)
from msgflux.tools.helpers import (
    RUNTIME_BACKGROUND_PARAM,
    build_call_parameters_for_response,
    coerce_tool_params,
    is_background_capable,
    is_reserved_tool_kind,
    normalize_background_capabilities,
    normalize_tool_capabilities,
    should_copy_injected_messages,
    should_dispatch_background,
)
from msgflux.tools.registration import ToolRegistrationTransaction
from msgflux.tools.responses import ToolCall, ToolResponses
from msgflux.tools.types import (
    ToolBackground,
    ToolBucket,
    ToolLibraryOperator,
    is_hidden_annotation,
    split_hidden_annotations,
)
from msgflux.utils.chat import generate_tool_json_schema
from msgflux.utils.inspect import fn_has_parameters, get_fn_param_defaults
from msgflux.utils.msgspec import restore_transport_value
from msgflux.utils.tenacity import apply_retry, default_tool_retry


class Tool(Module):
    """Tool is Module type that provide a json schema to tools."""

    def get_json_schema(self):
        return generate_tool_json_schema(self)


class MCPTool(Tool):
    """MCP Tool Proxy - wraps remote MCP tool as a Tool object.

    This allows MCP tools to be treated exactly like local tools,
    enabling polymorphism and unified telemetry.

    Args:
        name: Tool name (without namespace prefix)
        mcp_client: Connected MCP client
        mcp_tool_info: MCP tool metadata
        namespace: MCP server namespace
        config: Optional tool configuration

    Example:
        >>> mcp_tool = MCPTool(
        ...     name="read_file",
        ...     mcp_client=client,
        ...     mcp_tool_info=tool_info,
        ...     namespace="filesystem"
        ... )
        >>> result = mcp_tool(path="/file.txt")
    """

    def __init__(
        self,
        name: str,
        mcp_client: Any,  # MCPClient type
        mcp_tool_info: Any,  # MCPToolInfo type
        namespace: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Set full tool name with namespace
        full_name = f"{namespace}__{name}"
        self.set_name(full_name)
        self.register_buffer(
            "display_name",
            (config.get("display_name") or full_name) if config else full_name,
        )
        self.register_buffer(
            "usage_guidance",
            config.get("usage_guidance") if config else None,
        )

        # Store MCP-specific data
        self._mcp_client = mcp_client
        self._mcp_tool_info = mcp_tool_info
        self._namespace = namespace
        self._mcp_tool_name = name

        # Set description from MCP tool info
        if hasattr(mcp_tool_info, "description"):
            self.set_description(mcp_tool_info.description)

        # Store config
        tc = {**(config or {})}
        tc.setdefault("tool_kind", "tool")
        self.register_buffer("tool_config", tc)

        # Apply retry
        retry_config = tc.get("retry")
        self.forward = apply_retry(
            self.forward, retry_config, default=default_tool_retry
        )
        self.aforward = apply_retry(
            self.aforward, retry_config, default=default_tool_retry
        )

    def get_json_schema(self) -> Dict[str, Any]:
        """Convert MCP tool schema to standard tool JSON schema."""
        return convert_mcp_schema_to_tool_schema(self._mcp_tool_info, self._namespace)

    @set_tool_attributes(execution_type="remote", protocol="mcp")
    def forward(self, **kwargs) -> Any:
        """Execute MCP tool call."""
        # Call MCP tool (wrap async in sync)
        result = F.wait_for(self._mcp_client.call_tool, self._mcp_tool_name, kwargs)

        # Handle errors
        if result.isError:
            error_text = extract_tool_result_text(result)
            raise RuntimeError(f"MCP tool error: {error_text}")

        # Extract and return result
        return extract_tool_result_text(result)

    @aset_tool_attributes(execution_type="remote", protocol="mcp")
    async def aforward(self, **kwargs) -> Any:
        """Execute MCP tool call asynchronously."""
        # Call MCP tool
        result = await self._mcp_client.call_tool(self._mcp_tool_name, kwargs)

        # Handle errors
        if result.isError:
            error_text = extract_tool_result_text(result)
            raise RuntimeError(f"MCP tool error: {error_text}")

        # Extract and return result
        return extract_tool_result_text(result)


class LocalTool(Tool):
    """Local tool implementation."""

    def __init__(
        self,
        name: str,
        description: str,
        annotations: Dict[str, Any],
        tool_config: Dict[str, Any],
        impl: Callable,
        display_name: Optional[str] = None,
        transport_params: Optional[Dict[str, Any]] = None,
        usage_guidance: Optional[str] = None,
    ):
        super().__init__()
        self.set_name(name)
        self.set_description(description)
        self.register_buffer("display_name", display_name or name)
        self.register_buffer("usage_guidance", usage_guidance)
        self.set_annotations(annotations)
        self.register_buffer("tool_config", tool_config)
        self.register_buffer("transport_params", transport_params or {})
        self.impl = impl  # Not a buffer for now
        self._param_defaults = get_fn_param_defaults(impl)

        # Apply retry
        retry_config = tool_config.get("retry")
        self.forward = apply_retry(
            self.forward, retry_config, default=default_tool_retry
        )
        self.aforward = apply_retry(
            self.aforward, retry_config, default=default_tool_retry
        )

    def _restore_transport_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Restore transport-lowered tool params using the original annotations."""
        annotations = {
            name: hint
            for name, hint in self.get_module_annotations().items()
            if name != "return"
        }
        if not annotations:
            return kwargs
        restored = dict(kwargs)
        for param_name, type_hint in annotations.items():
            if param_name not in restored:
                continue
            restored[param_name] = restore_transport_value(
                restored[param_name],
                type_hint,
                restore_structs=True,
            )
        return restored

    def _strip_none_default_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Treat `null` tool arguments as omission when Python defaults exist.

        This mirrors the tool schema contract used by LocalTool/function tools:
        strict providers require every field in `required`, so optional/defaulted
        params are represented as nullable in the schema and mapped back to
        Python defaults here when the model emits `null`.
        """
        if not self._param_defaults:
            return kwargs
        return {
            key: value
            for key, value in kwargs.items()
            if not (key in self._param_defaults and value is None)
        }

    def _prepare_call_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = self._restore_transport_params(kwargs)
        return self._strip_none_default_kwargs(kwargs)

    @set_tool_attributes(execution_type="local")
    def forward(self, **kwargs):
        kwargs = self._prepare_call_kwargs(kwargs)
        if inspect.iscoroutinefunction(self.impl):
            return F.wait_for(self.impl, **kwargs)
        return self.impl(**kwargs)

    @aset_tool_attributes(execution_type="local")
    async def aforward(self, *args, **kwargs):
        kwargs = self._prepare_call_kwargs(kwargs)
        if hasattr(self.impl, "acall"):
            return await self.impl.acall(*args, **kwargs)
        elif inspect.iscoroutinefunction(self.impl):
            return await self.impl(*args, **kwargs)
        # `to_thread` preserves the runtime ContextVars used by injected handles.
        return await asyncio.to_thread(self.impl, *args, **kwargs)


def _inspect_tool_metadata(impl: Callable) -> ToolMetadata:  # noqa: C901
    """Extract normalized metadata from a callable tool."""
    tool_config = dotdict(deepcopy(getattr(impl, "tool_config", dotdict())))
    tool_config.setdefault("on_demand", False)
    tool_config.setdefault("exposed", True)

    name_overridden = tool_config.pop("name_overridden", None)
    configured_display_name = tool_config.get("display_name")
    configured_usage_guidance = tool_config.get("usage_guidance")

    # Case 1: Uninitialized or initialized class
    if inspect.isclass(impl) or callable(impl):
        if not callable(impl):
            raise NotImplementedError(
                "To transform a class in `nn.Tool`"
                " is necessary implement a `def __call__`"
            )

        doc = (
            getattr(impl, "description", None)
            or getattr(impl, "__doc__", None)
            or getattr(impl.__call__, "__doc__", None)
        )
        if doc is None:
            raise NotImplementedError(
                "To transform a class into a `nn.Tool` "
                "it is necessary to implement a docstring. "
                "Can be: a cls attr `self.docstring`, or"
                "a docstring in the class or in `def __call__`"
            )

        name = (
            name_overridden
            or getattr(impl, "name", None)
            or getattr(impl, "__name__", None)
        )
        display_name = configured_display_name or getattr(impl, "display_name", None)
        usage_guidance = configured_usage_guidance or getattr(
            impl, "usage_guidance", None
        )

        # Instantiate class first if needed, so we can get instance attributes
        class_annotation_source = impl if inspect.isclass(impl) else None
        if inspect.isclass(impl):
            impl = impl()  # Initialized
            display_name = display_name or getattr(impl, "display_name", None)
            usage_guidance = usage_guidance or getattr(impl, "usage_guidance", None)

        # Now extract annotations (after instantiation for classes)
        annotation_source = None
        annotations = getattr(impl, "annotations", None)
        if annotations is None:
            annotations = getattr(impl, "__annotations__", None)
            if annotations is not None:
                annotation_source = class_annotation_source or impl
        if annotations is None:
            annotations = getattr(impl.__call__, "__annotations__", None)
            if annotations is not None:
                annotation_source = impl.__call__
        if annotations is None:
            if fn_has_parameters(impl.__call__):
                raise NotImplementedError(
                    "To transform a class in `nn.Tool` is necessary "
                    "to implement annotations of types hint in "
                    "`self.annotations`, `self.__annotations__` or in `def __call__`"
                )
            annotations = {}
        annotations = _resolve_tool_annotations(annotation_source, annotations)

    # Case 2: Function
    elif inspect.isfunction(impl) or inspect.iscoroutinefunction(impl):
        if hasattr(impl, "__doc__") and impl.__doc__ is not None:
            doc = impl.__doc__
        else:
            raise NotImplementedError(
                "To transform a function into a `nn.Tool` "
                "is necessary to implement a docstring"
            )

        annotations = impl.__annotations__
        annotation_source = impl

        if annotations is None:
            if fn_has_parameters(impl):
                raise NotImplementedError(
                    "To transform a function into a `nn.Tool` "
                    "is necessary to implement parameters "
                    "annotations of types hint "
                )
            annotations = {}
        annotations = _resolve_tool_annotations(annotation_source, annotations)

        name = name_overridden or impl.__name__
        display_name = configured_display_name or getattr(impl, "display_name", None)
        usage_guidance = configured_usage_guidance or getattr(
            impl, "usage_guidance", None
        )

    else:
        raise ValueError(
            "The given object is not a callable function, class, or instance"
        )

    if tool_config.get("handoff", False):
        name = "transfer_to_" + name

    if isinstance(impl, ToolBucket):
        tool_kind = ToolBucket.tool_kind
    else:
        tool_kind = tool_config.get("tool_kind") or getattr(impl, "tool_kind", None)
        tool_kind = tool_kind or "tool"
    if not isinstance(tool_kind, str) or not tool_kind.strip():
        raise ValueError(f"The tool `{name}` must define a non-empty tool_kind.")
    tool_config["tool_kind"] = tool_kind

    declared_capabilities = tool_config.get("background_capabilities")
    if declared_capabilities is not None:
        if not is_background_capable(tool_config):
            raise ValueError(
                "`background_capabilities` requires `background=True` or "
                "`allow_background=True`."
            )
        tool_config["background_capabilities"] = normalize_background_capabilities(
            declared_capabilities
        )

    annotations, hidden_params = split_hidden_annotations(annotations)
    if hidden_params:
        tool_config["_hidden_params"] = hidden_params
    if tool_config.get("handle") is not None:
        tool_config["handle"] = normalize_handle_access(tool_config["handle"])
        if "handle" not in hidden_params:
            raise ValueError(
                "Tools configured with `handle` must declare `handle: mf.Hidden`."
            )

    if tool_config.get("handoff", False) or tool_config.get("disable_input", False):
        annotations = {}  # pass only the model state
    else:
        if tool_config.get("inject_message", False):
            annotations.pop("message", None)
        if tool_config.get("inject_messages", False):
            annotations.pop("messages", None)
        if tool_config.get("handle") is not None:
            annotations.pop("handle", None)
        if tool_config.get("inject_vars", False):
            annotations.pop("vars", None)
        if tool_config.get("allow_background", False) and not tool_config.get(
            "background", False
        ):
            annotations[RUNTIME_BACKGROUND_PARAM] = Optional[bool]

    if tool_config.get("spawn"):
        doc = "This tool will not generate a return. \n" + doc
    if tool_config.get("background"):
        doc = "This tool runs in the background and returns a task id. \n" + doc
    elif tool_config.get("allow_background", False):
        doc = (
            "This tool can run in the background when "
            f"`{RUNTIME_BACKGROUND_PARAM}=true`; otherwise it runs normally. \n" + doc
        )

    return ToolMetadata(
        name=name,
        description=doc,
        annotations=annotations,
        tool_config=tool_config,
        impl=impl,
        display_name=display_name or name,
        usage_guidance=usage_guidance,
    )


def _convert_metadata_to_local_tool(metadata: ToolMetadata) -> LocalTool:
    return LocalTool(
        name=metadata.name,
        description=metadata.description,
        annotations=metadata.annotations,
        tool_config=metadata.tool_config,
        impl=metadata.impl,
        display_name=metadata.display_name,
        usage_guidance=metadata.usage_guidance,
    )


def _resolve_tool_annotations(
    annotation_source: Callable | None,
    annotations: Mapping[str, Any],
) -> Dict[str, Any]:
    resolved = dict(annotations)
    if annotation_source is None:
        return resolved
    try:
        type_hints = get_type_hints(annotation_source)
    except Exception:
        return resolved
    if not type_hints:
        return resolved
    resolved.update(type_hints)
    return resolved


def _convert_module_to_nn_tool(impl: Callable) -> Tool:
    """Convert a callable in nn.Tool."""
    return _convert_metadata_to_local_tool(_inspect_tool_metadata(impl))


def _metadata_from_tool(tool: Tool) -> ToolMetadata:
    return ToolMetadata(
        name=tool.name,
        description=tool.get_module_description() or "",
        annotations=tool.get_module_annotations(),
        tool_config=getattr(tool, "tool_config", {}),
        impl=getattr(tool, "impl", tool),
        display_name=getattr(tool, "display_name", None) or tool.name,
        usage_guidance=getattr(tool, "usage_guidance", None),
        source_tool=tool,
    )


class ToolLibrary(Module, metaclass=AutoParams):
    """ToolLibrary is a Module type that manage tool calls over the tool library."""

    def __init__(
        self,
        name: str,
        tools: List[Callable],
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        task_store: Any | None = None,
    ):
        """Initialize the ToolLibrary.

        Args:
        name:
            Library name.
        tools:
            A list of callables.
        mcp_servers:
            List of MCP server configurations. Each config should contain:
            - name: Namespace for tools from this server
            - transport: "stdio" or "http"
            - For stdio: command, args, cwd, env
            - For http: base_url, headers
            - Optional: include_tools, exclude_tools, tool_config
        """
        super().__init__()
        self.set_name(f"{name}_tool_library")
        self.library = ModuleDict()
        self.register_buffer("tool_configs", {})
        self.register_buffer("tool_owners", {})
        self._buckets = ToolBucketManager(
            self.library,
            self.tool_configs,
            self.tool_owners,
            _metadata_from_tool,
        )
        self.register_buffer("mcp_clients", {})
        self._task_store = task_store
        self._agent_inbox: Optional[AgentInbox] = None
        self._disabled_background_task_tool_names: set[str] = set()
        self._handle: Optional[ToolLibraryHandle] = None
        self._background_dispatcher: Optional[BackgroundTaskDispatcher] = None
        for tool in tools:
            self.add(tool)
        if mcp_servers:
            self._initialize_mcp_clients(mcp_servers)

    def get_handle(self) -> ToolLibraryHandle:
        if self._handle is None:
            self._handle = ToolLibraryHandle(self)
        return self._handle

    def get_background_dispatcher(self) -> BackgroundTaskDispatcher:
        if self._background_dispatcher is None:
            self._background_dispatcher = BackgroundTaskDispatcher(self.get_handle())
        return self._background_dispatcher

    def _get_default_task_store(self) -> Any:
        if self._task_store is None:
            self._task_store = InMemoryTaskStore()
        return self._task_store

    def get_agent_inbox(self) -> AgentInbox:
        if self._agent_inbox is None:
            self._agent_inbox = AgentInbox(
                owner=self.name,
                store=InMemoryAgentInboxStore(),
            )
        return self._agent_inbox

    @property
    def _bucket_graph(self) -> ToolBucketGraph:
        """Return a read-only view over the current bucket ownership tree."""
        return self._buckets.graph

    def add(self, tool: Callable) -> str:
        """Add a local tool in library."""
        transaction = ToolRegistrationTransaction()
        try:
            tool_name = self._add(tool, transaction=transaction)
            if transaction.reconcile_background:
                self._reconcile_runtime_tools(transaction=transaction)
        except Exception:
            transaction.rollback()
            self._buckets.refresh_presentations()
            raise
        return tool_name

    def _add(
        self,
        tool: Callable,
        *,
        transaction: ToolRegistrationTransaction,
    ) -> str:
        """Register one tool as part of an existing registration transaction."""
        if isinstance(tool, ToolMetadata):
            metadata = tool
        elif isinstance(tool, Tool):
            metadata = _metadata_from_tool(tool)
        else:
            metadata = _inspect_tool_metadata(tool)

        metadata.tool_config = dotdict(metadata.tool_config)
        metadata.tool_config.setdefault("on_demand", False)
        metadata.tool_config.setdefault("exposed", True)
        if not isinstance(metadata.tool_config["exposed"], bool):
            raise TypeError("`exposed` must be a bool.")
        metadata.tool_config["capabilities"] = normalize_tool_capabilities(
            metadata.tool_config.get("capabilities")
        )
        if "inject_handle" in metadata.tool_config:
            raise ValueError(
                "`inject_handle` was removed; configure exact access with `handle`."
            )
        if metadata.tool_config.get("handle") is not None:
            metadata.tool_config["handle"] = normalize_handle_access(
                metadata.tool_config["handle"]
            )
            hidden_params = metadata.tool_config.get("_hidden_params", {})
            if "handle" not in hidden_params and not is_hidden_annotation(
                metadata.annotations.get("handle")
            ):
                raise ValueError(
                    "Tools configured with `handle` must declare `handle: mf.Hidden`."
                )

        self._bucket_graph.validate_unique_names(metadata)
        if is_background_capable(metadata.tool_config):
            ToolBackground.validate_background_capabilities(
                metadata.impl,
                metadata.tool_config,
            )

        # On-demand tools are held by the search bucket until explicit activation.
        if (
            metadata.tool_config.get("on_demand", False)
            and ToolSearchTool.name not in self.library
        ):
            self._add(ToolSearchTool(), transaction=transaction)

        self._register_tool(metadata, transaction=transaction)
        return metadata.name

    def remove(self, tool_name: str):
        node = self._bucket_graph.find_node(tool_name)
        if node is None:
            raise ValueError(f"The tool name `{tool_name}` is not in tool library")
        node_bucket = node.bucket
        if isinstance(node_bucket, ToolBucket) and node_bucket.tools:
            raise ValueError(
                f"The bucket tool `{tool_name}` still captures tools and cannot "
                "be removed."
            )
        config = self.tool_configs.get(tool_name, {})
        is_task_tool = ToolBackground.is_active_task_tool(
            library=self,
            tool_name=tool_name,
            config=config,
            base_tools=BASE_TASK_TOOLS,
            capability_tools=BACKGROUND_CAPABILITY_TOOLS,
            metadata_factory=_inspect_tool_metadata,
        )
        was_background = not is_reserved_tool_kind(config) and is_background_capable(
            config
        )
        owner = node.parent
        remove_empty_owner = False
        if owner is not None:
            self._buckets.release(owner, tool_name)
            owner_bucket = self._bucket_graph.require_bucket(owner)
            remove_empty_owner = bool(
                owner_bucket.expose_captured_names and not owner_bucket.tools
            )
        self._remove_registered_tool(tool_name)

        if remove_empty_owner:
            self.remove(owner)

        if is_task_tool:
            self._disabled_background_task_tool_names.add(tool_name)
        elif was_background:
            self._reconcile_runtime_tools()

    def _remove_registered_tool(self, tool_name: str) -> None:
        tool = self.library.get(tool_name)
        self._buckets.unbind(tool)
        self.tool_owners.pop(tool_name, None)
        if tool_name in self.library:
            self.library.pop(tool_name)
        self.tool_configs.pop(tool_name, None)

    def _remove_reconciled_tool(
        self,
        tool_name: str,
        *,
        transaction: ToolRegistrationTransaction | None = None,
    ) -> None:
        """Remove a runtime tool and preserve it for registration rollback."""
        tool = self.library.get(tool_name)
        config = self.tool_configs.get(tool_name)
        if tool is None or config is None:
            return
        position = list(self.library).index(tool_name)
        owner = self.tool_owners.get(tool_name)
        exposed = bool(config.get("exposed", True))
        if owner is not None:
            self._buckets.release(owner, tool_name, exposed=exposed)
        self._remove_registered_tool(tool_name)
        if transaction is not None:
            transaction.record(
                partial(
                    self._restore_reconciled_tool,
                    tool_name,
                    tool=tool,
                    config=config,
                    owner=owner,
                    exposed=exposed,
                    position=position,
                )
            )

    def _restore_reconciled_tool(
        self,
        tool_name: str,
        *,
        tool: Tool,
        config: Dict[str, Any],
        owner: str | None,
        exposed: bool,
        position: int,
    ) -> None:
        config["exposed"] = exposed
        trailing = [
            (name, self.library.pop(name)) for name in list(self.library)[position:]
        ]
        self.library.update({tool_name: tool})
        self.library.update(dict(trailing))
        self.tool_configs[tool_name] = config
        if owner is not None:
            self.tool_owners[tool_name] = owner

    def clear(self):
        self._buckets.unbind_all()
        self.library.clear()
        self.tool_configs.clear()
        self.tool_owners.clear()
        for mcp_data in self.mcp_clients.values():
            F.wait_for(mcp_data["client"].disconnect)
        self.mcp_clients.clear()
        self._disabled_background_task_tool_names.clear()
        if self._background_dispatcher is not None:
            self._background_dispatcher.clear()

    def _register_tool(
        self,
        metadata: ToolMetadata,
        *,
        transaction: ToolRegistrationTransaction,
    ) -> Tool:
        bucket = metadata.impl if isinstance(metadata.impl, ToolBucket) else None
        captures = []
        pending: list[ToolMetadata] = []
        if bucket is not None:
            captures = self._bucket_graph.validate_registration(
                metadata,
                _metadata_from_tool,
            )
            for captured in captures:
                bucket.validate_capture(_metadata_from_tool(captured.tool))
            pending = list(bucket.tools.values())

        parent_names = self._bucket_graph.matching_buckets(metadata)
        if parent_names:
            self._bucket_graph.require_bucket(parent_names[0]).validate_capture(
                metadata
            )

        # Convert callable metadata to the local executable representation when needed.
        tool = (
            metadata.source_tool
            if isinstance(metadata.source_tool, Tool)
            else _convert_metadata_to_local_tool(metadata)
        )

        tool_config = dotdict(metadata.tool_config)
        tool.register_buffer("tool_config", tool_config)
        self.tool_configs[tool.name] = tool_config
        self.library.update({tool.name: tool})
        if bucket is not None:
            self._buckets.bind(tool.name, bucket)
        transaction.record(
            partial(
                self._undo_registered_tool,
                tool.name,
                bucket=bucket,
                pending=pending,
            )
        )

        for captured in captures:
            self._buckets.capture(
                tool.name,
                _metadata_from_tool(captured.tool),
                transaction=transaction,
            )
        for candidate in pending:
            self._add(candidate, transaction=transaction)
        if parent_names:
            self._buckets.capture(
                parent_names[0],
                _metadata_from_tool(tool),
                transaction=transaction,
            )

        if bucket is not None:
            bucket.refresh()
            self._buckets.sync_presentation(tool.name, bucket)

        self._update_background_registration(tool, tool_config, transaction)
        return tool

    def _update_background_registration(
        self,
        tool: Tool,
        tool_config: Mapping[str, Any],
        transaction: ToolRegistrationTransaction,
    ) -> None:
        if is_reserved_tool_kind(tool_config):
            was_disabled = tool.name in self._disabled_background_task_tool_names
            self._disabled_background_task_tool_names.discard(tool.name)
            if was_disabled:
                transaction.record(
                    partial(
                        self._disabled_background_task_tool_names.add,
                        tool.name,
                    )
                )
        elif is_background_capable(tool_config):
            transaction.reconcile_background = True

    def _undo_registered_tool(
        self,
        tool_name: str,
        *,
        bucket: ToolBucket | None,
        pending: list[ToolMetadata],
    ) -> None:
        self._remove_registered_tool(tool_name)
        if bucket is not None:
            self._buckets.unbind(bucket, pending)
            bucket.refresh()

    def _activate_on_demand(self, owner_name: str, tool_name: str) -> str:
        return self._buckets.activate_on_demand(
            owner_name,
            tool_name,
            remove_owner=self.remove,
        )

    @staticmethod
    def _create_mcp_client(
        server_config: Mapping[str, Any],
        namespace: str,
    ) -> Any:
        transport_type = server_config.get("transport", "stdio")
        if transport_type == "stdio":
            command = server_config.get("command")
            if not command:
                raise ValueError(
                    f"MCP server '{namespace}' stdio transport requires 'command'"
                )
            return MCPClient.from_stdio(
                command=command,
                args=server_config.get("args"),
                cwd=server_config.get("cwd"),
                env=server_config.get("env"),
                timeout=server_config.get("timeout", 30.0),
            )
        if transport_type == "http":
            base_url = server_config.get("base_url")
            if not base_url:
                raise ValueError(
                    f"MCP server '{namespace}' http transport requires 'base_url'"
                )
            return MCPClient.from_http(
                base_url=base_url,
                timeout=server_config.get("timeout", 30.0),
                headers=server_config.get("headers"),
                auth=server_config.get("auth"),
            )
        raise ValueError(
            f"Unknown transport type: {transport_type}. "
            "Supported types: 'stdio', 'http'"
        )

    def _initialize_mcp_clients(self, mcp_servers: List[Dict[str, Any]]):
        """Initialize MCP clients from server configurations."""
        for server_config in mcp_servers:
            namespace = server_config.get("name")
            if not namespace:
                raise ValueError("MCP server config must include 'name' field")

            client = self._create_mcp_client(server_config, namespace)

            # Connect and list tools with error handling
            try:
                connection = F.wait_for(client.connect)
                if isinstance(connection, TaskError):
                    raise connection.exception

                all_tools = F.wait_for(client.list_tools, use_cache=False)
                if isinstance(all_tools, TaskError):
                    raise all_tools.exception

                # Apply filters
                include_tools = server_config.get("include_tools")
                exclude_tools = server_config.get("exclude_tools")
                filtered_tools = filter_tools(all_tools, include_tools, exclude_tools)

                # Create MCPTool for each remote tool
                tool_configs = server_config.get("tool_config", {})
                for mcp_tool_info in filtered_tools:
                    tool_config = tool_configs.get(mcp_tool_info.name, {})

                    # Create MCPTool instance
                    mcp_tool = MCPTool(
                        name=mcp_tool_info.name,
                        mcp_client=client,
                        mcp_tool_info=mcp_tool_info,
                        namespace=namespace,
                        config=tool_config,
                    )

                    self.add(mcp_tool)

                self.mcp_clients[namespace] = {
                    "client": client,
                    "tools": filtered_tools,
                    "tool_config": tool_configs,
                }

                logger.debug(
                    f"Successfully connected to MCP server `{namespace}` "
                    f"with {len(filtered_tools)} tools"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize MCP server '{namespace}': {e!s}",
                    exc_info=True,
                )
                # Continue with other servers instead of failing completely

    def get_tools(self) -> Iterator[Dict[str, Tool]]:
        return self.library.items()

    def get_tool(self, tool_name: str) -> Tool:
        """Return the executable wrapper for any canonical graph node."""
        tool = self.library.get(tool_name)
        if tool is None:
            raise ValueError(f"The tool `{tool_name}` is no longer available.")
        if not isinstance(tool, Tool):
            raise ValueError(f"Tool `{tool_name}` has no executable wrapper.")
        return tool

    def get_tool_names(self, owner: str | None = None) -> List[str]:
        """Get public names or the captured descendants of one bucket."""
        if owner is not None:
            node = self._bucket_graph.find_node(owner)
            if node is not None and node.bucket is not None:
                return [
                    child.name for child in self._bucket_graph.bucket_descendants(owner)
                ]
        names = [
            name
            for name in self.library
            if self.tool_configs.get(name, {}).get("exposed", True)
        ]
        for node in self._bucket_graph.iter_nodes():
            bucket = node.bucket
            if isinstance(bucket, ToolBucket) and bucket.expose_captured_names:
                names.extend(name for name in bucket.tools if name not in names)
        return names

    def get_tool_display_names(self) -> Dict[str, str]:
        """Return human-readable display names keyed by registered tool name."""
        display_names = {}
        for tool_name, tool in self.library.items():
            if not self.tool_configs.get(tool_name, {}).get("exposed", True):
                continue
            display_names[tool_name] = getattr(tool, "display_name", None) or tool_name

        return display_names

    def get_tool_usage_guidance(
        self, tool_names: Optional[set[str]] = None
    ) -> List[Dict[str, str]]:
        """Return usage guidance metadata for tools that define it."""
        guidance = []
        display_names = self.get_tool_display_names()

        for tool_name, tool in self.library.items():
            if not self.tool_configs.get(tool_name, {}).get("exposed", True):
                continue
            if tool_names is not None and tool_name not in tool_names:
                continue
            usage_guidance = getattr(tool, "usage_guidance", None)
            if usage_guidance:
                guidance.append(
                    {
                        "name": tool_name,
                        "display_name": display_names.get(tool_name, tool_name),
                        "guidance": usage_guidance,
                    }
                )

        return guidance

    def get_mcp_tool_names(self) -> List[str]:
        """Get names of all MCP tools (with namespace)."""
        tool_names = []
        for namespace, mcp_data in self.mcp_clients.items():
            for tool in mcp_data["tools"]:
                tool_names.append(f"{namespace}__{tool.name}")
        return tool_names

    def get_tool_json_schemas(self) -> List[Dict[str, Any]]:
        """Returns a list of JSON schemas from local and MCP tools."""
        return [
            tool.get_json_schema()
            for name, tool in self.library.items()
            if self.tool_configs.get(name, {}).get("exposed", True)
        ]

    def get_tool_annotations(self) -> Dict[str, Dict[str, Any]]:
        """Return local tool annotations keyed by tool name."""
        annotations = {}
        for tool_name, tool in self.library.items():
            if not self.tool_configs.get(tool_name, {}).get("exposed", True):
                continue
            annotations[tool_name] = {
                name: hint
                for name, hint in tool.get_module_annotations().items()
                if name != "return"
            }
        return annotations

    def set_agent_inbox(self, agent_inbox: AgentInbox) -> None:
        self._agent_inbox = agent_inbox

    def set_task_store(self, task_store: Any) -> None:
        if task_store is not None:
            self._task_store = task_store

    def get_task_store(self, task_store: Any = None) -> Any:
        if task_store is not None:
            return task_store
        context_task_store = get_execution_context().get("task_store")
        if context_task_store is not None:
            return context_task_store
        return self._get_default_task_store()

    def _reconcile_runtime_tools(
        self,
        *,
        transaction: ToolRegistrationTransaction | None = None,
    ) -> None:
        """Rebuild runtime-provided tools from the current ownership graph."""
        self._sync_background_task_tools(transaction=transaction)

    def _sync_background_task_tools(
        self,
        *,
        transaction: ToolRegistrationTransaction | None = None,
    ) -> None:
        ToolBackground.sync_task_tools(
            library=self,
            disabled_tool_names=self._disabled_background_task_tool_names,
            base_tools=BASE_TASK_TOOLS,
            capability_tools=BACKGROUND_CAPABILITY_TOOLS,
            metadata_factory=_inspect_tool_metadata,
            transaction=transaction,
        )

    # --- Tool Call Preparation ---

    def _build_call_params(  # noqa: C901
        self,
        *,
        tool: Tool,
        tool_name: str,
        tool_params: Any,
        config: Mapping[str, Any],
        message: Optional[Any],
        messages: List[Dict[str, Any]],
        vars: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if config.get("handoff", False) or config.get("disable_input", False):
            call_params: Dict[str, Any] = {}
        else:
            call_params = coerce_tool_params(tool_name, tool_params)

        for param_name in config.get("_hidden_params") or {}:
            call_params.pop(param_name, None)

        inject_vars = config.get("inject_vars", False)
        if inject_vars:
            if isinstance(inject_vars, list):
                missing = [key for key in inject_vars if key not in vars]
                if missing:
                    subject = "agent" if config.get("tool_kind") == "agent" else "tool"
                    raise ValueError(
                        f"The {subject} `{tool_name}` requires the injected "
                        f"parameter `{missing[0]}`, but it was not found."
                    )
                if config.get("tool_kind") == "agent":
                    call_params["vars"] = {key: vars[key] for key in inject_vars}
                else:
                    for key in inject_vars:
                        call_params[key] = vars[key]
            elif inject_vars is True:
                call_params["vars"] = vars

        if config.get("inject_messages", False):
            if should_copy_injected_messages(tool, config):
                call_params["messages"] = deepcopy(messages)
            else:
                call_params["messages"] = messages

        if config.get("inject_message", False):
            call_params["message"] = message

        impl = getattr(tool, "impl", tool)
        hidden_params = config.get("_hidden_params") or {}
        if isinstance(impl, ToolBucket) and "tools" in hidden_params:
            context = get_execution_context()
            scoped = self.get_handle().for_tool(
                tool_name=tool_name,
                agent_inbox=context.get("agent_inbox"),
                task_store=context.get("task_store"),
                message=message,
                messages=messages,
                vars=vars,
            )
            call_params["tools"] = ToolBucketHandle(scoped)

        if config.get("handle") is not None:
            context = get_execution_context()
            call_params["handle"] = self.get_handle().tool_view(
                access=config["handle"],
                tool_name=tool_name,
                agent_inbox=context.get("agent_inbox"),
                task_store=context.get("task_store"),
                message=message,
                messages=messages,
                vars=vars,
            )

        return call_params

    def _record_tool_activity(
        self,
        *,
        activity_recorder: Any,
        tool_name: str,
        parameters: Mapping[str, Any] | None,
    ) -> None:
        config = self.tool_configs.get(tool_name, {})
        if (
            activity_recorder is None
            or is_reserved_tool_kind(config)
            or ToolLibraryOperator.is_operator_tool(self.library.get(tool_name))
        ):
            return
        activity_recorder.tool_call(tool_name, parameters)

    def _prepare_tool_kwargs(
        self,
        *,
        tool: Tool,
        tool_name: str,
        tool_params: Any,
        config: Mapping[str, Any],
        message: Optional[Any],
        messages: List[Dict[str, Any]],
        vars: Mapping[str, Any],
        activity_recorder: Any,
    ) -> tuple[Dict[str, Any], dict[str, Any] | None]:
        call_params = self._build_call_params(
            tool=tool,
            tool_name=tool_name,
            tool_params=tool_params,
            config=config,
            message=message,
            messages=messages,
            vars=vars,
        )
        response_params = build_call_parameters_for_response(call_params)
        if response_params is not None:
            for param_name in config.get("_hidden_params") or {}:
                response_params.pop(param_name, None)
        self._record_tool_activity(
            activity_recorder=activity_recorder,
            tool_name=tool_name,
            parameters=response_params,
        )
        return call_params, response_params

    def _prepare_execution(
        self,
        *,
        tool_id: str,
        tool_name: str,
        arguments: Any,
        message: Any = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
        owner: str | None = None,
        exposed_only: bool = False,
        inline: bool = False,
    ) -> PreparedToolExecution:
        """Resolve one tool call and inject its runtime arguments."""
        node = self._bucket_graph.find_node(tool_name)
        if node is None or (exposed_only and not node.config.get("exposed", True)):
            raise ToolNotAvailableError(f"Tool `{tool_name}` not found.")
        if owner is not None:
            owner_node = self._bucket_graph.find_node(owner)
            if owner_node is None or owner_node.bucket is None:
                raise ValueError(f"Tool `{owner}` is not an executable bucket.")
            if not self._bucket_graph.is_descendant(owner, tool_name):
                raise ValueError(
                    f"Tool `{tool_name}` is outside bucket `{owner}` capture scope."
                )

        tool = self.get_tool(tool_name)
        config = node.config
        call_params, response_params = self._prepare_tool_kwargs(
            tool=tool,
            tool_name=tool_name,
            tool_params=arguments,
            config=config,
            message=message,
            messages=messages if messages is not None else [],
            vars=vars if vars is not None else {},
            activity_recorder=get_execution_context().get("task_activity_recorder"),
        )
        if inline:
            mode = "call"
        elif config.get("spawn", False):
            mode = "spawn"
        elif should_dispatch_background(config=config, call_params=call_params):
            mode = "background"
        elif config.get("call_as_response", False):
            mode = "response"
        else:
            mode = "call"
        return PreparedToolExecution(
            id=tool_id,
            name=tool_name,
            tool=tool,
            config=config,
            call_params=call_params,
            response_params=response_params,
            mode=mode,
        )

    @staticmethod
    def _is_immediate_execution(execution: PreparedToolExecution) -> bool:
        return execution.mode != "call"

    @staticmethod
    def _requires_serial_execution(execution: PreparedToolExecution) -> bool:
        impl = getattr(execution.tool, "impl", execution.tool)
        return bool(getattr(impl, "_serial_execution", False))

    @staticmethod
    def _execution_returns_directly(execution: PreparedToolExecution) -> bool:
        if execution.mode in {"spawn", "background"}:
            return False
        if execution.mode == "response":
            return True
        return bool(execution.config.get("return_direct", False))

    def _execute_prepared(self, execution: PreparedToolExecution) -> ToolCall:
        config = execution.config
        if execution.mode == "spawn":
            F.spawn(execution.tool, **execution.call_params)
            return ToolCall(
                id=execution.id,
                name=execution.name,
                parameters=execution.response_params,
                result=f"The `{execution.name}` tool was dispatched. "
                "This tool will not generate a return.",
            )
        if execution.mode == "background":
            return self.get_background_dispatcher().dispatch(
                tool=execution.tool,
                tool_id=execution.id,
                tool_name=execution.name,
                call_params=execution.call_params,
                config=config,
                response_params=execution.response_params,
            )
        if execution.mode == "response":
            return ToolCall(
                id=execution.id,
                name=execution.name,
                parameters=execution.response_params,
            )

        call_params = dict(execution.call_params)
        call_params["tool_call_id"] = execution.id
        result = execution.tool(**call_params)
        return ToolCall(
            id=execution.id,
            name=execution.name,
            parameters=execution.response_params,
            result=result,
        )

    async def _aexecute_prepared(
        self,
        execution: PreparedToolExecution,
    ) -> ToolCall:
        config = execution.config
        if execution.mode == "spawn":
            await F.aspawn(execution.tool, **execution.call_params)
            return ToolCall(
                id=execution.id,
                name=execution.name,
                parameters=execution.response_params,
                result=f"The `{execution.name}` tool was dispatched. "
                "This tool will not generate a return.",
            )
        if execution.mode == "background":
            return self.get_background_dispatcher().dispatch(
                tool=execution.tool,
                tool_id=execution.id,
                tool_name=execution.name,
                call_params=execution.call_params,
                config=config,
                response_params=execution.response_params,
            )
        if execution.mode == "response":
            return ToolCall(
                id=execution.id,
                name=execution.name,
                parameters=execution.response_params,
            )

        call_params = dict(execution.call_params)
        call_params["tool_call_id"] = execution.id
        result = await execution.tool.acall(**call_params)
        return ToolCall(
            id=execution.id,
            name=execution.name,
            parameters=execution.response_params,
            result=result,
        )

    @staticmethod
    def _coerce_execution_result(
        execution: PreparedToolExecution,
        result: ToolCall | TaskError,
    ) -> ToolCall:
        if not isinstance(result, TaskError):
            return result
        if isinstance(result.exception, TaskInterruptRequestedError):
            raise result.exception
        return ToolCall(
            id=execution.id,
            name=execution.name,
            parameters=execution.response_params,
            error=str(result),
        )

    def execute(
        self,
        tool_name: str,
        arguments: Any,
        *,
        message: Any = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
        tool_call_id: str = "",
    ) -> Any:
        """Execute one tool through the same runtime pipeline as ``forward``."""
        execution = self._prepare_execution(
            tool_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            message=message,
            messages=messages,
            vars=vars,
        )
        return self._execute_prepared(execution).result

    def _execute_scoped(
        self,
        owner: str,
        tool_name: str,
        arguments: Any,
        *,
        message: Any = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        execution = self._prepare_execution(
            tool_id="",
            tool_name=tool_name,
            arguments=arguments,
            message=message,
            messages=messages,
            vars=vars,
            owner=owner,
        )
        return self._execute_prepared(execution).result

    def _execute_inline(
        self,
        tool_name: str,
        arguments: Any,
        *,
        message: Any = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        execution = self._prepare_execution(
            tool_id="",
            tool_name=tool_name,
            arguments=arguments,
            message=message,
            messages=messages,
            vars=vars,
            inline=True,
        )
        return self._execute_prepared(execution).result

    async def aexecute(
        self,
        tool_name: str,
        arguments: Any,
        *,
        message: Any = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
        tool_call_id: str = "",
    ) -> Any:
        """Asynchronously execute one tool through the ``aforward`` pipeline."""
        execution = self._prepare_execution(
            tool_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            message=message,
            messages=messages,
            vars=vars,
        )
        return (await self._aexecute_prepared(execution)).result

    async def _aexecute_scoped(
        self,
        owner: str,
        tool_name: str,
        arguments: Any,
        *,
        message: Any = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        execution = self._prepare_execution(
            tool_id="",
            tool_name=tool_name,
            arguments=arguments,
            message=message,
            messages=messages,
            vars=vars,
            owner=owner,
        )
        return (await self._aexecute_prepared(execution)).result

    def forward(
        self,
        tool_callings: List[Tuple[str, str, Any]],
        message: Optional[Any] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> ToolResponses:
        """Execute model-originated tool calls with library config semantics."""
        prepared_calls: list[PreparedToolExecution] = []
        tool_calls: List[ToolCall] = []
        return_directly = bool(tool_callings)

        for tool_id, tool_name, arguments in tool_callings:
            try:
                execution = self._prepare_execution(
                    tool_id=tool_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    message=message,
                    messages=messages,
                    vars=vars,
                    exposed_only=True,
                )
            except ToolNotAvailableError as exc:
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=arguments,
                        error=f"Error: {exc}",
                    )
                )
                return_directly = False
                continue

            if self._is_immediate_execution(execution):
                tool_calls.append(self._execute_prepared(execution))
            else:
                prepared_calls.append(execution)
            if self._execution_returns_directly(execution):
                if execution.mode == "response":
                    return_directly = True
            else:
                return_directly = False

        if prepared_calls:
            if any(self._requires_serial_execution(call) for call in prepared_calls):
                results = tuple(
                    F.scatter_gather([partial(self._execute_prepared, call)])[0]
                    for call in prepared_calls
                )
            else:
                results = F.scatter_gather(
                    [partial(self._execute_prepared, call) for call in prepared_calls]
                )
            for execution, result in zip(prepared_calls, results):
                tool_calls.append(self._coerce_execution_result(execution, result))

        return ToolResponses(return_directly=return_directly, tool_calls=tool_calls)

    async def aforward(
        self,
        tool_callings: List[Tuple[str, str, Any]],
        message: Optional[Any] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> ToolResponses:
        """Asynchronously execute model-originated tool calls."""
        prepared_calls: list[PreparedToolExecution] = []
        tool_calls: List[ToolCall] = []
        return_directly = bool(tool_callings)

        for tool_id, tool_name, arguments in tool_callings:
            try:
                execution = self._prepare_execution(
                    tool_id=tool_id,
                    tool_name=tool_name,
                    arguments=arguments,
                    message=message,
                    messages=messages,
                    vars=vars,
                    exposed_only=True,
                )
            except ToolNotAvailableError as exc:
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=arguments,
                        error=f"Error: {exc}",
                    )
                )
                return_directly = False
                continue

            if self._is_immediate_execution(execution):
                tool_calls.append(await self._aexecute_prepared(execution))
            else:
                prepared_calls.append(execution)
            if self._execution_returns_directly(execution):
                if execution.mode == "response":
                    return_directly = True
            else:
                return_directly = False

        if prepared_calls:
            if any(self._requires_serial_execution(call) for call in prepared_calls):
                serial_results = []
                for call in prepared_calls:
                    result = await F.ascatter_gather(
                        [partial(self._aexecute_prepared, call)]
                    )
                    serial_results.append(result[0])
                results = tuple(serial_results)
            else:
                results = await F.ascatter_gather(
                    [partial(self._aexecute_prepared, call) for call in prepared_calls]
                )
            for execution, result in zip(prepared_calls, results):
                tool_calls.append(self._coerce_execution_result(execution, result))

        return ToolResponses(return_directly=return_directly, tool_calls=tool_calls)
