"""Composable extensions for ToolLibrary capabilities."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import msgflux.nn.functional as F
from msgflux.logger import logger
from msgflux.nn.hooks import Hook
from msgflux.nn.modules.module import Module
from msgflux.protocols.mcp import MCPClient, filter_tools
from msgflux.tools.builtin.task_tool import (
    BACKGROUND_CAPABILITY_TOOLS,
    BASE_TASK_TOOLS,
)
from msgflux.tools.builtin.tool_search import ToolSearchTool
from msgflux.tools.types import ToolBackground

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolLibrary

__all__ = [
    "BackgroundTasksExtension",
    "MCPServersExtension",
    "ToolLibraryExtension",
    "ToolLibraryExtensionHandle",
    "ToolSearchExtension",
]


class ToolLibraryExtension(Module):
    """A named package of tools and hooks owned by a ToolLibrary."""

    def __init__(self, name: str) -> None:
        super().__init__()
        if not isinstance(name, str) or not name.strip():
            raise ValueError("`name` must be a non-empty string")
        self.name = name
        self._library_ref = None

    @property
    def library(self) -> ToolLibrary:
        library = self._library_ref() if self._library_ref is not None else None
        if library is None:
            raise RuntimeError("The extension is not registered on a ToolLibrary")
        return library

    def _bind_library(self, library: ToolLibrary) -> None:
        self._library_ref = weakref.ref(library)

    def _unbind_library(self) -> None:
        self._library_ref = None

    def __getstate__(self):
        state = super().__getstate__()
        state["_library_ref"] = None
        return state

    def hooks(self) -> Iterable[Hook]:
        return ()

    def tools(self) -> Iterable[Any]:
        return ()

    def validate_tool(self, _library: ToolLibrary, definition: Any) -> None:
        """Validate a compiled definition before registration mutates state."""

    def on_tool_added(self, library: ToolLibrary, definition: Any) -> None:
        """React after a definition and executor are registered."""

    def on_tool_removed(self, library: ToolLibrary, definition: Any) -> None:
        """React after a definition and executor are removed."""

    def on_clear(self, _library: ToolLibrary) -> None:
        """Reset extension-owned state after the library is cleared."""

    def on_register(self, library: ToolLibrary) -> None:
        """Run local setup after contributions are installed."""

    def on_remove(self, library: ToolLibrary) -> None:
        """Clean up synchronously owned resources."""

    async def aon_remove(self, library: ToolLibrary) -> None:
        self.on_remove(library)


class ToolLibraryExtensionHandle:
    """Ownership handle returned by ToolLibrary.register_extension."""

    def __init__(self, library: ToolLibrary, name: str) -> None:
        self._library_ref = weakref.ref(library)
        self.name = name

    @property
    def active(self) -> bool:
        library = self._library_ref()
        return library is not None and library.has_extension(self.name)

    def remove(self) -> None:
        library = self._library_ref()
        if library is not None:
            library.remove_extension(self.name)

    async def aremove(self) -> None:
        library = self._library_ref()
        if library is not None:
            await library.aremove_extension(self.name)


class ToolSearchExtension(ToolLibraryExtension):
    """Install the builtin deferred-tool search bucket."""

    def __init__(self) -> None:
        super().__init__("tool_search")

    def tools(self):
        return (ToolSearchTool(),)


class BackgroundTasksExtension(ToolLibraryExtension):
    """Manage task controls derived from background-capable tools."""

    def __init__(self) -> None:
        super().__init__("background_tasks")
        self.register_buffer("disabled_tool_names", set())

    def validate_tool(self, _library: ToolLibrary, definition: Any) -> None:
        if ToolBackground.is_background_definition(definition):
            ToolBackground.validate_background_capabilities(definition)

    def on_tool_added(self, library: ToolLibrary, definition: Any) -> None:
        if ToolBackground.is_reserved_definition(definition):
            self.disabled_tool_names.discard(definition.name)
            return
        if ToolBackground.is_background_definition(definition):
            self.sync(library)

    def on_tool_removed(self, library: ToolLibrary, definition: Any) -> None:
        if ToolBackground.is_reserved_definition(definition):
            if self.is_active_task_tool(
                library=library,
                tool_name=definition.name,
                definition=definition,
            ):
                self.disabled_tool_names.add(definition.name)
            return
        if ToolBackground.is_background_definition(definition):
            self.sync(library)

    def on_clear(self, _library: ToolLibrary) -> None:
        self.disabled_tool_names.clear()

    def on_register(self, library: ToolLibrary) -> None:
        for definition in library.registry.definitions():
            self.validate_tool(library, definition)
        self.sync(library)

    def sync(self, library: ToolLibrary) -> None:
        ToolBackground.sync_task_tools(
            library=library,
            disabled_tool_names=self.disabled_tool_names,
            base_tools=BASE_TASK_TOOLS,
            capability_tools=BACKGROUND_CAPABILITY_TOOLS,
            definition_factory=library.inspect_tool_definition,
        )

    def is_active_task_tool(
        self,
        *,
        library: ToolLibrary,
        tool_name: str,
        definition: Any,
    ) -> bool:
        return ToolBackground.is_active_task_tool(
            library=library,
            tool_name=tool_name,
            definition=definition,
            base_tools=BASE_TASK_TOOLS,
            capability_tools=BACKGROUND_CAPABILITY_TOOLS,
            definition_factory=library.inspect_tool_definition,
        )

    def on_remove(self, library: ToolLibrary) -> None:
        task_tools = [
            tool_name
            for tool_name in tuple(library.library)
            if self.is_active_task_tool(
                library=library,
                tool_name=tool_name,
                definition=library.get_tool_definition(tool_name),
            )
        ]
        for tool_name in task_tools:
            library.remove(tool_name)
        self.disabled_tool_names.clear()


class MCPServersExtension(ToolLibraryExtension):
    """Connect MCP servers and contribute their remote tools."""

    def __init__(self, servers: Iterable[Mapping[str, Any]]) -> None:
        super().__init__("mcp_servers")
        self.servers = tuple(dict(server) for server in servers)
        self._tool_names: list[str] = []
        self._namespaces: list[str] = []

    def on_register(self, library: ToolLibrary) -> None:  # noqa: C901
        for server_config in self.servers:
            namespace = server_config.get("name")
            if not namespace:
                raise ValueError("MCP server config must include 'name' field")
            transport_type = server_config.get("transport", "stdio")
            if transport_type == "stdio":
                command = server_config.get("command")
                if not command:
                    raise ValueError(
                        f"MCP server '{namespace}' stdio transport requires 'command'"
                    )
                client = MCPClient.from_stdio(
                    command=command,
                    args=server_config.get("args"),
                    cwd=server_config.get("cwd"),
                    env=server_config.get("env"),
                    timeout=server_config.get("timeout", 30.0),
                )
            elif transport_type == "http":
                base_url = server_config.get("base_url")
                if not base_url:
                    raise ValueError(
                        f"MCP server '{namespace}' http transport requires 'base_url'"
                    )
                client = MCPClient.from_http(
                    base_url=base_url,
                    timeout=server_config.get("timeout", 30.0),
                    headers=server_config.get("headers"),
                    auth=server_config.get("auth"),
                )
            else:
                raise ValueError(
                    f"Unknown transport type: {transport_type}. "
                    "Supported types: 'stdio', 'http'"
                )

            added_names: list[str] = []
            try:
                F.wait_for(client.connect)
                all_tools = F.wait_for(client.list_tools, use_cache=False)
                filtered_tools = filter_tools(
                    all_tools,
                    server_config.get("include_tools"),
                    server_config.get("exclude_tools"),
                )
                tool_configs = server_config.get("tool_config", {})
                for tool_info in filtered_tools:
                    added_names.append(
                        library.add(
                            library.create_mcp_tool(
                                name=tool_info.name,
                                mcp_client=client,
                                mcp_tool_info=tool_info,
                                namespace=namespace,
                                config=tool_configs.get(tool_info.name, {}),
                            )
                        )
                    )
                library.mcp_clients[namespace] = {
                    "client": client,
                    "tools": filtered_tools,
                    "tool_config": tool_configs,
                }
                self._tool_names.extend(added_names)
                self._namespaces.append(namespace)
                logger.debug(
                    f"Successfully connected to MCP server `{namespace}` "
                    f"with {len(filtered_tools)} tools"
                )
            except Exception as exc:
                for tool_name in reversed(added_names):
                    try:
                        library.remove(tool_name)
                    except ValueError:
                        pass
                try:
                    F.wait_for(client.disconnect)
                except Exception as cleanup_exc:
                    logger.debug(
                        f"Failed to disconnect MCP server `{namespace}` after "
                        f"setup error: {cleanup_exc!s}"
                    )
                logger.error(
                    f"Failed to initialize MCP server '{namespace}': {exc!s}",
                    exc_info=True,
                )

    def on_remove(self, library: ToolLibrary) -> None:
        for tool_name in reversed(self._tool_names):
            try:
                library.remove(tool_name)
            except ValueError:
                pass
        self._tool_names.clear()
        for namespace in reversed(self._namespaces):
            mcp_data = library.mcp_clients.pop(namespace, None)
            if mcp_data is not None:
                F.wait_for(mcp_data["client"].disconnect)
        self._namespaces.clear()
