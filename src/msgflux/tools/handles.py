from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List

from msgflux.runtime.agent_inbox import ToolNotificationHandle
from msgflux.runtime.context import execution_context, get_execution_context
from msgflux.tools.types import ToolBucket

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolLibrary
    from msgflux.runtime.agent_inbox import AgentInbox


class ToolLibraryHandle:
    """Controlled handle exposed to runtime-aware tools."""

    def __init__(
        self,
        library: ToolLibrary,
        *,
        tool_name: str | None = None,
        agent_inbox: AgentInbox | None = None,
        task_store: Any = None,
    ):
        self._library = library
        self._tool_name = tool_name
        self._agent_inbox = agent_inbox
        self._task_store = task_store

    def for_tool(
        self,
        *,
        tool_name: str,
        agent_inbox: AgentInbox | None = None,
        task_store: Any = None,
    ) -> ToolLibraryHandle:
        return ToolLibraryHandle(
            self._library,
            tool_name=tool_name,
            agent_inbox=agent_inbox if agent_inbox is not None else self._agent_inbox,
            task_store=task_store if task_store is not None else self._task_store,
        )

    def add(self, tool: Callable) -> str:
        return self._library.add(tool)

    def remove(self, tool_name: str) -> str:
        self._library.remove(tool_name)
        return tool_name

    def get_agent_inbox(self) -> AgentInbox:
        if self._agent_inbox is not None:
            return self._agent_inbox
        return self._library.get_agent_inbox()

    def get_task_store(self) -> Any:
        return self._library.get_task_store(self._task_store)

    def list_tools(self) -> List[str]:
        return self._library.get_tool_names()

    def get_tool(self, tool_name: str) -> Any:
        if tool_name not in self._library.library:
            raise ValueError(f"The tool `{tool_name}` is no longer available.")
        return self._library.library[tool_name]

    def get_task_future(self, task_id: str) -> Any | None:
        return self._library.get_background_dispatcher().get_task_future(task_id)

    def get_task_inbox(self, task_id: str) -> AgentInbox | None:
        return self._library.get_background_dispatcher().get_task_inbox(task_id)

    def get_task(self) -> Any:
        task_handle = get_execution_context().get("task_handle")
        if task_handle is None:
            raise RuntimeError(
                "`handle.get_task()` is only available in background tools."
            )
        return task_handle

    def get_task_id(self) -> str:
        return self.get_task().task_id

    def get_notification(self) -> ToolNotificationHandle:
        if self._tool_name is None:
            raise RuntimeError(
                "`handle.get_notification()` is only available on a tool-scoped handle."
            )
        task_handle = get_execution_context().get("task_handle")
        ref = getattr(task_handle, "task_id", None)
        return self.build_notification_handle(
            tool_name=self._tool_name,
            ref=ref,
            agent_inbox=self.get_agent_inbox(),
        )

    def set_running(
        self,
        *,
        stage: str | None = None,
        message: str | None = None,
    ) -> Any:
        return self.get_task().set_running(stage=stage, message=message)

    def update_progress(
        self,
        *,
        stage: str | None = None,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
        percent: float | None = None,
    ) -> Any:
        return self.get_task().update_progress(
            stage=stage,
            message=message,
            current=current,
            total=total,
            percent=percent,
        )

    def notify(
        self,
        *,
        status: str,
        hint: str | None = None,
        metadata: Dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        source: str | None = None,
    ) -> Any:
        return self.get_notification().update(
            status,
            hint=hint,
            metadata=metadata,
            dedupe_key=dedupe_key,
            source=source,
        )

    def raise_if_interrupted(self) -> None:
        self.get_task().raise_if_interrupted()

    def raise_if_paused(self) -> None:
        self.get_task().raise_if_paused()

    def resume_background_agent_task(self, *, task: Any, message: str) -> str:
        with execution_context(task_store=self.get_task_store()):
            return self._library.get_background_dispatcher().resume_agent_task(
                task=task,
                message=message,
            )

    def list_on_demand_tools(self) -> List[str]:
        return list(self._library.on_demand_tools.keys())

    def describe_tool(self, tool_name: str) -> dict[str, Any]:
        metadata = self._library.on_demand_tools.get(tool_name)
        if metadata is None and tool_name in self._library.library:
            tool = self._library.library[tool_name]
            return {
                "name": tool.name,
                "display_name": getattr(tool, "display_name", None) or tool.name,
                "description": tool.description,
                "usage_guidance": getattr(tool, "usage_guidance", None),
                "tool_kind": getattr(tool, "tool_config", {}).get("tool_kind", "tool"),
            }
        if metadata is None:
            metadata = ToolBucket.find_captured_metadata(
                tool_name,
                self._library.library,
                self._library.tool_configs,
            )
        if metadata is None:
            raise ValueError(f"Tool `{tool_name}` not found.")
        return self._describe_metadata(metadata)

    @staticmethod
    def _describe_metadata(metadata: Any) -> dict[str, Any]:
        return {
            "name": metadata.name,
            "display_name": metadata.display_name or metadata.name,
            "description": metadata.description,
            "usage_guidance": metadata.usage_guidance,
            "tool_kind": metadata.tool_config.get("tool_kind", "tool"),
        }

    def search_on_demand_tools(
        self,
        *,
        query: str,
        max_results: int = 5,
    ) -> List[str]:
        query_lower = query.strip().lower()
        terms = [term for term in query_lower.split() if term]
        if not terms:
            return []

        matches = []
        for tool_name, metadata in self._library.on_demand_tools.items():
            name_parts = tool_name.lower().replace("__", " ").replace("_", " ")
            description = (metadata.description or "").lower()
            score = 0
            if query_lower == tool_name.lower():
                score += 100
            if query_lower in name_parts:
                score += 40
            for term in terms:
                if term in name_parts:
                    score += 15
                if description and term in description:
                    score += 5
            if score > 0:
                matches.append((score, tool_name))

        matches.sort(key=lambda item: (-item[0], item[1]))
        return [tool_name for _, tool_name in matches[:max_results]]

    def select_on_demand_tools(self, requested: List[str]) -> List[str]:
        resolved = []
        normalized = {
            tool_name.lower(): tool_name for tool_name in self._library.on_demand_tools
        }
        for tool_name in requested:
            match = normalized.get(tool_name.lower())
            if match is not None and match not in resolved:
                resolved.append(match)
        return resolved

    def activate_on_demand_tools(self, tool_names: List[str]) -> List[str]:
        activated = []
        for tool_name in tool_names:
            metadata = self._library.on_demand_tools.get(tool_name)
            if metadata is None:
                continue
            self._activate_on_demand_tool(metadata)
            activated.append(tool_name)
        self._library._sync_on_demand_operator_tools()
        return activated

    def _activate_on_demand_tool(self, metadata: Any) -> None:
        """Promote one on-demand tool while restoring it if registration fails."""
        tool_name = metadata.name
        original_config = self._library.tool_configs.get(tool_name)
        promoted = replace(
            metadata,
            tool_config={**metadata.tool_config, "on_demand": False},
        )

        # `ToolLibrary.add` rejects names still present in the on-demand registry.
        self._library.on_demand_tools.pop(tool_name)
        self._library.tool_configs.pop(tool_name, None)
        try:
            self._library.add(promoted)
        except Exception:
            self._library.on_demand_tools[tool_name] = metadata
            if original_config is not None:
                self._library.tool_configs[tool_name] = original_config
            raise

    def search_tools(
        self,
        *,
        query: str | None,
        select: List[str] | None,
        include_descriptions: bool,
        max_results: int,
    ) -> dict[str, Any]:
        """Search or promote on-demand tools through one stateful operation."""
        total = len(self._library.on_demand_tools)
        if select is not None:
            matches = self.select_on_demand_tools(select)
            loaded = self.activate_on_demand_tools(matches)
        else:
            matches = self.search_on_demand_tools(
                query=query or "",
                max_results=max_results,
            )
            loaded = []

        descriptions = []
        if include_descriptions:
            descriptions = [self.describe_tool(tool_name) for tool_name in matches]

        return {
            "query": query,
            "matches": matches,
            "loaded": loaded,
            "descriptions": descriptions,
            "total_on_demand_tools": total,
        }

    def build_notification_handle(
        self,
        *,
        tool_name: str,
        ref: str | None = None,
        agent_inbox: AgentInbox | None = None,
    ) -> ToolNotificationHandle:
        execution_context = get_execution_context()
        inbox = agent_inbox
        if inbox is None:
            inbox = execution_context.get("agent_inbox")
        if inbox is None:
            inbox = self.get_agent_inbox()
        return ToolNotificationHandle(
            inbox,
            ref=ref,
            metadata={"tool": tool_name},
        )
