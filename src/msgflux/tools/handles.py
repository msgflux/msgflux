from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping

from msgflux.runtime.agent_inbox import ToolNotificationHandle
from msgflux.runtime.context import get_execution_context
from msgflux.tools.helpers import RUNTIME_BACKGROUND_PARAM as _RUNTIME_BACKGROUND_PARAM

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolLibrary
    from msgflux.runtime.agent_inbox import AgentInbox
    from msgflux.tools.types import ToolMetadata


class ToolLibraryHandle:
    """Controlled handle exposed to runtime-aware tools."""

    def __init__(
        self,
        library: ToolLibrary,
        *,
        tool_name: str | None = None,
        agent_inbox: AgentInbox | None = None,
    ):
        self._library = library
        self._tool_name = tool_name
        self._agent_inbox = agent_inbox

    def for_tool(
        self,
        *,
        tool_name: str,
        agent_inbox: AgentInbox | None = None,
    ) -> ToolLibraryHandle:
        return ToolLibraryHandle(
            self._library,
            tool_name=tool_name,
            agent_inbox=agent_inbox or self._agent_inbox,
        )

    def add(self, tool: Callable) -> str:
        return self._library.add(tool)

    def add_runtime_tool(self, metadata: ToolMetadata) -> None:
        from msgflux.nn.modules.tool import LocalTool  # noqa: PLC0415

        name = metadata.name
        if name in self._library.library and name not in self.runtime_tool_names:
            raise ValueError(
                f"The runtime tool `{name}` conflicts with an existing tool."
            )
        self.runtime_tool_names.add(name)
        tool = LocalTool(
            name=name,
            description=metadata.description,
            annotations=metadata.annotations,
            tool_config=metadata.tool_config,
            impl=metadata.impl,
        )
        if name in self._library.library and self._library.tool_configs.get(name):
            raise ValueError(
                f"The runtime tool `{name}` conflicts with an existing tool."
            )
        self._library.library.update({name: tool})
        self._library.tool_configs[name] = metadata.tool_config

    def remove(self, tool_name: str) -> str:
        if tool_name in self.runtime_tool_names:
            raise ValueError(f"The runtime tool `{tool_name}` cannot be removed.")
        self._library.remove(tool_name)
        return tool_name

    @property
    def runtime_tool_names(self) -> set[str]:
        return self._library._runtime_tool_names

    @property
    def agent_inbox(self) -> AgentInbox:
        return self._agent_inbox or self._library.agent_inbox

    @property
    def task_store(self) -> Any:
        return self._library.task_store

    @property
    def task_runtime_tools(self) -> Any:
        return self._library.task_runtime_tools

    def list_tools(self) -> List[str]:
        return self._library.get_tool_names()

    def get_tool(self, tool_name: str) -> Any:
        if tool_name not in self._library.library:
            raise ValueError(f"The tool `{tool_name}` is no longer available.")
        return self._library.library[tool_name]

    def get_task_future(self, task_id: str) -> Any | None:
        return self._library.background_dispatcher.get_task_future(task_id)

    def get_task_inbox(self, task_id: str) -> AgentInbox | None:
        return self._library.background_dispatcher.get_task_inbox(task_id)

    @property
    def task(self) -> Any:
        task_handle = get_execution_context().get("task_handle")
        if task_handle is None:
            raise RuntimeError("`handle.task` is only available in background tools.")
        return task_handle

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def notification(self) -> ToolNotificationHandle:
        if self._tool_name is None:
            raise RuntimeError(
                "`handle.notification` is only available on a tool-scoped handle."
            )
        task_handle = get_execution_context().get("task_handle")
        ref = getattr(task_handle, "task_id", None)
        return self.build_notification_handle(
            tool_name=self._tool_name,
            ref=ref,
            agent_inbox=self.agent_inbox,
        )

    def set_running(
        self,
        *,
        stage: str | None = None,
        message: str | None = None,
    ) -> Any:
        return self.task.set_running(stage=stage, message=message)

    def update_progress(
        self,
        *,
        stage: str | None = None,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
        percent: float | None = None,
    ) -> Any:
        return self.task.update_progress(
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
        return self.notification.update(
            status,
            hint=hint,
            metadata=metadata,
            dedupe_key=dedupe_key,
            source=source,
        )

    def raise_if_interrupted(self) -> None:
        self.task.raise_if_interrupted()

    def raise_if_paused(self) -> None:
        self.task.raise_if_paused()

    def resume_background_agent_task(self, *, task: Any, message: str) -> str:
        return self._library.background_dispatcher.resume_agent_task(
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
                "tool_kind": getattr(tool, "tool_config", {}).get(
                    "tool_kind", "tool"
                ),
            }
        if metadata is None:
            raise ValueError(f"Tool `{tool_name}` not found.")
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
            metadata = self._library.on_demand_tools.pop(tool_name, None)
            if metadata is None:
                continue
            metadata.tool_config["on_demand"] = False
            self._library.tool_configs.pop(tool_name, None)
            self._library.add(metadata)
            activated.append(tool_name)
        self._library._sync_on_demand_runtime_tools()
        return activated

    def build_call_parameters_for_response(
        self,
        params: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if params is None:
            return None
        if hasattr(params, "to_dict"):
            parameters = params.to_dict()
        else:
            parameters = dict(params)
        for key in (
            "vars",
            "messages",
            "message",
            "task",
            "notification",
            "scope",
            "handle",
            "tool_call_id",
            _RUNTIME_BACKGROUND_PARAM,
        ):
            parameters.pop(key, None)
        return parameters

    def build_notification_handle(
        self,
        *,
        tool_name: str,
        ref: str | None = None,
        agent_inbox: AgentInbox | None = None,
    ) -> ToolNotificationHandle:
        execution_context = get_execution_context()
        inbox = agent_inbox or execution_context.get("agent_inbox") or self.agent_inbox
        return ToolNotificationHandle(
            inbox,
            ref=ref,
            metadata={"tool": tool_name},
        )
