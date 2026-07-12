from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List

from msgflux.runtime.agent_inbox import ToolNotificationHandle
from msgflux.runtime.context import execution_context, get_execution_context

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
