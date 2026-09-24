from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping

from msgflux.runtime.agent_inbox import ToolNotificationHandle
from msgflux.runtime.context import execution_context, get_execution_context

if TYPE_CHECKING:
    from msgflux.chat_messages import ChatMessages
    from msgflux.nn.modules.tool import ToolLibrary
    from msgflux.runtime.agent_inbox import AgentInbox
    from msgflux.tools.catalog import ToolRef
    from msgflux.tools.types import ToolBucketEntry


class ToolLibraryHandle:
    """Controlled handle exposed to runtime-aware tools."""

    def __init__(
        self,
        library: ToolLibrary,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        agent_inbox: AgentInbox | None = None,
        task_store: Any = None,
    ):
        self._library = library
        self._tool_name = tool_name
        self._tool_call_id = tool_call_id
        self._agent_inbox = agent_inbox
        self._task_store = task_store

    def for_tool(
        self,
        *,
        tool_name: str,
        agent_inbox: AgentInbox | None = None,
        task_store: Any = None,
        message: Any = None,
        messages: Any = None,
        vars: Mapping[str, Any] | None = None,  # noqa: A002
        tool_call_id: str | None = None,
        activity_recorder: Any = None,
    ) -> ToolLibraryHandle:
        if self._library.is_bucket(tool_name):
            return ToolBucketHandle(
                self._library,
                bucket_name=tool_name,
                agent_inbox=(
                    agent_inbox if agent_inbox is not None else self._agent_inbox
                ),
                task_store=task_store if task_store is not None else self._task_store,
                message=message,
                messages=messages,
                vars=vars,
                tool_call_id=tool_call_id,
                activity_recorder=activity_recorder,
            )
        return ToolLibraryHandle(
            self._library,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            agent_inbox=agent_inbox if agent_inbox is not None else self._agent_inbox,
            task_store=task_store if task_store is not None else self._task_store,
        )

    def add(self, tool: Callable) -> str:
        return self._library.add(tool)

    def remove(self, tool_name: str) -> str:
        self._library.remove(tool_name)
        return tool_name

    def load_tools(
        self,
        messages: ChatMessages,
        tool_names: List[str],
    ) -> List[str]:
        return self._library.load_tools(messages, tool_names)

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

    def get_ref(self, tool_name: str) -> ToolRef:
        return self._library.get_tool_ref(tool_name)

    def get_tool_definition(self, tool_name: str) -> Any:
        """Return the registered declaration, including execution requirements."""
        return self._library.get_tool_definition(tool_name)

    def __call__(
        self,
        tool: ToolRef | str,
        /,
        **arguments: Any,
    ) -> Any:
        return self._library.run(tool, arguments)

    async def acall(
        self,
        tool: ToolRef | str,
        /,
        **arguments: Any,
    ) -> Any:
        return await self._library.arun(tool, arguments)

    def get_task_future(self, task_id: str) -> Any | None:
        return self._library.get_background_dispatcher().get_task_future(task_id)

    def get_task_inbox(self, task_id: str) -> AgentInbox | None:
        return self._library.get_background_dispatcher().get_task_inbox(
            task_id,
            task_store=self.get_task_store(),
            agent_inbox=self.get_agent_inbox(),
        )

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
        ref = getattr(task_handle, "task_id", None) or self._tool_call_id
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
        metadata: Dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        source: str | None = None,
    ) -> Any:
        return self.get_notification().update(
            status,
            metadata=metadata,
            dedupe_key=dedupe_key,
            source=source,
        )

    def raise_if_interrupted(self) -> None:
        self.get_task().raise_if_interrupted()

    def raise_if_paused(self) -> None:
        self.get_task().raise_if_paused()

    def resume_background_agent_task(self, *, task: Any, message: str) -> str:
        with execution_context(
            task_store=self.get_task_store(),
            agent_inbox=self.get_agent_inbox(),
        ):
            return self._library.get_background_dispatcher().resume_agent_task(
                task=task,
                message=message,
            )

    def _get_background_agent_task(self, task_id: str) -> Any:
        task = self.get_task_store().get(task_id)
        if task is None:
            raise ValueError(f"Task `{task_id}` was not found.")
        if task.metadata.get("task_kind") != "agent":
            raise ValueError(f"Task `{task_id}` is not a background agent.")
        return task

    def recover_background_agent_task(self, task_id: str, *, message: str) -> str:
        """Claim an expired worker lease and resume an existing checkpoint."""
        task = self._get_background_agent_task(task_id)
        with execution_context(
            task_store=self.get_task_store(),
            agent_inbox=self.get_agent_inbox(),
        ):
            return self._library.get_background_dispatcher().resume_agent_task(
                task=task,
                message=message,
                recover_expired=True,
            )

    def reconcile_background_agent_task(self, task_id: str) -> str:
        """Claim an expired worker and persist its committed Agent result."""
        task = self._get_background_agent_task(task_id)
        with execution_context(
            task_store=self.get_task_store(),
            agent_inbox=self.get_agent_inbox(),
        ):
            return self._library.get_background_dispatcher().resume_agent_task(
                task=task,
                message="",
                recover_expired=True,
                reconcile_terminal=True,
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


class ToolBucketHandle(ToolLibraryHandle):
    """Scoped proxy used by a bucket to execute one of its captured tools."""

    def __init__(
        self,
        library: ToolLibrary,
        *,
        bucket_name: str,
        agent_inbox: AgentInbox | None = None,
        task_store: Any = None,
        message: Any = None,
        messages: Any = None,
        vars: Mapping[str, Any] | None = None,  # noqa: A002
        tool_call_id: str | None = None,
        activity_recorder: Any = None,
    ):
        super().__init__(
            library,
            tool_name=bucket_name,
            agent_inbox=agent_inbox,
            task_store=task_store,
        )
        self._bucket_name = bucket_name
        self._message = message
        self._messages = messages
        self._vars = vars
        self._tool_call_id = tool_call_id
        self._activity_recorder = activity_recorder

    def has_tool(self, tool_name: str) -> bool:
        return self._library.bucket_has_tool(self._bucket_name, tool_name)

    def list_captured_tools(self) -> List[str]:
        return self._library.get_bucket_tool_names(self._bucket_name)

    def list_entries(self) -> tuple[ToolBucketEntry, ...]:
        """Return captured-tool presentation without exposing executors."""
        return self._library.get_bucket_entries(self._bucket_name)

    def get_entry(self, tool_name: str) -> ToolBucketEntry:
        """Return one captured tool's execution-free presentation."""
        return self._library.get_bucket_entry(self._bucket_name, tool_name)

    def get_execution_namespace(self, tool_name: str) -> str:
        return self._library.get_bucket_execution_namespace(
            self._bucket_name,
            tool_name,
        )

    def get_tool_call_id(self) -> str | None:
        """Return the parent model call id carried by this scoped handle."""
        return self._tool_call_id

    def with_runtime(
        self,
        *,
        agent_inbox: AgentInbox | None = None,
        task_store: Any = None,
        activity_recorder: Any = None,
    ) -> ToolBucketHandle:
        """Copy this call scope while rebinding background runtime resources."""
        return ToolBucketHandle(
            self._library,
            bucket_name=self._bucket_name,
            agent_inbox=agent_inbox if agent_inbox is not None else self._agent_inbox,
            task_store=task_store if task_store is not None else self._task_store,
            message=self._message,
            messages=self._messages,
            vars=self._vars,
            tool_call_id=self._tool_call_id,
            activity_recorder=activity_recorder,
        )

    def __call__(
        self,
        tool_name: ToolRef | str,
        /,
        *,
        _runtime_arguments: Mapping[str, Any] | None = None,
        **arguments: Any,
    ) -> Any:
        tool_ref = (
            tool_name
            if not isinstance(tool_name, str)
            else self._library.get_tool_ref(tool_name)
        )
        return self._library.run(
            tool_ref,
            arguments,
            bucket_name=self._bucket_name,
            runtime_arguments=_runtime_arguments,
            message=self._message,
            messages=self._messages,
            vars=self._vars,
            parent_tool_call_id=self._tool_call_id,
            activity_recorder=self._activity_recorder,
        )

    async def acall(
        self,
        tool_name: ToolRef | str,
        /,
        *,
        _runtime_arguments: Mapping[str, Any] | None = None,
        **arguments: Any,
    ) -> Any:
        tool_ref = (
            tool_name
            if not isinstance(tool_name, str)
            else self._library.get_tool_ref(tool_name)
        )
        return await self._library.arun(
            tool_ref,
            arguments,
            bucket_name=self._bucket_name,
            runtime_arguments=_runtime_arguments,
            message=self._message,
            messages=self._messages,
            vars=self._vars,
            parent_tool_call_id=self._tool_call_id,
            activity_recorder=self._activity_recorder,
        )
