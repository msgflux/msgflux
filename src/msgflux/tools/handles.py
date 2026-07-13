from __future__ import annotations

import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping

from msgflux.exceptions import HandleAccessError
from msgflux.runtime.agent_inbox import ToolNotificationHandle
from msgflux.runtime.context import execution_context, get_execution_context


HANDLE_ACTIONS = {
    "notifications": frozenset({"publish"}),
    "task": frozenset({"read", "progress", "activity", "interrupt_check"}),
    "tasks": frozenset(
        {"list", "read", "wait", "output", "interrupt", "activity", "message"}
    ),
    "tools": frozenset({"list", "get", "register", "remove"}),
    "background": frozenset({"dispatch", "resume"}),
}


def normalize_handle_access(
    access: dict[str, list[str]] | None,
) -> dict[str, list[str]] | None:
    """Validate and freeze exact access configured for a tool handle."""
    if access is None:
        return None
    if not isinstance(access, dict):
        raise TypeError("`handle` must be a dict of domain to action list.")
    if not access:
        raise ValueError("`handle` must grant at least one action.")

    normalized: dict[str, list[str]] = {}
    for domain, actions in access.items():
        if not isinstance(domain, str) or domain not in HANDLE_ACTIONS:
            valid = ", ".join(sorted(HANDLE_ACTIONS))
            raise ValueError(
                f"Unknown handle domain `{domain}`. Valid domains: {valid}."
            )
        if not isinstance(actions, list):
            raise TypeError(f"`handle.{domain}` must be a list of actions.")
        if not actions:
            raise ValueError(f"`handle.{domain}` must grant at least one action.")
        if any(not isinstance(action, str) for action in actions):
            raise TypeError(f"`handle.{domain}` actions must be strings.")
        unknown = sorted(set(actions) - HANDLE_ACTIONS[domain])
        if unknown:
            valid = ", ".join(sorted(HANDLE_ACTIONS[domain]))
            raise ValueError(
                f"Unknown handle action `{domain}.{unknown[0]}`. "
                f"Valid actions: {valid}."
            )
        normalized[domain] = list(dict.fromkeys(actions))
    return normalized

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolLibrary
    from msgflux.runtime.agent_inbox import AgentInbox


class ToolLibraryHandle:
    """Privileged runtime handle used internally by a tool library."""

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

    def tool_view(
        self,
        *,
        access: dict[str, list[str]],
        tool_name: str,
        agent_inbox: AgentInbox | None = None,
        task_store: Any = None,
    ) -> ToolHandle:
        scoped = self.for_tool(
            tool_name=tool_name,
            agent_inbox=agent_inbox,
            task_store=task_store,
        )
        return ToolHandle(scoped, access)

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


class _HandleFacet:
    __slots__ = ("_handle", "_actions", "_domain")

    def __init__(self, handle: ToolLibraryHandle, domain: str, actions: frozenset[str]):
        self._handle = handle
        self._domain = domain
        self._actions = actions

    def _require(self, action: str) -> None:
        if action not in self._actions:
            raise HandleAccessError(self._domain, action)


class NotificationHandle(_HandleFacet):
    def publish(
        self,
        *,
        status: str,
        hint: str | None = None,
        metadata: Dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        source: str | None = None,
    ) -> Any:
        self._require("publish")
        return self._handle.notify(
            status=status,
            hint=hint,
            metadata=metadata,
            dedupe_key=dedupe_key,
            source=source,
        )


class CurrentTaskHandle(_HandleFacet):
    def read(self) -> Dict[str, Any]:
        self._require("read")
        task_handle = self._handle.get_task()
        task = task_handle._store.get(task_handle.task_id)
        if task is None:
            raise RuntimeError(f"Current task `{task_handle.task_id}` was not found.")
        return task.to_dict()

    def progress(
        self,
        *,
        stage: str | None = None,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
        percent: float | None = None,
    ) -> Any:
        self._require("progress")
        return self._handle.update_progress(
            stage=stage,
            message=message,
            current=current,
            total=total,
            percent=percent,
        )

    def activity(
        self,
        *,
        kind: str,
        summary: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        self._require("activity")
        recorder = get_execution_context().get("task_activity_recorder")
        if recorder is None:
            raise RuntimeError("`handle.task.activity()` requires a background task.")
        return recorder.add(kind=kind, summary=summary, metadata=metadata)

    def interrupt_check(self) -> None:
        self._require("interrupt_check")
        self._handle.raise_if_interrupted()
        self._handle.raise_if_paused()


class ToolsHandle(_HandleFacet):
    def list(self) -> List[str]:
        self._require("list")
        return self._handle.list_tools()

    def get(self, tool_name: str) -> Any:
        self._require("get")
        return self._handle.get_tool(tool_name)

    def register(self, tool: Callable) -> str:
        self._require("register")
        return self._handle.add(tool)

    def remove(self, tool_name: str) -> str:
        self._require("remove")
        return self._handle.remove(tool_name)


class TasksHandle(_HandleFacet):
    def _store(self) -> Any:
        return self._handle.get_task_store()

    def _record(self, task_id: str) -> Any | None:
        from msgflux.tools.builtin.task_tool import get_scoped_task

        return get_scoped_task(self._store(), task_id)

    def read(self, task_id: str) -> Dict[str, Any] | None:
        self._require("read")
        task = self._record(task_id)
        if task is None:
            return None
        from msgflux.tools.builtin.task_tool import (
            build_task_timing_fields,
            format_task_activity_entry,
        )

        payload = task.to_dict()
        payload.update(build_task_timing_fields(task))
        last_activity = self._store().get_last_activity(task_id)
        if last_activity is not None:
            payload["last_activity_summary"] = format_task_activity_entry(last_activity)
        return payload

    def list(self, *, status: str | None = None) -> list[Dict[str, Any]]:
        self._require("list")
        from msgflux.tools.builtin.task_tool import (
            build_task_timing_fields,
            format_task_activity_entry,
            task_is_in_current_scope,
        )

        result = []
        store = self._store()
        for task in store.list(status=status):
            if not task_is_in_current_scope(task):
                continue
            payload = task.to_dict()
            payload.update(build_task_timing_fields(task))
            last_activity = store.get_last_activity(task.task_id)
            if last_activity is not None:
                payload["last_activity_summary"] = format_task_activity_entry(
                    last_activity
                )
            result.append(payload)
        return result

    def output(self, task_id: str) -> Any:
        self._require("output")
        from msgflux.tools.builtin.task_tool import build_task_result

        return build_task_result(task_id=task_id, task=self._record(task_id))

    def wait(self, task_id: str, *, timeout: float | None = None) -> Any:
        self._require("wait")
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError(
                    f"`timeout` must be float, int or None, given `{type(timeout)}`"
                )
            if timeout < 0:
                raise ValueError("`timeout` must be greater than or equal to 0.")
        from msgflux.tools.builtin.task_tool import (
            build_task_result,
            build_task_timeout_result,
        )

        task = self._record(task_id)
        if task is None or task.status in {"completed", "failed", "interrupted"}:
            return build_task_result(task_id=task_id, task=task)
        future = self._handle.get_task_future(task_id)
        if future is not None:
            try:
                future.result(timeout=timeout)
            except FutureTimeoutError:
                return build_task_timeout_result(
                    task_id=task_id, task=self._record(task_id)
                )
            except Exception:
                pass
            return build_task_result(task_id=task_id, task=self._record(task_id))

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            task = self._record(task_id)
            if task is None or task.status in {"completed", "failed", "interrupted"}:
                return build_task_result(task_id=task_id, task=task)
            if deadline is not None and time.monotonic() >= deadline:
                return build_task_timeout_result(task_id=task_id, task=task)
            time.sleep(0.05)

    def interrupt(self, task_id: str) -> Dict[str, Any]:
        self._require("interrupt")
        task = self._record(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if task.status in {"completed", "failed", "interrupted"}:
            return {
                "task_id": task_id,
                "status": task.status,
                "message": "Task already reached a terminal state.",
            }
        store = self._store()
        store.request_interrupt(task_id)
        future = self._handle.get_task_future(task_id)
        if future is not None and future.cancel():
            interrupted = store.interrupt(
                task_id, reason="Task was cancelled before it started running."
            )
            return {
                "task_id": task_id,
                "status": "interrupted",
                "message": "Task interrupted before execution started.",
                "task_status": (
                    interrupted.status if interrupted is not None else "interrupted"
                ),
            }
        return {
            "task_id": task_id,
            "status": "interrupt_requested",
            "message": (
                "Interrupt requested. The task will interrupt at the next "
                "cooperative checkpoint."
            ),
        }

    def activity(self, task_id: str, *, limit: int | None = 10) -> Any:
        self._require("activity")
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError(f"`limit` must be int or None, given `{type(limit)}`")
            if limit <= 0:
                raise ValueError("`limit` must be greater than 0.")
        from msgflux.tools.builtin.task_tool import (
            format_task_activity_entry,
            get_task_background_capabilities,
        )

        task = self._record(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if "activity" not in get_task_background_capabilities(task):
            return {
                "task_id": task_id,
                "status": "unsupported",
                "error": "task_activity requires the task activity capability.",
            }
        return [
            format_task_activity_entry(item)
            for item in self._store().list_activity(task_id, limit=limit)
        ]

    def message(self, task_id: str, message: str) -> Dict[str, Any]:
        self._require("message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("`message` must be a non-empty string.")
        message = message.strip()
        from msgflux.tools.builtin.task_tool import (
            get_task_background_capabilities,
            truncate_activity_text,
        )

        task = self._record(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if "message" not in get_task_background_capabilities(task):
            return {
                "task_id": task_id,
                "status": "unsupported",
                "error": "task_message requires the task message capability.",
            }
        inbox = self._handle.get_task_inbox(task_id)
        if task.status == "running":
            if inbox is None:
                return {
                    "task_id": task_id,
                    "status": "unsupported",
                    "error": "The running task does not expose a message inbox.",
                }
            inbox.publish(
                {
                    "source": "task_message",
                    "ref": task_id,
                    "status": "message",
                    "hint": message,
                    "metadata": {"direction": "root_to_task"},
                }
            )
            self._store().add_activity(
                task_id,
                kind="message",
                summary=f"Root message: {truncate_activity_text(message)}",
                metadata={"direction": "root_to_task"},
            )
            return {
                "task_id": task_id,
                "status": "delivered",
                "message": "Message delivered to the running background agent.",
            }
        if task.metadata.get("task_kind") != "agent":
            return {
                "task_id": task_id,
                "status": "unsupported",
                "error": "Task continuation is not implemented for this task kind.",
            }
        return {
            "task_id": task_id,
            "status": "resumed",
            "message": self._handle.resume_background_agent_task(
                task=task, message=message
            ),
        }


class BackgroundHandle(_HandleFacet):
    def dispatch(self, *args: Any, **kwargs: Any) -> Any:
        self._require("dispatch")
        return self._handle._library.get_background_dispatcher().dispatch(
            *args, **kwargs
        )

    def resume(self, *, task_id: str, message: str) -> str:
        self._require("resume")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("`message` must be a non-empty string.")
        task = TasksHandle(self._handle, "tasks", frozenset({"read"}))._record(task_id)
        if task is None:
            raise ValueError(f"Task `{task_id}` was not found in the current scope.")
        return self._handle.resume_background_agent_task(
            task=task, message=message.strip()
        )


class ToolHandle:
    """Least-privilege runtime handle injected into a configured tool."""

    __slots__ = ("notifications", "task", "tasks", "tools", "background")

    def __init__(
        self,
        handle: ToolLibraryHandle,
        access: dict[str, list[str]],
    ):
        normalized = normalize_handle_access(
            {domain: list(actions) for domain, actions in access.items()}
        )
        assert normalized is not None
        self.notifications = NotificationHandle(
            handle, "notifications", frozenset(normalized.get("notifications", ()))
        )
        self.task = CurrentTaskHandle(
            handle, "task", frozenset(normalized.get("task", ()))
        )
        self.tasks = TasksHandle(
            handle, "tasks", frozenset(normalized.get("tasks", ()))
        )
        self.tools = ToolsHandle(
            handle, "tools", frozenset(normalized.get("tools", ()))
        )
        self.background = BackgroundHandle(
            handle, "background", frozenset(normalized.get("background", ()))
        )
