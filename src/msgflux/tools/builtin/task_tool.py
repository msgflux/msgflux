from __future__ import annotations

import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Collection, Dict, Optional

from msgflux.core.registry import Registry
from msgflux.runtime.context import get_execution_context
from msgflux.tools.helpers import (
    BACKGROUND_ACTIVITY_TOOL_KIND,
    BACKGROUND_MESSAGE_TOOL_KIND,
    DEFAULT_AGENT_BACKGROUND_CAPABILITIES,
    normalize_background_capabilities,
)
from msgflux.tools.types import Hidden, ToolBackground
from msgflux.utils.time import parse_utc_timestamp

_base_task_tools = Registry()
_background_activity_tools = Registry()
_background_message_tools = Registry()


@_base_task_tools
class TaskStatusTool(ToolBackground):
    name = "task_status"
    description = "Get the current status of a background task by task_id."
    annotations = {"task_id": str, "handle": Hidden, "return": Dict[str, Any]}

    def __call__(self, task_id: str, handle: Hidden) -> Dict[str, Any]:
        task_store = handle.get_task_store()
        task = get_scoped_task(task_store, task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        payload = task.to_dict()
        payload.update(build_task_timing_fields(task))
        last_activity = task_store.get_last_activity(task_id)
        if last_activity is not None:
            payload["last_activity_summary"] = format_task_activity_entry(last_activity)
        return payload


@_base_task_tools
class TaskListTool(ToolBackground):
    name = "task_list"
    description = "List background tasks registered in the current tool library."
    annotations = {
        "status": Optional[str],
        "handle": Hidden,
        "return": list[Dict[str, Any]],
    }

    def __call__(
        self,
        status: str | None = None,
        handle: Hidden = None,
    ) -> list[Dict[str, Any]]:
        task_store = handle.get_task_store()
        tasks = []
        for task in task_store.list(status=status):
            if not task_is_in_current_scope(task):
                continue
            payload = task.to_dict()
            payload.update(build_task_timing_fields(task))
            last_activity = task_store.get_last_activity(task.task_id)
            if last_activity is not None:
                payload["last_activity_summary"] = format_task_activity_entry(
                    last_activity
                )
            tasks.append(payload)
        return tasks


@_base_task_tools
class TaskOutputTool(ToolBackground):
    name = "task_output"
    description = "Get the final output of a background task by task_id."
    annotations = {"task_id": str, "handle": Hidden, "return": Any}

    def __call__(self, task_id: str, handle: Hidden) -> Any:
        task = get_scoped_task(handle.get_task_store(), task_id)
        return build_task_result(task_id=task_id, task=task)


@_base_task_tools
class TaskWaitTool(ToolBackground):
    name = "task_wait"
    description = (
        "Wait for a background task to finish. Returns the final output, failed "
        "payload, or a timeout status."
    )
    annotations = {
        "task_id": str,
        "timeout": Optional[float],
        "handle": Hidden,
        "return": Any,
    }

    def __call__(  # noqa: C901
        self,
        task_id: str,
        timeout: float | None = None,
        handle: Hidden = None,
    ) -> Any:
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError(
                    f"`timeout` must be float, int or None, given `{type(timeout)}`"
                )
            if timeout < 0:
                raise ValueError("`timeout` must be greater than or equal to 0.")

        task_store = handle.get_task_store()
        task = get_scoped_task(task_store, task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if task.status in {"completed", "failed", "interrupted"}:
            return build_task_result(task_id=task_id, task=task)

        future = handle.get_task_future(task_id)
        if future is not None:
            try:
                future.result(timeout=timeout)
            except FutureTimeoutError:
                task = get_scoped_task(task_store, task_id)
                return build_task_timeout_result(
                    task_id=task_id,
                    task=task,
                )
            except Exception:
                task = get_scoped_task(task_store, task_id)
                return build_task_result(task_id=task_id, task=task)
            task = get_scoped_task(task_store, task_id)
            return build_task_result(task_id=task_id, task=task)

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            task = get_scoped_task(task_store, task_id)
            if task is None or task.status in {"completed", "failed", "interrupted"}:
                return build_task_result(task_id=task_id, task=task)
            if deadline is not None and time.monotonic() >= deadline:
                return build_task_timeout_result(
                    task_id=task_id,
                    task=task,
                )
            time.sleep(0.05)


@_base_task_tools
class TaskInterruptTool(ToolBackground):
    name = "task_interrupt"
    description = (
        "Request a cooperative interrupt for a background task. Interrupts "
        "immediately only if the task has not started yet."
    )
    annotations = {"task_id": str, "handle": Hidden, "return": Dict[str, Any]}

    def __call__(self, task_id: str, handle: Hidden) -> Dict[str, Any]:
        task_store = handle.get_task_store()
        task = get_scoped_task(task_store, task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}

        if task.status in {"completed", "failed", "interrupted"}:
            return {
                "task_id": task_id,
                "status": task.status,
                "message": "Task already reached a terminal state.",
            }

        task_store.request_interrupt(task_id)
        future = handle.get_task_future(task_id)
        if future is not None and future.cancel():
            interrupted = task_store.interrupt(
                task_id,
                reason="Task was cancelled before it started running.",
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


@_background_activity_tools
class TaskActivityTool(ToolBackground):
    name = "task_activity"
    tool_kind = BACKGROUND_ACTIVITY_TOOL_KIND
    description = "List compact activity entries for a background task."
    annotations = {
        "task_id": str,
        "limit": Optional[int],
        "handle": Hidden,
        "return": Any,
    }

    def __call__(
        self,
        task_id: str,
        limit: int | None = 10,
        handle: Hidden = None,
    ) -> Any:
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError(f"`limit` must be int or None, given `{type(limit)}`")
            if limit <= 0:
                raise ValueError("`limit` must be greater than 0.")
        task_store = handle.get_task_store()
        task = get_scoped_task(task_store, task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if "activity" not in get_task_background_capabilities(task):
            return {
                "task_id": task_id,
                "status": "unsupported",
                "error": "task_activity requires the task activity capability.",
            }
        activity = task_store.list_activity(
            task_id,
            limit=limit,
        )
        return [format_task_activity_entry(item) for item in activity]


@_background_message_tools
class TaskMessageTool(ToolBackground):
    name = "task_message"
    tool_kind = BACKGROUND_MESSAGE_TOOL_KIND
    description = (
        "Send a message to a capable background task. Agent tasks can also "
        "resume from their checkpoint."
    )
    annotations = {
        "task_id": str,
        "message": str,
        "handle": Hidden,
        "return": Dict[str, Any],
    }

    def __call__(
        self,
        task_id: str,
        message: str,
        handle: Hidden,
    ) -> Dict[str, Any]:
        task_store = handle.get_task_store()
        task = get_scoped_task(task_store, task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if not isinstance(message, str) or not message.strip():
            raise ValueError("`message` must be a non-empty string.")

        capabilities = get_task_background_capabilities(task)
        if "message" not in capabilities:
            return {
                "task_id": task_id,
                "status": "unsupported",
                "error": "task_message requires the task message capability.",
            }

        task_inbox = handle.get_task_inbox(task_id)
        if task.status == "running":
            if task_inbox is None:
                return {
                    "task_id": task_id,
                    "status": "unsupported",
                    "error": "The running task does not expose a message inbox.",
                }
            task_inbox.publish(
                {
                    "source": "task_message",
                    "ref": task_id,
                    "status": "message",
                    "hint": message.strip(),
                    "metadata": {"direction": "root_to_task"},
                }
            )
            task_store.add_activity(
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
        resumed = handle.resume_background_agent_task(
            task=task,
            message=message.strip(),
        )
        return {
            "task_id": task_id,
            "status": "resumed",
            "message": resumed,
        }


BASE_TASK_TOOLS = _base_task_tools.to_list()
BACKGROUND_ACTIVITY_TOOLS = _background_activity_tools.to_list()
BACKGROUND_MESSAGE_TOOLS = _background_message_tools.to_list()
BACKGROUND_CAPABILITY_TOOLS = {
    "activity": BACKGROUND_ACTIVITY_TOOLS,
    "message": BACKGROUND_MESSAGE_TOOLS,
}


def task_is_in_current_scope(task: Any) -> bool:
    """Return whether a model-facing task belongs to the active execution."""
    context = get_execution_context()
    thread_id = context.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        return True
    if task.metadata.get("thread_id") != thread_id:
        return False

    namespace = context.get("namespace")
    task_namespace = task.metadata.get("namespace")
    return task_namespace is None or task_namespace == namespace


def get_scoped_task(task_store: Any, task_id: str) -> Any | None:
    task = task_store.get(task_id)
    if task is None or not task_is_in_current_scope(task):
        return None
    return task


def get_task_background_capabilities(task: Any) -> tuple[str, ...]:
    capabilities = task.metadata.get("background_capabilities")
    if capabilities is None:
        if task.metadata.get("task_kind") == "agent":
            return DEFAULT_AGENT_BACKGROUND_CAPABILITIES
        return ()
    return normalize_background_capabilities(capabilities)


def build_task_result(*, task_id: str, task: Any | None) -> Any:
    if task is None:
        return {"task_id": task_id, "status": "not_found"}
    if task.status == "completed":
        return task.result
    if task.status == "failed":
        return {"task_id": task_id, "status": task.status, "error": task.error}
    if task.status == "interrupted":
        return {
            "task_id": task_id,
            "status": task.status,
            "reason": task.metadata.get("interrupt_reason"),
        }
    return {
        "task_id": task_id,
        "status": task.status,
        "progress": task.progress.to_dict(),
    }


def build_task_timeout_result(
    *,
    task_id: str,
    task: Any | None,
) -> Dict[str, Any]:
    if task is None:
        return {"task_id": task_id, "status": "not_found"}
    payload = {
        "task_id": task_id,
        "status": "timeout",
        "task_status": task.status,
    }
    if task.status not in {"completed", "failed"}:
        if task.status == "interrupted":
            payload["reason"] = task.metadata.get("interrupt_reason")
            return payload
        payload["progress"] = task.progress.to_dict()
    elif task.status == "failed":
        payload["error"] = task.error
    return payload


def build_task_timing_fields(task: Any) -> Dict[str, Any]:
    started_at = task.created_at
    now = time.time()
    created_ts = parse_utc_timestamp(task.created_at)
    completed_ts = parse_utc_timestamp(task.completed_at)
    payload: Dict[str, Any] = {"started_at": started_at}
    if created_ts is None:
        return payload
    if completed_ts is not None:
        payload["elapsed_seconds"] = round(completed_ts - created_ts, 3)
    else:
        payload["running_for_seconds"] = round(now - created_ts, 3)
    return payload


def format_task_activity_entry(activity: Any) -> str:
    label_map = {
        "status": "Status",
        "progress": "Progress",
        "tool_call": "ToolCall",
        "error": "Error",
        "message": "Message",
    }
    label = label_map.get(activity.kind, activity.kind.title())
    return f"{label}: {activity.summary}"


def build_background_dispatch_result(
    *,
    task_id: str,
    tool_name: str,
    task_capabilities: Collection[str],
) -> str:
    actions = ["`task_status`"]
    if "activity" in task_capabilities:
        actions.append("`task_activity`")
    if "message" in task_capabilities:
        actions.append("`task_message`")
    actions.extend(["`task_interrupt`", "`task_wait`", "`task_output`"])
    return (
        f"The `{tool_name}` tool is running in the background with "
        f"task_id='{task_id}'. Use that task_id with "
        + ", ".join(actions[:-1])
        + f", or {actions[-1]}."
    )


def truncate_activity_text(value: str, *, limit: int = 140) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
