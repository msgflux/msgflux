from __future__ import annotations

import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from msgflux.core.registry import Registry
from msgflux.tools.types import ToolMetadata

BASE_TASK_TOOLS = Registry()
AGENT_TASK_TOOLS = Registry()


class TaskRuntimeTool:
    """Base object for task runtime tools registered through ToolLibraryHandle."""

    name: str
    description: str
    annotations: Dict[str, Any]

    def __init__(self, runtime: TaskRuntimeContext):
        self.runtime = runtime

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        return self(*args, **kwargs)

    def to_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description=self.description,
            annotations=self.annotations,
            tool_config={},
            impl=self,
        )


@BASE_TASK_TOOLS
class TaskStatusTool(TaskRuntimeTool):
    name = "task_status"
    description = "Get the current status of a background task by task_id."
    annotations = {"task_id": str, "return": str}

    def __call__(self, task_id: str) -> Dict[str, Any]:
        task = self.runtime.library_handle.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        payload = task.to_dict()
        payload.update(self.runtime.build_task_timing_fields(task))
        last_activity = self.runtime.library_handle.task_store.get_last_activity(
            task_id
        )
        if last_activity is not None:
            payload["last_activity_summary"] = self.runtime.format_task_activity_entry(
                last_activity
            )
        return payload


@BASE_TASK_TOOLS
class TaskListTool(TaskRuntimeTool):
    name = "task_list"
    description = "List background tasks registered in the current tool library."
    annotations = {"status": Optional[str], "return": str}

    def __call__(self, status: str | None = None) -> list[Dict[str, Any]]:
        tasks = []
        for task in self.runtime.library_handle.task_store.list(status=status):
            payload = task.to_dict()
            payload.update(self.runtime.build_task_timing_fields(task))
            last_activity = self.runtime.library_handle.task_store.get_last_activity(
                task.task_id
            )
            if last_activity is not None:
                payload["last_activity_summary"] = (
                    self.runtime.format_task_activity_entry(last_activity)
                )
            tasks.append(payload)
        return tasks


@BASE_TASK_TOOLS
class TaskOutputTool(TaskRuntimeTool):
    name = "task_output"
    description = "Get the final output of a background task by task_id."
    annotations = {"task_id": str, "return": str}

    def __call__(self, task_id: str) -> Any:
        task = self.runtime.library_handle.task_store.get(task_id)
        return self.runtime.build_task_result(task_id=task_id, task=task)


@BASE_TASK_TOOLS
class TaskWaitTool(TaskRuntimeTool):
    name = "task_wait"
    description = (
        "Wait for a background task to finish. "
        "Returns the final output, failed payload, or a timeout status."
    )
    annotations = {"task_id": str, "timeout": Optional[float], "return": str}

    def __call__(self, task_id: str, timeout: float | None = None) -> Any:  # noqa: C901
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError(
                    f"`timeout` must be float, int or None, given `{type(timeout)}`"
                )
            if timeout < 0:
                raise ValueError("`timeout` must be greater than or equal to 0.")

        task = self.runtime.library_handle.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if task.status in {"completed", "failed", "interrupted"}:
            return self.runtime.build_task_result(task_id=task_id, task=task)

        future = self.runtime.library_handle.get_task_future(task_id)
        if future is not None:
            try:
                future.result(timeout=timeout)
            except FutureTimeoutError:
                task = self.runtime.library_handle.task_store.get(task_id)
                return self.runtime.build_task_timeout_result(
                    task_id=task_id,
                    task=task,
                )
            except Exception:
                task = self.runtime.library_handle.task_store.get(task_id)
                return self.runtime.build_task_result(task_id=task_id, task=task)
            task = self.runtime.library_handle.task_store.get(task_id)
            return self.runtime.build_task_result(task_id=task_id, task=task)

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            task = self.runtime.library_handle.task_store.get(task_id)
            if task is None or task.status in {"completed", "failed", "interrupted"}:
                return self.runtime.build_task_result(task_id=task_id, task=task)
            if deadline is not None and time.monotonic() >= deadline:
                return self.runtime.build_task_timeout_result(
                    task_id=task_id,
                    task=task,
                )
            time.sleep(0.05)


@BASE_TASK_TOOLS
class TaskInterruptTool(TaskRuntimeTool):
    name = "task_interrupt"
    description = (
        "Request a cooperative interrupt for a background task. "
        "Interrupts immediately only if the task has not started yet."
    )
    annotations = {"task_id": str, "return": str}

    def __call__(self, task_id: str) -> Dict[str, Any]:
        task = self.runtime.library_handle.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}

        if task.status in {"completed", "failed", "interrupted"}:
            return {
                "task_id": task_id,
                "status": task.status,
                "message": "Task already reached a terminal state.",
            }

        self.runtime.library_handle.task_store.request_interrupt(task_id)
        future = self.runtime.library_handle.get_task_future(task_id)
        if future is not None and future.cancel():
            interrupted = self.runtime.library_handle.task_store.interrupt(
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


@AGENT_TASK_TOOLS
class TaskActivityTool(TaskRuntimeTool):
    name = "task_activity"
    description = "List compact activity entries for a background agent task."
    annotations = {"task_id": str, "limit": Optional[int], "return": str}

    def __call__(self, task_id: str, limit: int | None = 10) -> Any:
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError(f"`limit` must be int or None, given `{type(limit)}`")
            if limit <= 0:
                raise ValueError("`limit` must be greater than 0.")
        task = self.runtime.library_handle.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if task.metadata.get("task_kind") != "agent":
            return {
                "task_id": task_id,
                "status": "unsupported",
                "error": "task_activity is only available for background agent tasks.",
            }
        activity = self.runtime.library_handle.task_store.list_activity(
            task_id,
            limit=limit,
        )
        return [self.runtime.format_task_activity_entry(item) for item in activity]


@AGENT_TASK_TOOLS
class TaskMessageTool(TaskRuntimeTool):
    name = "task_message"
    description = (
        "Send a message to a background agent task. "
        "If it is still running, deliver the message to its inbox. "
        "If it already interrupted, resume the task from its checkpoint."
    )
    annotations = {"task_id": str, "message": str, "return": str}

    def __call__(self, task_id: str, message: str) -> Dict[str, Any]:
        task = self.runtime.library_handle.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if task.metadata.get("task_kind") != "agent":
            return {
                "task_id": task_id,
                "status": "unsupported",
                "error": "task_message is only available for background agent tasks.",
            }
        if not isinstance(message, str) or not message.strip():
            raise ValueError("`message` must be a non-empty string.")

        task_inbox = self.runtime.library_handle.get_task_inbox(task_id)
        if task.status == "running" and task_inbox is not None:
            task_inbox.publish(
                {
                    "source": "task_message",
                    "ref": task_id,
                    "status": "message",
                    "hint": message.strip(),
                    "metadata": {"direction": "root_to_task"},
                }
            )
            self.runtime.library_handle.task_store.add_activity(
                task_id,
                kind="message",
                summary=f"Root message: {self.runtime.truncate_activity_text(message)}",
                metadata={"direction": "root_to_task"},
            )
            return {
                "task_id": task_id,
                "status": "delivered",
                "message": "Message delivered to the running background agent.",
            }

        resumed = self.runtime.library_handle.resume_background_agent_task(
            task=task,
            message=message.strip(),
        )
        return {
            "task_id": task_id,
            "status": "resumed",
            "message": resumed,
        }


class TaskRuntimeContext:
    """Shared helpers used by task runtime tool objects."""

    def __init__(self, library_handle: Any):
        self.library_handle = library_handle

    def build_task_result(self, *, task_id: str, task: Any | None) -> Any:
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
        self, *, task_id: str, task: Any | None
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

    def build_task_timing_fields(self, task: Any) -> Dict[str, Any]:
        started_at = task.created_at
        now = time.time()
        created_ts = self.parse_utc_timestamp(task.created_at)
        completed_ts = self.parse_utc_timestamp(task.completed_at)
        payload: Dict[str, Any] = {"started_at": started_at}
        if created_ts is None:
            return payload
        if completed_ts is not None:
            payload["elapsed_seconds"] = round(completed_ts - created_ts, 3)
        else:
            payload["running_for_seconds"] = round(now - created_ts, 3)
        return payload

    def format_task_activity_entry(self, activity: Any) -> str:
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
        self,
        *,
        task_id: str,
        tool_name: str,
        task_kind: str,
    ) -> str:
        actions = ["`task_status`", "`task_interrupt`", "`task_wait`", "`task_output`"]
        if task_kind == "agent":
            actions.insert(1, "`task_activity`")
            actions.insert(2, "`task_message`")
        return (
            f"The `{tool_name}` tool is running in the background with "
            f"task_id='{task_id}'. Use that task_id with "
            + ", ".join(actions[:-1])
            + f", or {actions[-1]}."
        )

    @staticmethod
    def parse_utc_timestamp(value: str | None) -> float | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            return (
                datetime.fromisoformat(normalized).astimezone(timezone.utc).timestamp()
            )
        except ValueError:
            return None

    @staticmethod
    def truncate_activity_text(value: str, *, limit: int = 140) -> str:
        text = " ".join(str(value).split())
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."
