from __future__ import annotations

import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, Optional

from msgflux.core.registry import Registry
from msgflux.tools.config import tool_config
from msgflux.tools.types import Hidden
from msgflux.utils.time import parse_utc_timestamp

BASE_TASK_TOOLS = Registry()
AGENT_TASK_TOOLS = Registry()


@BASE_TASK_TOOLS
@tool_config(inject_handle=True)
def task_status(task_id: str, handle: Hidden) -> Dict[str, Any]:
    """Get the current status of a background task by task_id."""
    task = handle.task_store.get(task_id)
    if task is None:
        return {"task_id": task_id, "status": "not_found"}
    payload = task.to_dict()
    payload.update(build_task_timing_fields(task))
    last_activity = handle.task_store.get_last_activity(task_id)
    if last_activity is not None:
        payload["last_activity_summary"] = format_task_activity_entry(last_activity)
    return payload


@BASE_TASK_TOOLS
@tool_config(inject_handle=True)
def task_list(
    status: Optional[str] = None,
    handle: Hidden = None,
) -> list[Dict[str, Any]]:
    """List background tasks registered in the current tool library."""
    tasks = []
    for task in handle.task_store.list(status=status):
        payload = task.to_dict()
        payload.update(build_task_timing_fields(task))
        last_activity = handle.task_store.get_last_activity(task.task_id)
        if last_activity is not None:
            payload["last_activity_summary"] = format_task_activity_entry(
                last_activity
            )
        tasks.append(payload)
    return tasks


@BASE_TASK_TOOLS
@tool_config(inject_handle=True)
def task_output(task_id: str, handle: Hidden) -> Any:
    """Get the final output of a background task by task_id."""
    task = handle.task_store.get(task_id)
    return build_task_result(task_id=task_id, task=task)


@BASE_TASK_TOOLS
@tool_config(inject_handle=True)
def task_wait(
    task_id: str,
    timeout: Optional[float] = None,
    handle: Hidden = None,
) -> Any:  # noqa: C901
    """Wait for a background task to finish.

    Returns the final output, failed payload, or a timeout status.
    """
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError(
                f"`timeout` must be float, int or None, given `{type(timeout)}`"
            )
        if timeout < 0:
            raise ValueError("`timeout` must be greater than or equal to 0.")

    task = handle.task_store.get(task_id)
    if task is None:
        return {"task_id": task_id, "status": "not_found"}
    if task.status in {"completed", "failed", "interrupted"}:
        return build_task_result(task_id=task_id, task=task)

    future = handle.get_task_future(task_id)
    if future is not None:
        try:
            future.result(timeout=timeout)
        except FutureTimeoutError:
            task = handle.task_store.get(task_id)
            return build_task_timeout_result(
                task_id=task_id,
                task=task,
            )
        except Exception:
            task = handle.task_store.get(task_id)
            return build_task_result(task_id=task_id, task=task)
        task = handle.task_store.get(task_id)
        return build_task_result(task_id=task_id, task=task)

    deadline = None if timeout is None else time.monotonic() + float(timeout)
    while True:
        task = handle.task_store.get(task_id)
        if task is None or task.status in {"completed", "failed", "interrupted"}:
            return build_task_result(task_id=task_id, task=task)
        if deadline is not None and time.monotonic() >= deadline:
            return build_task_timeout_result(
                task_id=task_id,
                task=task,
            )
        time.sleep(0.05)


@BASE_TASK_TOOLS
@tool_config(inject_handle=True)
def task_interrupt(task_id: str, handle: Hidden) -> Dict[str, Any]:
    """Request a cooperative interrupt for a background task.

    Interrupts immediately only if the task has not started yet.
    """
    task = handle.task_store.get(task_id)
    if task is None:
        return {"task_id": task_id, "status": "not_found"}

    if task.status in {"completed", "failed", "interrupted"}:
        return {
            "task_id": task_id,
            "status": task.status,
            "message": "Task already reached a terminal state.",
        }

    handle.task_store.request_interrupt(task_id)
    future = handle.get_task_future(task_id)
    if future is not None and future.cancel():
        interrupted = handle.task_store.interrupt(
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
@tool_config(inject_handle=True)
def task_activity(
    task_id: str,
    limit: Optional[int] = 10,
    handle: Hidden = None,
) -> Any:
    """List compact activity entries for a background agent task."""
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError(f"`limit` must be int or None, given `{type(limit)}`")
        if limit <= 0:
            raise ValueError("`limit` must be greater than 0.")
    task = handle.task_store.get(task_id)
    if task is None:
        return {"task_id": task_id, "status": "not_found"}
    if task.metadata.get("task_kind") != "agent":
        return {
            "task_id": task_id,
            "status": "unsupported",
            "error": "task_activity is only available for background agent tasks.",
        }
    activity = handle.task_store.list_activity(
        task_id,
        limit=limit,
    )
    return [format_task_activity_entry(item) for item in activity]


@AGENT_TASK_TOOLS
@tool_config(inject_handle=True)
def task_message(task_id: str, message: str, handle: Hidden) -> Dict[str, Any]:
    """Send a message to a background agent task.

    If it is still running, deliver the message to its inbox. If it already
    interrupted, resume the task from its checkpoint.
    """
    task = handle.task_store.get(task_id)
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

    task_inbox = handle.get_task_inbox(task_id)
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
        handle.task_store.add_activity(
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

    resumed = handle.resume_background_agent_task(
        task=task,
        message=message.strip(),
    )
    return {
        "task_id": task_id,
        "status": "resumed",
        "message": resumed,
    }


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


def truncate_activity_text(value: str, *, limit: int = 140) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
