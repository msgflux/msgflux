from __future__ import annotations

from typing import Any, Optional

from msgflux.core.registry import Registry
from msgflux.runtime.context import get_execution_context
from msgflux.tools.helpers import (
    BACKGROUND_ACTIVITY_TOOL_KIND,
    BACKGROUND_MESSAGE_TOOL_KIND,
    BACKGROUND_TASK_TOOL_KIND,
    DEFAULT_AGENT_BACKGROUND_CAPABILITIES,
    normalize_background_capabilities,
)
from msgflux.tools.types import Hidden, ToolBackground, ToolBucket
from msgflux.utils.inspect import get_fn_param_defaults

_base_task_tools = Registry()
_background_activity_tools = Registry()
_background_message_tools = Registry()


@_base_task_tools
class TaskStatusTool(ToolBackground):
    name = "task_status"
    description = "Get task state."
    annotations = {"task_id": str, "handle": Hidden, "return": str}
    tool_config = {"handle": {"tasks": ["read"]}}

    def __call__(self, task_id: str, handle: Hidden) -> str:
        return handle.tasks.read(task_id)


@_base_task_tools
class TaskListTool(ToolBackground):
    name = "task_list"
    description = "List tasks."
    annotations = {
        "handle": Hidden,
        "return": str,
    }
    tool_config = {"handle": {"tasks": ["list"]}}

    def __call__(
        self,
        handle: Hidden = None,
    ) -> str:
        return handle.tasks.list()


@_base_task_tools
class TaskOutputTool(ToolBackground):
    name = "task_output"
    description = "Get a completed task result."
    annotations = {"task_id": str, "handle": Hidden, "return": Any}
    tool_config = {"handle": {"tasks": ["output"]}}

    def __call__(self, task_id: str, handle: Hidden) -> Any:
        return handle.tasks.output(task_id)


@_base_task_tools
class TaskWaitTool(ToolBackground):
    name = "task_wait"
    description = "Wait for and return a task result."
    annotations = {
        "task_id": str,
        "timeout": Optional[float],
        "handle": Hidden,
        "return": Any,
    }
    tool_config = {"handle": {"tasks": ["wait"]}}

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

        return handle.tasks.wait(task_id, timeout=timeout)


@_base_task_tools
class TaskInterruptTool(ToolBackground):
    name = "task_interrupt"
    description = "Request task interruption."
    annotations = {"task_id": str, "handle": Hidden, "return": str}
    tool_config = {"handle": {"tasks": ["interrupt"]}}

    def __call__(self, task_id: str, handle: Hidden) -> str:
        return handle.tasks.interrupt(task_id)


@_background_activity_tools
class TaskActivityTool(ToolBackground):
    name = "task_activity"
    tool_kind = BACKGROUND_ACTIVITY_TOOL_KIND
    description = "Get recent task activity."
    annotations = {
        "task_id": str,
        "handle": Hidden,
        "return": str,
    }
    tool_config = {"handle": {"tasks": ["activity"]}}

    def __call__(
        self,
        task_id: str,
        handle: Hidden = None,
    ) -> str:
        return handle.tasks.activity(task_id, limit=10)


@_background_message_tools
class TaskMessageTool(ToolBackground):
    name = "task_message"
    tool_kind = BACKGROUND_MESSAGE_TOOL_KIND
    description = "Message or resume an agent task."
    annotations = {
        "task_id": str,
        "message": str,
        "handle": Hidden,
        "return": str,
    }
    tool_config = {"handle": {"tasks": ["message"]}}

    def __call__(
        self,
        task_id: str,
        message: str,
        handle: Hidden,
    ) -> str:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("`message` must be a non-empty string.")
        return handle.tasks.message(task_id, message.strip())


class TaskTool(ToolBucket):
    """Expose captured background task controls through one mode-based tool."""

    name = "task_tool"
    display_name = "Task"
    capture = {
        "tool_kind": (
            f"{BACKGROUND_TASK_TOOL_KIND}|{BACKGROUND_ACTIVITY_TOOL_KIND}"
        ),
        "on_demand": False,
    }
    _base_description = "Inspect or control background work by mode."
    description = _base_description
    usage_guidance = None
    tool_config = {}
    annotations = {
        "mode": str,
        "task_id": Optional[str],
        "timeout": Optional[float],
        "tools": Hidden,
        "return": Any,
    }

    def refresh(self) -> None:
        modes = tuple(sorted(self._mode_name(name) for name in self.tools))
        available = ", ".join(modes) if modes else "none"
        self.description = f"{self._base_description} Available modes: {available}."

    def validate_capture(self, metadata: Any) -> None:
        super().validate_capture(metadata)
        public_params = set(metadata.annotations) - {"return"}
        unsupported = public_params - {"task_id", "timeout"}
        if unsupported:
            names = ", ".join(sorted(f"`{name}`" for name in unsupported))
            raise ValueError(
                f"Task mode `{metadata.name}` has unsupported parameters: {names}."
            )
        mode = self._mode_name(metadata.name)
        if any(self._mode_name(name) == mode for name in self.tools):
            raise ValueError(f"Duplicate task mode `{mode}` in bucket.")

    def __call__(
        self,
        mode: str,
        task_id: str | None = None,
        timeout: float | None = None,
        tools: Hidden = None,
    ) -> Any:
        metadata = self._resolve_mode(mode)
        accepted = set(metadata.annotations) - {"return"}
        supplied = {"task_id": task_id, "timeout": timeout}
        unexpected = [
            name
            for name, value in supplied.items()
            if value is not None and name not in accepted
        ]
        if unexpected:
            names = ", ".join(f"`{name}`" for name in unexpected)
            raise ValueError(f"Task mode `{mode}` does not accept {names}.")

        defaults = get_fn_param_defaults(metadata.impl)
        missing = [
            name
            for name in accepted
            if supplied.get(name) is None and name not in defaults
        ]
        if missing:
            names = ", ".join(f"`{name}`" for name in missing)
            raise ValueError(f"Task mode `{mode}` requires {names}.")

        params = {
            name: supplied[name]
            for name in accepted
            if supplied.get(name) is not None
        }
        return tools(metadata.name, **params)

    def _resolve_mode(self, mode: str):
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError("`mode` must be a non-empty string.")
        normalized = mode.strip().lower()
        for tool_name, metadata in self.tools.items():
            if self._mode_name(tool_name) == normalized:
                return metadata
        available = ", ".join(sorted(self._mode_name(name) for name in self.tools))
        raise ValueError(f"Unknown task mode `{mode}`. Available modes: {available}.")

    @staticmethod
    def _mode_name(tool_name: str) -> str:
        return tool_name.removeprefix("task_")


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


def build_task_result(*, task: Any | None) -> Any:
    if task is None:
        return "status=not_found"
    if task.status == "completed":
        return task.result
    return format_task_status(task)


def build_task_timeout_result(
    *,
    task: Any | None,
) -> str:
    if task is None:
        return "status=not_found"
    fields = ["status=timeout", f"task_status={task.status}"]
    if task.status not in {"completed", "failed", "interrupted"}:
        fields.extend(format_task_progress(task))
    return " ".join(fields)


def format_task_status(
    task: Any,
    *,
    include_id: bool = False,
    include_tool: bool = False,
) -> str:
    fields = []
    if include_id:
        fields.append(f"task_id={task.task_id}")
    fields.append(f"status={task.status}")
    if include_tool:
        fields.append(f"tool={task.tool_name}")
    if task.status not in {"completed", "failed", "interrupted"}:
        fields.extend(format_task_progress(task))
    if task.status == "failed" and task.error:
        fields.append(f"error={compact_task_text(task.error, limit=240)}")
    if task.status == "interrupted":
        reason = task.metadata.get("interrupt_reason")
        if reason:
            fields.append(f"reason={compact_task_text(reason, limit=240)}")
    return " ".join(fields)


def format_task_progress(task: Any) -> list[str]:
    progress = task.progress
    fields = []
    if progress.stage:
        fields.append(f"stage={compact_task_text(progress.stage)}")
    if progress.percent is not None:
        fields.append(f"progress={progress.percent:g}%")
    elif progress.current is not None or progress.total is not None:
        current = "?" if progress.current is None else progress.current
        total = "?" if progress.total is None else progress.total
        fields.append(f"progress={current}/{total}")
    if progress.message:
        fields.append(f"message={compact_task_text(progress.message)}")
    return fields


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
) -> str:
    return f"task_id={task_id} status=running"


def truncate_activity_text(value: str, *, limit: int = 140) -> str:
    return compact_task_text(value, limit=limit)


def compact_task_text(value: Any, *, limit: int = 140) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
