"""Send concise results from a subagent back to its parent agent."""

from __future__ import annotations

from typing import Mapping

from msgflux.runtime.context import ExecutionScope, get_execution_scope
from msgflux.tools.handles import ToolLibraryHandle
from msgflux.tools.types import Hidden

_SOURCE = "subagent"
_DIRECTION = "child_to_parent"


class SendParentMessageTool:
    """Send the parent agent a concise, actionable result.

    Args:
        message: Outcome, blocker, or answer the parent needs; no raw logs.
    """

    name = "send_parent_message"
    display_name = "Send Parent Message"
    annotations = {"message": str, "return": str}
    tool_config = {"runtime_inputs": ("handle",)}
    usage_guidance = (
        "Report outcomes, blockers, or answers the parent needs to act on. "
        "Write one to three sentences of conclusions instead of raw logs, "
        "command output, or step-by-step narration."
    )

    def __init__(self, parent: ToolLibraryHandle):
        if not isinstance(parent, ToolLibraryHandle):
            raise TypeError(
                f"`parent` must be a ToolLibraryHandle, given `{type(parent)}`"
            )
        self._parent = parent

    def __call__(self, message: str, *, handle: Hidden) -> str:
        text = self._message_text(message)
        scope = get_execution_scope()
        thread_id, parent_run_id, run_id = self._require_scope(scope)
        sender = scope.namespace
        reference, metadata = self._reference(
            scope=scope,
            run_id=run_id,
            handle=handle,
        )
        self._parent.get_agent_inbox(
            thread_id=thread_id,
            run_id=parent_run_id,
        ).message(
            f"[{sender}:{reference}] {text}",
            description=f"Message from subagent {sender}",
            source=_SOURCE,
            ref=reference,
            metadata={
                "agent_name": sender,
                "direction": _DIRECTION,
                **metadata,
            },
        )
        return "Message queued for the parent agent. Continue the task."

    @staticmethod
    def _message_text(message: str) -> str:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("`message` must contain non-empty text for the parent")
        return message.strip()

    @staticmethod
    def _require_scope(scope: ExecutionScope) -> tuple[str, str, str]:
        required = ("thread_id", "parent_run_id", "run_id")
        missing = []
        for field in required:
            value = getattr(scope, field, None)
            if not isinstance(value, str) or not value.strip():
                missing.append(field)
        if missing:
            names = ", ".join(f"`{field}`" for field in missing)
            raise RuntimeError(
                f"send_parent_message requires an active subagent scope with {names}"
            )
        return scope.thread_id, scope.parent_run_id, scope.run_id

    def _reference(
        self,
        *,
        scope: ExecutionScope,
        run_id: str,
        handle: ToolLibraryHandle,
    ) -> tuple[str, dict[str, str]]:
        """Return a routing reference that is stable across task resumes.

        A background task records its own checkpoint run id, so the task id
        stays stable when the worker restarts. Inherited ancestor tasks in
        inline nested children fail that identity check and fall back to the
        scope run id of the sending subagent.
        """
        task_id = self._task_id(handle)
        if task_id is not None:
            record = handle.get_task_store().get(task_id, include_result=False)
            if record is not None and self._record_matches(
                record,
                scope=scope,
                run_id=run_id,
            ):
                return task_id, {"ref_kind": "task", "task_id": task_id}
        return run_id, {"ref_kind": "run", "run_id": run_id}

    @staticmethod
    def _task_id(handle: ToolLibraryHandle) -> str | None:
        """Return the current background task id, if this call has one."""
        try:
            task = handle.get_task()
        except RuntimeError:
            return None
        task_id = getattr(task, "task_id", None)
        if isinstance(task_id, str) and task_id:
            return task_id
        return None

    @staticmethod
    def _record_matches(
        record: object,
        *,
        scope: ExecutionScope,
        run_id: str,
    ) -> bool:
        metadata = getattr(record, "metadata", None)
        if not isinstance(metadata, Mapping):
            return False
        recorded_run_id = metadata.get("checkpoint_run_id") or getattr(
            record, "task_id", None
        )
        if recorded_run_id != run_id:
            return False
        if metadata.get("checkpoint_namespace") != scope.namespace:
            return False
        return metadata.get("thread_id") == scope.thread_id


__all__ = ["SendParentMessageTool"]
