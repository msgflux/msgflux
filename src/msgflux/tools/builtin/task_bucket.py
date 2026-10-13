"""One typed interface for the background task controls available to a library."""

from __future__ import annotations

from typing import Any, Union

import msgspec

from msgflux.tools.handles import ToolBucketHandle
from msgflux.tools.types import ToolBucket, ToolLibraryOperator


class _Status(
    msgspec.Struct, tag="status", tag_field="operation", forbid_unknown_fields=True
):
    task_id: str


class _List(
    msgspec.Struct, tag="list", tag_field="operation", forbid_unknown_fields=True
):
    status: str | None = None


class _Output(
    msgspec.Struct, tag="output", tag_field="operation", forbid_unknown_fields=True
):
    task_id: str


class _Wait(
    msgspec.Struct, tag="wait", tag_field="operation", forbid_unknown_fields=True
):
    task_id: str
    timeout: float | None = None


class _Interrupt(
    msgspec.Struct, tag="interrupt", tag_field="operation", forbid_unknown_fields=True
):
    task_id: str


class _Activity(
    msgspec.Struct, tag="activity", tag_field="operation", forbid_unknown_fields=True
):
    task_id: str
    limit: int = 10


class _Message(
    msgspec.Struct, tag="message", tag_field="operation", forbid_unknown_fields=True
):
    task_id: str
    message: str


_REQUESTS = {
    "task_" + cls.__struct_config__.tag: cls
    for cls in (_Status, _List, _Output, _Wait, _Interrupt, _Activity, _Message)
}


class TaskTool(ToolBucket, ToolLibraryOperator):
    """Inspect, wait for, or control background tasks."""

    name = "task"
    display_name = "Task"
    capture = {
        "tool_kind": "background|background_activity|background_message",
        "defer_loading": False,
    }
    annotations = {"request": _Status, "return": Any}
    default_usage_guidance = (
        "Use the task_id returned by background dispatch. Choose status for compact "
        "state, activity for progress, output or wait for final results, and interrupt "
        "to stop a task. Pass the operation and its fields inside request. Results "
        "can be fetched repeatedly."
    )

    def __init__(self):
        self.refresh()

    def validate_capture(self, definition):
        super().validate_capture(definition)
        if definition.name not in _REQUESTS:
            raise ValueError("TaskTool captures only builtin task controls")

    def refresh(self, entries=()) -> None:
        classes = tuple(_REQUESTS[entry.name] for entry in entries)
        self._request_type = (
            Union[classes] if len(classes) > 1 else classes[0] if classes else _Status
        )
        self.description = (
            "Manage background tasks. Available operations: "
            + ", ".join(cls.__struct_config__.tag for cls in classes)
        )

    def patch_schema_annotations(self, annotations):
        return {**annotations, "request": self._request_type}

    def _arguments(self, request, handle):
        parsed = msgspec.convert(request, type=self._request_type, strict=True)
        arguments = msgspec.to_builtins(parsed)
        name = "task_" + arguments.pop("operation")
        if not handle.has_tool(name):
            raise ValueError(f"Task operation {name!r} is not available")
        return name, arguments

    def __call__(self, request, *, handle: ToolBucketHandle):
        name, arguments = self._arguments(request, handle)
        return handle(name, **arguments)

    async def acall(self, request, *, handle: ToolBucketHandle):
        name, arguments = self._arguments(request, handle)
        return await handle.acall(name, **arguments)
