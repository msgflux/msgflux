"""Focused tests for background tasks, task progress, notifications, and
library-aware tools."""

from concurrent.futures import CancelledError as FutureCancelledError
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import msgflux as mf
import pytest
from msgflux.chat_messages import ChatMessages
from msgflux.runtime.context import execution_context
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError, TaskInterruptRequestedError
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.02) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("Timed out waiting for condition.")


def _mock_model(text: str = "ok") -> MagicMock:
    model = MagicMock()
    model.model_type = "chat_completion"
    resp = Mock(spec=ModelResponse)
    resp.response_type = "text_generation"
    resp.consume.return_value = text
    resp.data = text
    resp.reasoning = None
    resp.metadata = {}
    model.return_value = resp
    return model


def _tool_call_response(
    tool_name: str, parameters: dict, *, call_id: str = "call_inner"
):
    response = ModelResponse()
    response.set_response_type("tool_call")
    agg = ToolCallAggregator()
    agg.process(0, call_id, tool_name, mf.msgspec_dumps(parameters))
    response.add(agg)
    response.reasoning = None
    response.metadata = {}
    return response


def _text_response(text: str):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    response.metadata = {}
    return response


class _ScriptedModel:
    def __init__(self, responses):
        self.model_type = "chat_completion"
        self._responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("Scripted model exhausted.")
        return self._responses.pop(0)


def _notification_messages(
    messages,
    *,
    source: str | None = None,
    status: str | None = None,
):
    result = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or "<notification>" not in content:
            continue
        if source is not None and f"source: {source}" not in content:
            continue
        if status is not None and f"status: {status}" not in content:
            continue
        result.append(message)
    return result


def _incoming_user_messages(messages):
    return [
        message
        for message in messages
        if isinstance(message.get("content"), str)
        and "<incoming_user_message>" in message["content"]
    ]


def test_background_tool_schema_excludes_task_handle():
    @mf.tool_config(background=True, inject_task=True)
    def background_tool(query: str, task) -> str:
        """Run a query in the background."""
        return query

    library = ToolLibrary(name="lib", tools=[background_tool])
    schema = next(
        item
        for item in library.get_tool_json_schemas()
        if item["function"]["name"] == "background_tool"
    )
    props = schema["function"]["parameters"].get("properties", {})

    assert "query" in props
    assert "task" not in props


def test_allow_background_tool_schema_includes_runtime_choice():
    @mf.tool_config(allow_background=True)
    def maybe_slow(query: str) -> str:
        """Run a query either inline or in the background."""
        return query

    library = ToolLibrary(name="lib", tools=[maybe_slow])
    schema = next(
        item
        for item in library.get_tool_json_schemas()
        if item["function"]["name"] == "maybe_slow"
    )
    props = schema["function"]["parameters"].get("properties", {})

    assert "query" in props
    assert "run_in_background" in props
    assert props["run_in_background"]["anyOf"] == [
        {"type": "boolean"},
        {"type": "null"},
    ]


def test_allow_background_runs_inline_by_default_and_strips_runtime_param():
    calls = []

    @mf.tool_config(allow_background=True)
    def maybe_slow(query: str) -> str:
        """Run a query either inline or in the background."""
        calls.append(query)
        return f"inline:{query}"

    library = ToolLibrary(name="lib", tools=[maybe_slow])

    default_result = library([("call_1", "maybe_slow", {"query": "a"})])
    explicit_inline = library(
        [
            (
                "call_2",
                "maybe_slow",
                {"query": "b", "run_in_background": False},
            )
        ]
    )

    assert default_result.tool_calls[0].result == "inline:a"
    assert explicit_inline.tool_calls[0].result == "inline:b"
    assert explicit_inline.tool_calls[0].parameters == {"query": "b"}
    assert calls == ["a", "b"]


def test_allow_background_dispatches_when_model_requests_background():
    @mf.tool_config(allow_background=True)
    def maybe_slow(query: str) -> str:
        """Run a query either inline or in the background."""
        return f"background:{query}"

    library = ToolLibrary(name="lib", tools=[maybe_slow])
    dispatch = library(
        [
            (
                "call_1",
                "maybe_slow",
                {"query": "a", "run_in_background": True},
            )
        ]
    )

    assert "task_id='" in dispatch.tool_calls[0].result
    assert dispatch.tool_calls[0].parameters == {"query": "a"}
    task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]

    _wait_until(
        lambda: (
            library([("call_2", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result["status"]
            == "completed"
        )
    )
    output = library([("call_3", "task_output", {"task_id": task_id})])

    assert output.tool_calls[0].result == "background:a"


def test_inject_handle_schema_excludes_handle():
    @mf.tool_config(inject_handle=True)
    def register_tool(handle, name: str) -> str:
        """Register a tool by name."""
        return name

    library = ToolLibrary(name="lib", tools=[register_tool])
    schema = next(
        item
        for item in library.get_tool_json_schemas()
        if item["function"]["name"] == "register_tool"
    )
    props = schema["function"]["parameters"].get("properties", {})

    assert "name" in props
    assert "handle" not in props


def test_inject_handle_response_parameters_exclude_handle():
    @mf.tool_config(inject_handle=True)
    def register_tool(handle, name: str) -> str:
        """Register a tool by name."""
        return name

    library = ToolLibrary(name="lib", tools=[register_tool])
    result = library([("call_1", "register_tool", {"name": "lookup"})])

    assert result.tool_calls[0].parameters == {"name": "lookup"}


def test_inject_notification_schema_excludes_notification_handle():
    @mf.tool_config(inject_notification=True)
    def publish_status(notification, name: str) -> str:
        """Publish a status notification."""
        return name

    library = ToolLibrary(name="lib", tools=[publish_status])
    schema = next(
        item
        for item in library.get_tool_json_schemas()
        if item["function"]["name"] == "publish_status"
    )
    props = schema["function"]["parameters"].get("properties", {})

    assert "name" in props
    assert "notification" not in props


def test_inject_handle_can_add_and_remove_tools():
    def multiply(x: int) -> int:
        """Multiply a number by two."""
        return x * 2

    @mf.tool_config(inject_handle=True)
    def add_multiplier(handle) -> list[str]:
        """Register the multiply tool."""
        handle.add(multiply)
        return handle.list_tools()

    @mf.tool_config(inject_handle=True)
    def remove_tool(handle, name: str) -> list[str]:
        """Remove a tool by name."""
        handle.remove(name)
        return handle.list_tools()

    library = ToolLibrary(name="lib", tools=[add_multiplier, remove_tool])

    add_result = library([("call_1", "add_multiplier", {})])
    assert "multiply" in add_result.tool_calls[0].result

    run_result = library([("call_2", "multiply", {"x": 4})])
    assert run_result.tool_calls[0].result == 8

    remove_result = library([("call_3", "remove_tool", {"name": "multiply"})])
    assert "multiply" not in remove_result.tool_calls[0].result
    assert "multiply" not in library.get_tool_names()


def test_inject_handle_can_add_background_tool_with_task_tools():
    @mf.tool_config(background=True, inject_task=True)
    def background_multiplier(value: int, task) -> int:
        """Multiply a number by two in the background."""
        task.update_progress(stage="work", message="Running", current=1, total=1)
        return value * 2

    @mf.tool_config(inject_handle=True)
    def add_background_multiplier(handle) -> list[str]:
        """Register a background tool."""
        handle.add(background_multiplier)
        return handle.list_tools()

    library = ToolLibrary(name="lib", tools=[add_background_multiplier])

    add_result = library([("call_1", "add_background_multiplier", {})])
    assert "background_multiplier" in add_result.tool_calls[0].result
    assert "task_status" in add_result.tool_calls[0].result
    assert "task_interrupt" in add_result.tool_calls[0].result
    assert "task_wait" in add_result.tool_calls[0].result
    assert "task_output" in add_result.tool_calls[0].result

    dispatch = library([("call_2", "background_multiplier", {"value": 4})])
    assert "task_id='" in dispatch.tool_calls[0].result
    assert "task_activity" not in dispatch.tool_calls[0].result

    _wait_until(
        lambda: (
            library([("call_3", "task_list", {})]).tool_calls[0].result[0]["status"]
            == "completed"
        )
    )

    task_id = library([("call_4", "task_list", {})]).tool_calls[0].result[0]["task_id"]
    output_result = library([("call_5", "task_output", {"task_id": task_id})])
    assert output_result.tool_calls[0].result == 8


def test_background_task_reports_progress_and_output():
    started = threading.Event()
    release = threading.Event()

    @mf.tool_config(background=True, inject_task=True)
    def long_job(value: int, task) -> int:
        """Run a long job in the background."""
        task.set_running(stage="prepare", message="Preparing")
        task.update_progress(stage="work", message="Halfway", current=1, total=2)
        started.set()
        release.wait(timeout=2.0)
        task.update_progress(stage="work", message="Finishing", current=2, total=2)
        return value * 2

    library = ToolLibrary(name="lib", tools=[long_job])

    dispatch = library([("call_1", "long_job", {"value": 21})])
    assert "task_status" in library.get_tool_names()
    assert "task_interrupt" in library.get_tool_names()
    assert "task_wait" in library.get_tool_names()
    assert "task_output" in library.get_tool_names()
    assert started.wait(timeout=1.0)
    assert "task_id='" in dispatch.tool_calls[0].result

    list_result = library([("call_2", "task_list", {})])
    task_id = list_result.tool_calls[0].result[0]["task_id"]

    get_result = library([("call_3", "task_status", {"task_id": task_id})])
    task_state = get_result.tool_calls[0].result
    assert task_state["status"] == "running"
    assert "started_at" in task_state
    assert isinstance(task_state["running_for_seconds"], float)
    assert task_state["metadata"]["supports_activity"] is False
    assert task_state["progress"]["stage"] == "work"
    assert task_state["progress"]["percent"] == 50.0

    release.set()
    _wait_until(
        lambda: (
            library([("call_4", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result["status"]
            == "completed"
        )
    )
    final_state = (
        library([("call_6", "task_status", {"task_id": task_id})]).tool_calls[0].result
    )
    assert "elapsed_seconds" in final_state

    output_result = library([("call_5", "task_output", {"task_id": task_id})])
    assert output_result.tool_calls[0].result == 42


def test_task_wait_returns_final_output():
    release = threading.Event()

    @mf.tool_config(background=True)
    def long_job(value: int) -> int:
        """Run a long job in the background."""
        release.wait(timeout=2.0)
        return value * 2

    library = ToolLibrary(name="lib", tools=[long_job])

    dispatch = library([("call_1", "long_job", {"value": 21})])
    assert "task_wait" in library.get_tool_names()
    assert "task_interrupt" in library.get_tool_names()
    task_id = library([("call_2", "task_list", {})]).tool_calls[0].result[0]["task_id"]
    assert f"task_id='{task_id}'" in dispatch.tool_calls[0].result
    assert "`task_wait`" in dispatch.tool_calls[0].result

    timer = threading.Timer(0.1, release.set)
    timer.start()
    try:
        wait_result = library(
            [("call_3", "task_wait", {"task_id": task_id, "timeout": 1.0})]
        )
    finally:
        timer.cancel()

    assert wait_result.tool_calls[0].result == 42


def test_task_wait_returns_timeout_payload_with_progress():
    release = threading.Event()

    @mf.tool_config(background=True, inject_task=True)
    def long_job(value: int, task) -> int:
        """Run a long job in the background."""
        task.update_progress(stage="work", message="Halfway", current=1, total=2)
        release.wait(timeout=2.0)
        return value * 2

    library = ToolLibrary(name="lib", tools=[long_job])

    library([("call_1", "long_job", {"value": 21})])
    task_id = library([("call_2", "task_list", {})]).tool_calls[0].result[0]["task_id"]

    wait_result = library(
        [("call_3", "task_wait", {"task_id": task_id, "timeout": 0.05})]
    )
    payload = wait_result.tool_calls[0].result

    assert payload["task_id"] == task_id
    assert payload["status"] == "timeout"
    assert payload["task_status"] == "running"
    assert payload["progress"]["stage"] == "work"
    assert payload["progress"]["percent"] == 50.0

    release.set()
    _wait_until(
        lambda: (
            library([("call_4", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result["status"]
            == "completed"
        )
    )


def test_task_wait_returns_failed_payload():
    @mf.tool_config(background=True)
    def failing_job() -> int:
        """Always fail."""
        raise RuntimeError("boom")

    library = ToolLibrary(name="lib", tools=[failing_job])

    library([("call_1", "failing_job", {})])
    task_id = library([("call_2", "task_list", {})]).tool_calls[0].result[0]["task_id"]

    wait_result = library(
        [("call_3", "task_wait", {"task_id": task_id, "timeout": 1.0})]
    )
    payload = wait_result.tool_calls[0].result

    assert payload["task_id"] == task_id
    assert payload["status"] == "failed"
    assert "boom" in payload["error"]


def test_cancelled_background_future_is_not_logged_as_error():
    library = ToolLibrary(name="lib", tools=[])
    future = Mock()
    future.result.side_effect = FutureCancelledError()

    with patch("msgflux.runtime.background.logger.error") as mock_error:
        library.background_dispatcher.log_task_failure(future)

    mock_error.assert_not_called()


def test_task_wait_falls_back_to_task_store_polling_without_future():
    @mf.tool_config(background=True)
    def placeholder() -> None:
        """Enable task runtime tools for the library."""
        return None

    library = ToolLibrary(name="lib", tools=[placeholder])
    task = library.task_store.create(tool_name="external_job")

    def complete_task():
        time.sleep(0.1)
        library.task_store.complete(task.task_id, 99)

    timer = threading.Thread(target=complete_task)
    timer.start()
    try:
        wait_result = library(
            [("call_1", "task_wait", {"task_id": task.task_id, "timeout": 1.0})]
        )
    finally:
        timer.join(timeout=1.0)

    assert wait_result.tool_calls[0].result == 99


def test_background_tool_dispatch_does_not_mention_task_activity():
    @mf.tool_config(background=True)
    def slow_pipeline(value: int) -> int:
        """Run a simple background tool."""
        return value * 2

    library = ToolLibrary(name="lib", tools=[slow_pipeline])

    dispatch = library([("call_1", "slow_pipeline", {"value": 4})])
    result = dispatch.tool_calls[0].result

    assert "`task_status`" in result
    assert "`task_interrupt`" in result
    assert "`task_wait`" in result
    assert "`task_output`" in result
    assert "`task_activity`" not in result
    assert "`task_message`" not in result
    assert "task_activity" not in library.get_tool_names()

