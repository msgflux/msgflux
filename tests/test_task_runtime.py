"""Focused tests for background tasks, task progress, notifications, and
library-aware tools."""

from concurrent.futures import CancelledError as FutureCancelledError
import threading
import time
from types import SimpleNamespace
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
from msgflux.tools.builtin import TaskActivityTool, TaskStatusTool
from msgflux.tools.builtin.task import (
    BACKGROUND_ACTIVITY_TOOLS,
    BACKGROUND_MESSAGE_TOOLS,
    BASE_TASK_TOOLS,
)
from msgflux.tasks import InMemoryTaskStore
from msgflux.tools import ToolBackground, ToolLibraryOperator


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


def test_hidden_handle_schema_excludes_handle():
    @mf.tool_config(background=True)
    def background_tool(
        query: str,
        handle: mf.Hidden,
    ) -> str:
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
    assert "handle" not in props


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


def test_tool_library_uses_context_task_store_without_replacing_default():
    @mf.tool_config(background=True)
    def slow_pipeline(value: int) -> int:
        """Run a simple background tool."""
        return value * 2

    library = ToolLibrary(name="lib", tools=[slow_pipeline])
    default_store = library.get_task_store()
    context_store = InMemoryTaskStore()

    with execution_context(task_store=context_store):
        dispatch = library([("call_1", "slow_pipeline", {"value": 4})])
        task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]

    _wait_until(
        lambda: (
            context_store.get(task_id) is not None
            and context_store.get(task_id).status == "completed"
        )
    )

    assert default_store.get(task_id) is None
    assert context_store.get(task_id).result == 8

    outside_status = (
        library([("call_2", "task_status", {"task_id": task_id})]).tool_calls[0].result
    )

    assert outside_status == {"task_id": task_id, "status": "not_found"}
    with execution_context(task_store=context_store):
        with_context_status = (
            library([("call_3", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result
        )
    assert with_context_status["status"] == "completed"


def test_background_task_tools_follow_background_tool_lifecycle():
    @mf.tool_config(background=True)
    def first_job(value: int) -> int:
        """Run the first background job."""
        return value

    @mf.tool_config(background=True)
    def second_job(value: int) -> int:
        """Run the second background job."""
        return value

    library = ToolLibrary(name="lib", tools=[first_job, second_job])

    assert "task_status" in library.get_tool_names()
    assert "task_wait" in library.get_tool_names()

    library.remove("first_job")
    assert "task_status" in library.get_tool_names()

    library.remove("second_job")
    assert "task_status" not in library.get_tool_names()
    assert "task_wait" not in library.get_tool_names()


def test_background_dispatch_retries_task_id_collision():
    release = threading.Event()

    @mf.tool_config(background=True)
    def slow_job() -> str:
        """Wait until the test releases the task."""
        release.wait(timeout=2.0)
        return "done"

    task_store = InMemoryTaskStore()
    task_store.create("existing", task_id="deadbeef")
    library = ToolLibrary(name="lib", tools=[slow_job])

    with execution_context(task_store=task_store):
        with patch(
            "msgflux.runtime.background.uuid4",
            side_effect=[
                SimpleNamespace(hex="deadbeef00000000"),
                SimpleNamespace(hex="cafebabe00000000"),
            ],
        ):
            dispatch = library([("call_1", "slow_job", {})])

    task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]
    assert task_id == "cafebabe"

    release.set()
    _wait_until(
        lambda: (
            task_store.get(task_id) is not None
            and task_store.get(task_id).status == "completed"
        )
    )


def test_background_task_tools_are_registered_as_classes():
    @mf.tool_config(background=True)
    def slow_pipeline(value: int) -> int:
        """Run a simple background tool."""
        return value

    library = ToolLibrary(name="lib", tools=[slow_pipeline])
    installed = library.library["task_status"]

    assert TaskStatusTool in BASE_TASK_TOOLS
    assert TaskActivityTool in BACKGROUND_ACTIVITY_TOOLS
    assert BACKGROUND_MESSAGE_TOOLS[0].name == "task_message"
    assert all(isinstance(task_class, type) for task_class in BASE_TASK_TOOLS)
    assert isinstance(installed.impl, TaskStatusTool)
    assert isinstance(installed.impl, ToolLibraryOperator)
    assert isinstance(installed.impl, ToolBackground)
    assert installed.tool_config["inject_handle"] is True
    assert installed.tool_config["tool_kind"] == "background"


def test_removed_background_task_tool_is_not_reinstalled_while_background_active():
    @mf.tool_config(background=True)
    def first_job(value: int) -> int:
        """Run the first background job."""
        return value

    @mf.tool_config(background=True)
    def second_job(value: int) -> int:
        """Run the second background job."""
        return value

    library = ToolLibrary(name="lib", tools=[first_job])
    library.remove("task_status")

    assert "task_status" not in library.get_tool_names()

    library.add(second_job)

    assert "task_status" not in library.get_tool_names()
    assert "task_wait" in library.get_tool_names()


def test_readded_background_task_tool_returns_to_the_background_lifecycle():
    @mf.tool_config(background=True)
    def job(value: int) -> int:
        """Run in the background."""
        return value

    library = ToolLibrary(name="lib", tools=[job])
    library.remove("task_status")
    library.add(TaskStatusTool)

    assert "task_status" in library.get_tool_names()

    library.remove("job")

    assert "task_status" not in library.get_tool_names()


def test_removing_task_tool_without_background_source_does_not_disable_it():
    @mf.tool_config(background=True)
    def job(value: int) -> int:
        """Run in the background."""
        return value

    library = ToolLibrary(name="lib", tools=[TaskStatusTool])
    library.remove("task_status")
    library.add(job)

    assert "task_status" in library.get_tool_names()


def test_agent_task_tools_follow_background_agent_lifecycle():
    @mf.tool_config(background=True)
    def slow_pipeline(value: int) -> int:
        """Run a simple background tool."""
        return value * 2

    worker = Agent(name="worker", model=_mock_model("done"))
    worker.tool_config = {"background": True}

    library = ToolLibrary(name="lib", tools=[slow_pipeline, worker])

    assert "task_status" in library.get_tool_names()
    assert "task_activity" in library.get_tool_names()
    assert "task_message" in library.get_tool_names()

    library.remove("worker")

    assert "task_status" in library.get_tool_names()
    assert "task_activity" not in library.get_tool_names()
    assert "task_message" not in library.get_tool_names()


def test_background_capabilities_control_task_tool_lifecycle():
    @mf.tool_config(background=True)
    def plain_job(value: int) -> int:
        """Run without additional task controls."""
        return value

    @mf.tool_config(background=True, background_capabilities=["activity"])
    def monitored_job(value: int) -> int:
        """Run with observable task activity."""
        return value

    library = ToolLibrary(name="lib", tools=[plain_job, monitored_job])

    assert "task_status" in library.get_tool_names()
    assert "task_activity" in library.get_tool_names()
    assert "task_message" not in library.get_tool_names()
    assert library.library["task_activity"].tool_config["tool_kind"] == (
        "background_activity"
    )

    library.remove("monitored_job")

    assert "task_status" in library.get_tool_names()
    assert "task_activity" not in library.get_tool_names()


def test_removing_inactive_optional_task_tool_does_not_disable_it():
    @mf.tool_config(background=True)
    def plain_job(value: int) -> int:
        """Run without additional task controls."""
        return value

    @mf.tool_config(background=True, background_capabilities=["activity"])
    def monitored_job(value: int) -> int:
        """Run with observable task activity."""
        return value

    library = ToolLibrary(name="lib", tools=[plain_job, TaskActivityTool])
    library.remove("task_activity")
    library.add(monitored_job)

    assert "task_activity" in library.get_tool_names()


def test_background_capabilities_validate_declaration():
    with pytest.raises(ValueError, match="requires `background=True`"):

        @mf.tool_config(background_capabilities=["activity"])
        def invalid_job() -> None:
            """Declare an invalid background capability."""

    with pytest.raises(ValueError, match="Unsupported background capabilities"):

        @mf.tool_config(background=True, background_capabilities=["resume"])
        def resume_capability_job() -> None:
            """Declare a removed background capability."""

    @mf.tool_config(background=True, background_capabilities=["message"])
    def generic_message_job() -> None:
        """Declare an unsupported generic messaging capability."""

    with pytest.raises(ValueError, match="only supported by agent sources"):
        ToolLibrary(name="lib", tools=[generic_message_job])


def test_hidden_handle_schema_excludes_handle_for_inline_tool():
    def register_tool(
        handle: mf.Hidden,
        name: str,
    ) -> str:
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


def test_hidden_handle_response_parameters_exclude_handle():
    def register_tool(
        name: str,
        handle: mf.Hidden = None,
    ) -> str:
        """Register a tool by name."""
        return name

    library = ToolLibrary(name="lib", tools=[register_tool])
    result = library([("call_1", "register_tool", {"name": "lookup"})])

    assert result.tool_calls[0].parameters == {"name": "lookup"}


def test_hidden_parameter_is_not_injected_without_tool_config():
    def hidden_tool(name: str, handle: mf.Hidden = None) -> str:
        """Hide a parameter without injecting it."""
        return f"{name}:{handle is None}"

    library = ToolLibrary(name="lib", tools=[hidden_tool])
    result = library([("call_1", "hidden_tool", {"name": "lookup"})])

    assert result.tool_calls[0].result == "lookup:True"


def test_hidden_parameter_is_ignored_from_model_params():
    def hidden_tool(name: str, secret: mf.Hidden[str] = "safe") -> str:
        """Hide a parameter from schema and runtime model params."""
        return f"{name}:{secret}"

    library = ToolLibrary(name="lib", tools=[hidden_tool])
    result = library([("call_1", "hidden_tool", {"name": "lookup", "secret": "model"})])

    assert result.tool_calls[0].result == "lookup:safe"
    assert result.tool_calls[0].parameters == {"name": "lookup"}


def test_hidden_handle_schema_excludes_notification_handle():
    def publish_status(
        handle: mf.Hidden,
        name: str,
    ) -> str:
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
    assert "handle" not in props


def test_injected_handle_can_add_and_remove_tools():
    def multiply(x: int) -> int:
        """Multiply a number by two."""
        return x * 2

    @mf.tool_config(inject_handle=True)
    def add_multiplier(handle: mf.Hidden) -> list[str]:
        """Register the multiply tool."""
        handle.add(multiply)
        return handle.list_tools()

    @mf.tool_config(inject_handle=True)
    def remove_tool(
        handle: mf.Hidden,
        name: str,
    ) -> list[str]:
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


def test_injected_handle_can_add_background_tool_with_task_tools():
    @mf.tool_config(background=True, inject_handle=True)
    def background_multiplier(
        value: int,
        handle: mf.Hidden,
    ) -> int:
        """Multiply a number by two in the background."""
        handle.update_progress(stage="work", message="Running", current=1, total=1)
        return value * 2

    @mf.tool_config(inject_handle=True)
    def add_background_multiplier(
        handle: mf.Hidden,
    ) -> list[str]:
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

    @mf.tool_config(background=True, inject_handle=True)
    def long_job(value: int, handle: mf.Hidden) -> int:
        """Run a long job in the background."""
        handle.set_running(stage="prepare", message="Preparing")
        handle.update_progress(stage="work", message="Halfway", current=1, total=2)
        started.set()
        release.wait(timeout=2.0)
        handle.update_progress(stage="work", message="Finishing", current=2, total=2)
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
    assert task_state["metadata"]["background_capabilities"] == []
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

    @mf.tool_config(background=True, inject_handle=True)
    def long_job(value: int, handle: mf.Hidden) -> int:
        """Run a long job in the background."""
        handle.update_progress(stage="work", message="Halfway", current=1, total=2)
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


def test_task_interrupt_interrupts_background_agent_at_next_checkpoint():
    slow_tool_started = threading.Event()
    release_tool = threading.Event()

    def slow_tool() -> str:
        """Block until released."""
        slow_tool_started.set()
        release_tool.wait(timeout=2.0)
        return "tool finished"

    worker_model = _ScriptedModel(
        [
            _tool_call_response("slow_tool", {}),
            _text_response("should not happen"),
        ]
    )
    worker = Agent(name="worker", model=worker_model, tools=[slow_tool])
    worker.tool_config = {"background": True}

    library = ToolLibrary(name="lib", tools=[worker])
    dispatch = library([("call_1", "worker", {"task": "Start worker."})])
    task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]

    assert slow_tool_started.wait(timeout=1.0)
    interrupt_result = (
        library([("call_2", "task_interrupt", {"task_id": task_id})])
        .tool_calls[0]
        .result
    )
    assert interrupt_result["status"] == "interrupt_requested"

    release_tool.set()
    _wait_until(
        lambda: (
            library([("call_3", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result["status"]
            == "interrupted"
        )
    )

    status = (
        library([("call_4", "task_status", {"task_id": task_id})]).tool_calls[0].result
    )
    assert status["status"] == "interrupted"
    assert status["metadata"]["background_capabilities"] == [
        "activity",
        "message",
    ]
    assert status["last_activity_summary"] == "Status: Task interrupted."


def test_cancelled_background_future_is_not_logged_as_error():
    library = ToolLibrary(name="lib", tools=[])
    future = Mock()
    future.result.side_effect = FutureCancelledError()

    with patch("msgflux.runtime.background.logger.error") as mock_error:
        library.get_background_dispatcher().log_task_failure(future)

    mock_error.assert_not_called()


def test_task_wait_falls_back_to_task_store_polling_without_future():
    @mf.tool_config(background=True)
    def placeholder() -> None:
        """Enable task control functions for the library."""
        return None

    library = ToolLibrary(name="lib", tools=[placeholder])
    task = library.get_task_store().create(tool_name="external_job")

    def complete_task():
        time.sleep(0.1)
        library.get_task_store().complete(task.task_id, 99)

    timer = threading.Thread(target=complete_task)
    timer.start()
    try:
        wait_result = library(
            [("call_1", "task_wait", {"task_id": task.task_id, "timeout": 1.0})]
        )
    finally:
        timer.join(timeout=1.0)

    assert wait_result.tool_calls[0].result == 99


def test_agent_injects_pending_task_notifications_as_system_note_messages():
    release = threading.Event()

    @mf.tool_config(background=True)
    def long_job(value: int) -> int:
        """Run a long job in the background."""
        release.wait(timeout=2.0)
        return value * 2

    agent = Agent(name="Assistant", model=_mock_model(), tools=[long_job])

    agent.tool_library([("call_1", "long_job", {"value": 5})])
    task_id = (
        agent.tool_library([("call_2", "task_list", {})])
        .tool_calls[0]
        .result[0]["task_id"]
    )

    release.set()
    _wait_until(
        lambda: (
            agent.tool_library([("call_3", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result["status"]
            == "completed"
        )
    )

    _wait_until(
        lambda: bool(
            _notification_messages(
                agent.inspect_model_execution_params("Continue.")["messages"],
                source="task",
                status="completed",
            )
        )
    )
    params = agent.inspect_model_execution_params("Continue.")
    notification_messages = _notification_messages(
        params["messages"],
        source="task",
        status="completed",
    )

    assert len(notification_messages) == 1
    assert notification_messages[0]["role"] == "system"
    content = notification_messages[0]["content"]
    assert "<system_note>" in content
    assert "<notification>" in content
    assert f"ref: {task_id}" in content
    assert "tool: long_job" in content
    assert f"task_output(task_id='{task_id}')" in content


def test_inspect_model_execution_params_does_not_consume_notifications():
    release = threading.Event()

    @mf.tool_config(background=True)
    def long_job(value: int) -> int:
        """Run a long job in the background."""
        release.wait(timeout=2.0)
        return value * 2

    model = _mock_model()
    agent = Agent(name="Assistant", model=model, tools=[long_job])

    agent.tool_library([("call_1", "long_job", {"value": 5})])
    task_id = (
        agent.tool_library([("call_2", "task_list", {})])
        .tool_calls[0]
        .result[0]["task_id"]
    )

    release.set()
    _wait_until(
        lambda: (
            agent.tool_library([("call_3", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result["status"]
            == "completed"
        )
    )

    _wait_until(
        lambda: bool(
            _notification_messages(
                agent.inspect_model_execution_params("Continue.")["messages"],
                source="task",
                status="completed",
            )
        )
    )
    params = agent.inspect_model_execution_params("Continue.")
    notification_messages = _notification_messages(
        params["messages"],
        source="task",
        status="completed",
    )
    assert len(notification_messages) == 1

    params = agent.inspect_model_execution_params("Continue again.")
    notification_messages = _notification_messages(
        params["messages"],
        source="task",
        status="completed",
    )
    assert len(notification_messages) == 1

    messages = ChatMessages()
    agent("Continue now.", messages=messages)

    model_messages = model.call_args.kwargs["messages"]
    notification_messages = _notification_messages(
        model_messages,
        source="task",
        status="completed",
    )
    assert len(notification_messages) == 1
    assert notification_messages[0]["role"] == "system"

    history_messages = messages.to_chatml()
    persisted_notifications = _notification_messages(
        history_messages,
        source="task",
        status="completed",
    )
    assert len(persisted_notifications) == 1
    assert persisted_notifications[0]["role"] == "system"
    notification_index = history_messages.index(persisted_notifications[0])
    user_index = next(
        index
        for index, message in enumerate(history_messages)
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and "Continue now." in message["content"]
    )
    assert notification_index < user_index

    params = agent.inspect_model_execution_params("Continue once more.")
    notification_messages = _notification_messages(params["messages"])
    assert notification_messages == []


def test_agent_control_interrupts_before_model_call():
    inbox = mf.AgentInbox(store=mf.InMemoryAgentInboxStore())
    model = _mock_model()
    agent = Agent(name="Assistant", model=model)
    agent.set_agent_inbox(inbox)

    inbox.interrupt(reason="operator requested interrupt")

    with pytest.raises(
        TaskInterruptRequestedError, match="operator requested interrupt"
    ):
        agent("Continue.")

    assert not model.called


def test_agent_control_pause_saves_checkpoint_before_model_call():
    inbox = mf.AgentInbox(store=mf.InMemoryAgentInboxStore())
    store = InMemoryCheckpointStore()
    model = _mock_model()
    agent = Agent(name="Assistant", model=model, checkpointer=store)
    agent.set_agent_inbox(inbox)
    scope = mf.ExecutionScope(thread_id="user_42", run_id="run_pause")

    inbox.pause(reason="wait for user input")

    with pytest.raises(TaskPauseRequestedError, match="wait for user input"):
        agent("Continue.", scope=scope)

    state = store.load_state("Assistant", "user_42", "run_pause")
    assert state is not None
    assert state["status"] == "paused"
    assert not model.called


def test_agent_incoming_user_message_is_injected_before_model_call():
    inbox = mf.AgentInbox(store=mf.InMemoryAgentInboxStore())
    model = _mock_model()
    agent = Agent(name="Assistant", model=model)
    agent.set_agent_inbox(inbox)

    inbox.user_message("I changed my mind.")
    agent("Continue.")

    incoming = _incoming_user_messages(model.call_args.kwargs["messages"])
    assert len(incoming) == 1
    assert incoming[0]["role"] == "user"
    assert "I changed my mind." in incoming[0]["content"]
    assert "<system_note>" not in incoming[0]["content"]


def test_agent_consumes_persisted_incoming_user_message_for_scope():
    store = mf.InMemoryAgentInboxStore()
    inbox = mf.AgentInbox(store=store)
    model = _mock_model()
    agent = Agent(name="Assistant", model=model, agent_inbox=inbox)
    scope = mf.ExecutionScope(thread_id="user_42", run_id="run_42")
    external_inbox = mf.AgentInbox(
        store=store,
        namespace="Assistant",
        thread_id="user_42",
        run_id="run_42",
    )

    external_inbox.user_message("Use the customer-visible tone.")
    agent("Continue.", scope=scope)

    incoming = _incoming_user_messages(model.call_args.kwargs["messages"])
    assert len(incoming) == 1
    assert incoming[0]["role"] == "user"
    assert "Use the customer-visible tone." in incoming[0]["content"]
    assert "<system_note>" not in incoming[0]["content"]
    assert external_inbox.peek() == []


def test_agent_drains_notifications_after_tool_call_before_next_model_call():
    @mf.tool_config(inject_handle=True)
    def publish_status(handle: mf.Hidden) -> str:
        """Publish an in-loop status update."""
        handle.get_notification().update(status="progress", hint="Tool completed.")
        return "ok"

    model = _ScriptedModel(
        [
            _tool_call_response("publish_status", {}),
            _text_response("done"),
        ]
    )
    agent = Agent(name="Assistant", model=model, tools=[publish_status])

    agent("Run tool.")

    assert len(model.calls) == 2
    notifications = _notification_messages(
        model.calls[1]["messages"],
        source="tool_status",
        status="progress",
    )
    assert len(notifications) == 1
    assert notifications[0]["role"] == "system"
    assert "Tool completed." in notifications[0]["content"]


def test_task_progress_notifications_are_persisted():
    started = threading.Event()
    release = threading.Event()

    @mf.tool_config(background=True, inject_handle=True)
    def long_job(value: int, handle: mf.Hidden) -> int:
        """Emit progress updates while running in the background."""
        handle.notify(
            source="task_progress",
            status="update",
            hint="Wait for the final completion notification before consuming output.",
            metadata={"tool_stage": "prepare"},
            dedupe_key=f"progress:{handle.get_task_id()}",
        )
        started.set()
        release.wait(timeout=2.0)
        return value * 2

    model = _mock_model()
    agent = Agent(
        name="Assistant",
        model=model,
        tools=[long_job],
    )

    agent.tool_library([("call_1", "long_job", {"value": 5})])
    task_id = (
        agent.tool_library([("call_2", "task_list", {})])
        .tool_calls[0]
        .result[0]["task_id"]
    )
    assert started.wait(timeout=1.0)

    messages = ChatMessages()
    agent("Continue.", messages=messages)

    model_messages = model.call_args.kwargs["messages"]
    progress_notifications = _notification_messages(
        model_messages,
        source="task_progress",
        status="update",
    )
    assert len(progress_notifications) == 1
    assert progress_notifications[0]["role"] == "system"
    assert f"ref: {task_id}" in progress_notifications[0]["content"]
    assert "tool_stage: prepare" in progress_notifications[0]["content"]

    persisted_notifications = _notification_messages(
        messages.to_chatml(),
        source="task_progress",
        status="update",
    )
    assert len(persisted_notifications) == 1
    assert persisted_notifications[0]["role"] == "system"

    release.set()
    _wait_until(
        lambda: (
            agent.tool_library([("call_3", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result["status"]
            == "completed"
        )
    )


def test_injected_handle_publishes_task_status_updates():
    started = threading.Event()
    release = threading.Event()

    @mf.tool_config(background=True, inject_handle=True)
    def long_job(value: int, handle: mf.Hidden) -> int:
        """Emit task status updates through the injected tool handle."""
        handle.get_notification().update(
            "prepare",
            hint="Background work has started.",
            metadata={"step": 1},
            dedupe_key="job-status",
        )
        started.set()
        release.wait(timeout=2.0)
        handle.get_notification().update(
            "process",
            metadata={"step": 2},
            dedupe_key="job-status",
        )
        return value * 3

    model = _mock_model()
    agent = Agent(
        name="Assistant",
        model=model,
        tools=[long_job],
    )

    agent.tool_library([("call_1", "long_job", {"value": 7})])
    task_id = (
        agent.tool_library([("call_2", "task_list", {})])
        .tool_calls[0]
        .result[0]["task_id"]
    )
    assert started.wait(timeout=1.0)

    messages = ChatMessages()
    agent("Continue.", messages=messages)

    status_notifications = _notification_messages(
        model.call_args.kwargs["messages"],
        source="tool_status",
        status="prepare",
    )
    assert len(status_notifications) == 1
    assert f"ref: {task_id}" in status_notifications[0]["content"]
    assert "tool: long_job" in status_notifications[0]["content"]
    assert "step: 1" in status_notifications[0]["content"]

    release.set()
    _wait_until(
        lambda: (
            agent.tool_library([("call_3", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result["status"]
            == "completed"
        )
    )

    agent("Continue again.", messages=messages)
    process_notifications = _notification_messages(
        model.call_args.kwargs["messages"],
        source="tool_status",
        status="process",
    )
    assert len(process_notifications) == 1


def test_nested_agent_uses_inherited_inbox_from_execution_context():
    parent_inbox = mf.AgentInbox(store=mf.InMemoryAgentInboxStore())
    child = Agent(name="child", model=_mock_model())

    with execution_context(agent_inbox=parent_inbox):
        effective_inbox = child._get_effective_agent_inbox()

    assert effective_inbox is parent_inbox


def test_background_agent_inherits_context_and_checkpoint_run_id():
    store = InMemoryCheckpointStore()
    worker = Agent(name="worker", model=_mock_model("worker-done"))
    worker.tool_config = {"background": True}

    library = ToolLibrary(name="lib", tools=[worker])

    with execution_context(
        thread_id="user_42",
        namespace="root_agent",
        run_id="run_root",
        root_run_id="run_root",
        checkpoint_store=store,
    ):
        dispatch = library([("call_1", "worker", {"task": "Solve this"})])

    assert "task_id='" in dispatch.tool_calls[0].result
    task_id = library([("call_2", "task_list", {})]).tool_calls[0].result[0]["task_id"]

    _wait_until(
        lambda: (
            library([("call_3", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result["status"]
            == "completed"
        )
    )

    task_state = (
        library([("call_4", "task_status", {"task_id": task_id})]).tool_calls[0].result
    )
    assert task_state["metadata"]["thread_id"] == "user_42"
    assert task_state["metadata"]["parent_run_id"] == "run_root"
    assert task_state["metadata"]["root_run_id"] == "run_root"
    assert task_state["metadata"]["checkpoint_thread_id"] == "user_42"
    assert task_state["metadata"]["checkpoint_run_id"] == task_id


def test_background_agent_dispatch_mentions_task_message_and_activity():
    worker = Agent(name="worker", model=_mock_model("done"))
    worker.tool_config = {"background": True}

    library = ToolLibrary(name="lib", tools=[worker])

    dispatch = library([("call_1", "worker", {"task": "Solve this"})])
    result = dispatch.tool_calls[0].result

    assert "`task_activity`" in result
    assert "`task_message`" in result
    assert "`task_interrupt`" in result
    assert "`task_wait`" in result
    assert "`task_output`" in result
    assert "task_message" in library.get_tool_names()
    assert "task_activity" in library.get_tool_names()
    assert "task_interrupt" in library.get_tool_names()


def test_injected_handle_can_add_background_agent_with_agent_task_tools():
    worker = Agent(name="worker", model=_mock_model("done"))
    worker.tool_config = {"background": True}

    @mf.tool_config(inject_handle=True)
    def add_worker(handle: mf.Hidden) -> list[str]:
        """Register a background agent."""
        handle.add(worker)
        return handle.list_tools()

    library = ToolLibrary(name="lib", tools=[add_worker])

    add_result = library([("call_1", "add_worker", {})]).tool_calls[0].result

    assert "worker" in add_result
    assert "task_status" in add_result
    assert "task_interrupt" in add_result
    assert "task_wait" in add_result
    assert "task_output" in add_result
    assert "task_activity" in add_result
    assert "task_message" in add_result


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


def test_background_activity_capability_is_available_for_non_agent_task():
    @mf.tool_config(background=True, background_capabilities=["activity"])
    def monitored_pipeline(value: int) -> int:
        """Run a monitored background pipeline."""
        return value * 2

    library = ToolLibrary(name="lib", tools=[monitored_pipeline])

    dispatch = library([("call_1", "monitored_pipeline", {"value": 4})])
    task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]

    assert "`task_activity`" in dispatch.tool_calls[0].result
    assert "`task_message`" not in dispatch.tool_calls[0].result
    assert "task_activity" in library.get_tool_names()
    assert "task_message" not in library.get_tool_names()

    task = (
        library([("call_2", "task_status", {"task_id": task_id})]).tool_calls[0].result
    )
    activity = (
        library([("call_3", "task_activity", {"task_id": task_id})])
        .tool_calls[0]
        .result
    )

    assert task["metadata"]["task_kind"] == "tool"
    assert task["metadata"]["background_capabilities"] == ["activity"]
    assert isinstance(activity, list)


def test_task_activity_is_unsupported_without_activity_capability():
    @mf.tool_config(background=True)
    def slow_pipeline(value: int) -> int:
        """Run a simple background tool."""
        return value * 2

    worker = Agent(name="worker", model=_mock_model("done"))
    worker.tool_config = {"background": True}

    library = ToolLibrary(name="lib", tools=[slow_pipeline, worker])

    dispatch = library([("call_1", "slow_pipeline", {"value": 4})])
    task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]

    activity = (
        library([("call_2", "task_activity", {"task_id": task_id})])
        .tool_calls[0]
        .result
    )

    assert activity["status"] == "unsupported"
    assert "activity capability" in activity["error"]


def test_task_activity_tracks_compact_subagent_tool_calls():
    def multiply(x: int) -> int:
        """Multiply by two."""
        return x * 2

    worker_model = _ScriptedModel(
        [
            _tool_call_response("multiply", {"x": 4}),
            _text_response("done"),
        ]
    )
    worker = Agent(name="worker", model=worker_model, tools=[multiply])
    worker.tool_config = {"background": True}

    library = ToolLibrary(name="lib", tools=[worker])
    dispatch = library([("call_1", "worker", {"task": "Multiply 4 by 2."})])
    task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]

    _wait_until(
        lambda: (
            library([("call_2", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result["status"]
            == "completed"
        )
    )

    activity = (
        library([("call_3", "task_activity", {"task_id": task_id})])
        .tool_calls[0]
        .result
    )

    assert any(entry == "Status: Task queued." for entry in activity)
    assert any(entry == "Status: Task running." for entry in activity)
    assert any("ToolCall: multiply({" in entry for entry in activity)
    assert all("ToolResult:" not in entry for entry in activity)


def test_task_message_resumes_completed_background_agent():
    store = InMemoryCheckpointStore()
    task_store = InMemoryTaskStore()
    worker_model = _ScriptedModel(
        [
            _text_response("first pass"),
            _text_response("resumed pass"),
        ]
    )
    worker = Agent(name="worker", model=worker_model)
    worker.tool_config = {"background": True}

    library = ToolLibrary(name="lib", tools=[worker])
    default_task_store = library.get_task_store()
    with execution_context(
        thread_id="user_42",
        namespace="root_agent",
        run_id="run_root",
        root_run_id="run_root",
        checkpoint_store=store,
        task_store=task_store,
    ):
        dispatch = library([("call_1", "worker", {"task": "Start worker."})])
        task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]
        _wait_until(
            lambda: (
                library([("call_2", "task_status", {"task_id": task_id})])
                .tool_calls[0]
                .result["status"]
                == "completed"
            )
        )

        message_result = (
            library(
                [
                    (
                        "call_3",
                        "task_message",
                        {"task_id": task_id, "message": "Continue."},
                    )
                ]
            )
            .tool_calls[0]
            .result
        )

        assert message_result["status"] == "resumed"

        _wait_until(
            lambda: (
                library([("call_4", "task_status", {"task_id": task_id})])
                .tool_calls[0]
                .result["status"]
                == "completed"
            )
        )
        output = (
            library([("call_5", "task_output", {"task_id": task_id})])
            .tool_calls[0]
            .result
        )

        assert output == "resumed pass"
        task_state = (
            library([("call_6", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result
        )
        resumed_run_id = task_state["metadata"]["checkpoint_run_id"]
        assert resumed_run_id != task_id
        assert store.load_state("worker", "user_42", task_id)["status"] == "completed"
        assert (
            store.load_state("worker", "user_42", resumed_run_id)["status"]
            == "completed"
        )
    assert default_task_store.get(task_id) is None
    assert task_store.get(task_id).status == "completed"


def test_task_message_resume_clears_previous_interrupt_reason():
    slow_tool_started = threading.Event()
    release_tool = threading.Event()

    def slow_tool() -> str:
        """Block until released."""
        slow_tool_started.set()
        release_tool.wait(timeout=2.0)
        return "tool finished"

    store = InMemoryCheckpointStore()
    worker_model = _ScriptedModel(
        [
            _tool_call_response("slow_tool", {}),
            _text_response("resumed pass"),
        ]
    )
    worker = Agent(name="worker", model=worker_model, tools=[slow_tool])
    worker.tool_config = {"background": True}

    library = ToolLibrary(name="lib", tools=[worker])
    with execution_context(
        thread_id="user_42",
        namespace="root_agent",
        run_id="run_root",
        root_run_id="run_root",
        checkpoint_store=store,
    ):
        dispatch = library([("call_1", "worker", {"task": "Start worker."})])
        task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]

        assert slow_tool_started.wait(timeout=1.0)
        interrupt_result = (
            library([("call_2", "task_interrupt", {"task_id": task_id})])
            .tool_calls[0]
            .result
        )
        assert interrupt_result["status"] == "interrupt_requested"

        release_tool.set()
        _wait_until(
            lambda: (
                library([("call_3", "task_status", {"task_id": task_id})])
                .tool_calls[0]
                .result["status"]
                == "interrupted"
            )
        )

        interrupted_state = (
            library([("call_4", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result
        )
        assert "interrupt_reason" in interrupted_state["metadata"]

        message_result = (
            library(
                [
                    (
                        "call_5",
                        "task_message",
                        {"task_id": task_id, "message": "Continue."},
                    )
                ]
            )
            .tool_calls[0]
            .result
        )
        assert message_result["status"] == "resumed"

        _wait_until(
            lambda: (
                library([("call_6", "task_status", {"task_id": task_id})])
                .tool_calls[0]
                .result["status"]
                == "completed"
            )
        )

        resumed_state = (
            library([("call_7", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result
        )
        assert "interrupt_reason" not in resumed_state["metadata"]

    state = store.load_state("worker", "user_42", task_id)
    assert state is not None
