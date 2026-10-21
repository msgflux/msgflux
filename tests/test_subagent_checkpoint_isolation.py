"""Regression tests for checkpoint history isolation across child agent runs."""

import asyncio
import json
import threading
import time

import msgflux as mf
from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.chat_messages import ChatMessages
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_context,
)
from msgflux.runtime.context import _HistoryOrigin, _history_origin_context
from msgflux.tasks import InMemoryTaskStore, SQLiteTaskStore
from msgflux.tools.builtin import AgentTool
from msgflux.tools.builtin.task_tool import TaskMessageTool


def _text_response(text: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    response.metadata = {}
    return response


def _tool_response(calls) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("tool_call")
    aggregate = ToolCallAggregator()
    for index, (call_id, name, parameters) in enumerate(calls):
        aggregate.process(index, call_id, name, json.dumps(parameters))
    response.add(aggregate)
    response.reasoning = None
    response.metadata = {}
    return response


def _wait_until(predicate, timeout: float = 4.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for child task")


class _CapturingModel:
    model_type = "chat_completion"

    def __init__(self, response="done", *, barrier=None):
        self.calls = []
        self.scopes = []
        self.response = response
        self.barrier = barrier
        self.lock = threading.Lock()
        self.callback = None

    def __call__(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
            self.scopes.append(get_execution_context().get("scope"))
        if self.callback is not None:
            self.callback(kwargs)
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        return _text_response(self.response)

    async def acall(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
            self.scopes.append(get_execution_context().get("scope"))
        if self.callback is not None:
            self.callback(kwargs)
        if self.barrier is not None:
            await asyncio.to_thread(self.barrier.wait, timeout=5)
        return _text_response(self.response)


class _NestedDispatchModel:
    model_type = "chat_completion"

    def __init__(self):
        self.calls = []
        self.responses = [
            _tool_response(
                [
                    ("nested-A", "agent", {"name": "worker", "message": "inner-A"}),
                    ("nested-B", "agent", {"name": "worker", "message": "inner-B"}),
                ]
            ),
            _text_response("outer done"),
        ]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)

    async def acall(self, **kwargs):
        return self(**kwargs)


def _messages(call):
    return [str(item.get("content", "")) for item in call["messages"]]


def _dispatch_background(library, call_id, marker, *, agent_name="worker"):
    result = (
        library(
            [
                (
                    call_id,
                    "agent",
                    {
                        "name": agent_name,
                        "message": marker,
                        "run_in_background": True,
                    },
                )
            ]
        )
        .tool_calls[0]
        .result
    )
    return result.split("task_id='")[1].split("'")[0]


def _wait_complete(library, task_id):
    _wait_until(
        lambda: (
            library([("status", "task_status", {"task_id": task_id})])
            .tool_calls[0]
            .result["status"]
            == "completed"
        )
    )


def test_agent_tool_child_histories_are_isolated_for_sequential_tasks():
    model = _CapturingModel()
    worker = Agent(name="worker", model=model)
    tools = [mf.tool_config(allow_background=True)(AgentTool()), worker]
    checkpoints = InMemoryCheckpointStore()
    tasks = InMemoryTaskStore()
    library = ToolLibrary(name="root", tools=tools, task_store=tasks)

    with execution_context(
        thread_id="shared-parent-thread",
        namespace="root",
        run_id="parent-run",
        root_run_id="parent-run",
        checkpoint_store=checkpoints,
        task_store=tasks,
    ):
        task_a = _dispatch_background(library, "call-a", "marker-A")
        _wait_complete(library, task_a)
        task_b = _dispatch_background(library, "call-b", "marker-B")
        _wait_complete(library, task_b)

    assert task_a != task_b
    call_a = next(
        call for call in model.calls if "marker-A" in " ".join(_messages(call))
    )
    call_b = next(
        call for call in model.calls if "marker-B" in " ".join(_messages(call))
    )
    assert "marker-A" not in " ".join(_messages(call_b))
    assert "marker-B" not in " ".join(_messages(call_a))
    metadata_a = tasks.get(task_a).metadata
    metadata_b = tasks.get(task_b).metadata
    assert metadata_a["checkpoint_thread_id"] == metadata_b["checkpoint_thread_id"]
    assert metadata_a["checkpoint_run_id"] != metadata_b["checkpoint_run_id"]


def test_agent_tool_parallel_child_histories_are_isolated():
    barrier = threading.Barrier(2)
    model = _CapturingModel(barrier=barrier)
    worker = Agent(name="worker", model=model)
    tools = [mf.tool_config(allow_background=True)(AgentTool()), worker]
    checkpoints = InMemoryCheckpointStore()
    tasks = InMemoryTaskStore()
    library = ToolLibrary(name="root", tools=tools, task_store=tasks)

    with execution_context(
        thread_id="parallel-parent-thread",
        namespace="root",
        run_id="parent-run",
        root_run_id="parent-run",
        checkpoint_store=checkpoints,
        task_store=tasks,
    ):
        task_a = _dispatch_background(library, "parallel-a", "parallel-A")
        task_b = _dispatch_background(library, "parallel-b", "parallel-B")
        _wait_complete(library, task_a)
        _wait_complete(library, task_b)

    assert len(model.calls) == 2
    histories = [" ".join(_messages(call)) for call in model.calls]
    assert any("parallel-A" in history for history in histories)
    assert any("parallel-B" in history for history in histories)
    assert all(not ("parallel-A" in h and "parallel-B" in h) for h in histories)


def test_foreground_agent_tool_calls_do_not_share_child_history():
    model = _CapturingModel()
    worker = Agent(name="worker", model=model)
    library = ToolLibrary(name="root", tools=[AgentTool(), worker])
    checkpoints = InMemoryCheckpointStore()

    with execution_context(
        thread_id="foreground-parent-thread",
        namespace="root",
        run_id="parent-run",
        checkpoint_store=checkpoints,
    ):
        library(
            [("foreground-a", "agent", {"name": "worker", "message": "foreground-A"})]
        )
        library(
            [("foreground-b", "agent", {"name": "worker", "message": "foreground-B"})]
        )

    call_a = next(
        call for call in model.calls if "foreground-A" in " ".join(_messages(call))
    )
    call_b = next(
        call for call in model.calls if "foreground-B" in " ".join(_messages(call))
    )
    assert "foreground-A" not in " ".join(_messages(call_b))
    assert "foreground-B" not in " ".join(_messages(call_a))


def test_partial_and_mismatched_explicit_agent_tool_scopes_are_fresh():
    model = _CapturingModel()
    worker = Agent(name="worker", model=model)
    agent_tool = AgentTool()
    library = ToolLibrary(name="root", tools=[agent_tool, worker])
    checkpoints = InMemoryCheckpointStore()
    thread_id = "partial-scope-thread"
    namespace = worker.get_module_name()

    with execution_context(checkpoint_store=checkpoints):
        worker(
            "parent-marker",
            scope=ExecutionScope(
                namespace=namespace, thread_id=thread_id, run_id="parent-run"
            ),
        )
    with execution_context(
        namespace=namespace,
        thread_id=thread_id,
        run_id="parent-run",
        checkpoint_store=checkpoints,
    ):
        for call_id, marker in (
            ("partial-A", "partial-A-marker"),
            ("partial-B", "partial-B-marker"),
        ):
            agent_tool(
                name="worker",
                message=marker,
                handle=library.get_handle().for_tool(
                    tool_name="agent", tool_call_id=call_id
                ),
                scope=ExecutionScope(),
            )

    call_a = next(
        call for call in model.calls if "partial-A-marker" in " ".join(_messages(call))
    )
    call_b = next(
        call for call in model.calls if "partial-B-marker" in " ".join(_messages(call))
    )
    history_a = " ".join(_messages(call_a))
    history_b = " ".join(_messages(call_b))
    assert "parent-marker" not in history_a + history_b
    assert "partial-B-marker" not in history_a
    assert "partial-A-marker" not in history_b
    scope_a = model.scopes[model.calls.index(call_a)]
    scope_b = model.scopes[model.calls.index(call_b)]
    assert scope_a.run_id != scope_b.run_id
    assert "parent-run" not in {scope_a.run_id, scope_b.run_id}

    with execution_context(checkpoint_store=checkpoints):
        worker(
            "source-marker",
            scope=ExecutionScope(
                namespace=namespace, thread_id=thread_id, run_id="source-run"
            ),
        )
    mismatched_origin = _HistoryOrigin(
        namespace=namespace,
        thread_id=thread_id,
        run_id="primary-run",
        source_run_id="source-run",
    )
    with (
        execution_context(
            namespace=namespace,
            thread_id=thread_id,
            run_id="primary-run",
            checkpoint_store=checkpoints,
        ),
        _history_origin_context(mismatched_origin),
    ):
        agent_tool(
            name="worker",
            message="explicit-scope-marker",
            handle=library.get_handle().for_tool(
                tool_name="agent", tool_call_id="primary-call"
            ),
            scope=ExecutionScope(
                namespace=namespace,
                thread_id=thread_id,
                run_id="explicit-other-run",
            ),
        )

    explicit_history = " ".join(_messages(model.calls[-1]))
    assert "explicit-scope-marker" in explicit_history
    assert "source-marker" not in explicit_history


def test_nested_same_profile_agent_tool_uses_fresh_scopes_and_histories():
    outer_model = _NestedDispatchModel()
    inner_model = _CapturingModel()
    outer = Agent(name="worker", model=outer_model)
    inner = Agent(name="worker", model=inner_model)
    outer.tool_library.add(AgentTool())
    outer.tool_library.add(inner)
    tasks = InMemoryTaskStore()
    library = ToolLibrary(
        name="root",
        tools=[mf.tool_config(allow_background=True)(AgentTool()), outer],
        task_store=tasks,
    )
    checkpoints = InMemoryCheckpointStore()

    with execution_context(
        thread_id="nested-parent-thread",
        namespace="root",
        run_id="parent-run",
        checkpoint_store=checkpoints,
    ):
        task_id = _dispatch_background(library, "outer-call", "outer-marker")
        _wait_complete(library, task_id)

    assert len(inner_model.calls) == 2
    histories = [" ".join(_messages(call)) for call in inner_model.calls]
    assert any("inner-A" in history for history in histories)
    assert any("inner-B" in history for history in histories)
    assert all("outer-marker" not in history for history in histories)
    assert all(
        not ("inner-A" in history and "inner-B" in history) for history in histories
    )
    inner_run_ids = [scope.run_id for scope in inner_model.scopes]
    assert len(set(inner_run_ids)) == 2
    assert all(scope.namespace == "worker" for scope in inner_model.scopes)
    task = tasks.get(task_id)
    assert task.metadata["checkpoint_run_id"] == task_id
    assert (
        checkpoints.load_state(
            task.metadata["checkpoint_namespace"],
            task.metadata["checkpoint_thread_id"],
            task_id,
        )["status"]
        == "completed"
    )


def test_direct_agent_thread_continuation_still_uses_latest_history():
    model = _CapturingModel()
    checkpoints = InMemoryCheckpointStore()
    agent = Agent(name="worker", model=model, checkpoint_store=checkpoints)

    agent("direct-A", scope=ExecutionScope(thread_id="direct-thread", run_id="run-A"))
    agent("direct-B", scope=ExecutionScope(thread_id="direct-thread", run_id="run-B"))

    second_history = " ".join(_messages(model.calls[-1]))
    assert "direct-A" in second_history
    assert "direct-B" in second_history


def test_scoped_checkpoint_queries_select_fresh_and_source_snapshots():
    checkpoints = InMemoryCheckpointStore()
    model = _CapturingModel()
    agent = Agent(name="worker", model=model, checkpoint_store=checkpoints)
    namespace = "worker"
    thread_id = "snapshot-thread"
    callbacks = []
    model.callback = lambda kwargs: callbacks.append(
        (
            agent.get_last_model_metadata(scope=get_execution_context().get("scope")),
            agent.get_reasoning_effort(scope=get_execution_context().get("scope")),
            kwargs.get("model_preference"),
        )
    )

    for run_id, marker, model_id, effort, preference in (
        ("run-A", "snapshot-A", "model-A", "low", "preference-A"),
        ("run-B", "snapshot-B", "model-B", "high", "preference-B"),
    ):
        messages = ChatMessages(
            [
                {"role": "user", "content": marker},
                {
                    "role": "assistant",
                    "content": f"answer {marker}",
                    "metadata": {
                        "model": {
                            "provider": "fake",
                            "model_id": model_id,
                            "api_mode": "chat",
                            "reasoning_effort": effort,
                        }
                    },
                },
                {"type": "model_configuration", "reasoning_effort": effort},
            ],
            thread_id=thread_id,
            namespace=namespace,
        )
        checkpoints.save_state(
            namespace,
            thread_id,
            run_id,
            {
                "status": "completed",
                "messages": messages._to_state(),
                "model_preference": preference,
            },
        )

    fresh_scope = ExecutionScope(
        namespace=namespace, thread_id=thread_id, run_id="fresh-B"
    )
    fresh_origin = _HistoryOrigin(
        namespace=namespace,
        thread_id=thread_id,
        run_id="fresh-B",
        source_run_id=None,
    )
    with _history_origin_context(fresh_origin):
        assert agent.get_last_model_metadata(scope=fresh_scope) is None
        restored, _, preference = agent._continue_thread_from_checkpoint(
            messages=ChatMessages(),
            vars={},
            model_preference=None,
            thread_id=thread_id,
            run_id=fresh_scope.run_id,
        )
        assert not restored
        assert preference is None
        agent("fresh-B prompt", scope=fresh_scope)
        assert callbacks[-1] == (None, None, None)

    resume_scope = ExecutionScope(
        namespace=namespace, thread_id=thread_id, run_id="resume-A"
    )
    resume_origin = _HistoryOrigin(
        namespace=namespace,
        thread_id=thread_id,
        run_id="resume-A",
        source_run_id="run-A",
    )
    with _history_origin_context(resume_origin):
        assert agent.get_last_model_metadata(scope=resume_scope) == {
            "provider": "fake",
            "model_id": "model-A",
            "api_mode": "chat",
        }
        assert agent.get_reasoning_effort(scope=resume_scope) == "low"
        restored, _, preference = agent._continue_thread_from_checkpoint(
            messages=ChatMessages(),
            vars={},
            model_preference=None,
            thread_id=thread_id,
            run_id=resume_scope.run_id,
        )
        restored_history = " ".join(
            str(item.get("content", "")) for item in restored.to_items()
        )
        assert "snapshot-A" in restored_history
        assert "snapshot-B" not in restored_history
        assert preference == "preference-A"
        agent("resume-A prompt", scope=resume_scope)
        assert callbacks[-1] == (
            {"provider": "fake", "model_id": "model-A", "api_mode": "chat"},
            "low",
            "preference-A",
        )


def test_task_resume_with_unknown_origin_fails_before_model_call(monkeypatch):
    model = _CapturingModel()
    worker = Agent(name="worker", model=model)
    tools = [mf.tool_config(allow_background=True)(AgentTool()), worker]
    checkpoints = InMemoryCheckpointStore()
    tasks = InMemoryTaskStore()
    library = ToolLibrary(name="root", tools=tools, task_store=tasks)

    with execution_context(
        thread_id="missing-origin-thread",
        namespace="root",
        run_id="parent-run",
        checkpoint_store=checkpoints,
        task_store=tasks,
    ):
        task_id = _dispatch_background(library, "missing-origin", "before-resume")
        _wait_complete(library, task_id)
        calls_before_resume = len(model.calls)

        # Simulate origin metadata loss after the real requeue writes the new
        # checkpoint run ID and before the background worker starts.
        requeue = tasks.requeue

        def requeue_without_origin(*args, **kwargs):
            updated = requeue(*args, **kwargs)
            tasks.update_metadata(task_id, {"checkpoint_origin_run_id": ""})
            return tasks.get(task_id) if updated is not None else None

        monkeypatch.setattr(tasks, "requeue", requeue_without_origin)
        result = TaskMessageTool()(
            task_id=task_id,
            message="must fail before model",
            handle=library.get_handle(),
        )
        assert result["status"] == "resumed"
        _wait_until(lambda: tasks.get(task_id).status == "failed")

    assert len(model.calls) == calls_before_resume
    assert tasks.get(task_id).status != "queued"


def test_task_resume_with_missing_source_checkpoint_fails_with_source_error(
    monkeypatch,
):
    model = _CapturingModel()
    worker = Agent(name="worker", model=model)
    tools = [mf.tool_config(allow_background=True)(AgentTool()), worker]
    checkpoints = InMemoryCheckpointStore()
    tasks = InMemoryTaskStore()
    library = ToolLibrary(name="root", tools=tools, task_store=tasks)

    with execution_context(
        thread_id="missing-source-thread",
        namespace="root",
        run_id="parent-run",
        checkpoint_store=checkpoints,
        task_store=tasks,
    ):
        task_id = _dispatch_background(library, "missing-source", "before-resume")
        _wait_complete(library, task_id)
        task = tasks.get(task_id)
        source_run_id = task.metadata["checkpoint_run_id"]
        source_namespace = task.metadata["checkpoint_namespace"]
        source_thread_id = task.metadata["checkpoint_thread_id"]
        assert (
            checkpoints.load_state(source_namespace, source_thread_id, source_run_id)
            is not None
        )
        calls_before_resume = len(model.calls)

        # Leave every checkpoint operation intact except the precise origin read.
        load_state = checkpoints.load_state

        def missing_source(namespace, thread_id, run_id):
            if (namespace, thread_id, run_id) == (
                source_namespace,
                source_thread_id,
                source_run_id,
            ):
                return None
            return load_state(namespace, thread_id, run_id)

        monkeypatch.setattr(checkpoints, "load_state", missing_source)
        result = TaskMessageTool()(
            task_id=task_id,
            message="resume from missing known source",
            handle=library.get_handle(),
        )
        assert result["status"] == "resumed"
        _wait_until(lambda: tasks.get(task_id).status == "failed")

    failed_task = tasks.get(task_id)
    assert len(model.calls) == calls_before_resume
    assert failed_task.status == "failed"
    assert source_run_id in failed_task.error
    assert "Expected checkpoint run" in failed_task.error


def test_sqlite_checkpoint_and_task_stores_keep_child_origins_independent(tmp_path):
    model = _CapturingModel()
    worker = Agent(name="worker", model=model)
    tools = [mf.tool_config(allow_background=True)(AgentTool()), worker]
    checkpoints = SQLiteCheckpointStore(str(tmp_path / "checkpoints.sqlite3"))
    tasks = SQLiteTaskStore(str(tmp_path / "tasks.sqlite3"))
    library = ToolLibrary(name="root", tools=tools, task_store=tasks)

    with execution_context(
        thread_id="sqlite-parent-thread",
        namespace="root",
        run_id="parent-run",
        checkpoint_store=checkpoints,
        task_store=tasks,
    ):
        task_a = _dispatch_background(library, "sqlite-a", "sqlite-A")
        _wait_complete(library, task_a)
        task_b = _dispatch_background(library, "sqlite-b", "sqlite-B")
        _wait_complete(library, task_b)

    call_b = next(
        call for call in model.calls if "sqlite-B" in " ".join(_messages(call))
    )
    assert "sqlite-A" not in " ".join(_messages(call_b))
    assert (
        tasks.get(task_a).metadata["checkpoint_thread_id"]
        == tasks.get(task_b).metadata["checkpoint_thread_id"]
    )
    assert (
        tasks.get(task_a).metadata["checkpoint_run_id"]
        != tasks.get(task_b).metadata["checkpoint_run_id"]
    )


def test_task_message_resumes_target_run_with_its_history_only():
    model = _CapturingModel()
    worker = Agent(name="worker", model=model)
    tools = [mf.tool_config(allow_background=True)(AgentTool()), worker]
    checkpoints = InMemoryCheckpointStore()
    tasks = InMemoryTaskStore()
    library = ToolLibrary(name="root", tools=tools, task_store=tasks)

    with execution_context(
        thread_id="resume-parent-thread",
        namespace="root",
        run_id="parent-run",
        checkpoint_store=checkpoints,
        task_store=tasks,
    ):
        task_a = _dispatch_background(library, "resume-a", "history-A")
        _wait_complete(library, task_a)
        task_b = _dispatch_background(library, "resume-b", "history-B")
        _wait_complete(library, task_b)
        # Resume lineage must come from the task's checkpoint scope/origin,
        # without depending on the original model tool-call ID being stored.
        tasks.update_metadata(task_a, {"tool_call_id": ""})
        tasks.update_metadata(task_b, {"tool_call_id": ""})
        assert not tasks.get(task_a).metadata.get("tool_call_id")
        assert not tasks.get(task_b).metadata.get("tool_call_id")
        resumed = (
            library(
                [("resume-A", "task_message", {"task_id": task_a, "message": "new-A"})]
            )
            .tool_calls[0]
            .result
        )
        assert resumed["status"] == "resumed"
        _wait_complete(library, task_a)
        resumed_b = (
            library(
                [("resume-B", "task_message", {"task_id": task_b, "message": "new-B"})]
            )
            .tool_calls[0]
            .result
        )
        assert resumed_b["status"] == "resumed"
        _wait_complete(library, task_b)

    resume_a_history = " ".join(_messages(model.calls[-2]))
    assert "history-A" in resume_a_history
    assert "new-A" in resume_a_history
    assert "history-B" not in resume_a_history
    resume_b_history = " ".join(_messages(model.calls[-1]))
    assert "history-B" in resume_b_history
    assert "new-B" in resume_b_history
    assert "history-A" not in resume_b_history
    assert "new-A" not in resume_b_history
    assert tasks.pending_messages(task_a) == []
    assert tasks.pending_messages(task_b) == []
    assert (
        tasks.get(task_a).metadata["checkpoint_run_id"]
        != tasks.get(task_b).metadata["checkpoint_run_id"]
    )


def test_background_agent_alias_uses_canonical_checkpoint_namespace_and_resumes():
    model = _CapturingModel()
    worker = Agent(name="worker_module", model=model)
    registered_worker = mf.tool_config(name_override="worker_alias")(worker)
    tools = [mf.tool_config(allow_background=True)(AgentTool()), registered_worker]
    checkpoints = InMemoryCheckpointStore()
    tasks = InMemoryTaskStore()
    library = ToolLibrary(name="root", tools=tools, task_store=tasks)

    with execution_context(
        thread_id="alias-parent-thread",
        namespace="root",
        run_id="parent-run",
        checkpoint_store=checkpoints,
        task_store=tasks,
    ):
        task_id = _dispatch_background(
            library, "alias-start", "alias-history", agent_name="worker_alias"
        )
        _wait_complete(library, task_id)

        task = tasks.get(task_id)
        assert task.metadata["checkpoint_namespace"] == worker.get_module_name()
        assert task.metadata["checkpoint_namespace"] != "worker_alias"
        assert (
            checkpoints.load_state(
                worker.get_module_name(),
                task.metadata["checkpoint_thread_id"],
                task.metadata["checkpoint_run_id"],
            )
            is not None
        )

        resumed = (
            library(
                [
                    (
                        "alias-resume",
                        "task_message",
                        {"task_id": task_id, "message": "alias-new-message"},
                    )
                ]
            )
            .tool_calls[0]
            .result
        )
        assert resumed["status"] == "resumed"
        _wait_complete(library, task_id)

    resumed_history = " ".join(_messages(model.calls[-1]))
    assert "alias-history" in resumed_history
    assert "alias-new-message" in resumed_history
    assert "checkpoint_namespace" in tasks.get(task_id).metadata
    assert (
        tasks.get(task_id).metadata["checkpoint_namespace"] == worker.get_module_name()
    )
