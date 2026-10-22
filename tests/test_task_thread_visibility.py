"""Task controls stay inside the active execution thread."""

from __future__ import annotations

import threading
import time

import msgflux as mf
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent, ToolLibrary
from msgflux.runtime import ExecutionScope, execution_context
from msgflux.tasks import InMemoryTaskStore


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for condition.")


def _thread(thread_id: str, *, namespace: str = "root"):
    run_id = f"run-{thread_id}"
    return execution_context(
        scope=ExecutionScope(
            thread_id=thread_id,
            namespace=namespace,
            run_id=run_id,
            root_run_id=run_id,
        )
    )


def _thread_with_checkpoint(thread_id: str, checkpoint_store):
    run_id = f"run-{thread_id}"
    return execution_context(
        scope=ExecutionScope(
            thread_id=thread_id,
            namespace="root",
            run_id=run_id,
            root_run_id=run_id,
        ),
        checkpoint_store=checkpoint_store,
    )


def _call(library, call_id: str, name: str, **params):
    return library([(call_id, name, params)]).tool_calls[0].result


def _bucket_call(library, call_id: str, operation: str, **params):
    request = {"operation": operation, **params}
    return library([(call_id, "task", {"request": request})]).tool_calls[0].result


@mf.tool_config(background=True, retry=False, background_capabilities=["activity"])
def _secret_job() -> str:
    """Return a thread-local secret."""
    return "THREAD_A_SECRET"


def _message_capable_agent() -> Agent:
    worker = Agent(name="message_worker", model=_ScriptedModel([]))
    worker.tool_config = {"background": True}
    return worker


def _text_response(text: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    response.metadata = {}
    return response


class _ScriptedModel:
    model_type = "chat_completion"
    model_id = "fake/scripted"

    def __init__(self, texts):
        self._responses = [_text_response(text) for text in texts]
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("Scripted model exhausted.")
        return self._responses.pop(0)

    async def acall(self, **kwargs):
        return self(**kwargs)


def test_direct_task_tools_hide_other_thread_tasks():
    store = InMemoryTaskStore()
    release = threading.Event()

    @mf.tool_config(background=True, retry=False)
    def slow_job() -> str:
        """Block until the test releases the task."""
        release.wait(timeout=5)
        return "SLOW_DONE"

    library = ToolLibrary(
        "direct-thread-scope",
        [_secret_job, slow_job, _message_capable_agent()],
        task_store=store,
    )
    try:
        with _thread("thread-A"):
            _call(library, "a-secret", "_secret_job")
        task_id = store.list()[0].task_id
        _wait_until(lambda: store.get(task_id).status == "completed")

        with _thread("thread-B"):
            assert _call(library, "b-list", "task_list") == []
            for index, name in enumerate(
                (
                    "task_status",
                    "task_output",
                    "task_interrupt",
                    "task_activity",
                )
            ):
                result = _call(
                    library,
                    f"b-{index}",
                    name,
                    task_id=task_id,
                )
                assert result == {"task_id": task_id, "status": "not_found"}
            assert _call(
                library,
                "b-wait",
                "task_wait",
                task_id=task_id,
                timeout=0.1,
            ) == {"task_id": task_id, "status": "not_found"}
            message = _call(
                library,
                "b-message",
                "task_message",
                task_id=task_id,
                message="foreign instruction",
            )
            assert message == {"task_id": task_id, "status": "not_found"}

        task = store.get(task_id)
        assert task.status == "completed"
        assert task.result == "THREAD_A_SECRET"

        with _thread("thread-A"):
            dispatch = _call(library, "a-slow", "slow_job")
        slow_id = dispatch.split("task_id='")[1].split("'")[0]
        _wait_until(lambda: store.get(slow_id).status == "running")

        with _thread("thread-B"):
            assert _call(
                library,
                "b-interrupt",
                "task_interrupt",
                task_id=slow_id,
            ) == {"task_id": slow_id, "status": "not_found"}
            assert (
                _call(
                    library,
                    "b-status",
                    "task_status",
                    task_id=slow_id,
                )["status"]
                == "not_found"
            )

        slow_task = store.get(slow_id)
        assert slow_task.status == "running"
        assert slow_task.metadata.get("interrupt_requested") is False

        release.set()
        with _thread("thread-A"):
            assert (
                _call(
                    library,
                    "a-wait",
                    "task_wait",
                    task_id=slow_id,
                    timeout=5.0,
                )
                == "SLOW_DONE"
            )
    finally:
        release.set()


def test_task_bucket_hides_other_thread_tasks():
    store = InMemoryTaskStore()
    library = ToolLibrary(
        "bucket-thread-scope",
        [_secret_job, _message_capable_agent()],
        task_store=store,
    )
    library.extensions["background_tasks"].enable_grouping()
    assert "task" in library.get_tool_names()

    with _thread("thread-A"):
        _call(library, "a-secret", "_secret_job")
    task_id = store.list()[0].task_id
    _wait_until(lambda: store.get(task_id).status == "completed")

    with _thread("thread-B"):
        assert _bucket_call(library, "b-list", "list") == []
        for index, operation in enumerate(
            ("status", "output", "interrupt", "activity")
        ):
            result = _bucket_call(
                library,
                f"b-{index}",
                operation,
                task_id=task_id,
            )
            assert result == {"task_id": task_id, "status": "not_found"}
        assert _bucket_call(
            library,
            "b-wait",
            "wait",
            task_id=task_id,
            timeout=0.1,
        ) == {"task_id": task_id, "status": "not_found"}
        message = _bucket_call(
            library,
            "b-message",
            "message",
            task_id=task_id,
            message="foreign instruction",
        )
        assert message == {"task_id": task_id, "status": "not_found"}

    with _thread("thread-A"):
        assert _bucket_call(library, "a-output", "output", task_id=task_id) == (
            "THREAD_A_SECRET"
        )


def test_cross_thread_message_cannot_resume_but_parent_can():
    store = InMemoryTaskStore()
    checkpoints = InMemoryCheckpointStore()
    model = _ScriptedModel(["worker-done", "resumed-by-A"])
    worker = Agent(name="worker", model=model)
    worker.tool_config = {"background": True}
    library = ToolLibrary("message-thread-scope", [worker], task_store=store)

    with _thread_with_checkpoint("thread-A", checkpoints):
        dispatch = _call(library, "a-dispatch", "worker", task="A private task")
    task_id = dispatch.split("task_id='")[1].split("'")[0]
    _wait_until(lambda: store.get(task_id).status == "completed")
    assert store.get(task_id).result == "worker-done"
    assert store.get(task_id).metadata["checkpoint_thread_id"] == "thread-A"
    assert store.get(task_id).metadata["checkpoint_namespace"] == "worker"

    with _thread_with_checkpoint("thread-B", checkpoints):
        result = _call(
            library,
            "b-message",
            "task_message",
            task_id=task_id,
            message="B injected instruction",
        )
        assert result == {"task_id": task_id, "status": "not_found"}

    assert len(model.calls) == 1
    task = store.get(task_id)
    assert task.status == "completed"
    assert task.result == "worker-done"

    with _thread_with_checkpoint("thread-A", checkpoints):
        listed = _call(library, "a-list", "task_list")
        assert [item["task_id"] for item in listed] == [task_id]
        assert (
            _call(
                library,
                "a-status",
                "task_status",
                task_id=task_id,
            )["status"]
            == "completed"
        )
        assert (
            _call(
                library,
                "a-message",
                "task_message",
                task_id=task_id,
                message="A follow-up",
            )["status"]
            == "resumed"
        )

    _wait_until(
        lambda: len(model.calls) == 2 and store.get(task_id).status == "completed"
    )
    assert store.get(task_id).result == "resumed-by-A"


def test_unscoped_host_task_tools_remain_global():
    store = InMemoryTaskStore()
    library = ToolLibrary(
        "host-thread-scope",
        [_secret_job],
        task_store=store,
    )

    with _thread("thread-A"):
        _call(library, "a-secret", "_secret_job")
    task_id = store.list()[0].task_id
    _wait_until(lambda: store.get(task_id).status == "completed")

    assert (
        _call(library, "host-status", "task_status", task_id=task_id)["status"]
        == "completed"
    )
    assert _call(library, "host-output", "task_output", task_id=task_id) == (
        "THREAD_A_SECRET"
    )
    assert [item["task_id"] for item in _call(library, "host-list", "task_list")] == [
        task_id
    ]
