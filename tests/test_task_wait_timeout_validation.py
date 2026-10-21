"""Non-finite task wait timeouts must not turn into unbounded waits."""

import math
import threading

import msgflux as mf
import pytest
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tasks import InMemoryTaskStore
from msgflux.tools.builtin import TaskWaitTool


@pytest.mark.parametrize("timeout", [math.nan, math.inf, -math.inf])
def test_task_wait_rejects_nonfinite_timeout_without_live_future(timeout):
    store = InMemoryTaskStore()
    task = store.create(tool_name="external_worker")
    library = ToolLibrary(
        name="nonfinite-task-wait",
        tools=[TaskWaitTool()],
        task_store=store,
    )
    finished = threading.Event()
    outcome = {}

    def call_wait():
        try:
            outcome["call"] = library(
                [
                    (
                        "wait",
                        "task_wait",
                        {"task_id": task.task_id, "timeout": timeout},
                    )
                ]
            ).tool_calls[0]
        except BaseException as error:
            outcome["error"] = error
        finally:
            finished.set()

    waiter = threading.Thread(target=call_wait)
    waiter.start()
    try:
        assert finished.wait(timeout=1)
        assert "finite" in outcome["call"].error
    finally:
        # This also releases the baseline NaN/+infinity poll loop if validation
        # regresses, so the test cannot leave a waiter thread behind.
        store.complete(task.task_id, "cleanup")
        waiter.join(timeout=2)
    assert not waiter.is_alive()


def test_task_wait_rejects_nonfinite_timeout_with_live_future():
    started = threading.Event()
    release = threading.Event()

    @mf.tool_config(background=True)
    def blocked_work() -> str:
        """Keep a real dispatcher future pending until the test releases it."""
        started.set()
        assert release.wait(timeout=3)
        return "done"

    library = ToolLibrary(name="live-nonfinite-task-wait", tools=[blocked_work])
    try:
        library([("dispatch", "blocked_work", {})])
        task_id = (
            library([("list", "task_list", {})]).tool_calls[0].result[0]["task_id"]
        )
        assert started.wait(timeout=2)

        response = library(
            [
                (
                    "wait",
                    "task_wait",
                    {"task_id": task_id, "timeout": math.inf},
                )
            ]
        )
        assert "finite" in response.tool_calls[0].error
    finally:
        release.set()
        if "task_id" in locals():
            future = library.get_background_dispatcher().get_task_future(task_id)
            if future is not None:
                future.result(timeout=3)
