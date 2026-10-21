from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, Thread

import pytest

from msgflux import TaskError
from msgflux._private.executor import Executor
from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.nn import functional as F


def _use_executor(monkeypatch, executor):
    monkeypatch.setattr(
        Executor,
        "get_instance",
        classmethod(lambda _cls: executor),
    )


@pytest.mark.parametrize("api", ["map_gather", "scatter_gather", "bcast_gather"])
def test_submit_errors_are_indexed_for_each_sync_functional_api(monkeypatch, api):
    class RejectedExecutor:
        def submit(self, *_args, **_kwargs):
            raise RuntimeError("executor rejected this item")

    _use_executor(monkeypatch, RejectedExecutor())

    if api == "map_gather":
        result = F.map_gather(lambda value: value, args_list=[(1,), (2,)])
    elif api == "scatter_gather":
        result = F.scatter_gather([lambda: 1, lambda: 2])
    else:
        result = F.bcast_gather([lambda: 1, lambda: 2])

    assert len(result) == 2
    assert all(isinstance(item, TaskError) for item in result)
    assert [item.index for item in result] == [0, 1]
    assert [str(item.exception) for item in result] == [
        "executor rejected this item",
        "executor rejected this item",
    ]


def test_sync_factory_failure_is_itemized_and_accepted_sibling_is_observed(
    monkeypatch,
):
    release_first = Event()
    first_started = Event()
    first_finished = Event()
    failing_factory_called = Event()
    error = RuntimeError("acall factory failed")

    def first():
        first_started.set()
        assert release_first.wait(timeout=3)
        first_finished.set()
        return "first result"

    class BrokenAsyncCallable:
        def __call__(self):
            return "sync fallback"

        def acall(self):
            failing_factory_called.set()
            raise error

    def release_after_factory_failure():
        assert failing_factory_called.wait(timeout=2)
        release_first.set()

    executor = Executor.__new__(Executor)
    executor.thread_pool = ThreadPoolExecutor(max_workers=2)
    executor._thread_local = threading.local()
    releaser = Thread(target=release_after_factory_failure)
    releaser.start()
    _use_executor(monkeypatch, executor)
    try:
        result = F.scatter_gather(
            [first, BrokenAsyncCallable(), lambda: "third result"], timeout=2
        )
    finally:
        release_first.set()
        releaser.join(timeout=2)
        executor.thread_pool.shutdown(wait=True)

    assert not releaser.is_alive()
    assert first_started.is_set()
    assert first_finished.is_set()
    assert result[0] == "first result"
    assert isinstance(result[1], TaskError)
    assert result[1].exception is error
    assert result[1].index == 1
    assert result[2] == "third result"


@pytest.mark.parametrize(
    "control_error",
    [
        AbortRequestedError("stop"),
        TaskPauseRequestedError("task", "pause"),
        TaskInterruptRequestedError("task", "interrupt"),
    ],
)
def test_control_error_from_later_submission_still_propagates(
    monkeypatch, control_error
):
    accepted = Future()
    accepted.set_result("accepted")

    class ControlOnSecondSubmit:
        calls = 0

        def submit(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return accepted
            raise control_error

    _use_executor(monkeypatch, ControlOnSecondSubmit())
    with pytest.raises(type(control_error)) as raised:
        F.scatter_gather([lambda: "first", lambda: "second"])

    assert raised.value is control_error


def test_wait_for_submit_failure_returns_task_error(monkeypatch):
    error = RuntimeError("executor is unavailable")

    class RejectedExecutor:
        def submit(self, *_args, **_kwargs):
            raise error

    _use_executor(monkeypatch, RejectedExecutor())
    result = F.wait_for(lambda: "never submitted")

    assert isinstance(result, TaskError)
    assert result.exception is error
    assert result.index == 0
