from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

import pytest

from msgflux import TaskError
from msgflux._private.executor import Executor
from msgflux.nn import functional as F


def _use_executor(monkeypatch, executor):
    monkeypatch.setattr(
        Executor,
        "get_instance",
        classmethod(lambda _cls: executor),
    )


@pytest.mark.parametrize("api", ["map_gather", "scatter_gather", "bcast_gather"])
def test_gather_timeout_returns_completed_results_without_waiting_for_slow_call(
    monkeypatch, api
):
    release_slow = Event()
    slow_started = Event()
    fast_finished = Event()
    caller_finished = Event()
    responses = []

    def work(kind):
        if kind == "fast":
            fast_finished.set()
            return "fast result"
        slow_started.set()
        assert release_slow.wait(timeout=3)
        return "slow result"

    with ThreadPoolExecutor(max_workers=2) as executor:
        _use_executor(monkeypatch, executor)

        def invoke():
            if api == "map_gather":
                result = F.map_gather(
                    work,
                    args_list=[("fast",), ("slow",)],
                    timeout=0.1,
                )
            elif api == "scatter_gather":
                result = F.scatter_gather(
                    [lambda: work("fast"), lambda: work("slow")],
                    timeout=0.1,
                )
            else:
                result = F.bcast_gather(
                    [lambda: work("fast"), lambda: work("slow")],
                    timeout=0.1,
                )
            responses.append(result)
            caller_finished.set()

        caller = Thread(target=invoke)
        caller.start()
        try:
            assert slow_started.wait(timeout=2)
            assert fast_finished.wait(timeout=2)
            returned_before_release = caller_finished.wait(timeout=1)
        finally:
            release_slow.set()
            caller.join(timeout=2)

        assert not caller.is_alive()
        assert returned_before_release
        assert responses[0][0] == "fast result"
        assert isinstance(responses[0][1], TaskError)
        assert isinstance(responses[0][1].exception, TimeoutError)
        assert responses[0][1].index == 1


def test_zero_timeout_returns_while_queued_call_continues_afterward(monkeypatch):
    release_blocker = Event()
    blocker_started = Event()
    target_started = Event()
    target_finished = Event()
    caller_finished = Event()
    responses = []

    def blocker():
        blocker_started.set()
        assert release_blocker.wait(timeout=3)

    def target():
        target_started.set()
        target_finished.set()
        return "late result"

    with ThreadPoolExecutor(max_workers=1) as executor:
        blocker_future = executor.submit(blocker)
        assert blocker_started.wait(timeout=2)
        _use_executor(monkeypatch, executor)

        def invoke():
            responses.append(F.wait_for(target, timeout=0))
            caller_finished.set()

        caller = Thread(target=invoke)
        caller.start()
        try:
            returned_before_release = caller_finished.wait(timeout=1)
            target_was_queued = not target_started.is_set()
        finally:
            release_blocker.set()
            caller.join(timeout=2)
            blocker_future.result(timeout=2)

        assert not caller.is_alive()
        assert returned_before_release
        assert target_was_queued
        assert isinstance(responses[0], TaskError)
        assert isinstance(responses[0].exception, TimeoutError)
        assert responses[0].index == 0
        assert target_started.wait(timeout=2)
        assert target_finished.is_set()


def test_none_timeout_waits_for_result_and_preserves_callable_exception(monkeypatch):
    error = ValueError("callable failed")

    def fail():
        raise error

    with ThreadPoolExecutor(max_workers=1) as executor:
        _use_executor(monkeypatch, executor)
        assert F.wait_for(lambda: "ready", timeout=None) == "ready"

        result = F.wait_for(fail, timeout=None)

    assert isinstance(result, TaskError)
    assert result.exception is error
    assert result.index == 0
