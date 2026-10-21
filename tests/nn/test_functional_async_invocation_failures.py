"""Async gather captures failures raised while invoking each callable."""

import asyncio
import gc
import warnings

import pytest

from msgflux import TaskError
from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.nn import functional as F


class _CallableFactory:
    def __init__(
        self,
        *,
        failure,
        fail_on_bad=True,
        result_value=None,
        control=None,
    ):
        self.failure = failure
        self.fail_on_bad = fail_on_bad
        self.result_value = result_value
        self.control = control
        self.completed = []
        self.coroutines = []
        self.error = ValueError("factory failed")

    def __call__(self, *values):
        if self.control is not None and values[0] == "control":
            raise self.control
        if self.fail_on_bad and values[0] == "bad":
            if self.failure == "raise":
                raise self.error
            return "not awaitable"
        value = self.result_value or values[-1]

        async def complete():
            self.completed.append(value)
            return f"result:{value}"

        coroutine = complete()
        self.coroutines.append(coroutine)
        return coroutine


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["map", "scatter", "broadcast"])
@pytest.mark.parametrize("failure", ["raise", "not_awaitable"])
async def test_async_gather_captures_invocation_failure_and_completes_sibling(
    kind, failure
):
    factory = _CallableFactory(failure=failure)
    sibling = _CallableFactory(
        failure=failure,
        fail_on_bad=False,
        result_value="good",
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", RuntimeWarning)
        if kind == "map":
            results = await F.amap_gather(
                factory,
                args_list=[("good",), ("bad",)],
            )
        elif kind == "scatter":
            results = await F.ascatter_gather(
                [sibling, factory],
                args_list=[("good",), ("bad",)],
            )
        else:
            results = await F.abcast_gather([sibling, factory], "bad")
        gc.collect()

    assert results[0] == "result:good"
    failure_result = results[1]
    assert isinstance(failure_result, TaskError)
    assert failure_result.index == 1
    if failure == "raise":
        assert failure_result.exception is factory.error
    else:
        assert isinstance(failure_result.exception, TypeError)
    completed = factory.completed if kind == "map" else sibling.completed
    assert completed == ["good"]
    assert not any("was never awaited" in str(item.message) for item in recorded)


@pytest.mark.asyncio
async def test_async_scatter_validates_callable_list_before_scheduling():
    with pytest.raises(TypeError, match="`to_send` must be a non-empty list"):
        await F.ascatter_gather([lambda: None, "not callable"])


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["map", "scatter", "broadcast"])
@pytest.mark.parametrize(
    "control",
    [
        AbortRequestedError("abort"),
        TaskPauseRequestedError("task", "pause"),
        TaskInterruptRequestedError("task", "interrupt"),
        asyncio.CancelledError("cancel"),
    ],
    ids=["abort", "pause", "interrupt", "cancel"],
)
async def test_synchronous_control_during_preparation_propagates_and_closes(
    kind, control
):
    factory = _CallableFactory(failure="control", control=control)
    sibling = _CallableFactory(
        failure="unused",
        fail_on_bad=False,
        result_value="good",
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(type(control)) as raised:
            if kind == "map":
                await F.amap_gather(
                    factory,
                    args_list=[("good",), ("control",)],
                )
            elif kind == "scatter":
                await F.ascatter_gather(
                    [sibling, factory],
                    args_list=[("good",), ("control",)],
                )
            else:
                await F.abcast_gather([sibling, factory], "control")
        gc.collect()

    assert raised.value is control
    created = factory.coroutines if kind == "map" else sibling.coroutines
    assert len(created) == 1
    assert created[0].cr_frame is None
    assert factory.completed == []
    assert sibling.completed == []
    assert not any("was never awaited" in str(item.message) for item in recorded)


@pytest.mark.asyncio
async def test_async_body_cancellation_result_keeps_existing_gather_behavior():
    async def cancel_in_body():
        raise asyncio.CancelledError("async body cancel")

    (result,) = await F.ascatter_gather([cancel_in_body])

    assert isinstance(result, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_preparation_cancel_does_not_cancel_callee_owned_future():
    release = asyncio.Event()
    finished = asyncio.Event()
    control = asyncio.CancelledError("stop preparation")

    async def external_work():
        try:
            await release.wait()
            return "external result"
        finally:
            finished.set()

    future = None

    def future_factory():
        nonlocal future
        future = asyncio.create_task(external_work())
        return future

    def cancelled_factory():
        raise control

    results = None
    try:
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(
                F.ascatter_gather([future_factory, cancelled_factory]),
                timeout=2,
            )

        assert raised.value is control
        assert future is not None
        assert not future.cancelled()
    finally:
        release.set()
        if future is not None:
            results = await asyncio.wait_for(
                asyncio.gather(future, return_exceptions=True),
                timeout=2,
            )

    assert results == ["external result"]
    assert finished.is_set()
