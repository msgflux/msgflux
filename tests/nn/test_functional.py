"""Tests for msgflux.nn.functional module."""

import asyncio
from functools import partial

import pytest

from msgflux import TaskError
from msgflux._private.executor import Executor
from msgflux.core.dotdict import dotdict
from msgflux.envs import envs
from msgflux.nn import functional as F


@pytest.fixture
def fresh_executor(monkeypatch):
    """Reset the singleton so tests can control the executor shape safely."""
    existing = Executor._instance
    if existing is not None:
        existing.shutdown()
        Executor._instance = None

    monkeypatch.setattr(envs, "executor_num_threads", 1)
    monkeypatch.setattr(envs, "executor_num_async_workers", 1)

    yield

    current = Executor._instance
    if current is not None:
        current.shutdown()
        Executor._instance = None


class TestMapGather:
    """Test suite for map_gather function."""

    def test_map_gather_basic(self):
        """Test basic map_gather functionality."""

        def add(x, y):
            return x + y

        results = F.map_gather(add, args_list=[(1, 2), (3, 4), (5, 6)])
        assert results == (3, 7, 11)

    def test_map_gather_with_kwargs(self):
        """Test map_gather with kwargs_list."""

        def multiply(x, y=2):
            return x * y

        results = F.map_gather(
            multiply,
            args_list=[(1,), (3,), (5,)],
            kwargs_list=[{"y": 3}, {"y": 4}, {"y": 5}],
        )
        assert results == (3, 12, 25)

    def test_map_gather_not_callable(self):
        """Test map_gather raises TypeError for non-callable."""
        with pytest.raises(TypeError, match="`to_send` must be a callable"):
            F.map_gather("not_callable", args_list=[(1,)])

    def test_map_gather_empty_args_list(self):
        """Test map_gather raises ValueError for empty args_list."""

        def dummy(x):
            return x

        with pytest.raises(ValueError, match="`args_list` must be a non-empty list"):
            F.map_gather(dummy, args_list=[])

    def test_map_gather_mismatched_kwargs_list(self):
        """Test map_gather raises ValueError for mismatched kwargs_list length."""

        def dummy(x, y=1):
            return x + y

        with pytest.raises(ValueError, match="`kwargs_list` must be a list"):
            F.map_gather(
                dummy,
                args_list=[(1,), (2,)],
                kwargs_list=[{"y": 2}],  # Length mismatch
            )


class TestScatterGather:
    """Test suite for scatter_gather function."""

    def test_scatter_gather_basic(self):
        """Test basic scatter_gather functionality."""

        def add(x, y):
            return x + y

        def multiply(x, y=2):
            return x * y

        callables = [add, multiply, add]
        args = [(1, 2), (3,), (10, 20)]
        results = F.scatter_gather(callables, args_list=args)

        assert results == (3, 6, 30)

    def test_scatter_gather_with_kwargs(self):
        """Test scatter_gather with kwargs."""

        def add(x, y):
            return x + y

        def multiply(x, y=2):
            return x * y

        callables = [add, multiply, add]
        args = [(1,), (), (10,)]
        kwargs = [{"y": 2}, {"x": 3, "y": 3}, {"y": 20}]

        results = F.scatter_gather(callables, args_list=args, kwargs_list=kwargs)
        assert results == (3, 9, 30)

    def test_scatter_gather_not_callable_list(self):
        """Test scatter_gather raises TypeError for non-callable list."""
        with pytest.raises(TypeError, match="`to_send` must be a non-empty list"):
            F.scatter_gather("not_a_list")

    def test_scatter_gather_nested_sync_work_executes_inline(self, fresh_executor):
        """Nested sync fan-out should not deadlock the shared thread pool."""

        def leaf(value):
            return value + 1

        def outer(value):
            return F.scatter_gather([partial(leaf, value)])[0]

        results = F.scatter_gather([outer], args_list=[(1,)])

        assert results == (2,)

    def test_scatter_gather_partial_preserves_acall(self, fresh_executor):
        """Partials of async-capable callables should still route through acall."""

        class AsyncCapable:
            def __call__(self, value):
                raise AssertionError("sync path should not be used")

            async def acall(self, value):
                return value + 1

        def outer(value):
            return F.scatter_gather([partial(AsyncCapable(), value=value)])[0]

        results = F.scatter_gather([outer], args_list=[(1,)])

        assert results == (2,)


class TestBcastGather:
    """Test suite for bcast_gather function."""

    def test_bcast_gather_basic(self):
        """Test basic bcast_gather functionality."""

        def square(x):
            return x * x

        def cube(x):
            return x * x * x

        results = F.bcast_gather([square, cube], 3)
        assert results == (9, 27)

    def test_bcast_gather_with_kwargs(self):
        """Test bcast_gather with kwargs."""

        def multiply(x, y=1):
            return x * y

        results = F.bcast_gather([multiply, multiply], 5, y=3)
        assert results == (15, 15)

    def test_bcast_gather_not_callable_list(self):
        """Test bcast_gather raises TypeError for non-callable list."""
        with pytest.raises(TypeError, match="`to_send` must be a non-empty list"):
            F.bcast_gather(["not_callable"], 1)

    def test_bcast_gather_supports_inplace_message_mutation(self):
        """Test bcast_gather with shared mutable message."""

        def add_field_a(msg):
            msg["field_a"] = "value_a"

        def add_field_b(msg):
            msg["field_b"] = "value_b"

        message = dotdict()
        results = F.bcast_gather([add_field_a, add_field_b], message)

        assert results == (None, None)
        assert message["field_a"] == "value_a"
        assert message["field_b"] == "value_b"

    def test_bcast_gather_returns_task_error(self):
        """Test bcast_gather captures callable failures as TaskError."""

        def fail(_):
            raise ValueError("boom")

        results = F.bcast_gather([fail], dotdict())

        assert len(results) == 1
        assert isinstance(results[0], TaskError)
        assert str(results[0]) == "Task 0 failed: boom"


class TestWaitFor:
    """Test suite for wait_for function."""

    def test_wait_for_basic(self):
        """Test basic wait_for functionality."""

        def square(x):
            return x * x

        result = F.wait_for(square, 5)
        assert result == 25

    def test_wait_for_not_callable(self):
        """Test wait_for raises TypeError for non-callable."""
        with pytest.raises(TypeError, match="`to_send` must be a callable"):
            F.wait_for("not_callable", 1)


class TestSpawn:
    """Test suite for spawn function."""

    def test_spawn_basic(self):
        """Test basic spawn functionality."""
        results = []

        def append_value(value):
            results.append(value)

        F.spawn(append_value, 42)
        # Give it a moment to execute
        import time

        time.sleep(0.1)

        assert 42 in results

    def test_spawn_not_callable(self):
        """Test spawn raises TypeError for non-callable."""
        with pytest.raises(TypeError, match="`to_send` must be a callable"):
            F.spawn("not_callable")


class TestWaitForEvent:
    """Test suite for wait_for_event function."""

    def test_wait_for_event_not_event(self):
        """Test wait_for_event raises TypeError for non-Event."""
        with pytest.raises(TypeError, match="`event` must be an instance"):
            F.wait_for_event("not_event")


class TestAsyncFunctions:
    """Test suite for async functional functions."""

    @pytest.mark.asyncio
    async def test_amap_gather_basic(self):
        """Test basic amap_gather functionality."""

        async def add(x, y):
            return x + y

        results = await F.amap_gather(add, args_list=[(1, 2), (3, 4), (5, 6)])
        assert results == (3, 7, 11)

    @pytest.mark.asyncio
    async def test_amap_gather_not_callable(self):
        """Test amap_gather raises TypeError for non-callable."""
        with pytest.raises(TypeError, match="`to_send` must be a callable"):
            await F.amap_gather("not_callable", args_list=[(1,)])

    @pytest.mark.asyncio
    async def test_ascatter_gather_basic(self):
        """Test basic ascatter_gather functionality."""

        async def add(x, y):
            return x + y

        async def multiply(x, y=2):
            return x * y

        callables = [add, multiply, add]
        args = [(1, 2), (3,), (10, 20)]
        results = await F.ascatter_gather(callables, args_list=args)

        assert results == (3, 6, 30)

    @pytest.mark.asyncio
    async def test_ascatter_gather_not_callable_list(self):
        """Test ascatter_gather raises TypeError for non-callable list."""
        with pytest.raises(TypeError, match="`to_send` must be a non-empty list"):
            await F.ascatter_gather("not_a_list")

    @pytest.mark.asyncio
    async def test_ascatter_gather_supports_inplace_message_mutation(self):
        """Test ascatter_gather with shared mutable message."""

        async def add_field_a(msg):
            msg["field_a"] = "value_a"

        async def add_field_b(msg):
            msg["field_b"] = "value_b"

        message = dotdict()
        results = await F.ascatter_gather(
            [add_field_a, add_field_b],
            args_list=[(message,), (message,)],
        )

        assert results == (None, None)
        assert message["field_a"] == "value_a"
        assert message["field_b"] == "value_b"

    @pytest.mark.asyncio
    async def test_await_for_event_basic(self):
        """Test basic await_for_event functionality."""
        event = asyncio.Event()

        async def setter():
            await asyncio.sleep(0.05)
            event.set()

        asyncio.create_task(setter())
        await F.await_for_event(event)

        assert event.is_set()

    @pytest.mark.asyncio
    async def test_await_for_event_not_event(self):
        """Test await_for_event raises TypeError for non-Event."""
        with pytest.raises(TypeError, match="`event` must be an instance"):
            await F.await_for_event("not_event")

    @pytest.mark.asyncio
    async def test_aspawn_basic(self):
        """Test basic aspawn functionality."""
        results = []

        async def append_value(value):
            results.append(value)

        await F.aspawn(append_value, 99)
        # Give it a moment to execute
        await asyncio.sleep(0.1)

        assert 99 in results

    @pytest.mark.asyncio
    async def test_aspawn_not_callable(self):
        """Test aspawn raises TypeError for non-callable."""
        with pytest.raises(TypeError, match="`to_send` must be a callable"):
            await F.aspawn("not_callable")
