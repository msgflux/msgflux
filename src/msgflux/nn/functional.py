# https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/
import asyncio
import concurrent.futures
import threading
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional, Tuple

from msgflux._private.executor import Executor
from msgflux.exceptions import TaskError
from msgflux.logger import logger
from msgflux.telemetry import Spans

__all__ = [
    "abcast_gather",
    "amap_gather",
    "ascatter_gather",
    "aspawn",
    "await_for_event",
    "bcast_gather",
    "map_gather",
    "scatter_gather",
    "spawn",
    "wait_for",
    "wait_for_event",
]


@Spans.instrument()
def map_gather(
    to_send: Callable,
    *,
    args_list: List[Tuple[Any, ...]],
    kwargs_list: Optional[List[Dict[str, Any]]] = None,
    timeout: Optional[float] = None,
) -> Tuple[Any, ...]:
    """Applies the `to_send` function to each set of arguments in `args_list`
    and `kwargs_list` using Executor and collects the results.

    Args:
        to_send:
            The callable function to be applied.
        args_list:
            Each tuple contains the positional argumentsvfor the corresponding callable
            in `to_send`. If `None`, no positional arguments are passed unless specified
            individually by an item in `kwargs_list`.
        kwargs_list:
            Each dictionary contains the named arguments for the corresponding callable
            in `to_send`. If `None`, no named arguments are passed unless specified
            individually by an item in `args_list`.
        timeout:
            Maximum time (in seconds) to wait for responses.

    Returns:
        A tuple containing the results of each call to the `f` function. If a call
        fails or times out, the corresponding result will be a `TaskError` instance.

    Raises:
        TypeError:
            If `f` is not callable.
        ValueError:
            If `args_list` is not a non-empty list or if `kwargs_list`
            (if provided) is not the same length as `args_list`.

    Examples:
        def add(x, y): return x + y
        results = F.map_gather(add, args_list=[(1, 2), (3, 4), (5, 6)])
        print(results)  # (3, 7, 11)

        def multiply(x, y=2): return x * y
        results = F.map_gather(multiply, args_list=[(1,), (3,), (5,)],
                            kwargs_list=[{'y': 3}, {'y': 4}, {'y': 5}])
        print(results)  # (3, 12, 25)

        results = F.map_gather(multiply, args_list=[(1,), (3,), (5,)])
        print(results)  # (2, 6, 10)
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    if not isinstance(args_list, list) or len(args_list) == 0:
        raise ValueError("`args_list` must be a non-empty list")

    if kwargs_list is not None:
        if not isinstance(kwargs_list, list) or len(kwargs_list) != len(args_list):
            raise ValueError(
                "`kwargs_list` must be a list with the same length as `args_list`"
            )

    executor = Executor.get_instance()
    futures = []

    for i in range(len(args_list)):
        args = args_list[i]
        kwargs = kwargs_list[i] if kwargs_list else {}
        futures.append(executor.submit(to_send, *args, **kwargs))

    concurrent.futures.wait(futures, timeout=timeout)
    responses: List[Any] = []
    for i, future in enumerate(futures):
        try:
            responses.append(future.result())
        except Exception as e:
            logger.error(str(e))
            responses.append(TaskError(exception=e, index=i))
    return tuple(responses)


@Spans.instrument()
def scatter_gather(
    to_send: List[Callable],
    args_list: Optional[List[Tuple[Any, ...]]] = None,
    kwargs_list: Optional[List[Dict[str, Any]]] = None,
    *,
    timeout: Optional[float] = None,
) -> Tuple[Any, ...]:
    """Sends different sets of arguments/kwargs to a list of modules
    and collects the responses.

    Each callable in `to_send` receives the positional arguments of
    the corresponding `tuple` in `args_list` and the named arguments
    of the corresponding `dict` in `kwargs_list`. If `args_list` or
    `kwargs_list` are not provided (or are `None`), the corresponding
    callables will be called without positional or named arguments,
    respectively, unless an empty list (`[]`) or empty tuple (`()`)
    is provided for a specific item.

    Args:
        to_send:
            List of callable objects (e.g. functions or `Module` instances).
        args_list:
            Each tuple contains the positional argumentsvfor the corresponding callable
            in `to_send`. If `None`, no positional arguments are passed unless specified
            individually by an item in `kwargs_list`.
        kwargs_list:
            Each dictionary contains the named arguments for the corresponding callable
            in `to_send`. If `None`, no named arguments are passed unless specified
            individually by an item in `args_list`.
        timeout:
            Maximum time (in seconds) to wait for responses.

    Returns:
        Tuple containing the responses for each callable. If an error or
        timeout occurs for a specific callable, its corresponding response
        in the tuple will be a `TaskError` instance.

    Raises:
        TypeError:
            If `to_send` is not a callable list.
        ValueError:
            If the lengths of `args_list` (if provided) or `kwargs_list`
            (if provided) do not match the length of `to_send`.

    Examples:
        def add(x, y): return x + y
        def multiply(x, y=2): return x * y
        callables = [add, multiply, add]

        # Example 1: Using only args_list
        args = [ (1, 2), (3,), (10, 20) ] # multiply will use its default y
        results = F.scatter_gather(callables, args_list=args)
        print(results) # (3, 6, 30)

        # Example 2: Using args_list e kwargs_list
        args = [ (1,), (), (10,) ]
        kwargs = [ {'y': 2}, {'x': 3, 'y': 3}, {'y': 20} ]
        results = F.scatter_gather(callables, args_list=args, kwargs_list=kwargs)
        print(results) # (3, 9, 30)

        # Example 3: Using only kwargs_list (useful if functions have
        # defaults or don't need positional args)
        def greet(name="World"): return f"Hello, {name}"
        def farewell(person_name): return f"Goodbye, {person_name}"
        funcs = [greet, greet, farewell]
        kwargs_for_funcs = [ {}, {'name': "Earth"}, {'person_name': "Commander"} ]
        results = F.scatter_gather(funcs, kwargs_list=kwargs_for_funcs)
        print(results) # ("Hello, World", "Hello, Earth", "Goodbye, Commander")
    """
    if not isinstance(to_send, list) or not all(callable(f) for f in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    executor = Executor.get_instance()
    futures = []
    for i, f in enumerate(to_send):
        args = args_list[i] if args_list and i < len(args_list) else ()
        kwargs = kwargs_list[i] if kwargs_list and i < len(kwargs_list) else {}
        futures.append(executor.submit(f, *args, **kwargs))

    concurrent.futures.wait(futures, timeout=timeout)
    responses: List[Any] = []
    for i, future in enumerate(futures):
        try:
            responses.append(future.result())
        except Exception as e:
            logger.error(str(e))
            responses.append(TaskError(exception=e, index=i))
    return tuple(responses)


@Spans.instrument()
def bcast_gather(
    to_send: List[Callable], *args: Any, timeout: Optional[float] = None, **kwargs: Any
) -> Tuple[Any, ...]:
    """Broadcasts arguments to multiple callables and gathers the responses.

    Args:
        to_send:
            List of callable objects (e.g. functions or `Module` instances).
        *args:
            Positional arguments.
        timeout:
            Maximum time (in seconds) to wait for responses.
        **kwargs:
            Named arguments.

    Returns:
        Tuple containing the responses.

    Raises:
        TypeError:
            If `to_send` is not a list of callables.

    Examples:
        def square(x): return x * x
        def cube(x): return x * x * x
        def fail(x): raise ValueError("Intentional error")

        # Example 1:
        results = F.bcast_gather([square, cube], 3)
        print(results)  # (9, 27)

        # Example 2: Simulate error
        results = F.bcast_gather([square, fail, cube], 2)
        print(results)  # (4, TaskError(...), 8)

        # Example 3: Timeout
        results = F.bcast_gather([square, cube], 4, timeout=0.01)
        print(results) # (16, 64)
    """
    if not to_send or not all(isinstance(f, Callable) for f in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    executor = Executor.get_instance()
    futures = [executor.submit(f, *args, **kwargs) for f in to_send]

    concurrent.futures.wait(futures, timeout=timeout)
    responses: List[Any] = []
    for i, future in enumerate(futures):
        try:
            responses.append(future.result())
        except Exception as e:
            logger.error(str(e))
            responses.append(TaskError(exception=e, index=i))
    return tuple(responses)


@Spans.instrument()
def wait_for(
    to_send: Callable, *args: Any, timeout: Optional[float] = None, **kwargs: Any
) -> Any:
    """Wait for a callable execution.

    Args:
        to_send:
            A callable object (e.g. functions or `Module` instances).
        *args:
            Positional arguments.
        timeout:
            Maximum time (in seconds) to wait for responses.
        **kwargs:
            Named arguments.

    Returns:
        Callable responses.

    Raises:
        TypeError:
            If `to_send` is not a callable.

    Examples:
        async def f1(x):
            return x * x

        # Example 1:
        results = F.wait_for(f1, 3)
        print(results) # 9
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    executor = Executor.get_instance()
    future = executor.submit(to_send, *args, **kwargs)
    concurrent.futures.wait([future], timeout=timeout)
    try:
        return future.result()
    except Exception as e:
        logger.error(str(e))
        return TaskError(exception=e, index=0)


@Spans.instrument()
def wait_for_event(event: Any) -> None:
    """Waits synchronously for an event to be set.

    This function will block until event.set() is called elsewhere.

    Args:
        event: The event to wait for.

    Raises:
        TypeError: If `event` is not an instance of asyncio.Event or threading.Event.
    """
    if isinstance(event, threading.Event):
        event.wait()
        return

    if isinstance(event, asyncio.Event):
        executor = Executor.get_instance()
        future = executor._submit_to_async_worker(event.wait())
        try:
            future.result()
        except Exception as e:
            logger.error(str(e))
        return

    raise TypeError("`event` must be an instance of asyncio.Event or threading.Event")


@Spans.instrument()
def spawn(to_send: Callable, *args: Any, **kwargs: Any) -> None:
    """Dispatches a task without waiting for a result.
    Uses the AsyncExecutorPool. The task is not tracked and no return is provided.

    Args:
        to_send:
            Callable object (function, async function, or module with .acall() method).
        *args:
            Positional arguments.
        **kwargs:
            Named arguments.

    Raises:
        TypeError: If `to_send` is not a callable.

    Examples:
        # Example 1:
        import time
        def print_message(message: str):
            time.sleep(1)
            print(f"[Sync] Message: {message}")
        F.spawn(print_message, "Hello from sync function")

        # Example 2:
        import asyncio
        async def async_print_message(message: str):
            await asyncio.sleep(1)
            print(f"[Async] Message: {message}")
        F.spawn(async_print_message, "Hello from async function")

        # Example 3 (with error):
        def failing_task():
            raise ValueError("This task failed!")
        F.spawn(failing_task)  # Error will be logged
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    def log_future(future: Future) -> None:
        """Callback to log exception of a Future."""
        try:
            future.result()
        except Exception as e:
            logger.error(f"Spawned task error: {e!s}", exc_info=True)

    executor = Executor.get_instance()
    future = executor.submit(to_send, *args, **kwargs)
    future.add_done_callback(log_future)


@Spans.ainstrument()
async def aspawn(to_send: Callable, *args, **kwargs) -> None:
    """Dispatches an async task without waiting for a result.
    The task is not tracked and no return is provided.

    Args:
        to_send:
            Callable object (async function or module with .acall() method).
        *args:
            Positional arguments.
        **kwargs:
            Named arguments.

    Raises:
        TypeError: If `to_send` is not a callable.

    Examples:
        # Example 1:
        import asyncio
        async def async_print_message(message: str):
            await asyncio.sleep(1)
            print(f"[Async] Message: {message}")
        await F.aspawn(async_print_message, "Hello from async function")

        # Example 2 (with error):
        async def failing_task():
            raise ValueError("This task failed!")
        await F.aspawn(failing_task)  # Error will be logged
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    async def run_task():
        """Wrapper to run the task and log errors."""
        try:
            if hasattr(to_send, "acall"):
                await to_send.acall(*args, **kwargs)
            elif asyncio.iscoroutinefunction(to_send):
                await to_send(*args, **kwargs)
            else:
                # Fall back to running sync function in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: to_send(*args, **kwargs))
        except Exception as e:
            logger.error(f"Fire-and-forget task error: {e!s}", exc_info=True)

    asyncio.create_task(run_task())  # noqa: RUF006


@Spans.ainstrument()
async def await_for_event(event: Any) -> None:
    """Waits asynchronously for an event to be set.

    This function will await until event.set() is called elsewhere.

    Args:
        event: The event to wait for.

    Raises:
        TypeError: If `event` is not an instance of asyncio.Event or threading.Event.

    Examples:
        # Example 1:
        import asyncio
        event = asyncio.Event()

        async def setter():
            await asyncio.sleep(1)
            event.set()

        asyncio.create_task(setter())
        await F.await_for_event(event)
        print("Event was set!")
    """
    if isinstance(event, asyncio.Event):
        await event.wait()
        return

    if isinstance(event, threading.Event):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, event.wait)
        return

    raise TypeError("`event` must be an instance of asyncio.Event or threading.Event")


@Spans.ainstrument()
async def amap_gather(
    to_send: Callable,
    *,
    args_list: List[Tuple[Any, ...]],
    kwargs_list: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Any, ...]:
    """Async version of map_gather. Applies the `to_send` async function to each
    set of arguments in `args_list` and `kwargs_list` and collects the results.

    Args:
        to_send:
            The async callable function to be applied.
        args_list:
            Each tuple contains the positional arguments for the corresponding callable
            in `to_send`. If `None`, no positional arguments are passed unless specified
            individually by an item in `kwargs_list`.
        kwargs_list:
            Each dictionary contains the named arguments for the corresponding callable
            in `to_send`. If `None`, no named arguments are passed unless specified
            individually by an item in `args_list`.

    Returns:
        A tuple containing the results of each call to the `to_send` function.

    Raises:
        TypeError:
            If `to_send` is not callable.
        ValueError:
            If `args_list` is not a non-empty list or if `kwargs_list`
            (if provided) is not the same length as `args_list`.
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    if not isinstance(args_list, list) or len(args_list) == 0:
        raise ValueError("`args_list` must be a non-empty list")

    if kwargs_list is not None:
        if not isinstance(kwargs_list, list) or len(kwargs_list) != len(args_list):
            raise ValueError(
                "`kwargs_list` must be a list with the same length as `args_list`"
            )

    tasks = []
    for i in range(len(args_list)):
        args = args_list[i]
        kwargs = kwargs_list[i] if kwargs_list else {}
        tasks.append(to_send(*args, **kwargs))

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            logger.error(str(response))
            results.append(TaskError(exception=response, index=i))
        else:
            results.append(response)

    return tuple(results)


@Spans.ainstrument()
async def ascatter_gather(
    to_send: List[Callable],
    args_list: Optional[List[Tuple[Any, ...]]] = None,
    kwargs_list: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Any, ...]:
    """Async version of scatter_gather. Sends different sets of arguments/kwargs
    to a list of async callables and collects the responses.

    Each callable in `to_send` receives the positional arguments of
    the corresponding `tuple` in `args_list` and the named arguments
    of the corresponding `dict` in `kwargs_list`. If `args_list` or
    `kwargs_list` are not provided (or are `None`), the corresponding
    callables will be called without positional or named arguments,
    respectively, unless an empty list (`[]`) or empty tuple (`()`)
    is provided for a specific item.

    Args:
        to_send:
            List of callable objects (e.g. async functions or `Module` instances
            with acall).
        args_list:
            Each tuple contains the positional arguments for the corresponding callable
            in `to_send`. If `None`, no positional arguments are passed unless specified
            individually by an item in `kwargs_list`.
        kwargs_list:
            Each dictionary contains the named arguments for the corresponding callable
            in `to_send`. If `None`, no named arguments are passed unless specified
            individually by an item in `args_list`.

    Returns:
        Tuple containing the responses for each callable. If an error occurs for a
        specific callable, its corresponding response in the tuple will be `None`.

    Raises:
        TypeError:
            If `to_send` is not a callable list.
        ValueError:
            If the lengths of `args_list` (if provided) or `kwargs_list`
            (if provided) do not match the length of `to_send`.
    """
    if not isinstance(to_send, list) or not all(callable(f) for f in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    tasks = []
    for i, f in enumerate(to_send):
        args = args_list[i] if args_list and i < len(args_list) else ()
        kwargs = kwargs_list[i] if kwargs_list and i < len(kwargs_list) else {}
        tasks.append(f(*args, **kwargs))

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            logger.error(str(response))
            results.append(TaskError(exception=response, index=i))
        else:
            results.append(response)

    return tuple(results)


@Spans.instrument()
async def abcast_gather(
    to_send: List[Callable], *args: Any, **kwargs: Any
) -> Tuple[Any, ...]:
    """Async version of bcast_gather. Broadcasts the same arguments to multiple
    async callables and gathers the responses concurrently.

    Args:
        to_send:
            List of callable objects (e.g. async functions or `Module` instances
            with acall).
        *args:
            Positional arguments broadcast to every callable.
        **kwargs:
            Named arguments broadcast to every callable.

    Returns:
        Tuple containing the responses for each callable. If an error occurs for a
        specific callable, its corresponding response in the tuple will be a
        `TaskError`.

    Raises:
        TypeError:
            If `to_send` is not a list of callables.

    Examples:
        async def square(x): return x * x
        async def cube(x): return x * x * x

        # Example 1:
        results = await F.abcast_gather([square, cube], 3)
        print(results)  # (9, 27)
    """
    if not isinstance(to_send, list) or not all(callable(f) for f in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    tasks = [f(*args, **kwargs) for f in to_send]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            logger.error(str(response))
            results.append(TaskError(exception=response, index=i))
        else:
            results.append(response)

    return tuple(results)
