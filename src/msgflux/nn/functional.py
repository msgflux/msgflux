# https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/
import asyncio
import concurrent.futures
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional, Tuple

from msgflux._private.executor import Executor
from msgflux.dotdict import dotdict
from msgflux.logger import logger
from msgflux.nn.modules.module import get_callable_name
from msgflux.telemetry.span import instrument

__all__ = [
    "background_task",
    "bcast_gather",
    "map_gather",
    "msg_bcast_gather",
    "msg_scatter_gather",
    "scatter_gather",
    "wait_for",
    "wait_for_event",
]


@instrument("msgflux.nn.F.map_gather")
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
        fails or times out, the corresponding result will be `None`.

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
    for future in futures:
        try:
            responses.append(future.result())
        except Exception as e:
            logger.error(str(e))
            responses.append(None)
    return tuple(responses)


@instrument("msgflux.nn.F.scatter_gather")
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
        in the tuple will be `None`.

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
    for future in futures:
        try:
            responses.append(future.result())
        except Exception as e:
            logger.error(str(e))
            responses.append(None)
    return tuple(responses)


@instrument("msgflux.nn.F.msg_scatter_gather")
def msg_scatter_gather(
    to_send: List[Callable],
    messages: List[dotdict],
    *,
    timeout: Optional[float] = None,
) -> Tuple[dotdict, ...]:
    """Scatter a list of messages to a list of modules and gather the responses.

    Args:
        to_send:
            List of callable objects (e.g. functions or `Module` instances).
        messages:
            List of `msgflux.dotdict` instances to be distributed.
        timeout:
            Maximum time (in seconds) to wait for responses.

    Returns:
        Tuple containing the messages updated with the responses.

    Raises:
        TypeError:
            If `messages` is not a list of `dotdict`, `to_send` is not a list
            of callables, or `prefix` is not a string.
    """
    if not messages or not all(isinstance(msg, dotdict) for msg in messages):
        raise TypeError(
            "`messages` must be a non-empty list of `msgflux.dotdict` instances"
        )

    if not to_send or not all(isinstance(f, Callable) for f in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    if len(messages) != len(to_send):
        raise ValueError(
            f"The size of `messages` ({len(messages)}) "
            f"must be equal to that of `to_send`: ({len(to_send)})"
        )

    executor = Executor.get_instance()
    futures = [executor.submit(f, msg) for f, msg in zip(to_send, messages)]

    concurrent.futures.wait(futures, timeout=timeout)
    for f, future in zip(to_send, futures):
        f_name = get_callable_name(f)
        try:
            future.result()
        except Exception as e:
            logger.error(f"Error in scattered task for `{f_name}`: {e}")
    return tuple(messages)


@instrument("msgflux.nn.F.bcast_gather")
def bcast_gather(
    to_send: List[Callable], *args, timeout: Optional[float] = None, **kwargs
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
        print(results)  # (4, None, 8)

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
    for future in futures:
        try:
            responses.append(future.result())
        except Exception as e:
            logger.error(str(e))
            responses.append(None)
    return tuple(responses)


@instrument("msgflux.nn.F.msg_bcast_gather")
def msg_bcast_gather(
    to_send: List[Callable],
    message: dotdict,
    *,
    timeout: Optional[float] = None,
) -> dotdict:
    """Broadcasts a single message to multiple modules and gathers the responses.

    Args:
        to_send:
            List of callable objects (e.g. functions or `Module` instances).
        message:
            Instance of `msgflux.dotdict` to broadcast.
        timeout:
            Maximum time (in seconds) to wait for responses.

    Returns:
        The original message with the module responses added.

    Raises:
        TypeError:
            If `message` is not an instance of `dotdict`, `to_send` is not a list
            of callables.
    """
    if not isinstance(message, dotdict):
        raise TypeError("`message` must be an instance of `msgflux.dotdict`")
    if not to_send or not all(isinstance(module, Callable) for module in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    executor = Executor.get_instance()
    futures = [executor.submit(f, message) for f in to_send]

    concurrent.futures.wait(futures, timeout=timeout)
    for f, future in zip(to_send, futures):
        f_name = get_callable_name(f)
        try:
            future.result()
        except Exception as e:
            logger.error(f"Error in scattered task for `{f_name}`: {e}")
    return message


@instrument("msgflux.nn.F.wait_for")
def wait_for(
    to_send: Callable, *args, timeout: Optional[float] = None, **kwargs
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
        return None


@instrument("msgflux.nn.F.wait_for_event")
def wait_for_event(event: asyncio.Event) -> None:
    """Waits synchronously for an asyncio.Event to be set.

    This function will block until event.set() is called elsewhere.

    Args:
        event: The asyncio.Event to wait for.

    Raises:
        TypeError: If `event` is not an instance of asyncio.Event.
    """
    if not isinstance(event, asyncio.Event):
        raise TypeError("`event` must be an instance of asyncio.Event")

    executor = Executor.get_instance()
    future = executor._submit_to_async_worker(event.wait())
    try:
        future.result()
    except Exception as e:
        logger.error(str(e))


@instrument("msgflux.nn.F.background_task")
def background_task(to_send: Callable, *args, **kwargs) -> None:
    """Executes a task in the background asynchronously without blocking,
    using the AsyncExecutorPool. This function is "fire-and-forget".

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
        F.background_task(print_message, "Hello from sync function")

        # Example 2:
        import asyncio
        async def async_print_message(message: str):
            await asyncio.sleep(1)
            print(f"[Async] Message: {message}")
        F.background_task(async_print_message, "Hello from async function")

        # Example 3 (with error):
        def failing_task():
            raise ValueError("This task failed!")
        F.background_task(failing_task)  # Error will be logged
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    def log_future(future: Future) -> None:
        """Callback to log exception of a Future."""
        try:
            future.result()
        except Exception as e:
            logger.error(f"Background task error: {e!s}", exc_info=True)

    executor = Executor.get_instance()
    future = executor.submit(to_send, *args, **kwargs)
    future.add_done_callback(log_future)
