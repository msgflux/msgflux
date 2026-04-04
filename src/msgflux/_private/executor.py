import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Tuple

try:
    import platform

    import uvloop

    if platform.python_implementation() != "PyPy":
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

from msgflux.envs import envs


class AsyncWorker:
    def __init__(self):
        """Initializes a worker with its own event loop in a separate thread."""
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def submit(self, coro):
        """Submits a coroutine to the worker's event loop."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def shutdown(self):
        """Terminates the event loop and worker thread."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()


class Executor:
    """Async pool to manage the execution of synchronous and asynchronous code.
    Executor distributes tasks among a ThreadPool or among Async Workers.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.num_threads = envs.executor_num_threads
        self.num_async_workers = envs.executor_num_async_workers
        self.async_worker_index = 0
        self._thread_local = threading.local()
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.num_threads,
            initializer=self._mark_thread_pool_worker,
        )
        self.async_workers = [AsyncWorker() for _ in range(self.num_async_workers)]

    @classmethod
    def get_instance(cls):
        """Returns the singleton instance in a thread-safe manner."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def submit(self, f: Callable, *args, **kwargs):
        """Submits a task to the appropriate pool based on the function type.
        Returns a Future to track the result.
        """
        mode, target = self._resolve_submission_target(f, args, kwargs)
        if mode == "async":
            return self._submit_to_async_worker(target)

        sync_callable, sync_args, sync_kwargs = target
        if self._is_in_thread_pool_worker():
            return self._execute_inline(sync_callable, *sync_args, **sync_kwargs)

        return self.thread_pool.submit(sync_callable, *sync_args, **sync_kwargs)

    def _submit_to_async_worker(self, coro):
        """Distribute a coroutine to an asynchronous worker using round-robin."""
        worker = self.async_workers[self.async_worker_index]
        self.async_worker_index = (self.async_worker_index + 1) % self.num_async_workers
        return worker.submit(coro)

    def _resolve_submission_target(
        self,
        f: Callable,
        args: Tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Tuple[str, Any]:
        """Resolve wrappers like functools.partial before deciding execution mode.

        This preserves `.acall()` routing for partially-bound Modules/Tools and
        avoids sending nested sync work back into the same saturated pool.
        """
        if isinstance(f, partial):
            merged_kwargs = dict(f.keywords or {})
            merged_kwargs.update(kwargs)
            merged_args = (*f.args, *args)
            return self._resolve_submission_target(f.func, merged_args, merged_kwargs)

        if hasattr(f, "acall"):
            return ("async", f.acall(*args, **kwargs))

        if asyncio.iscoroutinefunction(f):
            return ("async", f(*args, **kwargs))

        return ("sync", (f, args, kwargs))

    def _mark_thread_pool_worker(self) -> None:
        self._thread_local.in_thread_pool_worker = True

    def _is_in_thread_pool_worker(self) -> bool:
        return bool(getattr(self._thread_local, "in_thread_pool_worker", False))

    def _execute_inline(self, f: Callable, *args, **kwargs) -> Future:
        """Execute nested sync work inline and wrap it in a completed Future."""
        future: Future = Future()
        try:
            result = f(*args, **kwargs)
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)
        return future

    def shutdown(self):
        """Shutdown the executor, closing the pools."""
        self.thread_pool.shutdown()
        for worker in self.async_workers:
            worker.shutdown()

    def __del__(self):
        self.shutdown()
