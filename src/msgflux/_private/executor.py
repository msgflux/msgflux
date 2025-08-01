import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import uvloop

from msgflux.envs import envs


class AsyncWorker:
    def __init__(self):
        """Initializes a worker with its own event loop in a separate thread."""
        self.loop = uvloop.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever)
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
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)
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
        if hasattr(f, "acall"):
            coro = f.acall(*args, **kwargs)
            return self._submit_to_async_worker(coro)
        elif asyncio.iscoroutinefunction(f):
            coro = f(*args, **kwargs)
            return self._submit_to_async_worker(coro)
        else:
            return self.thread_pool.submit(f, *args, **kwargs)

    def _submit_to_async_worker(self, coro):
        """Distribute a coroutine to an asynchronous worker using round-robin."""
        worker = self.async_workers[self.async_worker_index]
        self.async_worker_index = (self.async_worker_index + 1) % self.num_async_workers
        return worker.submit(coro)

    def shutdown(self):
        """Shutdown the executor, closing the pools."""
        self.thread_pool.shutdown()
        for worker in self.async_workers:
            worker.shutdown()

    def __del__(self):
        self.shutdown()
