from unittest.mock import Mock

from msgflux._private import executor as executor_module
from msgflux._private.executor import AsyncWorker, Executor


def test_executor_shutdown_uses_non_blocking_pool_shutdown_during_finalization(
    monkeypatch,
):
    monkeypatch.setattr(executor_module.sys, "is_finalizing", lambda: True)

    executor = Executor.__new__(Executor)
    executor._shutdown = False
    executor._shutdown_lock = executor_module.threading.Lock()
    executor.thread_pool = Mock()
    executor.async_workers = [Mock()]

    executor.shutdown()

    executor.thread_pool.shutdown.assert_called_once_with(wait=False)
    executor.async_workers[0].shutdown.assert_called_once_with()


def test_executor_shutdown_is_idempotent(monkeypatch):
    monkeypatch.setattr(executor_module.sys, "is_finalizing", lambda: False)

    executor = Executor.__new__(Executor)
    executor._shutdown = False
    executor._shutdown_lock = executor_module.threading.Lock()
    executor.thread_pool = Mock()
    executor.async_workers = [Mock()]

    executor.shutdown()
    executor.shutdown()

    executor.thread_pool.shutdown.assert_called_once_with(wait=True)
    executor.async_workers[0].shutdown.assert_called_once_with()


def test_executor_del_suppresses_shutdown_errors_during_finalization(monkeypatch):
    monkeypatch.setattr(executor_module.sys, "is_finalizing", lambda: True)

    executor = Executor.__new__(Executor)
    executor.shutdown = Mock(side_effect=RuntimeError("cannot join thread"))

    executor.__del__()

    executor.shutdown.assert_called_once_with()


def test_executor_del_suppresses_shutdown_errors_outside_finalization(monkeypatch):
    monkeypatch.setattr(executor_module.sys, "is_finalizing", lambda: False)

    executor = Executor.__new__(Executor)
    executor.shutdown = Mock(side_effect=RuntimeError("shutdown failed"))

    executor.__del__()

    executor.shutdown.assert_called_once_with()


def test_async_worker_shutdown_skips_join_and_close_during_finalization(monkeypatch):
    monkeypatch.setattr(executor_module.sys, "is_finalizing", lambda: True)

    worker = AsyncWorker.__new__(AsyncWorker)
    worker._shutdown = False
    worker.loop = Mock()
    worker.loop.is_closed.return_value = False
    worker.thread = Mock()
    worker.thread.is_alive.return_value = True

    worker.shutdown()

    worker.loop.call_soon_threadsafe.assert_called_once_with(worker.loop.stop)
    worker.thread.join.assert_not_called()
    worker.loop.close.assert_not_called()
