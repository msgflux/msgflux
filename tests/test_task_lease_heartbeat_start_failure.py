from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_heartbeat_retries_after_thread_start_failure_in_isolated_process():
    script = """
import threading
import time

from msgflux.runtime.task_leases import TaskLeaseHeartbeats
from msgflux.tasks import InMemoryTaskStore, TaskHandle

store = InMemoryTaskStore()
store.create("worker", task_id="probe")
handle = TaskHandle("probe", store)
handle.start_worker(lease_seconds=10)
renewed = threading.Event()
original_renew = store.renew_worker

def observe_renew(*args, **kwargs):
    result = original_renew(*args, **kwargs)
    if result:
        renewed.set()
    return result

store.renew_worker = observe_renew
original_start = threading.Thread.start
start_error = RuntimeError("simulated thread-start failure")
failed = False

def fail_once(thread):
    global failed
    if not failed and thread.name == "msgflux-task-lease-heartbeat":
        failed = True
        raise start_error
    return original_start(thread)

threading.Thread.start = fail_once
try:
    try:
        TaskLeaseHeartbeats.register(handle, lease_seconds=0.3)
    except RuntimeError as error:
        assert error is start_error
    else:
        raise AssertionError("first registration should fail")
finally:
    threading.Thread.start = original_start

assert TaskLeaseHeartbeats._thread is None
assert id(handle) not in TaskLeaseHeartbeats._active
TaskLeaseHeartbeats.register(handle, lease_seconds=0.3)
try:
    assert renewed.wait(timeout=2), "retry did not renew the worker lease"
finally:
    TaskLeaseHeartbeats.unregister(handle)
    deadline = time.monotonic() + 2
    while TaskLeaseHeartbeats._thread is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert TaskLeaseHeartbeats._thread is None, "heartbeat thread did not stop"
"""
    result = subprocess.run(  # noqa: S603 - fixed interpreter and local fixture
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr
