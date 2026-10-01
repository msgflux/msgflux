"""Bounded output draining for already-launched subprocesses.

This helper does not launch processes and makes no sandbox or workspace claims;
the caller remains responsible for choosing and authorizing the process.
"""

from __future__ import annotations

import asyncio
import math
import os
import signal
from collections.abc import Awaitable, Callable
from typing import Literal

from msgflux.runtime.abort import AbortSignal, await_with_abort


class ProcessOutputLimitError(RuntimeError):
    """The combined process output exceeded the requested byte budget."""


async def drain_subprocess(  # noqa: C901
    process,
    on_output: Callable[[Literal["stdout", "stderr"], bytes], Awaitable[None]],
    *,
    max_output_bytes: int,
    timeout_seconds: float,
    abort_signal: AbortSignal | None = None,
    chunk_size: int = 65536,
    owns_process_group: bool = False,
) -> int:
    """Drain both pipes concurrently and return the process return code.

    ``process`` must be an asyncio subprocess that has already been started
    with pipes.  On cancellation, timeout, callback failure, or output-limit
    failure the process is terminated and reaped before the error is re-raised.
    On POSIX, set ``owns_process_group=True`` only for a process launched with
    ``start_new_session=True``. This also terminates descendants in that group.
    Otherwise the executor must manage descendants itself. Neither option is a
    sandbox: descendants may escape a process group without OS-level isolation.
    """
    if not callable(on_output):
        raise TypeError("on_output must be callable")
    if type(max_output_bytes) is not int or max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be a positive integer")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive")
    if chunk_size > 65536:
        raise ValueError("chunk_size must not exceed 65536")
    if process.stdout is None or process.stderr is None:
        raise ValueError("process must expose stdout and stderr pipes")
    if type(owns_process_group) is not bool:
        raise TypeError("owns_process_group must be a boolean")
    if owns_process_group:
        if os.name != "posix":
            raise NotImplementedError("Owned process groups require POSIX")
        try:
            group = os.getpgid(process.pid)
        except ProcessLookupError:
            group = None
        if group is not None and (group != process.pid or group == os.getpgrp()):
            raise ValueError("Process must own a separate process group")

    total = 0
    total_lock = asyncio.Lock()

    async def read_pipe(channel, stream):
        nonlocal total
        while True:
            data = await stream.read(chunk_size)
            if not data:
                return
            async with total_lock:
                total += len(data)
                if total > max_output_bytes:
                    raise ProcessOutputLimitError(
                        "Process output exceeded its byte limit"
                    )
            await on_output(channel, data)

    tasks = (
        asyncio.create_task(read_pipe("stdout", process.stdout)),
        asyncio.create_task(read_pipe("stderr", process.stderr)),
        asyncio.create_task(process.wait()),
    )
    completion = asyncio.gather(*tasks)

    async def run():
        await asyncio.shield(completion)
        return process.returncode

    async def cleanup():
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Retrieve an exception from the aggregate even if abort won the race.
        if completion.done() and not completion.cancelled():
            completion.exception()
        await _cleanup_process(process, owns_process_group=owns_process_group)

    try:
        return await asyncio.wait_for(
            await_with_abort(run(), abort_signal), timeout_seconds
        )
    except BaseException as exc:
        cleanup_task = asyncio.create_task(cleanup())
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        try:
            cleanup_task.result()
        except BaseException as cleanup_error:
            exc.add_note(f"Process cleanup failed: {cleanup_error}")
        raise


async def _cleanup_process(process, *, owns_process_group):  # noqa: C901
    def send(force):
        try:
            if owns_process_group:
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            elif process.returncode is None:
                process.kill() if force else process.terminate()
        except ProcessLookupError:
            pass

    send(False)

    async def discard(stream):
        if stream is None:
            return
        try:
            while await stream.read(65536):
                pass
        except OSError:
            pass

    drains = [
        asyncio.create_task(discard(process.stdout)),
        asyncio.create_task(discard(process.stderr)),
        asyncio.create_task(process.wait()),
    ]
    completion = asyncio.gather(*drains)
    try:
        try:
            await asyncio.wait_for(asyncio.shield(completion), timeout=1)
        except asyncio.TimeoutError:
            send(True)
            await asyncio.wait_for(asyncio.shield(completion), timeout=2)
    finally:
        for task in drains:
            if not task.done():
                task.cancel()
        await asyncio.gather(*drains, return_exceptions=True)
        if completion.done() and not completion.cancelled():
            completion.exception()
