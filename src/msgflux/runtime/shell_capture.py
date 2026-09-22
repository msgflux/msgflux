"""Per-call disk capture for shell batches, independent of process backends."""

from __future__ import annotations

import asyncio
import codecs
from contextlib import ExitStack
from dataclasses import replace
from itertools import chain
from tempfile import TemporaryFile

import msgspec

from msgflux.runtime.tool_results import ToolResultStore
from msgflux.tools.shell import ShellCommandResult, ShellResult


async def _disk_call(fn, *args):
    """Join an in-flight disk operation before its owner closes the files."""
    task = asyncio.create_task(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Repeated cancellation must not let a worker access closed files.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if not task.cancelled():
            task.exception()
        raise


def _text_chunks(captured):
    stream, offset, remaining = captured
    stream.seek(offset)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while remaining:
        chunk = stream.read(min(16384, remaining))
        if not chunk:
            raise OSError("Temporary process capture ended unexpectedly")
        remaining -= len(chunk)
        yield decoder.decode(chunk)
    yield decoder.decode(b"", final=True)


def _batch_chunks(parts):
    yield b'{"results":['
    for index, (status, returncode, stdout, stderr) in enumerate(parts):
        if index:
            yield b","
        yield b'{"status":' + msgspec.json.encode(status)
        yield b',"returncode":' + msgspec.json.encode(returncode)
        for name, stream in (("stdout", stdout), ("stderr", stderr)):
            yield b',"' + name.encode() + b'":"'
            for text in _text_chunks(stream):
                yield msgspec.json.encode(text)[1:-1]
            yield b'"'
        yield b"}"
    yield b"]}"


class ShellOutputCapture:
    """Host policy used by the offload extension, not a sandbox or access grant.

    A fresh set of temporary files is owned by each ``run`` invocation. Executors
    deliver bytes through an awaited callback; no unbounded output queue is used.
    Legacy executors may still buffer internally before calling that callback.
    """

    def __init__(
        self,
        store: ToolResultStore,
        *,
        max_inline_bytes: int,
        preview_bytes: int,
        max_capture_bytes: int,
    ):
        self.store = store
        self.max_inline_bytes = max_inline_bytes
        self.preview_bytes = preview_bytes
        self.max_capture_bytes = max_capture_bytes

    def _finish(self, parts):
        chunks = _batch_chunks(parts)
        prefix = bytearray()
        try:
            for chunk in chunks:
                if len(prefix) + len(chunk) > self.max_inline_bytes:
                    ref = self.store.put(
                        chain((bytes(prefix), chunk), chunks),
                        media_type="application/json",
                    )
                    break
                prefix.extend(chunk)
            else:
                return msgspec.json.decode(prefix, type=ShellResult)
        finally:
            chunks.close()
        per_field = self.preview_bytes // (2 * len(parts))

        def preview(captured):
            stream, offset, size = captured
            stream.seek(offset)
            # Replacement decoding can expand invalid input; bound encoded text.
            return (
                stream.read(min(per_field, size))
                .decode("utf-8", errors="replace")
                .encode("utf-8")[:per_field]
                .decode("utf-8", errors="ignore")
            )

        return ShellResult(
            results=tuple(
                ShellCommandResult(
                    status=status,
                    returncode=code,
                    stdout=preview(stdout),
                    stderr=preview(stderr),
                )
                for status, code, stdout, stderr in parts
            ),
            output_reference=ref,
        )

    async def run(self, environment, requests):
        remaining = self.max_capture_bytes
        parts = []
        with ExitStack() as stack:
            stdout = stack.enter_context(TemporaryFile(mode="w+b"))
            stderr = stack.enter_context(TemporaryFile(mode="w+b"))
            for prepared in requests:
                out_start, err_start = stdout.tell(), stderr.tell()
                if remaining <= 0:
                    stderr.write(b"Batch output limit exhausted; command not executed.")
                    parts.append(
                        (
                            "not_executed",
                            None,
                            (stdout, out_start, 0),
                            (stderr, err_start, stderr.tell() - err_start),
                        )
                    )
                    continue

                async def consume(channel, data, stdout=stdout, stderr=stderr):
                    nonlocal remaining
                    if len(data) > remaining:
                        raise RuntimeError("Process executor violated its output limit")
                    remaining -= len(data)
                    await _disk_call(
                        (stdout if channel == "stdout" else stderr).write, data
                    )

                request = replace(prepared, max_output_bytes=remaining)
                try:
                    result = await environment.arun(request, on_output=consume)
                except asyncio.TimeoutError:
                    status, code = "timed_out", None
                else:
                    status, code = "exited", result.returncode
                parts.append(
                    (
                        status,
                        code,
                        (stdout, out_start, stdout.tell() - out_start),
                        (stderr, err_start, stderr.tell() - err_start),
                    )
                )
            return await _disk_call(self._finish, parts)
