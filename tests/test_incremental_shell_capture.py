"""Deterministic coverage for incremental shell output capture."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
import msgspec

from msgflux.nn import ToolLibrary
from msgflux.nn.extensions import ToolOutputOffloadExtension
from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    LocalToolResultStore,
    PermissionSet,
    ProcessExecutor,
    ProcessResult,
    ProcessRequest,
    SandboxCapabilities,
    execution_context,
)
from msgflux.runtime.shell_capture import ShellOutputCapture
from msgflux.runtime.tool_results import ToolResultStore
from msgflux.tools.builtin import BashTool
from msgflux.tools.shell import ShellResult


class StreamingExecutor(ProcessExecutor):
    def __init__(self, streams, *, error=None):
        self.streams = streams
        self.error = error
        self.calls = 0

    @property
    def capabilities(self):
        return SandboxCapabilities(
            {"filesystem", "network", "process", "resource_limits"}
        )

    def supports_workspace(self, filesystem):
        return isinstance(filesystem, InMemoryWorkspace)

    async def execute(self, **kwargs):
        raise AssertionError("capture must use execute_stream")

    async def execute_stream(self, request, *, on_output, **kwargs):
        del request, kwargs
        self.calls += 1
        for channel, chunks in self.streams:
            for chunk in chunks:
                for offset in range(0, len(chunk), 32_000):
                    await on_output(channel, chunk[offset : offset + 32_000])
        if self.error is not None:
            raise self.error
        return ProcessResult(0)


class FailingStore(ToolResultStore):
    def put(self, chunks, *, media_type="application/octet-stream"):
        for _ in chunks:
            pass
        raise RuntimeError("sink failed")

    def get(self, result_id):
        raise KeyError(result_id)

    def iter_bytes(self, reference, *, offset=0, limit=None, chunk_size=65536):
        return iter(())


def _environment(executor):
    return ExecutionEnvironment(InMemoryWorkspace("capture"), executor)


async def _capture(capture, environment, requests):
    scope = ExecutionScope(
        namespace="capture",
        thread_id="capture",
        run_id="run",
        environment=environment,
        permissions=PermissionSet(["process.execute"]),
    )
    with execution_context(scope=scope):
        return await capture.run(environment, requests)


def _request(name="echo"):
    return ProcessRequest(("bash", "-c", name), max_output_bytes=2_000_000)


@pytest.mark.asyncio
async def test_incremental_capture_handles_large_split_utf8_and_recovers_full_result(
    tmp_path: Path,
):
    store = LocalToolResultStore(tmp_path / "results")
    capture = ShellOutputCapture(
        store, max_inline_bytes=128, preview_bytes=24, max_capture_bytes=1_200_000
    )
    utf8 = "á🌍".encode()
    executor = StreamingExecutor(
        [
            ("stdout", [b"A" * 700_000, utf8[:2], utf8[2:], b"B" * 400_000]),
            ("stderr", [b"ERR\n" * 20]),
        ]
    )
    result = await _capture(capture, _environment(executor), [_request()])
    assert isinstance(result, ShellResult)
    assert result.output_reference is not None
    store.verify(result.output_reference)
    data = store.read(result.output_reference, limit=1_200_000)
    assert b"A" * 1000 in data and b"B" * 1000 in data
    assert "á🌍" in data.decode()
    assert len(result.results[0].stdout.encode()) <= 24
    assert len(result.results[0].stderr.encode()) <= 24


@pytest.mark.asyncio
async def test_timeout_keeps_partial_stream_and_marks_command_timed_out(tmp_path):
    store = LocalToolResultStore(tmp_path / "results")
    capture = ShellOutputCapture(
        store, max_inline_bytes=64, preview_bytes=16, max_capture_bytes=1000
    )
    executor = StreamingExecutor(
        [("stdout", [b"partial output"]), ("stderr", [b"warning"])],
        error=asyncio.TimeoutError(),
    )
    result = await _capture(capture, _environment(executor), [_request()])
    assert result.results[0].status == "timed_out"
    assert result.output_reference is not None
    stored = store.read(result.output_reference)
    assert b"partial output" in stored and b"warning" in stored


@pytest.mark.asyncio
async def test_capture_quota_fails_closed_and_storage_failure_does_not_return_reference(
    tmp_path: Path,
):
    quota_capture = ShellOutputCapture(
        LocalToolResultStore(tmp_path / "capture-quota"),
        max_inline_bytes=32,
        preview_bytes=8,
        max_capture_bytes=64,
    )
    with pytest.raises(RuntimeError, match="output limit"):
        await _capture(
            quota_capture,
            _environment(StreamingExecutor([("stdout", [b"x" * 128])])),
            [_request()],
        )
    for store in (
        LocalToolResultStore(tmp_path / "quota", max_result_bytes=10),
        FailingStore(),
    ):
        capture = ShellOutputCapture(
            store, max_inline_bytes=32, preview_bytes=8, max_capture_bytes=256
        )
        executor = StreamingExecutor([("stdout", [b"x" * 128])])
        with pytest.raises((RuntimeError, ValueError)):
            await _capture(capture, _environment(executor), [_request()])


@pytest.mark.asyncio
async def test_concurrent_capture_calls_have_independent_results_and_previews(tmp_path):
    store = LocalToolResultStore(tmp_path / "results")
    capture = ShellOutputCapture(
        store, max_inline_bytes=32, preview_bytes=12, max_capture_bytes=1000
    )
    first = StreamingExecutor([("stdout", [b"first" * 100])])
    second = StreamingExecutor([("stdout", [b"second" * 100])])
    one, two = await asyncio.gather(
        _capture(capture, _environment(first), [_request("one")]),
        _capture(capture, _environment(second), [_request("two")]),
    )
    assert one.output_reference != two.output_reference
    assert b"first" in store.read(one.output_reference)
    assert b"second" in store.read(two.output_reference)
    assert one.results[0].stdout != two.results[0].stdout


def test_extension_removal_restores_optional_capture_binding_and_hidden_schema(
    tmp_path,
):
    extension = ToolOutputOffloadExtension(LocalToolResultStore(tmp_path))
    library = ToolLibrary("shell", [BashTool()], extensions=[extension])
    definition = library.get_tool_definition("bash")
    assert "shell_capture" not in definition.input_schema.get("properties", {})
    assert "shell_capture" not in definition.annotations
    assert any(
        binding.source == "shell_capture" and not binding.required
        for binding in definition.context.bindings
    )
    library.remove_extension(extension.name)
    assert library.get_tool_definition("bash").input_schema == definition.input_schema


@pytest.mark.asyncio
async def test_batch_uses_two_spools_and_keeps_command_boundaries(
    tmp_path, monkeypatch
):
    import msgflux.runtime.shell_capture as capture_module

    files = []
    temporary_file = capture_module.TemporaryFile

    def track_file(*args, **kwargs):
        stream = temporary_file(*args, **kwargs)
        files.append(stream)
        return stream

    monkeypatch.setattr(capture_module, "TemporaryFile", track_file)
    store = LocalToolResultStore(tmp_path)
    capture = ShellOutputCapture(
        store, max_inline_bytes=128, preview_bytes=32, max_capture_bytes=1000
    )
    executor = StreamingExecutor([("stdout", [b"abc\xf0"]), ("stderr", [b"warning"])])
    result = await _capture(capture, _environment(executor), [_request()] * 50)
    assert len(files) == 2
    assert all(stream.closed for stream in files)
    restored = msgspec.json.decode(b"".join(store.iter_bytes(result.output_reference)))
    assert len(restored["results"]) == 50
    assert all(item["stdout"] == "abc�" for item in restored["results"])
    assert all(item["stderr"] == "warning" for item in restored["results"])


@pytest.mark.asyncio
async def test_cancellation_joins_publication_worker_before_closing_spools(
    tmp_path, monkeypatch
):
    import msgflux.runtime.shell_capture as capture_module

    started, release = threading.Event(), threading.Event()
    files = []
    temporary_file = capture_module.TemporaryFile

    def track_file(*args, **kwargs):
        stream = temporary_file(*args, **kwargs)
        files.append(stream)
        return stream

    class SlowStore(LocalToolResultStore):
        def put(self, chunks, *, media_type="application/octet-stream"):
            started.set()
            if not release.wait(5):
                raise TimeoutError("test worker was not released")
            assert all(not stream.closed for stream in files)
            return super().put(chunks, media_type=media_type)

    monkeypatch.setattr(capture_module, "TemporaryFile", track_file)
    store = SlowStore(tmp_path)
    capture = ShellOutputCapture(
        store, max_inline_bytes=32, preview_bytes=8, max_capture_bytes=1000
    )
    task = asyncio.create_task(
        _capture(
            capture,
            _environment(StreamingExecutor([("stdout", [b"x" * 200])])),
            [_request()],
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert all(stream.closed for stream in files)
        entries = list(tmp_path.iterdir())
        assert len(entries) == 1
        store.verify(store.get(entries[0].name))
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
