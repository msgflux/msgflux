"""Plug-in output policy before tool events and durable feedback serialization."""

import asyncio
import threading
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest

from msgflux.nn import ToolLibrary
from msgflux.nn import Agent
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.extensions import ToolOutputOffloadExtension
from msgflux.nn.extensions.tool_output import _json_chunks
from msgflux.nn.hooks import Hook
from msgflux.nn.hooks.events import AfterTool
from msgflux.runtime import LocalToolResultStore, RuntimeResources, ToolResultRef
from msgflux.runtime import ExecutionScope
from msgflux.tools.config import tool_config


def make_library(tmp_path, value, *, max_result_bytes=100000):
    def produce() -> Any:
        """Produce an example result."""
        return value

    store = LocalToolResultStore(tmp_path, max_result_bytes=max_result_bytes)
    extension = ToolOutputOffloadExtension(store, max_inline_bytes=64, preview_bytes=8)
    return ToolLibrary("test", [produce], extensions=[extension]), store


@pytest.mark.parametrize(
    "value",
    ["🌍" * 100, {"rows": ['olá\n"\\' * 100, True, None, 1.5]}, ["x" * 100, {}]],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_offload_roundtrip_preserves_text_or_json(tmp_path, value, asynchronous):
    library, store = make_library(tmp_path, value)
    calls = [("call_1", "produce", {})]
    response = await library.acall(calls) if asynchronous else library(calls)
    result = response.tool_calls[0].result
    assert result["type"] == "tool_result_reference"
    assert result["truncated"] is True
    assert len(result["preview"].encode()) <= 8
    ref = msgspec.convert(result["reference"], type=ToolResultRef)
    stored = b"".join(store.iter_bytes(ref))
    assert (
        stored.decode() if isinstance(value, str) else msgspec.json.decode(stored)
    ) == value
    assert ref.media_type == (
        "text/plain; charset=utf-8" if isinstance(value, str) else "application/json"
    )
    store.verify(ref)


@pytest.mark.parametrize(
    "value", ["", "x" * 64, {"ok": True}, [1, 2], b"binary", object()]
)
def test_small_and_unsupported_top_level_results_are_unchanged(tmp_path, value):
    extension = ToolOutputOffloadExtension(
        LocalToolResultStore(tmp_path), max_inline_bytes=64, preview_bytes=0
    )
    original = AfterTool(tool_call_id="call", tool_name="produce", result=value)
    assert extension.hooks()[0].handle(original) is original
    assert list(tmp_path.iterdir()) == []


def test_extension_removal_restores_normal_results(tmp_path):
    library, store = make_library(tmp_path, "x" * 100)
    library.remove_extension("tool_output_offload")
    assert library([("call_1", "produce", {})]).tool_calls[0].result == "x" * 100
    assert list(store.root.iterdir()) == []


def test_background_task_retains_the_offloaded_descriptor(tmp_path):
    @tool_config(allow_background=True)
    def produce() -> dict:
        """Produce a report in a background task."""
        return {"data": "x" * 10000}

    store = LocalToolResultStore(tmp_path)
    library = ToolLibrary(
        "background",
        [produce],
        extensions=[
            ToolOutputOffloadExtension(store, max_inline_bytes=512, preview_bytes=8)
        ],
    )
    dispatch = library([("call_1", "produce", {"run_in_background": True})])
    task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]
    response = library([("call_2", "task_wait", {"task_id": task_id, "timeout": 3.0})])
    result = response.tool_calls[0].result
    assert result["type"] == "tool_result_reference"
    ref = msgspec.convert(result["reference"], type=ToolResultRef)
    assert msgspec.json.decode(store.read(ref))["data"] == "x" * 10000
    assert len(list(tmp_path.iterdir())) == 1


def test_return_direct_keeps_feedback_and_offloads_result(tmp_path):
    @tool_config(return_direct=True)
    def produce() -> str:
        """Return a large answer directly."""
        return "x" * 10000

    store = LocalToolResultStore(tmp_path)
    library = ToolLibrary(
        "direct",
        [produce],
        extensions=[
            ToolOutputOffloadExtension(store, max_inline_bytes=512, preview_bytes=8)
        ],
    )
    response = library([("call_1", "produce", {})])
    assert response.return_directly
    assert response.tool_calls[0].result["type"] == "tool_result_reference"


@pytest.mark.asyncio
async def test_async_offload_uses_worker_and_cancellation_does_not_wait_for_disk(
    tmp_path,
):
    loop = asyncio.get_running_loop()
    started, finished = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    caller_thread = threading.get_ident()

    class BlockingStore(LocalToolResultStore):
        def put(self, chunks, *, media_type="application/octet-stream"):
            assert threading.get_ident() != caller_thread
            loop.call_soon_threadsafe(started.set)
            try:
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release storage")
                return super().put(chunks, media_type=media_type)
            finally:
                loop.call_soon_threadsafe(finished.set)

    def produce() -> str:
        """Produce a result whose storage will be paused."""
        return "x" * 10000

    store = BlockingStore(tmp_path)
    library = ToolLibrary(
        "cancel",
        [produce],
        extensions=[
            ToolOutputOffloadExtension(store, max_inline_bytes=512, preview_bytes=8)
        ],
    )
    task = asyncio.create_task(library.acall([("call_1", "produce", {})]))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert not finished.is_set()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if started.is_set():
            await asyncio.wait_for(finished.wait(), timeout=2)
    # The non-cancellable sync write may complete, but it never returns a
    # descriptor to the cancelled call. No rollback/automatic retry is implied.
    entries = list(tmp_path.iterdir())
    assert len(entries) == 1
    store.verify(store.get(entries[0].name))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_tool_end_contains_reference_or_bounded_error_not_original(
    tmp_path, failure
):
    library, store = make_library(
        tmp_path, "SECRET" * 1000, max_result_bytes=10 if failure else 100000
    )
    events = [
        event async for event in library.stream_events([("call_1", "produce", {})])
    ]
    end = next(event for event in events if event.type == "tool.end")
    assert "SECRETSECRETSECRET" not in str(end.data)
    if failure:
        assert end.data["result"] is None
        assert "do not retry automatically" in end.data["error"]
        assert any(event.type == "handler.error" for event in events)
        assert list(store.root.iterdir()) == []
    else:
        assert end.data["result"]["type"] == "tool_result_reference"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_later_transform_failure_cannot_restore_original_output(
    tmp_path, asynchronous
):
    library, _ = make_library(tmp_path, "x" * 1000)

    def fail(_outcome):
        raise RuntimeError("sensitive details must not leak")

    library.register_lifecycle_hook(
        "transform_tool_output", Hook(event="transform_tool_output", handler=fail)
    )
    calls = [("call_1", "produce", {})]
    response = await library.acall(calls) if asynchronous else library(calls)
    call = response.tool_calls[0]
    assert call.result is None
    assert "sensitive" not in str(call.error)
    assert "processing failed" in str(call.error)


def test_regular_after_tool_runs_before_offload(tmp_path):
    library, store = make_library(tmp_path, "small")
    library.register_lifecycle_hook(
        "after_tool",
        Hook(
            event="after_tool",
            handler=lambda outcome: replace(outcome, result={"data": "x" * 1000}),
        ),
    )
    result = library([("call_1", "produce", {})]).tool_calls[0].result
    ref = msgspec.convert(result["reference"], type=ToolResultRef)
    assert msgspec.json.decode(store.read(ref))["data"] == "x" * 1000


def test_json_encoding_handles_large_escaped_keys_and_strings_in_bounded_chunks():
    value = {'"🌍\n' * 20000: ["\\\x00é" * 20000, 2, False, None]}
    chunks = list(_json_chunks(value, set()))
    assert max(map(len, chunks)) <= 8192 * 6
    assert msgspec.json.decode(b"".join(chunks)) == value


@pytest.mark.parametrize("value", [{1: "bad key"}, {"bad": object()}, [float("nan")]])
def test_invalid_json_fails_closed(tmp_path, value):
    library, store = make_library(tmp_path, value)
    call = library([("call_1", "produce", {})]).tool_calls[0]
    assert call.result is None
    assert "processing failed" in str(call.error)
    assert list(store.root.iterdir()) == []


def test_cyclic_and_deep_json_fail_closed(tmp_path):
    cyclic = []
    cyclic.append(cyclic)
    deep = []
    for _ in range(70):
        deep = [deep]
    for value in (cyclic, deep):
        library, store = make_library(tmp_path, value)
        call = library([("call_1", "produce", {})]).tool_calls[0]
        assert call.result is None
        assert "processing failed" in str(call.error)
        assert list(store.root.iterdir()) == []


def test_offloaded_descriptor_roundtrips_in_sqlite(tmp_path):
    library, store = make_library(tmp_path / "tool-results", {"data": "x" * 1000})
    response = library([("call_1", "produce", {})])
    resources = RuntimeResources(tmp_path)
    checkpoints = resources.checkpoint_store("thd_test")
    try:
        checkpoints.save_state(
            "main",
            "thd_test",
            "run_1",
            {"tool_call_id": "call_1", "result": response.tool_calls[0].result},
        )
        restored = checkpoints.load_state("main", "thd_test", "run_1")["result"]
    finally:
        checkpoints.close()
    assert restored == response.tool_calls[0].result
    store.verify(msgspec.convert(restored["reference"], type=ToolResultRef))


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_real_agent_checkpoint_and_next_model_turn_use_descriptor(
    tmp_path, asynchronous
):
    def produce() -> dict:
        """Produce a large report."""
        return {"data": "x" * 10000}

    calls = ToolCallAggregator()
    calls.process(0, "call_1", "produce", "{}")
    first = ModelResponse()
    first.set_response_type("tool_call")
    first.add(calls)
    final = ModelResponse()
    final.set_response_type("text_generation")
    final.add("done")
    resources = RuntimeResources(tmp_path)
    results = resources.tool_result_store()
    checkpoints = resources.checkpoint_store("thd_test")
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="reporter", model=model, tools=[produce], checkpoint_store=checkpoints
    )
    extension = ToolOutputOffloadExtension(
        results, max_inline_bytes=64, preview_bytes=8
    )
    agent.tool_library.register_extension(extension.name, extension)
    generator = (
        AsyncMock(side_effect=[first, final])
        if asynchronous
        else Mock(side_effect=[first, final])
    )
    if asynchronous:
        agent.generator.aforward = generator
    else:
        agent.generator.forward = generator
    scope = ExecutionScope(namespace="reporter", thread_id="thd_test", run_id="run_1")
    try:
        if asynchronous:
            await agent.acall("report", scope=scope)
        else:
            agent("report", scope=scope)
        state = checkpoints.load_state("reporter", "thd_test", "run_1")
        assert state["status"] == "completed"
        encoded = msgspec.json.encode(state)
        assert b"tool_result_reference" in encoded
        assert b"x" * 10000 not in encoded
        messages = generator.call_args_list[-1].kwargs["messages"]
        assert "tool_result_reference" in str(messages.to_chatml())
        assert "x" * 10000 not in str(messages.to_chatml())
    finally:
        checkpoints.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_inline_bytes": 0},
        {"max_inline_bytes": True},
        {"preview_bytes": -1},
        {"preview_bytes": True},
        {"max_inline_bytes": 2, "preview_bytes": 3},
    ],
)
def test_invalid_extension_limits(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ToolOutputOffloadExtension(LocalToolResultStore(tmp_path), **kwargs)
