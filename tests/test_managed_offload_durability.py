"""Managed artifact retention, quota, and publication-boundary integrations."""

import asyncio
import hashlib
import multiprocessing
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest

from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.extensions import ManagedToolOutputOffloadExtension
from msgflux.runtime import (
    ExecutionScope,
    LocalToolResultStore,
    ToolOutputOffloadConfig,
    ToolResultQuotaError,
    ToolResultRef,
)


def _response(text="done"):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


def _tool_response(name, call_id):
    calls = ToolCallAggregator()
    calls.process(0, call_id, name, "{}")
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _managed_agent(agent_dir, tools, config):
    model = Mock()
    model.model_type = "chat_completion"
    return Agent(
        name="managed",
        model=model,
        tools=tools,
        agent_dir=agent_dir,
        extensions=[ManagedToolOutputOffloadExtension(config)],
    )


def _tool_descriptors(messages):
    descriptors = []
    for message in messages.to_chatml():
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            descriptor = msgspec.json.decode(content)
        except msgspec.DecodeError:
            continue
        if isinstance(descriptor, dict) and descriptor.get("type") == (
            "tool_result_reference"
        ):
            descriptors.append(descriptor)
    return descriptors


@pytest.mark.asyncio
async def test_managed_reference_survives_agent_reopen_and_checkpoint_load(tmp_path):
    agent_dir = tmp_path / "managed-state"
    thread_id = "retained-thread"
    config = ToolOutputOffloadConfig(
        max_inline_bytes=64,
        preview_bytes=8,
        max_result_bytes=4096,
        max_store_bytes=16384,
    )
    effects = []

    def produce() -> str:
        """Produce a report retained by the managed thread."""
        effects.append("produced")
        return "durable-report-" * 80

    first = _managed_agent(agent_dir, [produce], config)
    first_calls = 0

    async def first_model(**_kwargs):
        nonlocal first_calls
        first_calls += 1
        if first_calls == 1:
            return _tool_response("produce", "produce-first")
        return _response("saved")

    first.generator.aforward = AsyncMock(side_effect=first_model)
    await first.acall(
        "Create a report", scope=ExecutionScope(thread_id=thread_id, run_id="run-1")
    )
    await first.aclose()

    restored = {}

    def produce_again() -> str:
        """A second call must not be needed to retrieve the old report."""
        effects.append("produced-again")
        return "unexpected"

    reopened = _managed_agent(agent_dir, [produce_again], config)

    async def reopened_model(**kwargs):
        restored["descriptors"] = _tool_descriptors(kwargs["messages"])
        return _response("reopened")

    reopened.generator.aforward = AsyncMock(side_effect=reopened_model)
    try:
        await reopened.acall(
            "Continue from saved history",
            scope=ExecutionScope(thread_id=thread_id, run_id="run-2"),
        )
        (descriptor,) = restored["descriptors"]
        reference = msgspec.convert(descriptor["reference"], type=ToolResultRef)
        bundle = reopened._owned_threads[thread_id].resources
        store = bundle.tool_result_store(create=False)
        assert store.get(reference.result_id) == reference
        store.verify(reference)
        assert store.read(reference) == ("durable-report-" * 80).encode()
        assert effects == ["produced"]
    finally:
        await reopened.aclose()


@pytest.mark.asyncio
async def test_managed_store_quota_failure_keeps_prior_result_and_does_not_replay(
    tmp_path,
):
    agent_dir = tmp_path / "quota-state"
    thread_id = "quota-thread"
    config = ToolOutputOffloadConfig(
        max_inline_bytes=32,
        preview_bytes=8,
        max_result_bytes=2048,
        max_store_bytes=700,
    )
    effects = []

    def first_result() -> str:
        """Publish one retained result before the root quota is exhausted."""
        effects.append("first")
        return "a" * 250

    def second_result() -> str:
        """Perform a second effect whose output exceeds remaining quota."""
        effects.append("second")
        return "b" * 400

    agent = _managed_agent(agent_dir, [first_result, second_result], config)
    model_calls = 0
    saved_reference = {}
    final_feedback = {}

    async def model(**kwargs):
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return _tool_response("first_result", "first-call")
        if model_calls == 2:
            descriptors = _tool_descriptors(kwargs["messages"])
            (descriptor,) = descriptors
            saved_reference["value"] = msgspec.convert(
                descriptor["reference"], type=ToolResultRef
            )
            return _tool_response("second_result", "second-call")
        final_feedback["messages"] = kwargs["messages"].to_chatml()
        return _response("quota reported")

    agent.generator.aforward = AsyncMock(side_effect=model)
    try:
        await agent.acall(
            "Run both operations", scope=ExecutionScope(thread_id=thread_id)
        )
        assert effects == ["first", "second"]
        assert model_calls == 3
        tool_content = "\n".join(
            str(message.get("content", ""))
            for message in final_feedback["messages"]
            if message.get("role") == "tool"
        )
        assert "ToolResultQuotaError" in tool_content
        assert "do not retry automatically" in tool_content

        store = agent._owned_threads[thread_id].resources.tool_result_store()
        old_reference = saved_reference["value"]
        store.verify(old_reference)
        assert store.read(old_reference) == b"a" * 250
        usage = store.usage()
        assert usage.results == 1
        assert usage.pending == 0
        assert list(store.root.glob("res_*")) == [store.root / old_reference.result_id]
    finally:
        await agent.aclose()


def _crash_after_managed_result_write(agent_dir: str) -> None:
    from msgflux.runtime.tool_results import LocalToolResultStore as Store

    marker = Path(agent_dir) / "effect.marker"
    thread_dir = Path(agent_dir) / "threads" / "prepublish-thread"
    thread_dir.mkdir(parents=True, exist_ok=True)

    def external_effect() -> str:
        """Persist a marker before producing the output to offload."""
        with marker.open("ab") as stream:
            stream.write(b"effect\n")
            stream.flush()
            os.fsync(stream.fileno())
        return "stage-after-write-" * 160

    config = ToolOutputOffloadConfig(
        max_inline_bytes=32,
        preview_bytes=8,
        max_result_bytes=8192,
        max_store_bytes=16384,
    )
    agent = _managed_agent(Path(agent_dir), [external_effect], config)
    write = Store._write

    def finish_write_then_exit(self, *args, **kwargs):
        write(self, *args, **kwargs)
        # _write has fsynced content and metadata. os._exit bypasses put()'s
        # exception cleanup and kills the process before the staging rename.
        os._exit(87)

    Store._write = finish_write_then_exit

    async def invoke_tool():
        with agent._resource_context(
            {"scope": ExecutionScope(thread_id="prepublish-thread")}
        ):
            await agent.tool_library.acall([("prepublish-call", "external_effect", {})])

    asyncio.run(invoke_tool())
    raise AssertionError("the injected process exit did not run")


def test_managed_crash_after_write_before_rename_counts_staging_and_requires_gc(
    tmp_path,
):
    context = multiprocessing.get_context("spawn")
    agent_dir = tmp_path / "crash-state"
    process = context.Process(
        target=_crash_after_managed_result_write, args=(str(agent_dir),)
    )
    process.start()
    try:
        process.join(timeout=20)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        assert process.exitcode == 87
        assert (agent_dir / "effect.marker").read_bytes() == b"effect\n"

        result_root = agent_dir / "threads" / "prepublish-thread" / "tool-results"
        entries = list(result_root.iterdir())
        (staging,) = entries
        assert staging.name.startswith(".pending-")
        assert not list(result_root.glob("res_*"))
        content = (staging / "content").read_bytes()
        metadata = msgspec.json.decode(
            (staging / "metadata.json").read_bytes(), type=ToolResultRef
        )
        assert metadata.size_bytes == len(content)
        assert metadata.sha256 == hashlib.sha256(content).hexdigest()
        assert metadata.result_id.startswith("res_")

        store = LocalToolResultStore(
            result_root,
            max_result_bytes=8192,
            max_store_bytes=16384,
        )
        usage = store.usage()
        assert usage.results == 0
        assert usage.pending == 1
        assert (
            usage.size_bytes
            == len(content) + (staging / "metadata.json").stat().st_size
        )
        with pytest.raises(FileNotFoundError):
            store.get(metadata.result_id)

        quota_store = LocalToolResultStore(
            result_root,
            max_result_bytes=8192,
            max_store_bytes=usage.size_bytes,
        )
        with pytest.raises(ToolResultQuotaError):
            quota_store.put([b"new result"])
        assert quota_store.usage().pending == 1
        assert not list(result_root.glob("res_*"))

        with pytest.raises(ValueError, match="quiescent"):
            store.collect_garbage([], dry_run=False)
        assert store.collect_garbage([], quiescent=True) == (staging.name,)
        assert store.collect_garbage([], quiescent=True, dry_run=False) == (
            staging.name,
        )
        assert store.usage().results == 0
        assert store.usage().pending == 0
        assert store.usage().size_bytes == 0
        assert store.put([b"after offline cleanup"]).size_bytes == len(
            b"after offline cleanup"
        )
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        if process.exitcode is not None:
            process.close()
