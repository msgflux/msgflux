"""Background Agent streams are finalized before their results are persisted."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import threading
from unittest.mock import Mock

import msgflux as mf
import pytest
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelStreamResponse
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook, OutputContext
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.context import execution_context
from msgflux.tasks import InMemoryTaskStore, SQLiteTaskStore
from msgflux.tools.builtin import AgentTool


@pytest.fixture(params=["memory", "sqlite"])
def background_task_store(request, tmp_path):
    if request.param == "memory":
        yield "memory", InMemoryTaskStore(), None
        return
    path = tmp_path / "tasks.sqlite3"
    store = SQLiteTaskStore(str(path))
    try:
        yield "sqlite", store, path
    finally:
        store.close()


def _tool_result(library, tool_name, call_id, parameters):
    return library([(call_id, tool_name, parameters)]).tool_calls[0].result


@pytest.mark.parametrize("failure_stage", [None, "transform", "provider"])
@pytest.mark.asyncio
async def test_background_agent_stream_task_waits_and_persists_final_output(  # noqa: C901
    background_task_store, failure_stage
):
    store_kind, task_store, database_path = background_task_store
    checkpoint_store = InMemoryCheckpointStore()
    stream_ready = threading.Event()
    release_stream = threading.Event()
    produced_streams = []
    producer_errors = []
    producer_threads = []
    task_id = None
    transform_inputs = []
    raw_output = "sha256=abc123 bytes=42"
    expected_output = f"normalized:{raw_output}"

    def transform_output(context: OutputContext):
        transform_inputs.append(context.output)
        if failure_stage == "transform":
            raise RuntimeError("stream output transform failed")
        return replace(context, output=f"normalized:{context.output}")

    async def produce_stream(**_kwargs):
        stream = ModelStreamResponse(mode="sync")
        stream.set_response_type("text_generation")
        produced_streams.append(stream)

        def finish_when_released():
            if not release_stream.wait(timeout=5):
                producer_errors.append(TimeoutError("stream release timed out"))
                return
            try:
                if failure_stage == "provider":
                    stream.add("partial provider output")
                    stream.finish(
                        status="failed",
                        error=RuntimeError("provider stream failed"),
                    )
                else:
                    stream.add("sha256=abc")
                    stream.add("123 bytes=42")
                    stream.finish()
            except BaseException as error:
                producer_errors.append(error)

        producer = threading.Thread(target=finish_when_released, daemon=True)
        producer_threads.append(producer)
        producer.start()
        stream_ready.set()
        return stream

    model = Mock(model_type="chat_completion")
    worker = Agent(
        name="stream_worker",
        model=model,
        config={"stream": True},
        hooks=[Hook(event="transform_output", handler=transform_output)],
    )
    worker.generator.aforward = produce_stream
    library = ToolLibrary(
        name="stream_tasks",
        tools=[mf.tool_config(allow_background=True)(AgentTool()), worker],
        task_store=task_store,
    )

    try:
        with execution_context(
            thread_id="stream-root-thread",
            namespace="stream-root",
            run_id="stream-root-run",
            root_run_id="stream-root-run",
            checkpoint_store=checkpoint_store,
        ):
            launch = _tool_result(
                library,
                "agent",
                "launch-stream",
                {
                    "name": "stream_worker",
                    "message": "Produce the digest summary",
                    "run_in_background": True,
                },
            )

        assert "task_id='" in launch
        task_id = launch.split("task_id='")[1].split("'")[0]
        assert await asyncio.to_thread(stream_ready.wait, 2)
        assert task_store.get(task_id).status == "running"

        timed_out = await asyncio.to_thread(
            _tool_result,
            library,
            "task_wait",
            "wait-before-stream-finish",
            {"task_id": task_id, "timeout": 0.02},
        )
        assert timed_out["status"] == "timeout"
        assert timed_out["task_status"] == "running"

        release_stream.set()
        settled = await asyncio.to_thread(
            _tool_result,
            library,
            "task_wait",
            "wait-for-stream-finish",
            {"task_id": task_id, "timeout": 2.0},
        )
        output = await asyncio.to_thread(
            _tool_result,
            library,
            "task_output",
            "read-stream-output",
            {"task_id": task_id},
        )
        record = task_store.get(task_id)

        if failure_stage is not None:
            expected_error = (
                "stream output transform failed"
                if failure_stage == "transform"
                else "provider stream failed"
            )
            assert settled["status"] == "failed"
            assert output["status"] == "failed"
            assert expected_error in settled["error"]
            assert expected_error in output["error"]
            assert record.status == "failed"
            assert expected_error in record.error
            assert "ModelStreamResponse" not in record.error
            if failure_stage == "provider":
                assert transform_inputs == []
                metadata = record.metadata
                checkpoint = checkpoint_store.load_state(
                    metadata["checkpoint_namespace"],
                    metadata["checkpoint_thread_id"],
                    metadata["checkpoint_run_id"],
                )
                assert checkpoint["status"] == "failed"
            else:
                assert transform_inputs == [raw_output]
        else:
            assert settled == expected_output
            assert output == expected_output
            assert record.status == "completed"
            assert record.result == expected_output
            assert transform_inputs == [raw_output]

        assert len(produced_streams) == 1
        assert not producer_errors
        if store_kind == "sqlite":
            reopened = SQLiteTaskStore(str(database_path))
            try:
                persisted = reopened.get(task_id)
                assert persisted.status == record.status
                assert persisted.result == record.result
                assert persisted.error == record.error
            finally:
                reopened.close()
    finally:
        release_stream.set()
        for producer in producer_threads:
            await asyncio.to_thread(producer.join, 2)
        if task_id is not None:
            try:
                await asyncio.to_thread(
                    _tool_result,
                    library,
                    "task_wait",
                    "cleanup-wait",
                    {"task_id": task_id, "timeout": 2.0},
                )
            except Exception as error:
                producer_errors.append(error)
