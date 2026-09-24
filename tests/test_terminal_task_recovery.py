"""Crash/restart coverage for terminal Agent checkpoints and task results."""

import multiprocessing
import os
import time

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.agent_inbox import AgentInbox, SQLiteAgentInboxStore
from msgflux.runtime.context import execution_context
from msgflux.tasks import SQLiteTaskStore


class _FixedModel:
    model_type = "chat_completion"

    def __init__(self, gate=None):
        self.gate = gate
        self.calls = 0

    def __call__(self, **_kwargs):
        self.calls += 1
        if self.gate is not None and not self.gate.wait(timeout=10):
            raise TimeoutError("Parent did not release the model")
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add("durable result")
        return response


def _crash_after_terminal_checkpoint(paths, connection, gate):
    checkpoints = SQLiteCheckpointStore(paths[0])
    tasks = SQLiteTaskStore(paths[1])
    inbox_store = SQLiteAgentInboxStore(paths[2])

    def exit_after_commit(_context):
        os._exit(73)

    worker = Agent(
        name="worker",
        model=_FixedModel(gate),
        checkpoint_store=checkpoints,
        hooks=[Hook(event="after_run_end", handler=exit_after_commit)],
    )
    worker.tool_config = {"background": True}
    library = ToolLibrary(name="lib", tools=[worker], task_store=tasks)
    library.set_agent_inbox(AgentInbox(owner="root", store=inbox_store))
    library.get_background_dispatcher().lease_seconds = 0.2

    with execution_context(thread_id="thread", run_id="root", root_run_id="root"):
        dispatch = library([("start", "worker", {"task": "Start"})])
        task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]
        connection.send(task_id)
        connection.close()

    gate.wait(timeout=10)


def test_crash_after_terminal_checkpoint_reconciles_without_model_call(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MSGTRACE_TELEMETRY_ENABLED", "false")
    monkeypatch.setenv("MSGTRACE_EXPORTER", "console")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    paths = tuple(
        str(tmp_path / filename)
        for filename in ("checkpoint.sqlite", "tasks.sqlite", "inbox.sqlite")
    )
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    gate = context.Event()
    process = context.Process(
        target=_crash_after_terminal_checkpoint, args=(paths, send, gate)
    )
    try:
        process.start()
        assert receive.poll(15)
        task_id = receive.recv()
        gate.set()
        process.join(timeout=15)
        assert process.exitcode == 73

        checkpoints = SQLiteCheckpointStore(paths[0])
        tasks = SQLiteTaskStore(paths[1])
        inbox_store = SQLiteAgentInboxStore(paths[2])
        try:
            checkpoint = checkpoints.load_state("worker", "thread", task_id)
            assert checkpoint["status"] == "completed"
            assert checkpoint["task_result"] == {"value": "durable result"}
            assert tasks.get(task_id).status == "running"

            deadline = time.monotonic() + 5
            while tasks.get_worker_lease(task_id).expires_at > time.time():
                assert time.monotonic() < deadline
                time.sleep(0.01)

            model = _FixedModel()
            worker = Agent(name="worker", model=model, checkpoint_store=checkpoints)
            worker.tool_config = {"background": True}
            library = ToolLibrary(name="lib", tools=[worker], task_store=tasks)
            library.set_agent_inbox(AgentInbox(owner="root", store=inbox_store))
            assert "reconciled" in library.reconcile_agent_task(task_id)
            assert tasks.get(task_id).status == "completed"
            assert tasks.get(task_id).result == "durable result"
            assert tasks.get_worker_lease(task_id) is None
            assert model.calls == 0
        finally:
            checkpoints.close()
            tasks.close()
            inbox_store.close()
    finally:
        gate.set()
        receive.close()
        send.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()
