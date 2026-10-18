"""Regression coverage for background worker cancellation terminalization."""

import asyncio
from unittest.mock import Mock

import msgflux as mf
import pytest
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.agent_inbox import AgentInbox, InMemoryAgentInboxStore
from msgflux.runtime.background import BackgroundTaskDispatcher
from msgflux.runtime.context import execution_context
from msgflux.tasks import SQLiteTaskStore, TaskHandle
from msgflux.tools.builtin import AgentTool


class _Library:
    def get_agent_inbox(self):
        return None


@pytest.mark.parametrize(
    ("reason", "expected_reason"),
    [
        ("unexpected tool cancellation", "unexpected tool cancellation"),
        ("", "CancelledError"),
    ],
)
def test_async_cancellation_interrupts_task_and_releases_worker(
    tmp_path, reason, expected_reason
):
    database_path = tmp_path / "tasks.sqlite3"
    task_store = SQLiteTaskStore(str(database_path))
    task_store.create("worker", task_id="cancelled")
    task_handle = TaskHandle("cancelled", task_store)

    def cancelled_tool():
        raise asyncio.CancelledError(reason)

    try:
        with pytest.raises(asyncio.CancelledError):
            BackgroundTaskDispatcher(_Library()).run_tool(
                tool=cancelled_tool,
                task_handle=task_handle,
                tool_name="worker",
                call_params={},
            )

        task = task_store.get("cancelled")
        assert task.status == "interrupted"
        assert task.metadata["interrupt_reason"] == expected_reason
        assert task_store.get_worker_lease("cancelled") is None
    finally:
        task_store.close()

    reopened = SQLiteTaskStore(str(database_path))
    try:
        task = reopened.get("cancelled")
        assert task.status == "interrupted"
        assert task.metadata["interrupt_reason"] == expected_reason
        assert reopened.get_worker_lease("cancelled") is None
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_scheduled_agent_cancellation_interrupts_and_cleans_up(tmp_path):
    database_path = tmp_path / "scheduled-tasks.sqlite3"
    task_store = SQLiteTaskStore(str(database_path))
    checkpoint_store = InMemoryCheckpointStore()
    inbox_store = InMemoryAgentInboxStore()
    worker = Agent(
        name="cancelled_worker",
        model=Mock(model_type="chat_completion"),
        config={"stream": True},
    )

    async def cancel_stream(**_kwargs):
        raise asyncio.CancelledError("cancelled async model call")

    worker.generator.aforward = cancel_stream
    library = ToolLibrary(
        name="cancelled_agent_tasks",
        tools=[mf.tool_config(allow_background=True)(AgentTool()), worker],
        task_store=task_store,
    )
    library.set_agent_inbox(AgentInbox(owner="root", store=inbox_store))

    try:
        with execution_context(
            thread_id="cancel-root-thread",
            namespace="cancel-root",
            run_id="cancel-root-run",
            root_run_id="cancel-root-run",
            checkpoint_store=checkpoint_store,
        ):
            launch = (
                library(
                    [
                        (
                            "launch-cancelled-agent",
                            "agent",
                            {
                                "name": "cancelled_worker",
                                "message": "Start cancellation repro",
                                "run_in_background": True,
                            },
                        )
                    ]
                )
                .tool_calls[0]
                .result
            )

        task_id = launch.split("task_id='")[1].split("'")[0]
        dispatcher = library.get_background_dispatcher()
        deadline = asyncio.get_running_loop().time() + 5
        while True:
            task = task_store.get(task_id)
            notices = [
                notice
                for notice in library.get_agent_inbox().peek()
                if notice.ref == task_id and notice.status == "interrupted"
            ]
            if (
                task.status not in {"queued", "running"}
                and dispatcher.get_task_future(task_id) is None
                and notices
            ):
                break
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.01)

        task = task_store.get(task_id)
        assert task.status == "interrupted"
        assert task.result is None
        assert task.error is None
        assert task.metadata["interrupt_reason"]
        assert "ModelStreamResponse" not in task.metadata["interrupt_reason"]
        assert task_store.get_worker_lease(task_id) is None

        assert dispatcher.get_task_future(task_id) is None
        assert len(notices) == 1

        task_store.close()
        reopened = SQLiteTaskStore(str(database_path))
        try:
            persisted = reopened.get(task_id)
            assert persisted.status == "interrupted"
            assert persisted.result is None
            assert persisted.error is None
            assert (
                persisted.metadata["interrupt_reason"]
                == task.metadata["interrupt_reason"]
            )
            assert reopened.get_worker_lease(task_id) is None
        finally:
            reopened.close()
    finally:
        task_store.close()
