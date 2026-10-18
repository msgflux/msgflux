"""Background dispatch failures must settle the task created before submit."""

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
from functools import partial

import msgflux as mf
import pytest

from msgflux._private.executor import Executor
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tasks import SQLiteTaskStore, TaskHandle


@pytest.mark.parametrize(
    ("submit_error", "expected_status", "expected_reason"),
    [
        (RuntimeError("executor unavailable"), "failed", "executor unavailable"),
        (
            FutureCancelledError("submission cancelled"),
            "interrupted",
            "submission cancelled",
        ),
        (
            asyncio.CancelledError("async submission cancelled"),
            "interrupted",
            "async submission cancelled",
        ),
    ],
)
def test_executor_submission_failure_settles_durable_task(
    tmp_path, monkeypatch, submit_error, expected_status, expected_reason
):
    @mf.tool_config(background=True)
    def worker() -> str:
        """Return a short worker result."""
        return "done"

    database_path = tmp_path / "dispatch-tasks.sqlite3"
    task_store = SQLiteTaskStore(str(database_path))
    library = ToolLibrary(
        name="dispatch_failure", tools=[worker], task_store=task_store
    )
    definition = library.get_tool_definition("worker")
    original_submit = Executor.submit

    def reject_background_worker(self, function, *args, **kwargs):
        candidate = function.func if isinstance(function, partial) else function
        if getattr(candidate, "__name__", None) == "run_tool":
            raise submit_error
        return original_submit(self, function, *args, **kwargs)

    monkeypatch.setattr(Executor, "submit", reject_background_worker)
    dispatcher = library.get_background_dispatcher()

    try:
        with pytest.raises(type(submit_error)) as raised:
            dispatcher.dispatch(
                tool=definition.executor,
                definition=definition,
                tool_id="launch",
                tool_name="worker",
                call_params={},
                visible_params={},
            )
        assert raised.value is submit_error

        (task,) = task_store.list()
        assert task.status == expected_status
        if expected_status == "failed":
            assert task.error == expected_reason
        else:
            assert task.metadata["interrupt_reason"] == expected_reason
        assert task_store.get_worker_lease(task.task_id) is None
        assert dispatcher.get_task_future(task.task_id) is None
        notices = [
            notice
            for notice in library.get_agent_inbox().peek()
            if notice.ref == task.task_id and notice.status == expected_status
        ]
        assert len(notices) == 1
    finally:
        task_store.close()

    reopened = SQLiteTaskStore(str(database_path))
    try:
        (persisted,) = reopened.list()
        assert persisted.status == expected_status
        assert reopened.get_worker_lease(persisted.task_id) is None
        if expected_status == "failed":
            assert persisted.error == expected_reason
        else:
            assert persisted.metadata["interrupt_reason"] == expected_reason
    finally:
        reopened.close()


def test_normal_resume_submission_failure_does_not_leave_task_queued(
    tmp_path, monkeypatch
):
    from unittest.mock import Mock

    from msgflux.nn import Agent

    worker = Agent(name="worker", model=Mock(model_type="chat_completion"))
    worker.tool_config = {"background": True}
    database_path = tmp_path / "resume-tasks.sqlite3"
    task_store = SQLiteTaskStore(str(database_path))
    library = ToolLibrary(name="resume_failure", tools=[worker], task_store=task_store)
    inbox = library.get_agent_inbox()
    task_store.create(
        "worker",
        task_id="resume-task",
        metadata={
            "task_kind": "agent",
            "checkpoint_namespace": "worker",
            "checkpoint_thread_id": "resume-thread",
            "checkpoint_run_id": "resume-task",
            "inbox_store_id": inbox.store.routing_id,
            "checkpoint_store_id": None,
            "workspace_reference": None,
            "task_resume_params": {"name": "worker"},
            "resume_generation": 0,
            "parent_run_id": None,
            "root_run_id": None,
            "account_name": None,
            "background_capabilities": [],
        },
    )
    task_store.complete("resume-task", "previous result")
    original_submit = Executor.submit

    def reject_background_worker(self, function, *args, **kwargs):
        candidate = function.func if isinstance(function, partial) else function
        if getattr(candidate, "__name__", None) == "run_tool":
            raise RuntimeError("resume executor unavailable")
        return original_submit(self, function, *args, **kwargs)

    monkeypatch.setattr(Executor, "submit", reject_background_worker)
    dispatcher = library.get_background_dispatcher()

    try:
        with pytest.raises(RuntimeError, match="resume executor unavailable"):
            dispatcher.resume_agent_task(
                task=task_store.get("resume-task"), message="Continue"
            )

        resumed = task_store.get("resume-task")
        assert resumed.status == "failed"
        assert resumed.error == "resume executor unavailable"
        assert task_store.get_worker_lease("resume-task") is None
        assert dispatcher.get_task_future("resume-task") is None
        notices = [
            notice
            for notice in library.get_agent_inbox().peek()
            if notice.ref == "resume-task" and notice.status == "failed"
        ]
        assert len(notices) == 1
    finally:
        task_store.close()

    reopened = SQLiteTaskStore(str(database_path))
    try:
        persisted = reopened.get("resume-task")
        assert persisted.status == "failed"
        assert persisted.error == "resume executor unavailable"
        assert reopened.get_worker_lease("resume-task") is None
    finally:
        reopened.close()


def test_notification_failure_does_not_rewrite_completed_task_or_repeat_tool(
    tmp_path, monkeypatch
):
    calls = []

    @mf.tool_config(background=True)
    def worker() -> str:
        calls.append("ran")
        return "finished"

    task_store = SQLiteTaskStore(str(tmp_path / "notification-tasks.sqlite3"))
    task_store.create("worker", task_id="notification-failure")
    library = ToolLibrary(
        name="notification_failure", tools=[worker], task_store=task_store
    )
    dispatcher = library.get_background_dispatcher()
    handle = TaskHandle("notification-failure", task_store, tool_name="worker")

    def fail_notification(**_kwargs):
        raise RuntimeError("inbox unavailable")

    monkeypatch.setattr(dispatcher, "publish_task_notification", fail_notification)
    try:
        with pytest.raises(RuntimeError, match="inbox unavailable"):
            dispatcher.run_tool(
                tool=library.get_tool_definition("worker").executor,
                task_handle=handle,
                tool_name="worker",
                call_params={},
            )
        task = task_store.get("notification-failure")
        assert task.status == "completed"
        assert task.result == "finished"
        assert task.error is None
        assert task_store.get_worker_lease(task.task_id) is None
        assert calls == ["ran"]
    finally:
        task_store.close()

    reopened = SQLiteTaskStore(str(tmp_path / "notification-tasks.sqlite3"))
    try:
        persisted = reopened.get("notification-failure")
        assert persisted.status == "completed"
        assert persisted.result == "finished"
        assert persisted.error is None
        assert reopened.get_worker_lease(persisted.task_id) is None
        assert calls == ["ran"]
    finally:
        reopened.close()


def test_terminal_persistence_failure_preserves_submit_error_without_notice(
    tmp_path, monkeypatch, caplog
):
    @mf.tool_config(background=True)
    def worker() -> str:
        """Return a short worker result."""
        return "done"

    task_store = SQLiteTaskStore(str(tmp_path / "persist-failure.sqlite3"))
    library = ToolLibrary(name="persist_failure", tools=[worker], task_store=task_store)
    definition = library.get_tool_definition("worker")
    dispatcher = library.get_background_dispatcher()
    submit_error = RuntimeError("original submit failure")
    original_submit = Executor.submit

    def reject_background_worker(self, function, *args, **kwargs):
        candidate = function.func if isinstance(function, partial) else function
        if getattr(candidate, "__name__", None) == "run_tool":
            raise submit_error
        return original_submit(self, function, *args, **kwargs)

    def reject_persistence(*_args, **_kwargs):
        raise OSError("task store unavailable")

    monkeypatch.setattr(Executor, "submit", reject_background_worker)
    monkeypatch.setattr(task_store, "fail", reject_persistence)
    try:
        with pytest.raises(RuntimeError) as raised:
            dispatcher.dispatch(
                tool=definition.executor,
                definition=definition,
                tool_id="launch",
                tool_name="worker",
                call_params={},
                visible_params={},
            )
        assert raised.value is submit_error
        (task,) = task_store.list()
        assert task.status == "queued"
        assert task_store.get_worker_lease(task.task_id) is None
        assert not [
            notice
            for notice in library.get_agent_inbox().peek()
            if notice.ref == task.task_id
        ]
        assert "Could not persist background task submission failure" in caplog.text
    finally:
        task_store.close()


def test_failure_notification_error_does_not_replace_submit_error(
    tmp_path, monkeypatch
):
    @mf.tool_config(background=True)
    def worker() -> str:
        """Return a short worker result."""
        return "done"

    database_path = tmp_path / "notification-submit-failure.sqlite3"
    task_store = SQLiteTaskStore(str(database_path))
    library = ToolLibrary(
        name="notification_submit_failure", tools=[worker], task_store=task_store
    )
    definition = library.get_tool_definition("worker")
    dispatcher = library.get_background_dispatcher()
    submit_error = RuntimeError("original submit failure")
    notification_calls = []
    original_submit = Executor.submit

    def reject_background_worker(self, function, *args, **kwargs):
        candidate = function.func if isinstance(function, partial) else function
        if getattr(candidate, "__name__", None) == "run_tool":
            raise submit_error
        return original_submit(self, function, *args, **kwargs)

    def reject_notification(**_kwargs):
        notification_calls.append(True)
        raise RuntimeError("inbox unavailable")

    monkeypatch.setattr(Executor, "submit", reject_background_worker)
    monkeypatch.setattr(dispatcher, "publish_task_notification", reject_notification)
    try:
        with pytest.raises(RuntimeError) as raised:
            dispatcher.dispatch(
                tool=definition.executor,
                definition=definition,
                tool_id="launch",
                tool_name="worker",
                call_params={},
                visible_params={},
            )
        assert raised.value is submit_error
        (task,) = task_store.list()
        assert task.status == "failed"
        assert task.error == "original submit failure"
        assert notification_calls == [True]
    finally:
        task_store.close()

    reopened = SQLiteTaskStore(str(database_path))
    try:
        (persisted,) = reopened.list()
        assert persisted.status == "failed"
        assert persisted.error == "original submit failure"
    finally:
        reopened.close()
