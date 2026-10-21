"""A task settlement or notice error must not replace the tool failure."""

import asyncio

import pytest

from msgflux.exceptions import AbortRequestedError, TaskPauseRequestedError
from msgflux.runtime.background import BackgroundTaskDispatcher
from msgflux.tasks import SQLiteTaskStore, TaskHandle


class _Library:
    def get_agent_inbox(self):
        return None


def _raises(error):
    def tool():
        raise error

    return tool


def test_tool_failure_survives_failed_task_settlement(tmp_path, monkeypatch):
    store = SQLiteTaskStore(str(tmp_path / "failed-settlement.sqlite3"))
    store.create("worker", task_id="failed-settlement")
    handle = TaskHandle("failed-settlement", store)
    dispatcher = BackgroundTaskDispatcher(_Library())
    primary = ValueError("tool rejected input")
    secondary = RuntimeError("store unavailable")

    def fail(_error):
        raise secondary

    monkeypatch.setattr(handle, "fail", fail)
    try:
        with pytest.raises(ValueError) as raised:
            dispatcher.run_tool(
                tool=_raises(primary),
                task_handle=handle,
                tool_name="worker",
                call_params={},
            )
        assert raised.value is primary
    finally:
        store.close()


def test_tool_cancellation_survives_failed_interrupt_settlement(
    tmp_path, monkeypatch, caplog
):
    store = SQLiteTaskStore(str(tmp_path / "cancelled-settlement.sqlite3"))
    store.create("worker", task_id="cancelled-settlement")
    handle = TaskHandle("cancelled-settlement", store)
    dispatcher = BackgroundTaskDispatcher(_Library())
    primary = asyncio.CancelledError("tool cancelled")
    notices = []
    monkeypatch.setattr(
        dispatcher,
        "publish_task_notification",
        lambda **kwargs: notices.append(kwargs["status"]),
    )

    def interrupt(*, reason=None):
        del reason
        raise asyncio.CancelledError("persistence cancelled")

    monkeypatch.setattr(handle, "interrupt", interrupt)
    try:
        with pytest.raises(asyncio.CancelledError) as raised:
            dispatcher.run_tool(
                tool=_raises(primary),
                task_handle=handle,
                tool_name="worker",
                call_params={},
            )
        assert raised.value is primary
        assert store.get("cancelled-settlement").status == "running"
        assert notices == []
        assert "Could not persist background task failure" in caplog.text
    finally:
        store.close()


@pytest.mark.parametrize(
    ("primary", "expected_status"),
    [
        (AbortRequestedError("worker stopped"), "interrupted"),
        (TaskPauseRequestedError("paused-task", "awaiting approval"), "paused"),
    ],
)
def test_abort_and_pause_keep_their_task_classification(
    tmp_path, monkeypatch, primary, expected_status
):
    store = SQLiteTaskStore(str(tmp_path / f"{expected_status}.sqlite3"))
    task_id = f"{expected_status}-task"
    store.create("worker", task_id=task_id)
    handle = TaskHandle(task_id, store)
    dispatcher = BackgroundTaskDispatcher(_Library())
    notices = []
    monkeypatch.setattr(
        dispatcher,
        "publish_task_notification",
        lambda **kwargs: notices.append(kwargs["status"]),
    )
    try:
        with pytest.raises(type(primary)) as raised:
            dispatcher.run_tool(
                tool=_raises(primary),
                task_handle=handle,
                tool_name="worker",
                call_params={},
            )
        assert raised.value is primary
        assert store.get(task_id).status == expected_status
        assert notices == [expected_status]
    finally:
        store.close()


@pytest.mark.parametrize(
    ("primary", "settle_method", "log_message"),
    [
        (
            AbortRequestedError("worker stopped"),
            "interrupt",
            "Could not persist background task failure",
        ),
        (
            TaskPauseRequestedError("paused-task", "awaiting approval"),
            "pause",
            "Could not persist background task failure",
        ),
    ],
)
def test_abort_and_pause_survive_cancelled_persistence_without_notice(
    tmp_path, monkeypatch, caplog, primary, settle_method, log_message
):
    store = SQLiteTaskStore(str(tmp_path / "control-settlement.sqlite3"))
    task_id = "control-settlement-task"
    store.create("worker", task_id=task_id)
    handle = TaskHandle(task_id, store)
    dispatcher = BackgroundTaskDispatcher(_Library())
    notices = []
    monkeypatch.setattr(
        dispatcher,
        "publish_task_notification",
        lambda **kwargs: notices.append(kwargs["status"]),
    )

    def cancel_persistence(**_kwargs):
        raise asyncio.CancelledError("persistence cancelled")

    monkeypatch.setattr(handle, settle_method, cancel_persistence)
    try:
        with pytest.raises(type(primary)) as raised:
            dispatcher.run_tool(
                tool=_raises(primary),
                task_handle=handle,
                tool_name="worker",
                call_params={},
            )
        assert raised.value is primary
        assert store.get(task_id).status == "running"
        assert notices == []
        assert log_message in caplog.text
    finally:
        store.close()


@pytest.mark.parametrize(
    ("primary", "secondary"),
    [
        (ValueError("tool failed"), RuntimeError("inbox unavailable")),
        (
            asyncio.CancelledError("tool cancelled"),
            RuntimeError("inbox unavailable"),
        ),
        (
            ValueError("tool failed"),
            asyncio.CancelledError("notice cancelled"),
        ),
        (
            asyncio.CancelledError("tool cancelled"),
            asyncio.CancelledError("notice cancelled"),
        ),
    ],
)
def test_tool_outcome_survives_notification_failure(
    tmp_path, monkeypatch, primary, secondary
):
    store = SQLiteTaskStore(str(tmp_path / "notification-failure.sqlite3"))
    task_id = "notification-failure"
    store.create("worker", task_id=task_id)
    handle = TaskHandle(task_id, store)
    dispatcher = BackgroundTaskDispatcher(_Library())

    def fail_notification(**_kwargs):
        raise secondary

    monkeypatch.setattr(dispatcher, "publish_task_notification", fail_notification)
    try:
        with pytest.raises(type(primary)) as raised:
            dispatcher.run_tool(
                tool=_raises(primary),
                task_handle=handle,
                tool_name="worker",
                call_params={},
            )
        assert raised.value is primary
        task = store.get(task_id)
        expected = (
            "interrupted" if isinstance(primary, asyncio.CancelledError) else "failed"
        )
        assert task.status == expected
        if expected == "failed":
            assert task.error == "tool failed"
    finally:
        store.close()
