"""Durable revisioned workspace policy snapshots in SQLiteServiceStore."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from msgflux.runtime.permissions import ResourcePermission
from msgflux.runtime.service.records import ServiceConflictError, ServiceThread
from msgflux.runtime.service.store import SQLiteServiceStore
from msgflux.runtime.workspace.policy import WorkspacePolicy


def _policy(thread_id="thread-1", *, permissions=("filesystem.read",), mode="never"):
    return WorkspacePolicy(
        thread_id=thread_id,
        permissions=permissions,
        resources=(
            ResourcePermission("workspace:project:/README.md", "filesystem.read"),
        ),
        approval_policy=mode,
    )


def _store(path=":memory:"):
    store = SQLiteServiceStore(path)
    store.bind_thread(ServiceThread("thread-1", "assistant"))
    return store


def test_policy_read_is_empty_and_does_not_create_a_default(tmp_path):
    store = _store(tmp_path / "service.sqlite3")
    try:
        assert store.workspace_policy("thread-1") is None
        assert store.workspace_policy_history("thread-1") == ()
        count = store._connection.execute(
            "SELECT count(*) FROM service_workspace_policies"
        ).fetchone()[0]
        assert count == 0
    finally:
        store.close()


def test_policy_history_rows_are_append_only():
    store = _store()
    try:
        saved = store.update_workspace_policy(_policy(), expected_revision=0)
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store._connection.execute(
                "UPDATE service_workspace_policy_history SET approval_policy='on-request' "
                "WHERE thread_id=? AND revision=?",
                (saved.thread_id, saved.revision),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store._connection.execute(
                "DELETE FROM service_workspace_policy_history "
                "WHERE thread_id=? AND revision=?",
                (saved.thread_id, saved.revision),
            )
        assert store.workspace_policy_history("thread-1") == (saved,)
    finally:
        store.close()


def test_policy_updates_append_reopen_and_return_store_revision(tmp_path):
    path = tmp_path / "service.sqlite3"
    store = _store(path)
    try:
        first = store.update_workspace_policy(_policy(), expected_revision=0)
        assert first.revision == 1
        assert first.updated_at
        second = store.update_workspace_policy(
            _policy(
                permissions=("filesystem.read", "filesystem.write"), mode="on-request"
            ),
            expected_revision=first.revision,
        )
        assert second.revision == 2
        assert second.permission_set().grants == {
            "filesystem.read",
            "filesystem.write",
        }
        assert second.approval_policy == "on-request"
        assert tuple(
            item.revision for item in store.workspace_policy_history("thread-1")
        ) == (
            1,
            2,
        )
    finally:
        store.close()

    reopened = SQLiteServiceStore(path)
    try:
        assert reopened.workspace_policy("thread-1") == second
        assert reopened.workspace_policy_history("thread-1") == (first, second)
    finally:
        reopened.close()


def test_policy_compare_and_swap_rejects_stale_revision_without_history_row():
    store = _store()
    try:
        first = store.update_workspace_policy(_policy(), expected_revision=0)
        with pytest.raises(ServiceConflictError, match="Workspace policy changed"):
            store.update_workspace_policy(
                _policy(permissions=("filesystem.write",)), expected_revision=0
            )
        assert store.workspace_policy("thread-1") == first
        assert store.workspace_policy_history("thread-1") == (first,)
    finally:
        store.close()


def test_policy_updates_without_expected_revision_are_serialized(tmp_path):
    path = tmp_path / "service.sqlite3"
    first = _store(path)
    try:
        current = first.update_workspace_policy(_policy(), expected_revision=0)
        second = SQLiteServiceStore(path)
        try:
            proposals = (
                _policy(permissions=("filesystem.write",)),
                _policy(permissions=("process.execute",), mode="on-request"),
            )

            def update(pair):
                connection, proposal = pair
                return connection.update_workspace_policy(proposal)

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = tuple(
                    executor.map(
                        update, ((first, proposals[0]), (second, proposals[1]))
                    )
                )
            assert {item.revision for item in results} == {2, 3}
            assert tuple(
                item.revision for item in first.workspace_policy_history("thread-1")
            ) == (1, 2, 3)
            assert first.workspace_policy("thread-1").revision == 3
            assert current.revision == 1
        finally:
            second.close()
    finally:
        first.close()


def test_policy_expected_revision_race_allows_only_one_writer(tmp_path):
    path = tmp_path / "service.sqlite3"
    first = _store(path)
    second = SQLiteServiceStore(path)
    try:
        current = first.update_workspace_policy(_policy(), expected_revision=0)

        def update(connection, policy):
            try:
                return connection.update_workspace_policy(
                    policy, expected_revision=current.revision
                )
            except ServiceConflictError as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = tuple(
                executor.map(
                    lambda args: update(*args),
                    (
                        (first, _policy(permissions=("filesystem.write",))),
                        (second, _policy(permissions=("process.execute",))),
                    ),
                )
            )
        assert sum(isinstance(item, WorkspacePolicy) for item in outcomes) == 1
        assert sum(isinstance(item, ServiceConflictError) for item in outcomes) == 1
        assert len(first.workspace_policy_history("thread-1")) == 2
    finally:
        second.close()
        first.close()


def test_policy_store_migrates_existing_thread_table_without_policy_rows(tmp_path):
    path = tmp_path / "old-service.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE service_threads (thread_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO service_threads (thread_id, agent_id) VALUES (?, ?)",
        ("thread-1", "assistant"),
    )
    connection.commit()
    connection.close()

    store = SQLiteServiceStore(path)
    try:
        assert store.thread("thread-1") == ServiceThread("thread-1", "assistant")
        assert store.workspace_policy("thread-1") is None
        saved = store.update_workspace_policy(_policy(), expected_revision=0)
        assert saved.revision == 1
    finally:
        store.close()


@pytest.mark.parametrize(
    "policy",
    [
        {"thread_id": "thread-1", "permissions": ("filesystem.read",)},
        None,
    ],
)
def test_policy_store_rejects_invalid_record_and_revision(policy):
    store = _store()
    try:
        with pytest.raises((TypeError, ValueError)):
            store.update_workspace_policy(policy)
        with pytest.raises(ValueError, match="expected_revision"):
            store.update_workspace_policy(_policy(), expected_revision=True)
        assert store.workspace_policy("thread-1") is None
        assert store.workspace_policy_history("thread-1") == ()
    finally:
        store.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thread_id": "", "permissions": ()},
        {"thread_id": "thread-1", "permissions": ("filesystem.*",)},
        {"thread_id": "thread-1", "permissions": (), "approval_policy": "yolo"},
        {"thread_id": "thread-1", "permissions": (), "revision": True},
    ],
)
def test_workspace_policy_record_rejects_invalid_fields(kwargs):
    with pytest.raises((TypeError, ValueError)):
        WorkspacePolicy(**kwargs)


def test_policy_methods_require_existing_thread_and_history_is_foreign_keyed():
    store = SQLiteServiceStore()
    try:
        with pytest.raises(KeyError):
            store.workspace_policy("unknown")
        with pytest.raises(KeyError):
            store.workspace_policy_history("unknown")
        with pytest.raises(KeyError):
            store.update_workspace_policy(_policy("unknown"), expected_revision=0)
        assert (
            store._connection.execute(
                "SELECT count(*) FROM service_workspace_policy_history"
            ).fetchone()[0]
            == 0
        )
    finally:
        store.close()
