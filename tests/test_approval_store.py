import asyncio
import json
import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from msgflux.data.stores import Store
from msgflux.runtime import (
    ApprovalBinding,
    ApprovalConflictError,
    ApprovalExpiredError,
    ExecutionScope,
    InMemoryApprovalStore,
    PermissionSet,
    SQLiteApprovalStore,
    execution_context,
)


def make_binding(**overrides):
    values = {
        "namespace": "tools",
        "thread_id": "thread",
        "run_id": "run",
        "principal": "user:42",
        "tool_call_id": "call:1",
        "tool_name": "write",
        "tool_revision": "impl:v1",
        "policy_version": "policy:v1",
        "arguments": {"path": "private-name", "content": "secret-value"},
        "resources": {"root": "/workspace"},
        "required_permissions": ("filesystem.write",),
    }
    values.update(overrides)
    return ApprovalBinding.from_call(**values)


def scope_for(binding):
    return ExecutionScope(
        namespace=binding.namespace,
        thread_id=binding.thread_id,
        run_id=binding.run_id,
        principal=binding.principal,
        permissions=PermissionSet(binding.required_permissions),
    )


@pytest.fixture(params=["in_memory", "sqlite"])
def journal(request, tmp_path):
    clock = [100.0]
    options = (
        {"path": str(tmp_path / "approvals.sqlite3")}
        if request.param == "sqlite"
        else {}
    )
    store = Store.approval(request.param, clock=lambda: clock[0], **options)
    yield store, clock
    store.close()


def approve(store, binding):
    store.request(binding, request_id="request", expires_at=200)
    return store.decide("tools", "request", approved=True, decided_by="reviewer:1")


def test_factory_registration():
    assert set(Store.providers()["approval"]) == {"in_memory", "sqlite"}
    with pytest.raises(ValueError, match="not registered"):
        Store.approval("missing")


def test_lifecycle_and_idempotency(journal):
    store, _ = journal
    binding = make_binding()
    pending = store.request(binding, request_id="request", expires_at=200)
    assert store.request(binding, request_id="request", expires_at=200) == pending
    assert store.pending("tools", "thread", "run") == [pending]
    assert store.pending("tools", "thread", "another-run") == []
    assert store.get("other-namespace", "request") is None
    approved = approve(store, binding)
    assert approve(store, binding) == approved
    assert store.pending("tools", "thread", "run") == []
    with execution_context(scope=scope_for(binding)):
        consumed = store.consume("request", binding=binding)
        with pytest.raises(ApprovalConflictError, match="not available"):
            store.consume("request", binding=binding)
    assert consumed.status == "consumed"
    assert [event.status for event in store.events("tools", "request")] == [
        "pending",
        "approved",
        "consumed",
    ]
    assert [event.revision for event in store.events("tools", "request")] == [1, 2, 3]
    with pytest.raises(FrozenInstanceError):
        consumed.status = "approved"
    with pytest.raises(FrozenInstanceError):
        consumed.binding.principal = "admin"
    assert "secret-value" not in json.dumps(asdict(consumed))
    assert "private-name" not in repr(store.events("tools", "request"))


@pytest.mark.parametrize(
    "changes",
    [
        {"arguments": {"path": "changed"}},
        {"resources": {"root": "/"}},
        {"tool_revision": "impl:v2"},
        {"policy_version": "policy:v2"},
        {"tool_name": "other"},
        {"tool_call_id": "call:2"},
        {"required_permissions": ()},
        {"principal": "user:other"},
        {"run_id": "other"},
        {"thread_id": "other"},
    ],
)
def test_binding_changes_never_reuse_decision(journal, changes):
    store, _ = journal
    original = make_binding()
    approve(store, original)
    changed = make_binding(**changes)
    with pytest.raises(ApprovalConflictError, match="already bound"):
        store.request(changed, request_id="request", expires_at=200)
    with execution_context(scope=scope_for(changed)):
        with pytest.raises(ApprovalConflictError, match="binding changed"):
            store.consume("request", binding=changed)
    assert store.get("tools", "request").status == "approved"


def test_denial_and_conflicting_reviewers_are_final(journal):
    store, _ = journal
    binding = make_binding()
    store.request(binding, request_id="request", expires_at=200)
    denied = store.decide("tools", "request", approved=False, decided_by="reviewer:1")
    assert (
        store.decide("tools", "request", approved=False, decided_by="reviewer:1")
        == denied
    )
    for approved, reviewer in [(True, "reviewer:1"), (False, "reviewer:2")]:
        with pytest.raises(ApprovalConflictError):
            store.decide("tools", "request", approved=approved, decided_by=reviewer)
    with execution_context(scope=scope_for(binding)):
        with pytest.raises(ApprovalConflictError):
            store.consume("request", binding=binding)
    assert len(store.events("tools", "request")) == 2


@pytest.mark.parametrize("approved", [False, True])
def test_expiry_is_persisted_and_does_not_revive_after_clock_rollback(
    journal, approved
):
    store, clock = journal
    binding = make_binding()
    store.request(binding, request_id="request", expires_at=200)
    if approved:
        approve(store, binding)
    clock[0] = 200
    if approved:
        with execution_context(scope=scope_for(binding)):
            with pytest.raises(ApprovalExpiredError):
                store.consume("request", binding=binding)
    else:
        with pytest.raises(ApprovalExpiredError):
            store.decide("tools", "request", approved=True, decided_by="reviewer:1")
    assert store.get("tools", "request").status == "expired"
    assert store.pending("tools", "thread", "run") == []
    events = store.events("tools", "request")
    assert events[-1].status == "expired"
    clock[0] = 100
    assert store.get("tools", "request").status == "expired"
    assert store.events("tools", "request") == events


def test_live_authority_is_required_even_after_approval(journal):
    store, _ = journal
    binding = make_binding()
    approve(store, binding)
    scopes = [
        ExecutionScope(**scope_for(binding).to_dict()),
        replace(scope_for(binding), permissions=PermissionSet()),
        replace(scope_for(binding), principal="other"),
    ]
    for scope in scopes:
        with execution_context(scope=scope):
            with pytest.raises(PermissionError):
                store.consume("request", binding=binding)
    assert store.get("tools", "request").status == "approved"
    assert len(store.events("tools", "request")) == 2


def test_invalid_requests_do_not_create_state(journal):
    store, _ = journal
    for deadline in (100, 99):
        with pytest.raises(ApprovalExpiredError):
            store.request(make_binding(), request_id="request", expires_at=deadline)
    for deadline in (float("inf"), float("nan"), True, "200"):
        with pytest.raises((TypeError, ValueError)):
            store.request(make_binding(), request_id="request", expires_at=deadline)
    assert store.get("tools", "request") is None
    assert store.events("tools", "request") == []


def test_binding_canonicalization_and_strict_json():
    source = {"b": [1, {"nested": True}], "a": "x"}
    binding = make_binding(arguments=source)
    assert binding == make_binding(arguments={"a": "x", "b": [1, {"nested": True}]})
    source["b"].append(2)
    assert binding != make_binding(arguments=source)
    for args in ({1: "x"}, {"x": object()}, {"x": (1, 2)}, {"x": float("nan")}):
        with pytest.raises((TypeError, ValueError)):
            make_binding(arguments=args)


@pytest.mark.asyncio
async def test_async_conformance_and_one_use(journal):
    store, _ = journal
    binding = make_binding()
    await store.arequest(binding, request_id="request", expires_at=200)
    assert len(await store.apending("tools", "thread", "run")) == 1
    await store.adecide("tools", "request", approved=True, decided_by="reviewer:1")
    with execution_context(scope=scope_for(binding)):
        results = await asyncio.gather(
            *(store.aconsume("request", binding=binding) for _ in range(8)),
            return_exceptions=True,
        )
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, ApprovalConflictError) for result in results) == 7
    assert (await store.aget("tools", "request")).status == "consumed"
    assert len(await store.aevents("tools", "request")) == 3


def test_sqlite_restart_and_transaction_rollback(tmp_path):
    path = str(tmp_path / "approvals.db")
    store = SQLiteApprovalStore(path, clock=lambda: 100)
    binding = make_binding()
    store.request(binding, request_id="request", expires_at=200)
    store.close()
    store = SQLiteApprovalStore(path, clock=lambda: 100)
    # Fault injection: journal failure must roll back the record update as well.
    store._conn.execute("""CREATE TRIGGER fail_audit BEFORE INSERT ON runtime_approval_events
        BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="audit unavailable"):
        store.decide("tools", "request", approved=True, decided_by="reviewer:1")
    assert store.get("tools", "request").status == "pending"
    assert len(store.events("tools", "request")) == 1
    store._conn.execute("DROP TRIGGER fail_audit")
    approve(store, binding)
    store.close()
    store = SQLiteApprovalStore(path, clock=lambda: 100)
    with execution_context(scope=scope_for(binding)):
        store.consume("request", binding=binding)
    assert len(store.events("tools", "request")) == 3
    store.close()


def _consume_in_process(path, binding, barrier, results):
    store = SQLiteApprovalStore(path, clock=lambda: 100)
    try:
        barrier.wait(timeout=10)
        with execution_context(scope=scope_for(binding)):
            try:
                store.consume("request", binding=binding)
                results.put("consumed")
            except ApprovalConflictError:
                results.put("conflict")
    finally:
        store.close()


def test_sqlite_independent_processes_cannot_consume_twice(tmp_path):
    path = str(tmp_path / "approvals.db")
    binding = make_binding()
    store = SQLiteApprovalStore(path, clock=lambda: 100)
    approve(store, binding)
    store.close()
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_consume_in_process, args=(path, binding, barrier, results)
        )
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        assert sorted(results.get(timeout=20) for _ in processes) == [
            "conflict",
            "consumed",
        ]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
    store = SQLiteApprovalStore(path, clock=lambda: 100)
    assert store.get("tools", "request").status == "consumed"
    assert len(store.events("tools", "request")) == 3
    store.close()


def test_sqlite_competing_decisions(tmp_path):
    path = str(tmp_path / "approvals.db")
    stores = [SQLiteApprovalStore(path, clock=lambda: 100) for _ in range(2)]
    stores[0].request(make_binding(), request_id="request", expires_at=200)

    def decide(index):
        try:
            return stores[index].decide(
                "tools", "request", approved=bool(index), decided_by=f"reviewer:{index}"
            )
        except ApprovalConflictError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(decide, range(2)))
    assert sum(isinstance(result, ApprovalConflictError) for result in results) == 1
    assert len(stores[0].events("tools", "request")) == 2
    for store in stores:
        store.close()
