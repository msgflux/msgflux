"""An old daemon lease must prove quiescence across journal connections."""

import multiprocessing
import os
import signal
from pathlib import Path

import pytest

from msgflux.runtime.service import AgentService, SQLiteServiceStore
from msgflux.runtime.service.local.leases import LocalOwnerLease
from msgflux.runtime.service.records import ServiceRecoveryRequiredError

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX lifetime locks")


def _owner_process(path, pipe):
    store = SQLiteServiceStore(path)
    service = AgentService(store=store)
    lease = LocalOwnerLease(service)
    try:
        pipe.send(service._owner_id)
        pipe.recv()
    finally:
        lease.close()
        store.close()
        pipe.close()


@pytest.fixture
def service_owners(tmp_path):
    path = tmp_path / "service.sqlite3"
    stores = [SQLiteServiceStore(path), SQLiteServiceStore(path)]
    services = [AgentService(store=store) for store in stores]
    leases = [LocalOwnerLease(service) for service in services]
    try:
        yield services, leases
    finally:
        for lease in leases:
            lease.close()
        for store in stores:
            store.close()


def test_independent_owners_share_journal_but_live_owner_cannot_be_reclaimed(
    service_owners,
):
    services, leases = service_owners
    with pytest.raises(ServiceRecoveryRequiredError, match="still held"):
        with leases[1].reclaim(services[0]._owner_id):
            pytest.fail("a live owner must remain fenced")


def test_closed_owner_can_be_reclaimed_and_lock_remains_for_future_recovery(
    service_owners,
):
    services, leases = service_owners
    path = Path(services[0].store.owner_lease(services[0]._owner_id)[0])
    leases[0].close()
    leases[0].close()
    with leases[1].reclaim(services[0]._owner_id):
        assert path.is_file()
        # A concurrent recovery cannot use the same quiescence evidence.
        with pytest.raises(ServiceRecoveryRequiredError, match="still held"):
            with leases[1].reclaim(services[0]._owner_id):
                pytest.fail("recovery must retain the old owner's lock")
    with leases[1].reclaim(services[0]._owner_id):
        assert path.is_file()


@pytest.mark.parametrize("replacement", ["missing", "replaced", "symlink", "mode"])
def test_missing_or_changed_owner_file_never_proves_quiescence(
    service_owners, replacement
):
    services, leases = service_owners
    path = Path(services[0].store.owner_lease(services[0]._owner_id)[0])
    leases[0].close()
    if replacement == "mode":
        path.chmod(0o644)
    else:
        saved = path.with_suffix(".old")
        path.rename(saved)
        if replacement == "replaced":
            path.touch(mode=0o600)
        elif replacement == "symlink":
            path.symlink_to(saved)
    with pytest.raises(ServiceRecoveryRequiredError):
        with leases[1].reclaim(services[0]._owner_id):
            pytest.fail("changed evidence must fail closed")
    if replacement == "missing":
        assert not path.exists()


def test_unregistered_embedded_owner_cannot_be_reclaimed(service_owners):
    services, leases = service_owners
    embedded = AgentService(store=services[0].store)
    with pytest.raises(ServiceRecoveryRequiredError, match="no registered lifetime"):
        with leases[1].reclaim(embedded._owner_id):
            pytest.fail("embedded ownership has no lifetime evidence")


def test_closed_current_lease_cannot_authorize_recovery(service_owners):
    services, leases = service_owners
    leases[0].close()
    leases[1].close()
    with pytest.raises(ServiceRecoveryRequiredError, match="lease is closed"):
        with leases[1].reclaim(services[0]._owner_id):
            pytest.fail("a closed owner cannot adopt work")


def test_recovery_body_error_is_not_misreported_as_owner_lease_failure(service_owners):
    services, leases = service_owners
    leases[0].close()
    failure = OSError("workspace validation failed")
    with pytest.raises(OSError) as caught:
        with leases[1].reclaim(services[0]._owner_id):
            raise failure
    assert caught.value is failure
    with leases[1].reclaim(services[0]._owner_id):
        pass


def test_memory_journal_cannot_publish_restart_evidence():
    store = SQLiteServiceStore()
    try:
        with pytest.raises(ValueError, match="persistent journal"):
            LocalOwnerLease(AgentService(store=store))
    finally:
        store.close()


def test_registration_failure_releases_owner_lock(tmp_path, monkeypatch):
    import fcntl

    store = SQLiteServiceStore(tmp_path / "service.sqlite3")
    service = AgentService(store=store)

    def fail_registration(*args, **kwargs):
        raise OSError("cannot persist lease evidence")

    monkeypatch.setattr(store, "register_owner_lease", fail_registration)
    try:
        with pytest.raises(OSError, match="cannot persist"):
            LocalOwnerLease(service)
        path = tmp_path / ".service.sqlite3.owners" / f"{service._owner_id}.lock"
        with path.open("r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert store.owner_lease(service._owner_id) is None
    finally:
        store.close()


def test_sigkill_releases_registered_owner_lease_without_cleanup(tmp_path):
    path = tmp_path / "service.sqlite3"
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_owner_process, args=(str(path), child))
    process.start()
    child.close()
    store = None
    lease = None
    try:
        assert parent.poll(15), "owner process did not register its lease"
        old_owner = parent.recv()
        store = SQLiteServiceStore(path)
        lease = LocalOwnerLease(AgentService(store=store))
        with pytest.raises(ServiceRecoveryRequiredError, match="still held"):
            with lease.reclaim(old_owner):
                pytest.fail("the owner process is still alive")
        os.kill(process.pid, signal.SIGKILL)
        process.join(10)
        assert process.exitcode == -signal.SIGKILL
        with lease.reclaim(old_owner):
            assert Path(store.owner_lease(old_owner)[0]).is_file()
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()
        if lease is not None:
            lease.close()
        if store is not None:
            store.close()
