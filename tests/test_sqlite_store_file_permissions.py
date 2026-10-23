"""Direct SQLite adapters create private databases without changing caller modes."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_STORE_NAMES = ("checkpoint", "task", "inbox", "approval", "service", "workspace")


def _collect_file_modes(root: Path, mask: int = 0):
    # This function runs only in a child process: never change pytest's umask.
    from msgflux.data.stores import SQLiteCheckpointStore
    from msgflux.runtime.agent_inbox.providers.sqlite import SQLiteAgentInboxStore
    from msgflux.runtime.approvals.providers.sqlite import SQLiteApprovalStore
    from msgflux.runtime.service import SQLiteServiceStore
    from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
    from msgflux.tasks.providers.sqlite import SQLiteTaskStore

    factories = (
        SQLiteCheckpointStore,
        SQLiteTaskStore,
        SQLiteAgentInboxStore,
        SQLiteApprovalStore,
        SQLiteServiceStore,
        SQLiteWorkspaceRegistry,
    )
    os.chdir(root)
    root.chmod(0o755)
    os.umask(mask)
    observations = {}
    for name, factory in zip(_STORE_NAMES, factories):
        path = root / (name + ".sqlite3")
        store = factory(str(path))
        try:
            connection = getattr(store, "_conn", None)
            if connection is None:
                connection = store._connection
            connection.execute("CREATE TABLE private_probe (value TEXT)")
            connection.execute("INSERT INTO private_probe VALUES ('synthetic history')")
            connection.commit()
            modes = {
                candidate.name: stat.S_IMODE(candidate.stat().st_mode)
                for suffix in ("", "-wal", "-shm")
                if (candidate := Path(str(path) + suffix)).exists()
            }
        finally:
            store.close()

        path.chmod(0o640)
        inode = path.stat().st_ino
        reopened = factory(str(path))
        try:
            connection = getattr(reopened, "_conn", None)
            if connection is None:
                connection = reopened._connection
            observations[name] = {
                "new_modes": modes,
                "existing_mode": stat.S_IMODE(path.stat().st_mode),
                "same_inode": path.stat().st_ino == inode,
                "parent_mode": stat.S_IMODE(root.stat().st_mode),
                "value": connection.execute(
                    "SELECT value FROM private_probe"
                ).fetchone()[0],
            }
        finally:
            reopened.close()

        trailing = root / ("trailing-" + name + ".sqlite3")
        store = factory(str(trailing) + os.sep)
        store.close()
        observations[name]["trailing_mode"] = stat.S_IMODE(trailing.stat().st_mode)

        memory = factory(":memory:")
        memory.close()
    assert not (root / ":memory:").exists()
    before_temporary = set(root.iterdir())
    temporary = SQLiteCheckpointStore("")
    temporary.close()
    assert set(root.iterdir()) == before_temporary
    return observations


@pytest.fixture(scope="module", params=[0, 0o777], ids=["permissive", "owner-masked"])
def direct_store_modes(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("direct-sqlite-modes")
    code = (
        "import json, sys; from pathlib import Path; "
        "from tests.test_sqlite_store_file_permissions import _collect_file_modes; "
        "print(json.dumps(_collect_file_modes(Path(sys.argv[1]), int(sys.argv[2]))))"
    )
    # Run only this test module with the current interpreter and a pytest temp path.
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, str(root), str(request.param)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
        env={
            **os.environ,
            "MSGTRACE_TELEMETRY_ENABLED": "false",
            "MSGFLUX_CONFIGURE_LOGGING": "false",
        },
    )
    return json.loads(result.stdout)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permission bits")
@pytest.mark.parametrize("store_name", _STORE_NAMES)
def test_new_databases_and_sidecars_are_private(direct_store_modes, store_name):
    modes = direct_store_modes[store_name]["new_modes"]
    assert modes
    assert all(mode == 0o600 for mode in modes.values()), modes


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permission bits")
@pytest.mark.parametrize("store_name", _STORE_NAMES)
def test_reopening_preserves_caller_permissions_and_data(
    direct_store_modes, store_name
):
    result = direct_store_modes[store_name]
    assert result["existing_mode"] == 0o640
    assert result["same_inode"]
    assert result["parent_mode"] == 0o755
    assert result["value"] == "synthetic history"


def test_service_missing_parent_preserves_sqlite_error_contract(tmp_path):
    import sqlite3
    from msgflux.runtime.service import SQLiteServiceStore

    path = tmp_path / "missing" / "service.sqlite3"
    with pytest.raises(sqlite3.OperationalError) as raised:
        SQLiteServiceStore(path)
    assert raised.value.sqlite_errorcode == sqlite3.SQLITE_CANTOPEN
    assert isinstance(raised.value.__cause__, FileNotFoundError)
    assert not path.parent.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink behavior")
def test_direct_adapter_preserves_existing_symlink_path(tmp_path):
    from msgflux.data.stores import SQLiteCheckpointStore

    target = tmp_path / "target.sqlite3"
    link = tmp_path / "link.sqlite3"
    link.symlink_to(target.name)
    store = SQLiteCheckpointStore(str(link))
    try:
        assert link.is_symlink()
        assert target.is_file()
    finally:
        store.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permission bits")
@pytest.mark.parametrize("store_name", _STORE_NAMES)
def test_trailing_separator_keeps_sqlite_filename_behavior(
    direct_store_modes, store_name
):
    assert direct_store_modes[store_name]["trailing_mode"] == 0o600
