"""Live activity listings through CLI subprocesses and a real local daemon."""

import json
import os

import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX local daemon")

from msgflux.runtime.service import SQLiteServiceStore
from msgflux.runtime.service.local import connect_local_service
from msgflux.runtime.service.local.cli import main
from tests.test_local_service_integration import (
    FACTORY,
    daemon_workspace,
    _until,
    _wait_completed,
)
from tests.test_local_service_operations import _command


@pytest.mark.parametrize(
    "resource,args",
    [
        ("agents", ["ls"]),
        ("threads", ["ls"]),
        ("runs", ["ls", "--active", "--agent", "coding"]),
    ],
)
@pytest.mark.asyncio
async def test_missing_daemon_listings_do_not_start_service(tmp_path, resource, args):
    runtime = tmp_path / "absent"
    code, out, err = await _command(runtime, resource, *args)
    assert code == 1 and not out and "stopped" in err
    assert not runtime.exists()


@pytest.mark.parametrize(
    "resource,args,message",
    [
        ("agents", ["ls", "thread"], "does not accept thread"),
        ("threads", ["ls", "thread"], "does not accept thread"),
        ("agents", ["ls", "--agent", "coding"], "only accepted with runs ls"),
        ("threads", ["stop", "t", "--active"], "only accepted with runs ls"),
    ],
)
def test_listing_arguments_are_validated(resource, args, message):
    with pytest.raises(SystemExit, match=message):
        main(args, resource=resource)


@pytest.mark.asyncio
async def test_registered_agents_and_threads_use_live_workers(daemon_workspace):
    project, _b, _c, runtime = daemon_workspace
    factory = project / "daemon_fixture.py"
    factory.write_text(
        factory.read_text()
        .replace(
            "    def create_agent(thread):",
            "    def create_agent(thread):\n"
            "        with (runtime_dir / 'session_creations').open('a') as stream:\n"
            "            stream.write('session\\n')",
        )
        .replace(
            '    service.register("main", create_agent)',
            '    service.register("coding", create_agent)\n    service.register("explorer", create_agent)',
        )
    )
    client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        code, out, err = await _command(runtime, "agents", "ls")
        assert code == 0 and not err
        assert json.loads(out)["agents"] == [
            {"agent_id": "coding", "active_runs": 0, "state": "idle"},
            {"agent_id": "explorer", "active_runs": 0, "state": "idle"},
        ]
        assert not (runtime / "model_calls").exists()
        assert not (runtime / "session_creations").exists()
        assert (
            json.loads((await _command(runtime, "threads", "ls"))[1])["threads"] == []
        )
        a = await client.open_thread("coding", thread_id="a", cwd=project)
        b = await client.open_thread("coding", thread_id="b", cwd=project)
        c = await client.open_thread("explorer", thread_id="c")
        orphan = await client.open_thread("explorer", thread_id="orphan")
        journal = SQLiteServiceStore(runtime / "service.sqlite3")
        try:
            stale = journal.admit(orphan.thread_id, "old", "block", "main")
            assert journal.claim(stale, "former-owner")
        finally:
            journal.close()
        ar = await client.prompt(a.thread_id, "block", request_id="a")
        br = await client.prompt(b.thread_id, "block", request_id="b")
        await _until(
            lambda: (
                (runtime / "model_calls").exists()
                and len((runtime / "model_calls").read_text().splitlines()) == 2
            )
        )
        code, out, err = await _command(runtime, "agents", "ls")
        assert code == 0 and not err
        assert json.loads(out)["agents"] == [
            {"agent_id": "coding", "active_runs": 2, "state": "running"},
            {"agent_id": "explorer", "active_runs": 0, "state": "idle"},
        ]
        code, out, err = await _command(
            runtime, "agents", "ls", "--active", as_json=False
        )
        assert code == 0 and not err
        assert "ACTIVE_RUNS" in out and "coding" in out and "explorer" not in out
        code, out, err = await _command(runtime, "threads", "ls")
        assert code == 0 and not err
        rows = {row["thread_id"]: row for row in json.loads(out)["threads"]}
        assert rows[a.thread_id]["cwd"] == str(project)
        assert (
            rows[a.thread_id]["active_runs"] == 1
            and rows[b.thread_id]["state"] == "running"
        )
        assert (
            rows[c.thread_id]["state"] == "idle"
            and rows[orphan.thread_id]["active_runs"] == 0
        )
        code, out, err = await _command(runtime, "threads", "ls", "--active")
        assert code == 0 and not err
        assert {row["thread_id"] for row in json.loads(out)["threads"]} == {
            a.thread_id,
            b.thread_id,
        }
        code, out, err = await _command(
            runtime, "runs", "ls", "--active", "--agent", "coding"
        )
        assert code == 0 and not err
        assert {row["run_id"] for row in json.loads(out)["runs"]} == {
            ar.run_id,
            br.run_id,
        }
        code, out, err = await _command(
            runtime,
            "runs",
            "ls",
            a.thread_id,
            "--active",
            "--agent",
            "coding",
            as_json=False,
        )
        assert code == 0 and not err and ar.run_id in out and br.run_id not in out
        code, out, err = await _command(
            runtime, "runs", "ls", "--active", "--agent", "explorer"
        )
        assert code == 0 and not err and json.loads(out)["runs"] == []
        code, out, err = await _command(
            runtime, "runs", "ls", "--active", "--agent", "missing"
        )
        assert code == 1 and not out and "Unknown agent" in err
        await client.interrupt(a.thread_id, ar.run_id)
        await client.interrupt(b.thread_id, br.run_id)
        assert (await _wait_completed(client, a.thread_id, "a")).status == "interrupted"
        assert (await _wait_completed(client, b.thread_id, "b")).status == "interrupted"
        code, out, err = await _command(runtime, "agents", "ls", "--active")
        assert code == 0 and not err and json.loads(out)["agents"] == []
        assert (runtime / "boots").read_text().splitlines() == ["boot"]
        assert len((runtime / "model_calls").read_text().splitlines()) == 2
        assert len((runtime / "session_creations").read_text().splitlines()) == 2
    finally:
        await client.aclose()


def test_agent_help_shows_listing_options_without_server_launch_flags(capsys):
    with pytest.raises(SystemExit) as exited:
        main(["--help"], resource="agents")
    assert exited.value.code == 0
    help_text = capsys.readouterr().out
    assert "--active" in help_text and "--runtime-dir" in help_text
    for irrelevant in (
        "--factory",
        "--config",
        "--stop-timeout",
        "--agent",
        "--model",
        "thread_id",
        "run_id",
    ):
        assert irrelevant not in help_text
