"""Local daemon integration across independent frontend processes."""

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import msgspec
import pytest
import pytest_asyncio

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Local daemon uses POSIX locks"
)

from msgflux.runtime.service.http import AgentServiceClient
from msgflux.runtime.service.local.discovery import connect_local_service
from msgflux.runtime.service.local.files import process_lock, read_record, write_record
from msgflux.runtime.service.local.records import LocalServiceRecord

FACTORY = "daemon_fixture:create_service"
_FACTORY_CODE = """
import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, Mock
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentWorkspace
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.tools.builtin import BashTool


def create_service(runtime_dir):
    with (runtime_dir / "boots").open("a") as stream:
        stream.write("boot\\n")
    service = AgentService(store=SQLiteServiceStore(runtime_dir / "service.sqlite3"))

    def create_agent(thread):
        thread_id = thread.thread_id
        checkpoints = SQLiteCheckpointStore(
            str(runtime_dir / (hashlib.sha256(thread_id.encode()).hexdigest() + ".sqlite3"))
        )
        model = Mock()
        model.model_type = "chat_completion"
        workspace = AgentWorkspace.local(thread.cwd) if thread.cwd is not None else None
        agent = Agent(
            name="main", model=model, tools=[BashTool()],
            checkpoint_store=checkpoints, workspace=workspace,
        )

        async def respond(**kwargs):
            with (runtime_dir / "model_calls").open("a") as stream:
                stream.write("call\\n")
            prompt = kwargs["messages"].to_chatml()[-1]["content"]
            if prompt.startswith("workspace:"):
                value = prompt.split(":", 1)[1]
                calls = ToolCallAggregator()
                calls.process(
                    0, "workspace-call", "bash",
                    json.dumps({"command": f"printf '%s' '{value}' > marker.txt; cat marker.txt"}),
                )
                response = ModelResponse()
                response.set_response_type("tool_call")
                response.add(calls)
                return response
            if prompt == "block":
                while not (runtime_dir / "release").exists():
                    get_execution_scope().abort_signal.raise_if_aborted()
                    await asyncio.sleep(0.02)
            response = ModelResponse()
            response.set_response_type("text_generation")
            response.add("daemon response: " + prompt)
            return response

        agent.generator.aforward = AsyncMock(side_effect=respond)
        return AgentSession(agent, on_close=checkpoints.close)

    service.register("main", create_agent)
    return service
"""


async def _until(predicate, timeout=8):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(wait(), timeout)


async def _shutdown_record(runtime_dir):
    record = read_record(runtime_dir)
    if record is None:
        return
    client = AgentServiceClient(record.url, token=record.token, timeout=1)
    try:
        health = await client.health()
        assert health.instance_id == record.instance_id
        # This fixture started the server and verified its current identity.
        os.kill(record.pid, signal.SIGTERM)
    finally:
        await client.aclose()

    def stopped():
        if read_record(runtime_dir) is not None:
            return False
        with process_lock(runtime_dir / "daemon.lock") as handle:
            return handle is not None

    await _until(stopped)


@pytest_asyncio.fixture
async def daemon_workspace(tmp_path):
    project = tmp_path / "project-a"
    project.mkdir()
    project_b = tmp_path / "project-b"
    project_b.mkdir()
    project_c = tmp_path / "project-c"
    project_c.mkdir()
    for frontend in (project, project_b, project_c):
        (frontend / "daemon_fixture.py").write_text(_FACTORY_CODE)
    runtime_dir = tmp_path / "runtime"
    yield project, project_b, project_c, runtime_dir
    if runtime_dir.exists():
        await _shutdown_record(runtime_dir)


def _starter(project, runtime_dir):
    code = """
import asyncio, json, sys
from msgflux.runtime.service.local.discovery import connect_local_service
async def main():
    client = await connect_local_service("daemon_fixture:create_service", runtime_dir=sys.argv[1], startup_timeout=15)
    try:
        print(json.dumps({"instance_id": (await client.health()).instance_id}), flush=True)
    finally:
        await client.aclose()
asyncio.run(main())
"""
    return subprocess.Popen(  # noqa: S603 - fixture code and paths
        [sys.executable, "-c", code, str(runtime_dir)],
        cwd=project,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


async def _finish_starter(process):
    try:
        stdout, stderr = await asyncio.to_thread(process.communicate, timeout=25)
        assert process.returncode == 0, stderr
        return json.loads(stdout.strip())
    finally:
        if process.poll() is None:
            process.kill()
            await asyncio.to_thread(process.wait)


async def _wait_completed(client, thread_id, request_id, timeout=8):
    async def wait():
        while True:
            receipt = await client.receipt(thread_id, request_id)
            if receipt.status not in {"accepted", "running"}:
                return receipt
            await asyncio.sleep(0.02)

    return await asyncio.wait_for(wait(), timeout)


@pytest.mark.asyncio
async def test_concurrent_starter_processes_exit_and_leave_one_reusable_daemon(
    daemon_workspace,
):
    project, project_b, _project_c, runtime_dir = daemon_workspace
    starters = [_starter(project, runtime_dir) for _ in range(2)]
    identities = await asyncio.gather(
        *(_finish_starter(process) for process in starters)
    )
    assert identities[0] == identities[1]
    assert (runtime_dir / "boots").read_text().splitlines() == ["boot"]
    record = read_record(runtime_dir)
    assert record.instance_id == identities[0]["instance_id"]
    assert runtime_dir.stat().st_mode & 0o077 == 0
    assert (runtime_dir / "daemon.json").stat().st_mode & 0o077 == 0
    # Both launchers ended; a fresh process can still reach the same backend.
    third = await _finish_starter(_starter(project_b, runtime_dir))
    assert third == identities[0]
    assert (runtime_dir / "boots").read_text().splitlines() == ["boot"]


@pytest.mark.asyncio
async def test_detached_client_run_survives_and_history_persists_across_restart(
    daemon_workspace,
):
    project, _project_b, _project_c, runtime_dir = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime_dir, cwd=project)
    old = read_record(runtime_dir)
    try:
        thread = await client.open_thread("main", thread_id="durable-local")
        async with client.watch(thread.thread_id) as observer:
            receipt = await client.prompt(thread.thread_id, "block", request_id="one")
            assert (await asyncio.wait_for(anext(observer), 3)).type == "run.start"
        await client.aclose()
        other = await connect_local_service(
            FACTORY, runtime_dir=runtime_dir, cwd=project
        )
        try:
            assert (await other.health()).instance_id == old.instance_id
            (runtime_dir / "release").touch()

            async def settled():
                while (await other.receipt(thread.thread_id, "one")).status in {
                    "accepted",
                    "running",
                }:
                    await asyncio.sleep(0.02)

            await asyncio.wait_for(settled(), 5)
        finally:
            await other.aclose()
        await _shutdown_record(runtime_dir)
        restarted = await connect_local_service(
            FACTORY, runtime_dir=runtime_dir, cwd=project
        )
        try:
            assert (await restarted.health()).instance_id != old.instance_id
            snapshot = await restarted.snapshot(thread.thread_id)
            assert "daemon response: block" in str(snapshot.messages)
            duplicate = await restarted.prompt(
                thread.thread_id, "block", request_id="one"
            )
            assert (
                duplicate.run_id == receipt.run_id and duplicate.status == "completed"
            )
            assert (runtime_dir / "model_calls").read_text().splitlines() == ["call"]
        finally:
            await restarted.aclose()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_reused_daemon_binds_and_restores_distinct_thread_workspaces(
    daemon_workspace,
):
    project_a, project_b, project_c, runtime_dir = daemon_workspace
    original_process_cwd = Path.cwd()
    client_a = await connect_local_service(
        FACTORY, runtime_dir=runtime_dir, cwd=project_a
    )
    first_record = read_record(runtime_dir)
    client_b = await connect_local_service(
        FACTORY, runtime_dir=runtime_dir, cwd=project_b
    )
    try:
        assert (await client_b.health()).instance_id == first_record.instance_id
        assert read_record(runtime_dir) == first_record
        assert (runtime_dir / "boots").read_text().splitlines() == ["boot"]

        thread_a = await client_a.open_thread(
            "main", thread_id="workspace-a", cwd=project_a
        )
        thread_b = await client_b.open_thread(
            "main", thread_id="workspace-b", cwd=project_b
        )
        assert thread_a.cwd == str(project_a.resolve())
        assert thread_b.cwd == str(project_b.resolve())
        await client_a.prompt(
            thread_a.thread_id, "workspace:alpha", request_id="write-a"
        )
        await client_b.prompt(
            thread_b.thread_id, "workspace:beta", request_id="write-b"
        )
        result_a, result_b = await asyncio.gather(
            _wait_completed(client_a, thread_a.thread_id, "write-a"),
            _wait_completed(client_b, thread_b.thread_id, "write-b"),
        )
        assert result_a.status == result_b.status == "completed"
        assert (project_a / "marker.txt").read_text() == "alpha"
        assert (project_b / "marker.txt").read_text() == "beta"
        assert not (project_a / "marker.txt").samefile(project_b / "marker.txt")
        assert Path.cwd() == original_process_cwd
        assert read_record(runtime_dir).cwd == str(project_a.resolve())
    finally:
        await client_a.aclose()
        await client_b.aclose()

    await _shutdown_record(runtime_dir)
    # This frontend becomes the new daemon bootstrap cwd. An omitted cwd on
    # the existing thread must still recover its persisted project root.
    restarted = await connect_local_service(
        FACTORY, runtime_dir=runtime_dir, cwd=project_c
    )
    try:
        assert read_record(runtime_dir).cwd == str(project_c.resolve())
        restored_a = await restarted.open_thread("main", thread_id="workspace-a")
        restored_b = await restarted.open_thread("main", thread_id="workspace-b")
        assert restored_a.cwd == str(project_a.resolve())
        assert restored_b.cwd == str(project_b.resolve())
        await restarted.prompt(
            restored_a.thread_id, "workspace:after-restart", request_id="write-again"
        )
        final = await _wait_completed(restarted, restored_a.thread_id, "write-again")
        assert final.status == "completed"
        assert (project_a / "marker.txt").read_text() == "after-restart"
        assert (project_b / "marker.txt").read_text() == "beta"
        assert Path.cwd() == original_process_cwd
    finally:
        await restarted.aclose()


@pytest.mark.asyncio
async def test_stale_pid_record_does_not_signal_unrelated_process(daemon_workspace):
    project, _project_b, _project_c, runtime_dir = daemon_workspace
    runtime_dir.mkdir(mode=0o700)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    stale = LocalServiceRecord(
        instance_id="stale",
        factory=FACTORY,
        cwd=str(project.resolve()),
        url="http://127.0.0.1:1",
        pid=unrelated.pid,
        token="stale-secret",
    )
    write_record(runtime_dir, stale)
    try:
        client = await connect_local_service(
            FACTORY, runtime_dir=runtime_dir, cwd=project
        )
        try:
            assert (await client.health()).instance_id != stale.instance_id
            assert unrelated.poll() is None
        finally:
            await client.aclose()
    finally:
        unrelated.terminate()
        await asyncio.to_thread(unrelated.wait)


@pytest.mark.asyncio
async def test_unreachable_live_owner_is_not_replaced_or_signaled(daemon_workspace):
    project, _project_b, _project_c, runtime_dir = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime_dir, cwd=project)
    original = read_record(runtime_dir)
    await client.aclose()
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        write_record(
            runtime_dir,
            msgspec.structs.replace(
                original, url="http://127.0.0.1:1", pid=unrelated.pid
            ),
        )
        with pytest.raises(RuntimeError):
            await connect_local_service(
                FACTORY, runtime_dir=runtime_dir, cwd=project, startup_timeout=2
            )
        assert unrelated.poll() is None
        assert (runtime_dir / "boots").read_text().splitlines() == ["boot"]
    finally:
        write_record(runtime_dir, original)
        unrelated.terminate()
        await asyncio.to_thread(unrelated.wait)


@pytest.mark.asyncio
async def test_factory_mismatch_preserves_current_runtime(daemon_workspace):
    project, project_b, _project_c, runtime_dir = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime_dir, cwd=project)
    original = read_record(runtime_dir)
    try:
        with pytest.raises(RuntimeError):
            await connect_local_service(
                "daemon_fixture:other", runtime_dir=runtime_dir, cwd=project
            )
        other_frontend = await connect_local_service(
            FACTORY, runtime_dir=runtime_dir, cwd=project_b
        )
        try:
            assert (await other_frontend.health()).instance_id == original.instance_id
        finally:
            await other_frontend.aclose()
        assert (await client.health()).instance_id == original.instance_id
        assert read_record(runtime_dir) == original
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_explicit_stop_with_live_sse_cleans_runtime_and_settles_owned_run(
    daemon_workspace,
):
    project, _project_b, _project_c, runtime_dir = daemon_workspace
    client = await connect_local_service(FACTORY, runtime_dir=runtime_dir, cwd=project)
    original = read_record(runtime_dir)
    try:
        thread = await client.open_thread("main", thread_id="shutdown-active")
        async with client.watch(thread.thread_id) as observer:
            receipt = await client.prompt(
                thread.thread_id, "block", request_id="active"
            )
            assert (await asyncio.wait_for(anext(observer), 3)).type == "run.start"
            await _shutdown_record(runtime_dir)
        new = await connect_local_service(FACTORY, runtime_dir=runtime_dir, cwd=project)
        try:
            assert (await new.health()).instance_id != original.instance_id
            restored = await new.receipt(thread.thread_id, "active")
            assert restored.run_id == receipt.run_id
            assert restored.status == "interrupted"
        finally:
            await new.aclose()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_connector_waits_for_foreground_owner_initialization(daemon_workspace):
    project, _project_b, _project_c, runtime_dir = daemon_workspace
    source = project / "daemon_fixture.py"
    source.write_text(
        source.read_text()
        + """
async def create_delayed(runtime_dir):
    (runtime_dir / "initializing").touch()
    await asyncio.sleep(0.5)
    return create_service(runtime_dir)
"""
    )
    factory = "daemon_fixture:create_delayed"
    foreground = subprocess.Popen(  # noqa: S603 - fixture code and paths
        [
            sys.executable,
            "-m",
            "msgflux.runtime.service.local.cli",
            "--factory",
            factory,
            "--runtime-dir",
            str(runtime_dir),
            "--cwd",
            str(project),
        ],
        cwd=project,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        await _until((runtime_dir / "initializing").exists)
        client = await connect_local_service(
            factory, runtime_dir=runtime_dir, cwd=project
        )
        try:
            assert read_record(runtime_dir).pid == foreground.pid
            assert (runtime_dir / "boots").read_text().splitlines() == ["boot"]
        finally:
            await client.aclose()
        await _shutdown_record(runtime_dir)
        await asyncio.to_thread(foreground.wait, timeout=5)
        assert foreground.returncode == 0
    finally:
        if foreground.poll() is None:
            foreground.terminate()
            await asyncio.to_thread(foreground.wait, timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_startup_releases_own_child_and_kernel_lock(
    daemon_workspace, cancel
):
    project, _project_b, _project_c, runtime_dir = daemon_workspace
    source = project / "daemon_fixture.py"
    source.write_text(
        source.read_text()
        + """
async def create_hanging(runtime_dir):
    (runtime_dir / "initializing").touch()
    while True:
        await asyncio.sleep(1)
"""
    )
    operation = asyncio.create_task(
        connect_local_service(
            "daemon_fixture:create_hanging",
            runtime_dir=runtime_dir,
            cwd=project,
            startup_timeout=3 if not cancel else 15,
        )
    )
    if cancel:
        await _until((runtime_dir / "initializing").exists)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        with pytest.raises(TimeoutError):
            await operation
    assert read_record(runtime_dir) is None
    with process_lock(runtime_dir / "daemon.lock") as handle:
        assert handle is not None
    with process_lock(runtime_dir / "startup.lock") as handle:
        assert handle is not None
