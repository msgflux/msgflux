"""Restart through the authenticated daemon, with real independent processes."""

import asyncio
import os
import subprocess
import sys

import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="POSIX daemon lifetime locks"
)

from msgflux.runtime.service.local.discovery import (
    connect_local_service,
    restart_local_service,
)
from msgflux.runtime.service.local.files import process_lock, read_record
from tests.test_local_service_integration import (
    FACTORY,
    daemon_workspace,
    _until,
    _wait_completed,
)


def _delay_factory_cleanup(project):
    path = project / "daemon_fixture.py"
    source = path.read_text()
    original = "        return AgentSession(agent, on_close=checkpoints.close)"
    replacement = "\n".join(
        [
            "        async def close():",
            "            while not (runtime_dir / 'release').exists():",
            "                await asyncio.sleep(0.02)",
            "            checkpoints.close()",
            "        return AgentSession(agent, on_close=close)",
        ]
    )
    assert original in source
    path.write_text(source.replace(original, replacement))


@pytest.mark.asyncio
async def test_restart_reloads_factory_and_preserves_history(daemon_workspace):
    project, _b, _c, runtime = daemon_workspace
    first = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        old = read_record(runtime)
        thread = await first.open_thread(
            "main", thread_id="restart-history", cwd=project
        )
        receipt = await first.prompt(thread.thread_id, "hello", request_id="before")
        assert (
            await _wait_completed(first, thread.thread_id, "before")
        ).status == "completed"
        factory_file = project / "daemon_fixture.py"
        source = factory_file.read_text()
        factory_file.write_text(
            source.replace("daemon response: ", "updated daemon response: ")
        )
        replacement = await restart_local_service(runtime_dir=runtime)
        try:
            new = read_record(runtime)
            assert new.instance_id != old.instance_id
            assert new.token != old.token
            assert new.factory == old.factory and new.cwd == old.cwd
            assert "daemon response: hello" in str(
                (await replacement.snapshot(thread.thread_id)).messages
            )
            assert (
                await replacement.receipt(thread.thread_id, "before")
            ).run_id == receipt.run_id
            await replacement.prompt(thread.thread_id, "after", request_id="after")
            assert (
                await _wait_completed(replacement, thread.thread_id, "after")
            ).status == "completed"
            assert "updated daemon response: after" in str(
                (await replacement.snapshot(thread.thread_id)).messages
            )
            assert (runtime / "boots").read_text().splitlines() == ["boot", "boot"]
        finally:
            await replacement.aclose()
    finally:
        await first.aclose()


@pytest.mark.asyncio
async def test_cli_restart_returns_and_leaves_replacement_running(daemon_workspace):
    project, other, _c, runtime = daemon_workspace
    first = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    old = read_record(runtime)
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "-m",
                "msgflux.runtime.service.local.cli",
                "restart",
                "--runtime-dir",
                str(runtime),
            ],
            cwd=other,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        current = read_record(runtime)
        assert current.instance_id != old.instance_id
        assert current.cwd == str(project)
        assert old.token not in result.stdout and current.token not in result.stdout
        assert "Restarted AgentService" in result.stdout
        client = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=other)
        try:
            assert (await client.health()).instance_id == current.instance_id
            assert (runtime / "boots").read_text().splitlines() == ["boot", "boot"]
        finally:
            await client.aclose()
    finally:
        await first.aclose()


@pytest.mark.asyncio
async def test_restart_drains_owned_run_and_concurrent_connect_reuses_replacement(
    daemon_workspace,
):
    project, _b, _c, runtime = daemon_workspace
    _delay_factory_cleanup(project)
    first = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    restarting = connecting = None
    try:
        thread = await first.open_thread("main", thread_id="restart-drain")
        await first.prompt(thread.thread_id, "block", request_id="blocked")
        await _until((runtime / "model_calls").exists)
        restarting = asyncio.create_task(restart_local_service(runtime_dir=runtime))
        await asyncio.sleep(0.2)
        assert not restarting.done()
        connecting = asyncio.create_task(
            connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
        )
        await asyncio.sleep(0.1)
        assert not connecting.done()
        assert (runtime / "boots").read_text().splitlines() == ["boot"]
        (runtime / "release").touch()
        replacement, concurrent = await asyncio.wait_for(
            asyncio.gather(restarting, connecting), 15
        )
        try:
            assert (await replacement.health()).instance_id == (
                await concurrent.health()
            ).instance_id
            settled = await replacement.receipt(thread.thread_id, "blocked")
            assert settled.status == "interrupted"
            assert "block" in str(
                (await replacement.snapshot(thread.thread_id)).messages
            )
            assert (runtime / "model_calls").read_text().splitlines() == ["call"]
            assert (runtime / "boots").read_text().splitlines() == ["boot", "boot"]
        finally:
            await replacement.aclose()
            await concurrent.aclose()
    finally:
        (runtime / "release").touch()
        for task in (restarting, connecting):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await first.aclose()


@pytest.mark.asyncio
async def test_restart_timeout_does_not_force_stop_or_start_another_owner(
    daemon_workspace,
):
    project, _b, _c, runtime = daemon_workspace
    _delay_factory_cleanup(project)
    first = await connect_local_service(FACTORY, runtime_dir=runtime, cwd=project)
    try:
        thread = await first.open_thread("main", thread_id="restart-timeout")
        await first.prompt(thread.thread_id, "block", request_id="blocked")
        await _until((runtime / "model_calls").exists)
        with pytest.raises(TimeoutError):
            await restart_local_service(runtime_dir=runtime, restart_timeout=0.5)
        with process_lock(runtime / "daemon.lock") as lock:
            assert lock is None
        assert (runtime / "boots").read_text().splitlines() == ["boot"]
        (runtime / "release").touch()

        def stopped():
            with process_lock(runtime / "daemon.lock") as lock:
                return lock is not None

        await _until(stopped)
        replacement = await connect_local_service(
            FACTORY, runtime_dir=runtime, cwd=project
        )
        try:
            assert (
                await replacement.receipt(thread.thread_id, "blocked")
            ).status == "interrupted"
            assert (runtime / "boots").read_text().splitlines() == ["boot", "boot"]
        finally:
            await replacement.aclose()
    finally:
        (runtime / "release").touch()
        await first.aclose()
