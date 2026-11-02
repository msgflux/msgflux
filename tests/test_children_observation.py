"""Compact parent/child observation without hydrating sessions or answers."""

import argparse
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2
import pytest

from msgflux.runtime.service import (
    AdmissionReceipt,
    ChildThread,
    ChildThreadsPage,
    ServiceThread,
)
from msgflux.runtime.service.http.app import create_service_app
from msgflux.runtime.service.http.client import AgentServiceClient
from msgflux.runtime.service.local.children import show_children
from tests.test_coding_cli import coding_cli_project
from tests.test_service_parent_threads import _host


def _child(thread="child", run="run", status="running", request="request"):
    return ChildThread(
        ServiceThread(thread, "worker", parent_thread_id="parent"),
        1,
        AdmissionReceipt(thread, request, run, status) if run else None,
    )


def _page(*children):
    return ChildThreadsPage("parent", children)


def _args(**kwargs):
    return argparse.Namespace(
        thread_id="parent",
        wait=True,
        wait_timeout=1,
        limit=None,
        after_child=None,
        exclude_run=None,
        json=True,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_children_http_pagination_never_calls_factory(tmp_path):
    service, store = _host(tmp_path)
    app = create_service_app(service, token="secret")
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://test"
        ) as http:
            client = AgentServiceClient("http://test", token="secret", client=http)
            await client.open_thread("main", thread_id="parent")
            for name in ("child-a", "child-b", "child-c"):
                await client.open_thread(
                    "main", thread_id=name, parent_thread_id="parent"
                )
            # The projection must not load the stored prompt into its response.
            store.admit("child-a", "request", "PRIVATE-HISTORY-" * 10000, "namespace")
            page = await client.child_threads("parent", limit=2)
            assert [child.thread.thread_id for child in page.children] == [
                "child-a",
                "child-b",
            ]
            assert page.children[0].receipt.status == "accepted"
            assert page.next_cursor is not None
            second = await client.child_threads(
                "parent", limit=2, after=page.next_cursor
            )
            assert [child.thread.thread_id for child in second.children] == ["child-c"]
            assert second.next_cursor is None
            from msgflux.utils.msgspec import msgspec_dumps

            assert "PRIVATE-HISTORY" not in msgspec_dumps(page)
    finally:
        await service.aclose()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "failed", "paused", "interrupted"])
async def test_wait_reports_already_settled_child_without_answer(status, capsys):
    client = SimpleNamespace(
        child_threads=AsyncMock(return_value=_page(_child(status=status)))
    )
    await show_children(client, _args())
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "settled"
    assert result["child"]["status"] == status
    assert "output" not in result["child"]
    assert client.child_threads.await_count == 1


@pytest.mark.asyncio
async def test_wait_observes_transition_and_excludes_handled_run(capsys):
    client = SimpleNamespace(
        child_threads=AsyncMock(
            side_effect=[
                _page(
                    _child(run="handled", status="completed"), _child("other", "next")
                ),
                _page(
                    _child(run="handled", status="completed"),
                    _child("other", "next", "failed"),
                ),
            ]
        )
    )
    args = _args()
    args.exclude_run = ["handled"]
    await show_children(client, args)
    result = json.loads(capsys.readouterr().out)
    assert result["child"]["run_id"] == "next"
    assert result["child"]["status"] == "failed"


@pytest.mark.asyncio
async def test_wait_pins_original_request_when_child_already_has_new_turn(capsys):
    client = SimpleNamespace(
        child_threads=AsyncMock(
            side_effect=[
                _page(_child()),
                _page(_child(run="new", request="new-request")),
            ]
        ),
        receipt=AsyncMock(
            return_value=AdmissionReceipt("child", "request", "run", "completed")
        ),
    )
    await show_children(client, _args())
    result = json.loads(capsys.readouterr().out)
    assert result["child"]["run_id"] == "run"
    client.receipt.assert_awaited_once_with("child", "request")


@pytest.mark.asyncio
async def test_wait_timeout_includes_initial_http_request_and_never_interrupts(capsys):
    async def blocked(*args, **kwargs):
        await asyncio.Event().wait()

    client = SimpleNamespace(
        child_threads=AsyncMock(side_effect=blocked), interrupt=AsyncMock()
    )
    args = _args()
    args.wait_timeout = 0.01
    with pytest.raises(SystemExit) as error:
        await show_children(client, args)
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "timeout"
    client.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_transport_timeout_is_not_claimed_as_observation_deadline():
    client = SimpleNamespace(
        child_threads=AsyncMock(side_effect=TimeoutError("transport"))
    )
    with pytest.raises(TimeoutError, match="transport"):
        await show_children(client, _args())


@pytest.mark.asyncio
@pytest.mark.parametrize("children", [(), (_child(run=None),)])
async def test_empty_batch_does_not_wait_for_future_children(children, capsys):
    client = SimpleNamespace(child_threads=AsyncMock(return_value=_page(*children)))
    await show_children(client, _args())
    assert json.loads(capsys.readouterr().out)["status"] == "empty"
    assert client.child_threads.await_count == 1


@pytest.mark.asyncio
async def test_cli_launches_linked_child_and_waits_for_compact_outcome(
    coding_cli_project,
):
    import sys
    from tests.test_coding_cli import _command, _shutdown
    from msgflux.runtime.service.local import connect_local_service

    project, config, runtime = coding_cli_project
    client = await connect_local_service(
        config=config, runtime_dir=runtime, cwd=project
    )
    try:
        await client.open_thread(
            "worker", thread_id="parent", cwd=project, interactive=False
        )
        code, stdout, stderr = await _command(
            project,
            config,
            runtime,
            "-p",
            "alpha",
            "--parent",
            "parent",
            "--detach",
            "--json",
        )
        assert code == 0, stderr + stdout
        admitted = json.loads(stdout)
        child = next(
            thread
            for thread in await client.threads()
            if thread.thread_id == admitted["thread_id"]
        )
        assert child.parent_thread_id == "parent"
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "msgflux.cli",
            "threads",
            "children",
            "parent",
            "--wait",
            "--wait-timeout",
            "5",
            "--json",
            "--runtime-dir",
            str(runtime),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            assert process.returncode == 0, stderr.decode()
            outcome = json.loads(stdout)
            assert outcome["status"] == "settled"
            assert outcome["child"]["thread_id"] == child.thread_id
            assert outcome["child"]["run_id"] == admitted["run_id"]
            assert "FINAL" not in stdout.decode()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
    finally:
        await client.aclose()
        await _shutdown(runtime)
