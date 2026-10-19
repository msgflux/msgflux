from __future__ import annotations

import asyncio

import pytest

from msgflux.runtime import (
    AgentWorkspace,
    ExecutionEnvironment,
    ExecutionScope,
    LocalWorkspace,
    PermissionSet,
    ProcessRequest,
    execution_context,
)
from msgflux.runtime.workspace.docker_executor import DockerProcessExecutor
from msgflux.runtime.workspace.receipts import bind_command_receipt_persist


def _environment(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "msgflux.runtime.workspace.docker_executor.shutil.which",
        lambda _name: "/usr/bin/docker",
    )
    root = tmp_path / "workspace"
    root.mkdir()
    socket_path = tmp_path / "docker.sock"
    socket_path.touch()
    filesystem = LocalWorkspace("docker-receipt", root)
    executor = DockerProcessExecutor(
        filesystem, image="trusted-image", socket_path=str(socket_path)
    )
    executor._prefix = ("docker", "--host", f"unix://{socket_path}")
    executor._command = lambda *_args: ("docker", "create")
    environment = ExecutionEnvironment(filesystem, executor)
    workspace = AgentWorkspace.from_environment(
        environment,
        permissions=PermissionSet(
            ["process.execute"],
            [filesystem.permission("/", "process.workspace")],
        ),
    )
    scope = ExecutionScope(workspace=workspace, thread_id="thread", run_id="run")
    return environment, executor, scope


@pytest.mark.asyncio
async def test_created_container_id_is_persisted_when_launched_checkpoint_fails(
    tmp_path, monkeypatch
):
    environment, executor, scope = _environment(tmp_path, monkeypatch)

    container_id = "a" * 64
    killed = []

    async def create(*_args, **_kwargs):
        return container_id

    async def run_cli(arguments, **_kwargs):
        if arguments[0] == "kill":
            killed.append(tuple(arguments))
        return 0, b"", b""

    executor._create = create
    executor._run_cli = run_cli
    attempts = []
    persisted = []
    primary_error = OSError("injected launched receipt failure")

    def persist(receipt):
        attempts.append(receipt.state)
        if receipt.state == "launched":
            raise primary_error
        persisted.append(receipt)

    with (
        execution_context(scope=scope),
        bind_command_receipt_persist(persist),
    ):
        with pytest.raises(OSError) as raised:
            await environment.arun(ProcessRequest(("echo", "hello")))

    assert raised.value is primary_error
    assert attempts == ["intent", "intent", "launched", "unknown"]
    assert persisted[-1].state == "unknown"
    assert persisted[-1].resource["container_id"] == container_id
    assert killed == [("kill", container_id)]


@pytest.mark.asyncio
async def test_cancelled_container_creation_persists_exact_id_during_cleanup(
    tmp_path, monkeypatch
):
    environment, executor, scope = _environment(tmp_path, monkeypatch)
    container_id = "b" * 64
    creation_started = asyncio.Event()
    allow_creation = asyncio.Event()
    killed = []
    persisted = []

    async def create(*_args, **_kwargs):
        creation_started.set()
        await allow_creation.wait()
        return container_id

    async def run_cli(arguments, **_kwargs):
        if arguments[0] == "kill":
            killed.append(tuple(arguments))
        return 0, b"", b""

    executor._create = create
    executor._run_cli = run_cli

    def persist(receipt):
        persisted.append(receipt)

    async def run():
        try:
            await environment.arun(ProcessRequest(("echo", "hello")))
        except BaseException as error:
            observed.append(error)
            raise

    observed = []
    with (
        execution_context(scope=scope),
        bind_command_receipt_persist(persist),
    ):
        task = asyncio.create_task(run())
        await creation_started.wait()
        task.cancel("cancel while Docker create is in flight")
        asyncio.get_running_loop().call_later(0.01, allow_creation.set)
        with pytest.raises(asyncio.CancelledError) as raised:
            await task

    assert observed[0] is raised.value
    assert persisted[-1].state == "unknown"
    assert persisted[-1].resource["container_id"] == container_id
    assert killed == [("kill", container_id)]
