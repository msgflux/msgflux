"""Real Docker command receipts survive abrupt controller death (opt in)."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import shlex
import shutil
import subprocess
import time

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime import (
    AgentTaskRecovery,
    AgentWorkspace,
    DockerWorkspaceBackend,
    PermissionSet,
    ResourcePermission,
)
from msgflux.runtime.context import ExecutionScope, execution_context
from msgflux.runtime.agent_inbox import AgentInbox, SQLiteAgentInboxStore
from msgflux.runtime.workspace.receipts import CommandExecution
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
from msgflux.tasks import SQLiteTaskStore
from msgflux.tools.builtin import BashTool


WORKSPACE_ID = "command-recovery-docker"
IMAGE = "python:3.12-slim"


class _Model:
    model_type = "chat_completion"

    def __init__(self, gate, command):
        self.gate = gate
        self.command = command
        self.calls = 0

    def __call__(self, **_kwargs):
        self.calls += 1
        if not self.gate.wait(30):
            raise TimeoutError("parent did not release model gate")
        response = ModelResponse()
        response.set_response_type("tool_call")
        calls = ToolCallAggregator()
        calls.process(
            0, "docker-recovery-call", "bash", json.dumps({"command": self.command})
        )
        response.add(calls)
        response.reasoning = None
        response.metadata = {}
        return response

    async def acall(self, **kwargs):
        return self(**kwargs)


def _permissions():
    return PermissionSet(
        {"filesystem.read", "filesystem.write", "process.execute", "process.workspace"},
        {ResourcePermission(f"workspace:{WORKSPACE_ID}:/", "process.workspace")},
    )


def _controller(paths, root, image_id, gate, pipe, crash_state):
    checkpoints = SQLiteCheckpointStore(paths[0])
    tasks = SQLiteTaskStore(paths[1])
    inbox_store = SQLiteAgentInboxStore(paths[2])
    registry = SQLiteWorkspaceRegistry(paths[3])
    backend = DockerWorkspaceBackend(root, image=image_id, registry=registry)
    workspace = asyncio.run(
        AgentWorkspace.open(
            backend,
            WORKSPACE_ID,
            permissions=_permissions(),
            write_guarantee="cooperative_compare",
        )
    )
    marker = "/workspace/container-started"
    stop = "/workspace/container-stop"
    if crash_state == "running":
        code = (
            "import pathlib,time; "
            f"pathlib.Path({marker!r}).write_text('started'); "
            f"p=pathlib.Path({stop!r}); "
            "exec('while not p.exists(): time.sleep(0.05)')"
        )
    else:
        code = (
            f"import pathlib,sys; pathlib.Path({marker!r}).write_text('completed'); "
            "print('command-finished'); print('command-warning', file=sys.stderr)"
        )
    command = "python -c " + shlex.quote(code)
    if crash_state in {"created", "completed", "exited_without_receipt"}:
        original_update = CommandExecution.update

        async def crash_after_persist(execution, state, **kwargs):
            resource = kwargs.get("resource") or {}
            if crash_state == "exited_without_receipt" and state == "completed":
                os._exit(84)
            result = await original_update(execution, state, **kwargs)
            if (
                crash_state == "created"
                and state == "launched"
                and resource.get("image_id")
            ):
                os._exit(82)
            if crash_state == "completed" and state == "completed":
                os._exit(83)
            return result

        CommandExecution.update = crash_after_persist
    worker = Agent(
        name="worker",
        model=_Model(gate, command),
        checkpoint_store=checkpoints,
        workspace=workspace,
        tools=[BashTool()],
    )
    worker.tool_config = {"background": True}
    library = ToolLibrary(name="docker-host", tools=[worker], task_store=tasks)
    library.set_agent_inbox(AgentInbox(owner="host", store=inbox_store))
    library.get_background_dispatcher().lease_seconds = 0.5
    with execution_context(
        scope=ExecutionScope(
            thread_id="docker-command-thread",
            run_id="docker-root-run",
            root_run_id="docker-root-run",
            workspace=workspace,
        ),
        checkpoint_store=checkpoints,
    ):
        dispatch = library([("start", "worker", {"task": "run command"})])
        task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]
        pipe.send(task_id)
        pipe.close()
        gate.wait(30)
        time.sleep(30)


def _receipts(tasks, task_id):
    return [
        item.metadata["receipt"]
        for item in tasks.list_activity(task_id)
        if item.kind == "command_receipt"
        and isinstance(item.metadata.get("receipt"), dict)
    ]


def _wait_for(predicate, timeout=35):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.1)
    raise AssertionError("timed out waiting for Docker receipt/resource")


def _container_exited(docker, receipt):
    resource = receipt.get("resource") or {}
    container_id = resource.get("container_id")
    if not container_id:
        return False
    try:
        status = subprocess.check_output(  # noqa: S603 -- exact receipt-owned ID inspection
            [docker, "inspect", "--format", "{{.State.Status}}", container_id],
            text=True,
            timeout=5,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return status == "exited"


def _assert_exact_container(docker, container_id, execution_id, image_id, status):
    details = json.loads(
        subprocess.check_output(  # noqa: S603 -- inspect exact receipt-owned ID
            [docker, "inspect", "--format", "{{json .}}", container_id],
            text=True,
            timeout=10,
        )
    )
    assert details["Id"] == container_id
    assert details["Image"] == image_id
    assert details["State"]["Status"] == status
    assert details["Config"]["Labels"]["msgflux.execution_id"] == execution_id


def _assert_unrelated_container_is_not_adopted(docker, image_id, workspace, receipt):
    execution_id = "unrelated-" + str(time.time_ns())
    container_id = subprocess.check_output(  # noqa: S603 -- fixed Docker CLI and labels
        [
            docker,
            "run",
            "--detach",
            "--label",
            "msgflux.executor=ephemeral",
            "--label",
            "msgflux.command=1",
            "--label",
            f"msgflux.execution_id={execution_id}",
            image_id,
            "python",
            "-c",
            "import time; time.sleep(60)",
        ],
        text=True,
        timeout=20,
    ).strip()
    try:
        unrelated = json.loads(json.dumps(receipt))
        unrelated["resource"]["container_id"] = container_id
        inspected = asyncio.run(workspace.ainspect_command(unrelated))
        assert inspected.classification == "blocked", inspected
        terminated = asyncio.run(workspace.aterminate_command(unrelated))
        assert terminated.classification == "blocked", terminated
        state = subprocess.check_output(  # noqa: S603 -- inspect exact test-owned ID
            [docker, "inspect", "--format", "{{.State.Running}}", container_id],
            text=True,
            timeout=10,
        ).strip()
        assert state == "true", "unrelated container was changed"
    finally:
        subprocess.run(  # noqa: S603 -- remove only this helper's exact ID
            [docker, "rm", "--force", container_id],
            check=False,
            timeout=15,
            capture_output=True,
        )


@pytest.mark.parametrize(
    "crash_state", ["running", "created", "completed", "exited_without_receipt"]
)
def test_docker_command_crash_boundary_preserves_exact_receipt_and_resource(  # noqa: C901
    tmp_path, crash_state
):
    if os.environ.get("MSGFLUX_TEST_DOCKER") != "1":
        pytest.skip("Set MSGFLUX_TEST_DOCKER=1 for real Docker command recovery")
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable")
    try:
        image_id = subprocess.check_output(  # noqa: S603 -- fixed Docker CLI and local image lookup
            [docker, "image", "inspect", IMAGE, "--format", "{{.Id}}"],
            text=True,
            timeout=10,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"{IMAGE} is not installed locally; no image pull attempted: {exc}")
    if not image_id.startswith("sha256:"):
        pytest.skip("Docker did not return an immutable image ID")

    root = tmp_path / "workspace"
    root.mkdir()
    paths = tuple(
        str(tmp_path / name)
        for name in (
            "checkpoints.sqlite",
            "tasks.sqlite",
            "inbox.sqlite",
            "registry.sqlite",
        )
    )
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    gate = context.Event()
    controller = context.Process(
        target=_controller,
        args=(paths, str(root), image_id, gate, child, crash_state),
    )
    stores = workspace = None
    owned_container_id = owned_execution_id = None
    try:
        controller.start()
        assert parent.poll(35), "controller did not persist a task"
        task_id = parent.recv()
        gate.set()
        if crash_state == "running":
            _wait_for((root / "container-started").exists)
            controller.kill()
            controller.join(10)
            expected_exit = -9
        else:
            task_reader = SQLiteTaskStore(paths[1])
            try:
                if crash_state == "created":
                    _wait_for(
                        lambda: any(
                            (item.get("resource") or {}).get("image_id")
                            for item in _receipts(task_reader, task_id)
                        )
                    )
                    expected_exit = 82
                elif crash_state == "completed":
                    _wait_for(
                        lambda: any(
                            item.get("state") == "completed"
                            for item in _receipts(task_reader, task_id)
                        )
                    )
                    expected_exit = 83
                else:
                    _wait_for((root / "container-started").exists)
                    _wait_for(
                        lambda: any(
                            _container_exited(docker, receipt)
                            for receipt in _receipts(task_reader, task_id)
                        )
                    )
                    expected_exit = 84
                controller.join(10)
            finally:
                task_reader.close()
        assert controller.exitcode == expected_exit

        registry = SQLiteWorkspaceRegistry(paths[3])
        backend = DockerWorkspaceBackend(root, image=image_id, registry=registry)
        record = registry.get_record(WORKSPACE_ID)
        workspace = asyncio.run(
            AgentWorkspace.reconnect(
                backend,
                WORKSPACE_ID,
                record.identity,
                permissions=_permissions(),
                write_guarantee="cooperative_compare",
            )
        )
        checkpoints = SQLiteCheckpointStore(paths[0])
        tasks = SQLiteTaskStore(paths[1])
        inbox_store = SQLiteAgentInboxStore(paths[2])
        stores = (inbox_store, tasks, checkpoints, registry)
        _wait_for(
            lambda: (
                (lease := tasks.get_worker_lease(task_id)) is None
                or lease.expires_at <= time.time()
            ),
            timeout=10,
        )
        receipt = _wait_for(lambda: _receipts(tasks, task_id))[-1]
        assert receipt["state"] == (
            "completed" if crash_state == "completed" else "launched"
        )
        assert receipt["resource"]["container_id"]
        assert receipt["resource"]["image_id"] == image_id
        owned_container_id = receipt["resource"]["container_id"]
        owned_execution_id = receipt["execution_id"]
        _assert_exact_container(
            docker,
            owned_container_id,
            owned_execution_id,
            image_id,
            {
                "running": "running",
                "created": "created",
                "completed": "exited",
                "exited_without_receipt": "exited",
            }[crash_state],
        )

        worker = Agent(
            name="worker",
            model=_Model(context.Event(), "false"),
            checkpoint_store=checkpoints,
            tools=[BashTool()],
        )
        worker.tool_config = {"background": True}
        library = ToolLibrary(name="docker-host", tools=[worker], task_store=tasks)
        library.set_agent_inbox(AgentInbox(owner="host", store=inbox_store))
        coordinator = AgentTaskRecovery(library, workspace=workspace)
        report = coordinator.inspect(task_id)
        assert report.classification == "uncertain", report

        inspected = asyncio.run(workspace.ainspect_command(receipt))
        expected_inspection = {
            "running": "running",
            "created": "unknown",
            "completed": "completed",
            "exited_without_receipt": "completed",
        }[crash_state]
        assert inspected.classification == expected_inspection, inspected
        assert inspected.resource_status == (
            "unchecked" if crash_state == "completed" else "present"
        ), inspected
        if crash_state == "running":
            stopped = asyncio.run(workspace.aterminate_command(receipt))
            assert stopped.classification == "completed", stopped
            assert stopped.resource_status == "present", stopped
            assert stopped.returncode == 137, stopped
            _assert_unrelated_container_is_not_adopted(
                docker, image_id, workspace, receipt
            )
        elif crash_state == "created":
            assert inspected.returncode is None, inspected
            assert not (root / "container-started").exists()
        elif crash_state == "completed":
            assert inspected.returncode == 0, inspected
            assert (root / "container-started").read_text() == "completed"
            assert b"command-finished" in inspected.stdout
            assert b"command-warning" in inspected.stderr
            _assert_exact_container(
                docker, owned_container_id, owned_execution_id, image_id, "exited"
            )
        elif crash_state == "exited_without_receipt":
            assert inspected.returncode == 0, inspected
            assert (root / "container-started").read_text() == "completed"
            assert b"command-finished" in inspected.stdout
            assert b"command-warning" in inspected.stderr
            _assert_exact_container(
                docker, owned_container_id, owned_execution_id, image_id, "exited"
            )
            assert inspected.stdout
        with pytest.raises(RuntimeError, match="uncertain"):
            coordinator.recover(task_id, "never replay", worker_stopped=True)
    finally:
        if controller.is_alive():
            controller.kill()
        controller.join(timeout=5)
        if workspace is not None:
            asyncio.run(workspace.aclose())
        if stores:
            for store in stores:
                store.close()
        if owned_container_id is not None:
            try:
                observed_id = subprocess.check_output(  # noqa: S603 -- exact receipt-owned ID lookup
                    [docker, "inspect", "--format", "{{.Id}}", owned_container_id],
                    text=True,
                    timeout=10,
                ).strip()
                observed_execution_id = subprocess.check_output(  # noqa: S603 -- verify receipt label before cleanup
                    [
                        docker,
                        "inspect",
                        "--format",
                        '{{index .Config.Labels "msgflux.execution_id"}}',
                        owned_container_id,
                    ],
                    text=True,
                    timeout=10,
                ).strip()
                if (
                    observed_id == owned_container_id
                    and observed_execution_id == owned_execution_id
                ):
                    subprocess.run(  # noqa: S603 -- remove only the verified exact owned ID
                        [docker, "rm", "--force", owned_container_id],
                        check=False,
                        timeout=10,
                        capture_output=True,
                    )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
        parent.close()
        child.close()


@pytest.mark.asyncio
async def test_docker_command_timeout_stops_process_before_late_side_effect(tmp_path):
    if os.environ.get("MSGFLUX_TEST_DOCKER") != "1":
        pytest.skip("Set MSGFLUX_TEST_DOCKER=1 for real Docker command recovery")
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable")
    try:
        image_id = subprocess.check_output(  # noqa: S603 -- fixed Docker CLI and local image lookup
            [docker, "image", "inspect", IMAGE, "--format", "{{.Id}}"],
            text=True,
            timeout=10,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"{IMAGE} is not installed locally; no image pull attempted: {exc}")
    if not image_id.startswith("sha256:"):
        pytest.skip("Docker did not return an immutable image ID")

    root = tmp_path / "workspace"
    root.mkdir()
    workspace_id = "docker-command-timeout"
    registry = SQLiteWorkspaceRegistry(tmp_path / "workspace.sqlite")
    workspace = await AgentWorkspace.open(
        DockerWorkspaceBackend(
            root,
            image=image_id,
            registry=registry,
        ),
        workspace_id,
        permissions=PermissionSet(
            {
                "filesystem.read",
                "filesystem.write",
                "process.execute",
                "process.workspace",
            },
            {ResourcePermission(f"workspace:{workspace_id}:/", "process.workspace")},
        ),
        write_guarantee="cooperative_compare",
    )
    try:
        marker = root / "after-timeout.txt"
        code = (
            "import pathlib,time; time.sleep(3); "
            f"pathlib.Path({str(marker)!r}).write_text('late')"
        )
        command = "python -c " + shlex.quote(code)
        with pytest.raises(TimeoutError):
            await workspace.arun(command, timeout=0.75)
        await asyncio.sleep(3.5)
        assert not marker.exists(), "timed-out Docker command ran after its deadline"
    finally:
        await workspace.aclose()
        registry.close()
