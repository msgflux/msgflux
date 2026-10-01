"""Opt-in local Docker execution; the daemon and image are trusted host services."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
from uuid import uuid4

import msgspec

from msgflux.runtime.isolation import SandboxCapabilities
from msgflux.runtime.workspace.backend import WorkspaceBinding
from msgflux.runtime.workspace.command_inspection import (
    command_inspection,
    receipt_identity_error,
    receipt_mapping,
    receipt_resource,
)
from msgflux.runtime.workspace.environment import ProcessExecutor, ProcessResult
from msgflux.runtime.workspace.local import LocalWorkspace, LocalWorkspaceBackend
from msgflux.runtime.workspace.process_capture import drain_subprocess
from msgflux.runtime.workspace.receipts import (
    MAX_RECEIPT_OUTPUT_BYTES,
    get_command_execution,
)

_CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")


async def _discard(_channel, _data):
    return None


async def _cleanup_cli_launch(launch):
    try:
        process = await asyncio.shield(launch)
    except BaseException:
        return
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        await drain_subprocess(
            process,
            _discard,
            max_output_bytes=65536,
            timeout_seconds=3,
            owns_process_group=True,
        )
    except BaseException:
        return


class _DockerSocketUnavailableError(ConnectionError):
    """The recorded local Docker socket cannot be inspected."""


class _DockerSocketChangedError(PermissionError):
    """The configured Docker socket no longer identifies the recorded daemon."""


def _execution_name(execution_id):
    suffix = hashlib.sha256(execution_id.encode("utf-8")).hexdigest()[:32]
    return f"msgflux-{suffix}"


def _valid_daemon_identity(value):
    return bool(
        isinstance(value, dict)
        and isinstance(value.get("path"), str)
        and os.path.isabs(value["path"])
        and type(value.get("device")) is int
        and type(value.get("inode")) is int
    )


class DockerLimits(msgspec.Struct, frozen=True, kw_only=True):
    memory_bytes: int = 256 * 1024 * 1024
    pids: int = 64
    cpus: int = 1
    tmp_bytes: int = 16 * 1024 * 1024

    def __post_init__(self):
        for value in (self.memory_bytes, self.pids, self.cpus, self.tmp_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError("Docker limits must be positive integers")


class DockerProcessExecutor(ProcessExecutor):
    """Linux/local-daemon executor with network disabled and an explicit mount.

    Requires process.execute and the exact workspace:/ grant process.workspace.
    That grant authorizes all mounted files, including writes and deletion;
    individual filesystem grants are NOT interpreted as a recursive grant.
    Images are never pulled implicitly. Use a host-pinned, trusted image ID.
    """

    capabilities = SandboxCapabilities(
        {"filesystem", "network", "process", "resource_limits"}
    )
    requires_workspace_process_grant = True

    def __init__(
        self,
        filesystem: LocalWorkspace,
        *,
        image: str,
        limits: DockerLimits | None = None,
        socket_path: str = "/var/run/docker.sock",
    ):
        if not isinstance(filesystem, LocalWorkspace):
            raise TypeError("Docker requires a LocalWorkspace")
        if (
            not isinstance(image, str)
            or not image
            or image.startswith("-")
            or "\0" in image
        ):
            raise ValueError("Expected a trusted Docker image name or ID")
        if (
            not isinstance(socket_path, str)
            or not os.path.isabs(socket_path)
            or "\0" in socket_path
        ):
            raise ValueError("Docker requires an absolute local Unix socket")
        self.filesystem = filesystem
        self.image = image
        self._socket_path = os.fspath(socket_path)
        self.limits = limits or DockerLimits()
        if not isinstance(self.limits, DockerLimits):
            raise TypeError("Expected DockerLimits")
        self._docker = shutil.which("docker")
        if self._docker is None:
            raise FileNotFoundError("Docker CLI is not installed")
        self._prefix = (self._docker, "--host", f"unix://{socket_path}")

    def supports_workspace(self, filesystem):
        return filesystem is self.filesystem

    def _daemon_identity(self):
        """Capture the configured Unix socket's host identity on demand."""
        path = os.path.realpath(self._socket_path)
        try:
            stat_result = os.stat(self._socket_path)
        except OSError as error:
            raise _DockerSocketUnavailableError(
                "Configured Docker socket is unavailable"
            ) from error
        return {
            "path": path,
            "device": stat_result.st_dev,
            "inode": stat_result.st_ino,
        }

    def _verify_daemon_identity(self, expected):
        if not _valid_daemon_identity(expected):
            raise _DockerSocketChangedError(
                "Receipt has no recorded Docker socket identity"
            )
        try:
            current = self._daemon_identity()
        except _DockerSocketUnavailableError:
            raise
        if current != expected:
            raise _DockerSocketChangedError(
                "Configured Docker socket identity changed since command launch"
            )

    def _command(self, request, name, execution=None):
        root = "/" + "/".join(self.filesystem._root_parts)
        if root == "/" or "," in root or "\n" in root:
            raise ValueError("Unsafe workspace mount root")
        os.close(self.filesystem._open_root())
        uid, gid = os.getuid(), os.getgid()
        if uid == 0:
            raise PermissionError("Run this adapter as a non-root workspace owner")
        limits = self.limits
        command = [
            *self._prefix,
            "create",
            "--pull=never",
            "--name",
            name,
            "--label",
            "msgflux.executor=ephemeral",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user",
            f"{uid}:{gid}",
            "--pids-limit",
            str(limits.pids),
            "--memory",
            str(limits.memory_bytes),
            "--memory-swap",
            str(limits.memory_bytes),
            "--cpus",
            str(limits.cpus),
            "--init",
            "--no-healthcheck",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,noexec,size={limits.tmp_bytes}",  # noqa: S108
            "--mount",
            f"type=bind,src={root},dst=/workspace,bind-recursive=disabled,bind-propagation=rprivate",
            "--workdir",
            "/workspace" + request.cwd.rstrip("/"),
            "--entrypoint",
            request.argv[0],
        ]
        if execution is None:
            command.extend(("--log-driver=none",))
        else:
            reference = execution.receipt.workspace_reference or {}
            identity = reference.get("identity", {})
            command.extend(
                (
                    "--label",
                    "msgflux.command=1",
                    "--label",
                    f"msgflux.execution_id={execution.execution_id}",
                    "--label",
                    f"msgflux.workspace_id={reference.get('workspace_id', '')}",
                    "--label",
                    f"msgflux.workspace_generation={identity.get('generation', '')}",
                    "--log-driver=json-file",
                    "--log-opt",
                    "max-size="
                    f"{max(1, min(64, (request.max_output_bytes + 1023) // 1024))}k",
                    "--log-opt",
                    "max-file=1",
                )
            )
        command.extend((self.image, *request.argv[1:]))
        return tuple(command)

    async def _remove(self, container_id, *, expected_name, execution=None):
        resource = (
            receipt_resource(execution.receipt) if execution is not None else None
        )
        daemon_identity = resource.get("daemon_identity") if resource else None
        details = await self._container_details(
            container_id, expected_daemon_identity=daemon_identity
        )
        if details is None:
            return
        if execution is None:
            labels = (details.get("Config") or {}).get("Labels") or {}
            valid = (
                details.get("Id") == container_id
                and details.get("Name", "").lstrip("/") == expected_name
                and labels.get("msgflux.executor") == "ephemeral"
                and (details.get("Config") or {}).get("Image") == self.image
            )
        else:
            valid = self._matches_receipt(details, execution.receipt)
        if not valid:
            raise RuntimeError("Docker container ownership changed before cleanup")
        if daemon_identity is not None:
            self._verify_daemon_identity(daemon_identity)
        process = await asyncio.create_subprocess_exec(
            *self._prefix,
            "rm",
            "--force",
            "--volumes",
            container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._client_env(),
            start_new_session=True,
        )
        try:
            code = await drain_subprocess(
                process,
                _discard,
                max_output_bytes=16384,
                timeout_seconds=15,
                owns_process_group=True,
            )
        except BaseException:
            raise
        if code:
            raise RuntimeError(f"Container cleanup failed; reconcile {container_id}")

    @staticmethod
    def _client_env():
        # No application credentials are copied into the container or CLI env.
        return {
            "PATH": os.defpath,
            "HOME": "/nonexistent",
            "DOCKER_CONFIG": "/nonexistent",
        }

    async def _create(self, command, *, expected_daemon_identity=None):
        if expected_daemon_identity is not None:
            self._verify_daemon_identity(expected_daemon_identity)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._client_env(),
            start_new_session=True,
        )

        output = bytearray()

        async def capture(channel, data):
            if channel == "stdout":
                output.extend(data)

        code = await drain_subprocess(
            process,
            capture,
            max_output_bytes=4096,
            timeout_seconds=30,
            owns_process_group=True,
        )
        if code:
            raise RuntimeError(
                "Docker container creation failed; check daemon/image/policy"
            )
        container_id = output.decode("ascii", errors="strict").strip()
        if not _CONTAINER_ID.fullmatch(container_id):
            raise RuntimeError("Docker create returned an invalid container ID")
        return container_id

    async def _verify_container_id(self, container_id):
        if not isinstance(container_id, str) or not _CONTAINER_ID.fullmatch(
            container_id
        ):
            raise ValueError("A full recorded Docker container ID is required")

    async def _container_details(self, container_id, *, expected_daemon_identity=None):
        await self._verify_container_id(container_id)
        code, stdout, stderr = await self._run_cli(
            ("inspect", "--format", "{{json .}}", container_id),
            max_output_bytes=65536,
            expected_daemon_identity=expected_daemon_identity,
        )
        if code:
            message = stderr.decode("utf-8", errors="replace")
            if "No such object" in message or "No such container" in message:
                return None
            raise ConnectionError("Docker daemon inspection failed")
        try:
            details = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "Docker inspect returned invalid container data"
            ) from error
        if not isinstance(details, dict):
            raise RuntimeError("Docker inspect returned an invalid record")
        return details

    async def _run_cli(
        self,
        arguments,
        *,
        max_output_bytes,
        timeout_seconds=15,
        expected_daemon_identity=None,
    ):
        if expected_daemon_identity is not None:
            self._verify_daemon_identity(expected_daemon_identity)
        launch = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *self._prefix,
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._client_env(),
                start_new_session=True,
            )
        )
        try:
            process = await asyncio.shield(launch)
        except BaseException as error:
            cleanup = asyncio.create_task(_cleanup_cli_launch(launch))
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            try:
                cleanup.result()
            except BaseException as cleanup_error:
                error.add_note(f"Docker CLI launch cleanup failed: {cleanup_error}")
            raise
        stdout = bytearray()
        stderr = bytearray()

        async def capture(channel, data):
            (stdout if channel == "stdout" else stderr).extend(data)

        code = await drain_subprocess(
            process,
            capture,
            max_output_bytes=max_output_bytes,
            timeout_seconds=timeout_seconds,
            owns_process_group=True,
        )
        return code, bytes(stdout), bytes(stderr)

    def _expected_labels(self, receipt):
        record = receipt_mapping(receipt)
        reference = record["workspace_reference"]
        identity = reference["identity"]
        return {
            "msgflux.executor": "ephemeral",
            "msgflux.command": "1",
            "msgflux.execution_id": record["execution_id"],
            "msgflux.workspace_id": reference["workspace_id"],
            "msgflux.workspace_generation": identity["generation"],
        }

    def _matches_receipt(self, details, receipt):
        record = receipt_mapping(receipt)
        resource = receipt_resource(receipt) or {}
        config = details.get("Config") or {}
        labels = config.get("Labels") or {}
        expected_id = resource.get("container_id")
        expected_image = resource.get("image")
        return bool(
            isinstance(expected_id, str)
            and _CONTAINER_ID.fullmatch(expected_id)
            and details.get("Id") == expected_id
            and all(
                labels.get(key) == value
                for key, value in self._expected_labels(receipt).items()
            )
            and config.get("Image") == expected_image == self.image
            and resource.get("daemon_config_revision")
            == record["workspace_reference"]["identity"]["config_revision"]
            and _valid_daemon_identity(resource.get("daemon_identity"))
            and (
                resource.get("image_id") is None
                or details.get("Image") == resource.get("image_id")
            )
            and resource.get("name") == details.get("Name", "").lstrip("/")
        )

    async def inspect_command(self, receipt):  # noqa: C901
        """Inspect the persisted container ID; names and labels are not authority."""
        error = receipt_identity_error(
            receipt, self.filesystem, backend=self.filesystem.identity.backend
        )
        if error is not None:
            return command_inspection(receipt, "blocked", "mismatch", error)
        try:
            os.close(self.filesystem._open_root())
        except OSError as error:
            return command_inspection(receipt, "blocked", "unavailable", str(error))

        record = receipt_mapping(receipt)
        if record["state"] == "completed" and record.get("returncode") is not None:
            return command_inspection(
                receipt,
                "completed",
                "unchecked",
                stdout=record.get("stdout"),
                stderr=record.get("stderr"),
                returncode=record["returncode"],
            )
        resource = receipt_resource(receipt)
        container_id = resource.get("container_id") if resource else None
        if not isinstance(container_id, str) or not _CONTAINER_ID.fullmatch(
            container_id
        ):
            return command_inspection(
                receipt,
                "unknown",
                "unchecked",
                "receipt has no exact Docker container ID; "
                "name discovery is not adopted",
            )
        if not isinstance(resource.get("image_id"), str):
            return command_inspection(
                receipt,
                "blocked",
                "unchecked",
                "receipt has no verified image identity",
            )
        daemon_identity = resource.get("daemon_identity")
        if not _valid_daemon_identity(daemon_identity):
            return command_inspection(
                receipt,
                "blocked",
                "unchecked",
                "receipt has no recorded Docker socket identity",
            )
        try:
            details = await self._container_details(
                container_id, expected_daemon_identity=daemon_identity
            )
        except _DockerSocketChangedError as error:
            return command_inspection(receipt, "blocked", "mismatch", str(error))
        except _DockerSocketUnavailableError as error:
            return command_inspection(receipt, "blocked", "unavailable", str(error))
        except (OSError, RuntimeError, ConnectionError, asyncio.TimeoutError):
            return command_inspection(
                receipt, "blocked", "unavailable", "Docker daemon is unavailable"
            )
        if details is None:
            return command_inspection(
                receipt, "unknown", "missing", "recorded Docker container is absent"
            )
        if not self._matches_receipt(details, receipt):
            return command_inspection(
                receipt,
                "blocked",
                "mismatch",
                "container ID, labels, image or daemon configuration changed",
            )

        state = details.get("State") or {}
        if state.get("Running") is True:
            return command_inspection(
                receipt, "running", "present", "recorded container is still running"
            )
        if state.get("Status") != "exited":
            return command_inspection(
                receipt,
                "unknown",
                "present",
                "container has no confirmed completed execution",
            )
        returncode = state.get("ExitCode")
        if type(returncode) is not int:
            return command_inspection(
                receipt,
                "unknown",
                "present",
                "Docker has no verifiable exit code for the container",
            )
        try:
            log_code, logs, errors = await self._run_cli(
                ("logs", "--tail", "200", container_id),
                max_output_bytes=131072,
                expected_daemon_identity=daemon_identity,
            )
        except (OSError, RuntimeError, ConnectionError, asyncio.TimeoutError):
            return command_inspection(
                receipt,
                "unknown",
                "present",
                "container exit status is known but bounded logs are unavailable",
                returncode=returncode,
            )
        if log_code:
            return command_inspection(
                receipt,
                "unknown",
                "present",
                "container exit status is known but bounded logs are unavailable",
                returncode=returncode,
            )
        return command_inspection(
            receipt,
            "completed",
            "present",
            "outcome recovered from Docker; persist reconciliation before resuming",
            returncode=returncode,
            stdout=logs[:MAX_RECEIPT_OUTPUT_BYTES],
            stderr=errors[:MAX_RECEIPT_OUTPUT_BYTES],
        )

    async def terminate_command(self, receipt):  # noqa: C901
        """Stop an explicitly selected, verified container and retain its evidence."""
        error = receipt_identity_error(
            receipt, self.filesystem, backend=self.filesystem.identity.backend
        )
        if error is not None:
            return command_inspection(receipt, "blocked", "mismatch", error)
        inspection = await self.inspect_command(receipt)
        if inspection.classification != "running":
            return inspection
        resource = receipt_resource(receipt)
        container_id = resource["container_id"]
        try:
            code, _stdout, _stderr = await self._run_cli(
                ("kill", container_id),
                max_output_bytes=16384,
                expected_daemon_identity=resource.get("daemon_identity"),
            )
        except _DockerSocketChangedError as error:
            return command_inspection(receipt, "blocked", "mismatch", str(error))
        except _DockerSocketUnavailableError as error:
            return command_inspection(receipt, "blocked", "unavailable", str(error))
        except (OSError, RuntimeError, ConnectionError, asyncio.TimeoutError):
            return command_inspection(
                receipt,
                "unknown",
                "unavailable",
                "Docker daemon could not confirm the termination request",
            )
        if code:
            return command_inspection(
                receipt,
                "unknown",
                "unavailable",
                "Docker could not confirm the termination request",
            )
        deadline = asyncio.get_running_loop().time() + 10
        while asyncio.get_running_loop().time() < deadline:
            try:
                details = await self._container_details(
                    container_id,
                    expected_daemon_identity=resource.get("daemon_identity"),
                )
            except _DockerSocketChangedError as error:
                return command_inspection(receipt, "blocked", "mismatch", str(error))
            except _DockerSocketUnavailableError as error:
                return command_inspection(receipt, "blocked", "unavailable", str(error))
            except (OSError, RuntimeError, ConnectionError, asyncio.TimeoutError):
                return command_inspection(
                    receipt,
                    "unknown",
                    "unavailable",
                    "Docker daemon became unavailable after termination",
                )
            if details is None:
                return command_inspection(
                    receipt,
                    "unknown",
                    "missing",
                    "container disappeared after termination; outcome is unavailable",
                )
            if not self._matches_receipt(details, receipt):
                return command_inspection(
                    receipt,
                    "blocked",
                    "mismatch",
                    "container ownership changed during termination",
                )
            if not (details.get("State") or {}).get("Running", False):
                return await self.inspect_command(receipt)
            await asyncio.sleep(0.1)
        return command_inspection(
            receipt,
            "unknown",
            "present",
            "termination was requested but the container remains running",
        )

    async def execute_stream(  # noqa: C901
        self,
        request,
        *,
        filesystem,
        permissions,
        requirements,
        abort_signal,
        on_output,
    ):
        self.capabilities.require(requirements)
        if not self.supports_workspace(filesystem):
            raise PermissionError("Executor belongs to another workspace")
        if permissions.missing(("process.execute",)) or permissions.missing_resources(
            (filesystem.permission("/", "process.workspace"),)
        ):
            raise PermissionError("Explicit whole-workspace process grant required")
        if abort_signal is not None:
            abort_signal.raise_if_aborted()
        execution = get_command_execution()
        name = (
            _execution_name(execution.execution_id)
            if execution is not None
            else "msgflux-" + uuid4().hex
        )
        command = self._command(request, name, execution)
        container_id = None
        resource = None
        daemon_identity = None
        completed = False
        failure = None
        create_task = None
        launch_task = None
        process = None
        stdout = bytearray()
        stderr = bytearray()
        try:
            if execution is not None:
                reference = execution.receipt.workspace_reference or {}
                identity = reference.get("identity", {})
                daemon_identity = self._daemon_identity()
                resource = {
                    "name": name,
                    "image": self.image,
                    "daemon_config_revision": identity.get("config_revision"),
                    "daemon_identity": daemon_identity,
                }
                await execution.update("intent", resource=resource)
            create_task = asyncio.create_task(
                self._create(command, expected_daemon_identity=daemon_identity)
            )
            container_id = await asyncio.shield(create_task)
            if execution is not None:
                reference = execution.receipt.workspace_reference or {}
                identity = reference.get("identity", {})
                resource = {
                    "container_id": container_id,
                    "name": name,
                    "image": self.image,
                    "daemon_config_revision": identity.get("config_revision"),
                    "daemon_identity": daemon_identity,
                }
                # Save the exact container ID before start/attach can fail.
                await execution.update("launched", resource=resource)
                details = await self._container_details(
                    container_id,
                    expected_daemon_identity=resource.get("daemon_identity"),
                )
                if details is None or not self._matches_receipt(
                    details, execution.receipt
                ):
                    raise RuntimeError(
                        "Created container failed ownership verification"
                    )
                resource["image_id"] = details.get("Image")
                if (
                    not isinstance(resource["image_id"], str)
                    or not resource["image_id"]
                ):
                    raise RuntimeError("Docker inspect did not identify the image")
                await execution.update("launched", resource=resource)

            if daemon_identity is not None:
                self._verify_daemon_identity(daemon_identity)
            launch_task = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *self._prefix,
                    "start",
                    "--attach",
                    container_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.DEVNULL,
                    env=self._client_env(),
                    start_new_session=True,
                )
            )
            process = await asyncio.shield(launch_task)

            async def capture(channel, data):
                if execution is not None:
                    target = stdout if channel == "stdout" else stderr
                    remaining = MAX_RECEIPT_OUTPUT_BYTES - len(target)
                    if remaining > 0:
                        target.extend(data[:remaining])
                await on_output(channel, data)

            code = await drain_subprocess(
                process,
                capture,
                max_output_bytes=request.max_output_bytes,
                timeout_seconds=request.timeout_seconds,
                abort_signal=abort_signal,
                owns_process_group=True,
            )
            if execution is not None:
                await execution.update(
                    "completed",
                    resource=resource,
                    returncode=code,
                    stdout=bytes(stdout),
                    stderr=bytes(stderr),
                )
            completed = True
            return ProcessResult(code)
        except BaseException as exc:
            failure = exc
            raise
        finally:

            async def cleanup():  # noqa: C901
                nonlocal container_id, process, resource
                if create_task is not None:
                    try:
                        container_id = await asyncio.shield(create_task)
                    except BaseException:  # noqa: S110
                        # Creation failure has no exact resource to inspect/remove.
                        pass
                if launch_task is not None:
                    try:
                        process = process or await asyncio.shield(launch_task)
                    except BaseException:  # noqa: S110
                        pass
                if process is not None and process.returncode is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        await drain_subprocess(
                            process,
                            _discard,
                            max_output_bytes=65536,
                            timeout_seconds=3,
                            owns_process_group=True,
                        )
                    except BaseException:  # noqa: S110
                        pass
                if container_id is None:
                    return
                if execution is not None and not completed:
                    if resource is None or resource.get("container_id") != container_id:
                        reference = execution.receipt.workspace_reference or {}
                        identity = reference.get("identity", {})
                        resource = {
                            "container_id": container_id,
                            "name": name,
                            "image": self.image,
                            "daemon_config_revision": identity.get("config_revision"),
                            "daemon_identity": daemon_identity,
                        }
                        try:
                            await execution.update("launched", resource=resource)
                        except BaseException as error:
                            if failure is not None:
                                failure.add_note(
                                    "Could not persist exact Docker container ID: "
                                    f"{error}"
                                )
                    # Killing the attached Docker CLI does not stop the daemon
                    # container. Stop it explicitly, then retain it for host
                    # inspection and log reconciliation.
                    try:
                        await self._run_cli(
                            ("kill", container_id),
                            max_output_bytes=16384,
                            expected_daemon_identity=daemon_identity,
                        )
                    except BaseException:  # noqa: S110
                        pass
                    return
                await self._remove(
                    container_id, expected_name=name, execution=execution
                )

            task = asyncio.create_task(cleanup())
            cleanup_error = None
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except BaseException as error:
                    cleanup_error = error
                    break
            try:
                task.result()
            except BaseException as error:
                cleanup_error = error
            if cleanup_error is not None:
                if failure is None:
                    raise cleanup_error
                failure.add_note(
                    "Container cleanup/termination failed; host reconciliation "
                    f"required: {cleanup_error}"
                )


class DockerWorkspaceBackend(LocalWorkspaceBackend):
    """Local files with per-command ephemeral Docker isolation for Bash."""

    def __init__(
        self,
        root,
        *,
        image: str,
        limits: DockerLimits | None = None,
        socket_path: str = "/var/run/docker.sock",
        registry=None,
    ):
        super().__init__(root, registry=registry)
        self.image = image
        self.limits = limits or DockerLimits()
        self.socket_path = socket_path

    def _backend_kind(self):
        return "docker-local-posix"

    def _configuration_revision(self):
        if self._persistent_registry is None:
            # Identity is process-local without a registry; no daemon probe needed.
            return super()._configuration_revision()
        if not (
            re.fullmatch(r"sha256:[0-9a-fA-F]{64}", self.image)
            or re.search(r"@sha256:[0-9a-fA-F]{64}$", self.image)
        ):
            raise ValueError(
                "Persistent Docker workspaces require a pinned image digest"
            )
        if not os.path.isabs(self.socket_path) or "\0" in self.socket_path:
            raise ValueError("Docker requires an absolute local Unix socket")
        socket_stat = os.stat(self.socket_path)
        limits = self.limits
        config = {
            "schema": 1,
            "image": self.image,
            "daemon_endpoint": os.path.realpath(self.socket_path),
            "daemon_device": socket_stat.st_dev,
            "daemon_inode": socket_stat.st_ino,
            "mount_policy": "bind-recursive-disabled:rprivate:read-only-rootfs",
            "network": "none",
            "uid": os.getuid(),
            "gid": os.getgid(),
            "limits": {
                "memory_bytes": limits.memory_bytes,
                "pids": limits.pids,
                "cpus": limits.cpus,
                "tmp_bytes": limits.tmp_bytes,
            },
        }
        return hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _bind(self, filesystem):
        executor = DockerProcessExecutor(
            filesystem,
            image=self.image,
            limits=self.limits,
            socket_path=self.socket_path,
        )
        return WorkspaceBinding(self, filesystem, executor, ownership="borrowed")
