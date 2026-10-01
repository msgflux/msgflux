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
from msgflux.runtime.workspace.environment import ProcessExecutor, ProcessResult
from msgflux.runtime.workspace.local import LocalWorkspace, LocalWorkspaceBackend
from msgflux.runtime.workspace.process_capture import drain_subprocess


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
        self.limits = limits or DockerLimits()
        if not isinstance(self.limits, DockerLimits):
            raise TypeError("Expected DockerLimits")
        self._docker = shutil.which("docker")
        if self._docker is None:
            raise FileNotFoundError("Docker CLI is not installed")
        self._prefix = (self._docker, "--host", f"unix://{socket_path}")

    def supports_workspace(self, filesystem):
        return filesystem is self.filesystem

    def _command(self, request, name):
        root = "/" + "/".join(self.filesystem._root_parts)
        if root == "/" or "," in root or "\n" in root:
            raise ValueError("Unsafe workspace mount root")
        os.close(self.filesystem._open_root())
        uid, gid = os.getuid(), os.getgid()
        if uid == 0:
            raise PermissionError("Run this adapter as a non-root workspace owner")
        limits = self.limits
        return (
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
            "--log-driver=none",
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,noexec,size={limits.tmp_bytes}",  # noqa: S108
            "--mount",
            f"type=bind,src={root},dst=/workspace,bind-recursive=disabled,bind-propagation=rprivate",
            "--workdir",
            "/workspace" + request.cwd.rstrip("/"),
            "--entrypoint",
            request.argv[0],
            self.image,
            *request.argv[1:],
        )

    async def _remove(self, name):
        process = await asyncio.create_subprocess_exec(
            *self._prefix,
            "rm",
            "--force",
            "--volumes",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=self._client_env(),
        )
        try:
            await asyncio.wait_for(process.wait(), 15)
        except BaseException:
            process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise RuntimeError(f"Container cleanup failed; reconcile {name}")

    @staticmethod
    def _client_env():
        # No application credentials are copied into the container or CLI env.
        return {
            "PATH": os.defpath,
            "HOME": "/nonexistent",
            "DOCKER_CONFIG": "/nonexistent",
        }

    async def _create(self, command):
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._client_env(),
            start_new_session=True,
        )

        async def discard(channel, data):
            del channel, data

        code = await drain_subprocess(
            process,
            discard,
            max_output_bytes=16384,
            timeout_seconds=30,
            owns_process_group=True,
        )
        if code:
            raise RuntimeError(
                "Docker container creation failed; check daemon/image/policy"
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
        name = "msgflux-" + uuid4().hex
        command = self._command(request, name)
        process = None
        failure = None
        creation = asyncio.create_task(self._create(command))
        # Shield launch: cancellation must not lose the just-created subprocess.
        launch = None
        try:
            await asyncio.shield(creation)
            launch = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *self._prefix,
                    "start",
                    "--attach",
                    name,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    stdin=asyncio.subprocess.DEVNULL,
                    env=self._client_env(),
                    start_new_session=True,
                )
            )
            process = await asyncio.shield(launch)
            code = await drain_subprocess(
                process,
                on_output,
                max_output_bytes=request.max_output_bytes,
                timeout_seconds=request.timeout_seconds,
                abort_signal=abort_signal,
                owns_process_group=True,
            )
            return ProcessResult(code)
        except BaseException as exc:
            failure = exc
            raise
        finally:

            async def cleanup():
                try:
                    await creation
                    if launch is not None:
                        launched = process or await launch
                        if launched.returncode is None:
                            try:
                                launched.kill()
                            except ProcessLookupError:
                                pass

                            async def discard(channel, data):
                                del channel, data

                            await drain_subprocess(
                                launched,
                                discard,
                                max_output_bytes=65536,
                                timeout_seconds=3,
                                owns_process_group=True,
                            )
                finally:
                    # Killing the CLI is insufficient: the daemon owns children.
                    await self._remove(name)

            task = asyncio.create_task(cleanup())
            cleanup_cancelled = None
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as exc:
                    cleanup_cancelled = exc
                    continue
                except BaseException:
                    break
            try:
                task.result()
            except BaseException as error:
                if failure is None:
                    raise
                failure.add_note(f"{error}; reconcile container {name}")
            if cleanup_cancelled is not None and failure is None:
                raise cleanup_cancelled


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
