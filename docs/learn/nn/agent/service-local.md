# Local Service Discovery And Startup

`connect_local_service()` connects to a matching local [HTTP/SSE backend](service-http.md)
and starts one when no process owns the runtime. Several frontends can reuse the
same AgentService. Closing a client or exiting its launcher leaves the daemon
running, including while an Agent works or waits for approval.

Install `msgflux[service]` for server support. The process manager in this increment
uses POSIX locks; the native HTTP client remains usable on other platforms.

## Define The Trusted Factory

The factory is application code, supplied as `module:callable`. The service
factory receives the runtime directory as a `Path` and returns an AgentService,
directly or through an async function. Each registered Agent factory receives a
`ServiceThread` binding with its immutable `thread_id`, `agent_id`, and optional
project `cwd`. Models, credentials, workspace grants and tools are configured by
that code, rather than by discovery files or remote request bodies.

Save this as `my_backend.py` in your project:

```python
import re

import msgflux as mf
from msgflux.coding import CodingCheckpointExtension
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.nn import Agent
from msgflux.runtime import AgentService, AgentSession, AgentWorkspace, SQLiteServiceStore


def create_service(runtime_dir):
    service = AgentService(store=SQLiteServiceStore(runtime_dir / "service.sqlite3"))

    async def create_agent(thread):
        # Application policy keeps thread IDs safe as directory names.
        if not re.fullmatch(r"[A-Za-z0-9_-]+", thread.thread_id):
            raise ValueError("Unsupported persistent thread ID")
        if thread.cwd is None:
            raise ValueError("This coding service requires a project cwd")
        folder = runtime_dir.parent / "threads" / thread.thread_id
        folder.mkdir(parents=True, exist_ok=True)
        model = workspace = checkpoints = None

        async def close():
            if model is not None:
                await model.aclose()
            if workspace is not None:
                await workspace.aclose()
            if checkpoints is not None:
                checkpoints.close()

        try:
            model = mf.Model.chat_completion("openai/gpt-6-luna", reasoning_effort="medium")
            workspace = AgentWorkspace.local(thread.cwd)
            checkpoints = SQLiteCheckpointStore(str(folder / "checkpoints.sqlite3"))
            agent = Agent(
                name="main", model=model, workspace=workspace,
                checkpoint_store=checkpoints, config={"stream": True},
            )
            agent.register_extension("coding_checkpoints", CodingCheckpointExtension())
            return AgentSession(agent, on_close=close)
        except BaseException:
            await close()
            raise

    service.register("main", create_agent)
    return service
```

This example configures persistent admission and checkpoint stores, a workspace,
and the coding checkpoint policy. It leaves tools empty; pass the tools and
permissions appropriate to your application when constructing the Agent.
Provider credentials belong in the server environment or the provider's configured
authentication source. The daemon inherits its launcher's environment.

The runner owns the returned service and closes its admission journal on shutdown.
Factories must release resources on failure and provide cleanup callbacks for their
successful sessions. Callbacks must drain any delegated work they own before
closing its dependencies.

## Connect On Demand

```python
import asyncio
from pathlib import Path

from msgflux.runtime.service.local import connect_local_service


async def main():
    client = await connect_local_service(
        "my_backend:create_service", cwd=Path.cwd(), startup_timeout=30
    )
    try:
        health = await client.health()
        print("Runtime:", health.instance_id)
        thread = await client.open_thread("main", cwd=Path.cwd())
        receipt = await client.prompt(
            thread.thread_id, "Explain this project", request_id="explain-1"
        )
        print("Thread:", thread.thread_id, "Run:", receipt.run_id)
    finally:
        await client.aclose()


asyncio.run(main())
```

The factory module must be importable from a stable installed location or the
selected launch working directory. `connect_local_service(..., cwd=...)` supplies
the working directory used to import and launch the factory when a daemon needs to
start. If a matching daemon is already healthy, a caller from another directory
reuses it; its own launch `cwd` does not select a new Agent workspace. Pass the
desired project path to `client.open_thread("main", cwd=Path.cwd())` instead.
That path is canonicalized and stored in the immutable thread binding. Reopening
without a path returns the existing binding, while a different explicit path
conflicts. The trusted factory can reject missing or disallowed roots and decides
whether to create a workspace there; the path grants no permissions by itself.
The runtime does not change process working directory. Different project
frontends can reuse the same daemon and open separate threads with their own
`cwd`; the service factory remains registered once for the backend. Keep the
factory module in a stable installed location, and use `thread.thread_id` for its
checkpoint directory so project paths do not choose or collide in storage. Use a
separate runtime directory for separate application configurations or factories.

Concurrent launchers coordinate startup and return clients for the same ready
instance. Discovery requires a matching authenticated health identity before
returning a connection. Initialization and shutdown of an existing owner are waited
for within `startup_timeout`; an owner with published metadata but an invalid
identity/health response is reported as requiring recovery and is not replaced.

## Gracefully Restart The Daemon

`restart_local_service()` asks the authenticated current instance to shut down,
waits for its lifetime lock to be released, and starts a replacement under the
same startup lock used by concurrent connectors. With no arguments it uses the
factory and launch directory recorded by the current daemon:

```python
import asyncio

from msgflux.runtime.service.local import restart_local_service


async def main():
    client = await restart_local_service(restart_timeout=45)
    try:
        print("New runtime:", (await client.health()).instance_id)
    finally:
        await client.aclose()


asyncio.run(main())
```

This example returns a client connected to the replacement instance. The
restart timeout bounds the complete stop and startup wait. If graceful shutdown
is accepted but the old daemon does not release its lock before the deadline,
the call raises `TimeoutError`; the old daemon continues closing its resources.
No replacement starts while it holds the lifetime lock. A connection attempt
may report `ServiceRecoveryRequiredError` while the old HTTP server is no longer
healthy; retry after shutdown completes. The timeout does not force-kill a worker
or signal a process.

Restart uses the existing service shutdown behavior: active runs receive a
cooperative interruption request, and shutdown joins their workers before closing
factory resources. Their receipts and checkpoints retain the interruption state.
Restart does not automatically replay prompts or resume paused approvals.

Pass `factory` and `cwd` to select new trusted launch configuration:

```python
client = await restart_local_service(
    "my_backend:create_service", runtime_dir="~/.msgflux/runtime",
    cwd="/absolute/project/path", restart_timeout=45,
)
```

An explicit factory must match the recorded factory while its daemon is alive.
If metadata is unavailable, supply a factory explicitly; a dead daemon's PID is
never used to infer or signal an owner. A healthy older daemon that does not
provide authenticated graceful shutdown raises `ServiceRecoveryRequiredError`;
stop or upgrade it through its existing operator-managed mechanism before
restarting.

The CLI keeps the existing foreground invocation unchanged. `serve` is the
default action; `restart` derives the factory and launch directory from private
runtime metadata and prints the new instance identity and loopback URL:

```bash
uv run --extra service msgflux-service restart \
  --runtime-dir ~/.msgflux/runtime --restart-timeout 45
```

The returned object is an ordinary `AgentServiceClient`. Use `watch()` to get an
atomic snapshot and future events. Reconnecting after a server restart requires
calling `connect_local_service()` again so the client receives the new URL/token;
old connections are not silently redirected or retried.

## Run In The Foreground

For an explicitly managed persistent process:

```bash
uv run --extra service msgflux-service \
  --factory my_backend:create_service \
  --cwd /absolute/project/path
```

The explicit spelling `msgflux-service serve --factory ...` is equivalent.

The equivalent Python entry point is
`python -m msgflux.runtime.service.local.cli`. A foreground server and clients
using on-demand discovery share the same lifetime lock. A competing server fails
without overwriting the active instance's metadata.

Ctrl+C or an explicit operating-system stop requests shutdown. Active SSE responses
are given a bounded graceful shutdown interval; the service then joins its owned
foreground work and closes factory resources. Agent interruption remains cooperative,
so non-cooperative application code can delay resource cleanup. There is no idle
shutdown timer and no shutdown request when a client closes.

## Files And Identity

The default directory is `~/.msgflux/runtime/`. Override it with `runtime_dir` in
Python or `--runtime-dir` on the command line.

```text
~/.msgflux/runtime/
├── daemon.json
├── daemon.lock
├── startup.lock
└── daemon.log
```

| File | Purpose |
| --- | --- |
| `daemon.json` | Versioned instance ID, factory, cwd, loopback URL, informational PID, and private connection token |
| `daemon.lock` | Kernel lock held for the daemon's process lifetime |
| `startup.lock` | Coordinates frontend discovery, spawn and readiness |
| `daemon.log` | Startup/server diagnostics for on-demand launches |

The selected directory is owned by the current user and secured to mode 0700.
Metadata and lock/log files use mode 0600; metadata is published atomically and
symlink metadata/lock paths are rejected. The token is local runtime authentication,
not a provider credential. Do not share the metadata file. Metadata representations
redact the token.

The backend binds only to `127.0.0.1` on an available port. `GET /v1/health` requires
the bearer token and returns the runtime instance ID and protocol version without
constructing an Agent. A PID from a discovery file is never used to signal or kill
a process. Stale metadata may be replaced when no owner holds the lifetime lock;
failed startup cleanup operates only on the child handle created by that launcher.

Factory-owned conversation stores stay separate from discovery files. The example
adds `runtime/service.sqlite3` and `threads/<thread_id>/checkpoints.sqlite3`; the
process manager does not choose conversation storage or model credentials itself.

## Failure And Recovery

A launcher may exit immediately after connecting; the daemon's lifecycle is
independent of that frontend's event loop. If startup fails or times out, inspect
`daemon.log`. A canceled/failed startup releases the launcher's own child and startup
lock; it does not terminate a process identified by stale metadata.

A process crash releases its kernel lock. The next connection can start a new
instance while preserving application-owned persistent stores. This does not
implicitly retry started Agent inputs or assert quiescence of surviving external
commands. Existing checkpoint, workspace, command-receipt and approval recovery
checks remain authoritative. A trusted factory/host can perform explicit recovery
before serving clients; network clients cannot manufacture `worker_stopped` evidence.
