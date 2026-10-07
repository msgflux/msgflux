# Agent Service Over HTTP

The native HTTP adapter serves an existing [AgentService](service.md) using JSON
and Server-Sent Events (SSE). A separate application can prompt a configured Agent,
observe its progress, detach, and reconnect while execution remains owned by the
service. Coding Agents retain the same workspace, checkpoint and inbox behavior.

Install the optional server dependencies:

```bash
uv add "msgflux[service]"
```

In a repository checkout, use `uv run --extra service ...`. Litestar and Uvicorn
belong to this extra; importing the client does not require Litestar. Request and
response contracts use `msgspec.Struct` with strict JSON decoding.

For a shared local process, [local discovery/startup](service-local.md) provides
on-demand connection and a foreground command without creating another runtime
when an existing instance is healthy.

## Run A Server

Save this as `server.py` and set `MSGFLUX_SERVICE_TOKEN` to a secret shared with
authorized clients. Configure the provider credentials in the server environment.

```python
import asyncio
import os

import uvicorn
import msgflux as mf
from msgflux.coding import CodingCheckpointExtension
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.nn import Agent
from msgflux.runtime import AgentService, AgentSession, AgentWorkspace, SQLiteServiceStore
from msgflux.runtime.service.http import create_service_app
from msgflux.tools.builtin import ReadFileTool


async def main():
    journal = SQLiteServiceStore()
    service = AgentService(store=journal)

    def create_session(thread):
        if thread.cwd is None:
            raise ValueError("This coding service requires a project cwd")
        model = mf.Model.chat_completion("openai/gpt-6-luna", reasoning_effort="medium")
        workspace = AgentWorkspace.local(thread.cwd)
        agent = Agent(
            name="main",
            model=model,
            workspace=workspace,
            checkpoint_store=InMemoryCheckpointStore(),
            tools=[ReadFileTool()],
            config={"stream": True},
        )
        agent.register_extension("coding_checkpoints", CodingCheckpointExtension())

        async def close():
            await model.aclose()
            await workspace.aclose()

        return AgentSession(agent, on_close=close)

    service.register("main", create_session)
    app = create_service_app(service, token=os.environ["MSGFLUX_SERVICE_TOKEN"])
    try:
        await uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=8765, workers=1)
        ).serve()
    finally:
        await service.aclose()
        journal.close()


asyncio.run(main())
```

The factory creates a distinct Agent for each thread. The example grants a read
tool and uses memory-backed stores; it demonstrates process-local reconnection,
not persistence across server restarts. The host configures model credentials,
permissions, principal and tools. The thread's optional `cwd` is an absolute
canonical host path stored in its immutable service binding. A client may request
a project location when opening the thread; the trusted factory decides whether
to use that path and which workspace and permissions to grant. A path alone
grants no tool access. Reopening the thread without a path returns its stored
binding, and supplying a different path conflicts. Clients cannot replace model
settings or grants through the HTTP payload.

This adapter runs in a single process around one service. Local daemon discovery/startup is available through the [local process API](service-local.md).
Coordination across independent server workers remains a separate integration. Do not run multiple Uvicorn workers against one in-memory service.

## Connect And Observe

Save this as `client.py` and run it with the same service token:

```python
import asyncio
import os
from pathlib import Path

from msgflux.runtime.service.http import AgentServiceClient


async def main():
    client = AgentServiceClient(
        "http://127.0.0.1:8765", token=os.environ["MSGFLUX_SERVICE_TOKEN"]
    )
    try:
        # The thread is bound to this project path for its lifetime.
        thread = await client.open_thread("main", cwd=Path.cwd())
        async with client.watch(thread.thread_id) as observer:
            print("Existing history:", observer.snapshot.messages)
            receipt = await client.prompt(
                thread.thread_id,
                "Read pyproject.toml and explain the project",
                request_id="explain-project-1",
            )
            async for event in observer:
                if event.type == "message.delta":
                    print(event.data.get("delta", ""), end="", flush=True)
                if (
                    event.run_id == receipt.run_id
                    and len(event.source_path) == 1
                    and event.type in {"run.end", "run.error", "run.paused"}
                ):
                    break
        # The terminal event is published before the journal finalization.
        while True:
            settled = await client.receipt(thread.thread_id, receipt.request_id)
            if settled.status not in {"accepted", "running"}:
                break
            await asyncio.sleep(0.05)
        print("\nStatus:", settled.status)
    finally:
        await client.aclose()


asyncio.run(main())
```

The client creates and closes its own connection pool. An optional injected
`httpx2.AsyncClient` remains owned by the caller. SSE read timeouts are disabled
for observation; normal requests have a bounded timeout. Calls do not retry or
switch providers automatically.

`client.prompt()` returns an `AdmissionReceipt` after admission. Stable request
IDs deduplicate repeated inputs within a thread. `watch()` exposes a portable
`SnapshotRecord`, then iterates `EventRecord` objects. History is a tuple of chat
message dictionaries, and live tools/tasks/approvals are JSON dictionaries; the
client does not reconstruct executable resources from serialized objects.

Saved run discovery uses `await client.runs(thread_id)`, which returns a tuple
of typed `RunSummary` records containing only `run_id`, `status`, and
`updated_at` metadata, newest first. This includes checkpoints created before
the service admission journal, which lets a client discover runs for resuming
through the normal recovery API. Threads without a checkpoint store return an
empty tuple. Saved checkpoint state, serialized config, and credentials are
never included.

```python
saved = await client.runs(thread_id)
for run in saved:
    print(run.run_id, run.status, run.updated_at)
```

This lists resumable checkpoint identities without loading saved state into the
client. Pass a selected identity to the explicit resume operation when the
service's recovery rules allow it.

## Detach And Reconnect

```python
async with client.watch(thread_id) as observer:
    print(observer.snapshot.active_runs)
    # Consume any desired events, then leave this block.

# The producer remains owned by the server.
async with client.watch(thread_id) as reattached:
    print(reattached.snapshot.messages)
    async for event in reattached:
        print(event.type)
```

The initial snapshot and subscription are captured together by `Agent.watch()`.
Reconnection starts with current history and live state, followed by future events.
Old token deltas are not a durable replay log. `Last-Event-ID` is rejected with
422; the adapter does not advertise resumable delta cursors.

Each observer has a buffer limit of 1024 events by default. The host can configure
`event_buffer_limit` when creating the app, including `None` for an unbounded
buffer. Overflow emits an SSE error asking the client to reconnect, closes that
observer, and leaves the Agent running. The client raises `AgentServiceHTTPError`;
open another watcher to obtain a fresh snapshot. There are no heartbeat frames or
automatic reconnect loops in this increment.

## Run Controls And Recovery

```python
await client.steer(thread_id, receipt.run_id, "Focus on integration tests")
interrupted = await client.interrupt(thread_id, receipt.run_id)
resumed = await client.resume_checkpoint(thread_id, receipt.run_id)
```

Steering returns the published notification as a JSON dictionary and is not a
follow-up queue. Interruption targets an explicit run and returns a boolean.
An unfinished run already paused or failed in this service can resume through its
normal recovery checks. Approval decisions remain a trusted host operation in
this increment; the HTTP resume route does not itself approve a tool invocation.

HTTP clients cannot assert `worker_stopped` or import uncertain executions after
a process restart. A trusted host must establish old-worker quiescence and use
the Python recovery API first. Persistent reconnection requires a file-backed
admission journal and checkpoint stores, plus persistent inbox/task stores when
those features must survive restart. The HTTP adapter reuses these stores and
creates no second conversation history.

## Endpoints And Errors

All routes require `Authorization: Bearer <token>`.

| Method | Route | Result |
| --- | --- | --- |
| GET | `/v1/health` | Authenticated runtime identity and protocol version |
| GET | `/v1/agents` | Registered agent IDs |
| GET / POST | `/v1/threads` | List bindings / open a thread |
| GET | `/v1/threads/{thread_id}/snapshot` | Portable thread snapshot |
| GET | `/v1/threads/{thread_id}/runs` | Saved run metadata summaries |
| POST | `/v1/threads/{thread_id}/prompt` | Admission receipt |
| GET | `/v1/threads/{thread_id}/requests/{request_id}` | Current receipt |
| GET | `/v1/threads/{thread_id}/watch` | Initial snapshot and continuous SSE events |
| POST | `/v1/threads/{thread_id}/runs/{run_id}/interrupt` | Interruption result |
| POST | `/v1/threads/{thread_id}/runs/{run_id}/steer` | Published notification |
| POST | `/v1/threads/{thread_id}/runs/{run_id}/resume` | Recovery receipt |

Thread-open bodies contain `agent_id` and may include `thread_id` and an absolute
`cwd`. Prompt bodies contain `prompt` and `request_id`; steer bodies contain
`content`; resume bodies are empty JSON objects. Unknown fields are rejected.
The native API is separate from any future Chat Completions compatibility adapter.

Errors have the JSON shape `{"code": "...", "message": "..."}`. Missing or invalid
authentication returns 401, unknown resources return 404, conflicts/busy threads
and recovery requirements return 409, and invalid payloads return 422. Unexpected
server errors return a generic 500 message and are logged on the host. Model
failures settle their run receipts and appear as `run.error` events, rather than
turning accepted prompts into transport validation errors.

## Ownership And Coding Sessions

`create_service_app()` borrows the service by default. Set `close_service=True`
when application shutdown owns service shutdown; borrowed stores still remain
host-owned. Resource callbacks must drain any delegated work they own before
releasing its model/workspace resources.

An existing embedded coding session can expose the same backend:

```python
session = CodingSession(agent)
app = create_service_app(session.service, token=service_token)
try:
    await uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8765)).serve()
finally:
    await session.aclose()
```

Clients attach to `session.thread_id`. This one-Agent convenience serves that
conversation; register per-thread factories on a shared AgentService to serve
multiple independent conversations. The application still owns the Agent's model,
workspace and supplied stores. A future Telegram or other channel adapter can use
the native client while keeping platform identity, authorization, message formatting
and delivery retries in the adapter.


## Thread-bound Client

`AgentSessionClient` binds a native HTTP client to one existing service thread.
The service host still creates the `AgentSession` and owns its Agent, workspace,
stores, credentials, and execution lifecycle. `AgentSessionClient` is a generic remote thread facade, including for coding
conversations. It does not construct or expose the Agent. Observation is
separate from admission, using `watch()` rather than a producer `stream()`.

Connect through local service discovery, bind the current project root, and
observe a run:

```python
import asyncio
from pathlib import Path

from msgflux.runtime.service.http import AgentSessionClient
from msgflux.runtime.service.local import connect_local_service


async def main():
    client = await connect_local_service("my_app:build_service", cwd=Path.cwd())
    session = await AgentSessionClient.open(
        client, agent_id="main", cwd=Path.cwd()
    )
    try:
        async with session.watch() as observer:
            print("Existing history:", observer.snapshot.messages)
            receipt = await session.prompt(
                "Explain the package layout", request_id="layout-1"
            )
            async for event in observer:
                if event.run_id == receipt.run_id and len(event.source_path) == 1:
                    print(event.type)
                    if event.type in {"run.end", "run.error", "run.paused"}:
                        break
        settled = await session.wait(receipt.request_id)
        print("Status:", settled.status)
        print("Saved run summaries:", await session.runs())
    finally:
        await client.aclose()


asyncio.run(main())
```

The stable request ID lets a caller safely check admission after a network
uncertainty without silently submitting a second prompt. To reconnect after a
watch disconnect, open another `session.watch()` context and replace the local
view with its new `observer.snapshot`; missed event deltas are not replayed.
`wait()` polls the receipt and does not observe events or cancel execution if
its caller is cancelled. Use `cancel(run_id)` for an explicit interruption.
`runs()` and `latest_run()` expose only service-provided run summaries, not
checkpoint contents or configuration. The remote facade has no automatic
reconnect loop. `session.aclose()` is an optional no-op because the facade
borrows its client and owns no background tasks; a UI can close its active watch
context on shutdown, while the host closes the shared HTTP client.
