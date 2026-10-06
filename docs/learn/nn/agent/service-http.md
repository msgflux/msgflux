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

    def create_session(thread_id):
        model = mf.Model.chat_completion("openai/gpt-6-luna", reasoning_effort="medium")
        workspace = AgentWorkspace.local(".")
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
workspace, permissions, principal and tools. Clients cannot replace those
settings through the HTTP payload.

This adapter runs in a single process around one service. Daemon discovery,
automatic startup and coordination across independent server workers are separate
integrations. Do not run multiple Uvicorn workers against one in-memory service.

## Connect And Observe

Save this as `client.py` and run it with the same service token:

```python
import asyncio
import os

from msgflux.runtime.service.http import AgentServiceClient


async def main():
    client = AgentServiceClient(
        "http://127.0.0.1:8765", token=os.environ["MSGFLUX_SERVICE_TOKEN"]
    )
    try:
        thread = await client.open_thread("main")
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
| GET | `/v1/agents` | Registered agent IDs |
| GET / POST | `/v1/threads` | List bindings / open a thread |
| GET | `/v1/threads/{thread_id}/snapshot` | Portable thread snapshot |
| POST | `/v1/threads/{thread_id}/prompt` | Admission receipt |
| GET | `/v1/threads/{thread_id}/requests/{request_id}` | Current receipt |
| GET | `/v1/threads/{thread_id}/watch` | Initial snapshot and continuous SSE events |
| POST | `/v1/threads/{thread_id}/runs/{run_id}/interrupt` | Interruption result |
| POST | `/v1/threads/{thread_id}/runs/{run_id}/steer` | Published notification |
| POST | `/v1/threads/{thread_id}/runs/{run_id}/resume` | Recovery receipt |

Prompt bodies contain `prompt` and `request_id`; steer bodies contain `content`;
resume bodies are empty JSON objects. Unknown fields are rejected. The native
API is separate from any future Chat Completions compatibility adapter.

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
