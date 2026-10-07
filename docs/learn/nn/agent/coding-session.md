# Coding Session

`CodingSession` is a Python API for one coding conversation. It wraps
[AgentService](service.md) so applications can submit requests, observe progress,
interrupt a run, and reconnect without assembling a service factory themselves.
It does not require a terminal interface or an HTTP framework.

The session is the application-facing facade for one thread. `AgentService` owns
its executions; `AgentSession` is the host-created binding containing the Agent
and runtime dependencies. See [their responsibilities and lifetimes](service.md#service-agent-session-and-coding-session).
`session.prompt(text)` returns an admission receipt; `session.stream(text)`
observes a prompted run until its attempt settles.

## Embedded Session

Configure the Agent's model, workspace, tools and permissions normally. The
session creates an embedded service and an in-memory admission journal.

```python
import asyncio

import msgflux as mf
from msgflux.coding import CodingSession
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.nn import Agent
from msgflux.runtime import AgentWorkspace
from msgflux.tools.builtin import ReadFileTool


async def main():
    model = mf.Model.chat_completion("openai/gpt-6-luna", reasoning_effort="medium")
    workspace = AgentWorkspace.local(".")
    agent = Agent(
        name="main",
        model=model,
        workspace=workspace,
        tools=[ReadFileTool()],
        checkpoint_store=InMemoryCheckpointStore(),
        config={"stream": True},
    )
    session = CodingSession(agent)
    try:
        async for event in session.stream("Read pyproject.toml and explain the project."):
            if event.type == "message.delta":
                print(event.data.get("delta", ""), end="", flush=True)
        print("\nThread:", session.thread_id)
    finally:
        await session.aclose()
        await workspace.aclose()
        await model.aclose()


asyncio.run(main())
```

Set `OPENAI_API_KEY` and run from a directory containing `pyproject.toml`. The
example gives the Agent a read tool; the session does not select providers, add
tools, or change workspace permissions. Its namespace is the Agent's module name.
Embedded coding sessions checkpoint the canonical context before each model
request when a checkpoint store is configured.

`stream()` observes a service-owned execution. Closing its iterator detaches the
observer; execution continues while the service remains alive. Use `cancel(run_id)`
for explicit interruption and `aclose()` for embedded runtime shutdown. An
ordinary model failure raises `RuntimeError`; an approval pause raises
`TaskPauseRequestedError` after its events have been delivered.

## Prompt And Reconnect

Separate admission from observation when a client may disconnect:

```python
async with session.watch() as observer:
    print(observer.snapshot.messages)
    receipt = await session.prompt("Explain the tests", request_id="explain-tests-1")
    event = await anext(observer)
    print(event.type)
# Only the observer is closed here.
settled = await session.wait(receipt.request_id)
snapshot = await session.snapshot()
print(settled.status, snapshot.messages)
```

`watch()` atomically subscribes and captures the existing Agent snapshot. Later
observers receive the current snapshot and future events; old token deltas are
not replayed. `wait()` waits for the selected admission without coupling its
execution to the waiting client. `receipt(request_id)` reads its current status.

Use stable request IDs for input retries. Reusing an ID with the same prompt
returns the same run; a different prompt raises `ServiceConflictError`. A thread
admits one foreground execution at a time. Additional fronts can observe that
same execution.

```python
await session.steer(receipt.run_id, "Focus on the integration tests")
await session.cancel(receipt.run_id)
```

These calls target an explicit live run. Steering enters the AgentInbox at the
Agent's normal model-request boundary. It is not a follow-up queue and does not
deduplicate repeated steering submissions. The Agent's default inbox is in
memory; supply `agent_inbox` with a persistent store when queued messages must
survive reconstruction. Host-provided `task_store` can also be injected through
the session constructor.

## Persistent State

Memory-backed defaults do not survive process exit. For persistent state, supply
both an admission journal and an Agent checkpoint store:

```python
from msgflux.coding import CodingSession
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.runtime import SQLiteServiceStore

checkpoints = SQLiteCheckpointStore("checkpoints.sqlite3")
journal = SQLiteServiceStore("service.sqlite3")
agent = make_agent(checkpoints)  # application factory, with a stable Agent name
session = CodingSession(
    agent,
    thread_id="my-thread",
    checkpoint_store=checkpoints,
    service_store=journal,
)
try:
    receipt = await session.prompt("Inspect the repository", request_id="inspect-1")
    print((await session.wait(receipt.request_id)).status)
finally:
    await session.aclose()
    journal.close()
    checkpoints.close()
```

The example writes at explicit paths. The session does not create a home
configuration directory. The journal owns admission identities and status;
checkpoints remain authoritative for conversation history and execution state.
User-supplied stores, Agent, model and workspace remain owned by the application.

`runs()`, `latest_run()` and `saved_state(run_id)` inspect the configured checkpoint
store. A completed or interrupted run is terminal: submit a new prompt to continue
the conversation. Resume an unfinished run explicitly:

```python
async for event in session.resume(run_id, worker_stopped=True):
    print(event.type)
```

A trusted host may assert `worker_stopped=True` only after establishing that the
old worker stopped and reconciling uncertain effects. Do not copy this value from
an untrusted client request. Locally settled paused runs can resume without that
assertion. Checkpoints predating an admission journal require explicit import
under this same quiescence condition. Recovery retains the existing run identity
and saved context; it does not submit the old user message again. Workspace,
command-receipt and approval validation still apply.

## Shared Runtime

An application managing several threads can register factories once and attach
coding facades to existing service threads:

```python
from pathlib import Path

from msgflux.coding import CodingCheckpointExtension, CodingSession
from msgflux.runtime import AgentService, AgentSession, AgentWorkspace, SQLiteServiceStore


def create_agent_session(thread):
    if thread.cwd is None:
        raise ValueError("This coding service requires a project cwd")
    agent = make_agent(thread.thread_id)  # application factory
    workspace = AgentWorkspace.local(thread.cwd)
    agent.workspace = workspace
    agent.register_extension("coding_checkpoints", CodingCheckpointExtension())
    return AgentSession(agent, on_close=workspace.aclose)


journal = SQLiteServiceStore("service.sqlite3")
service = AgentService(store=journal)
service.register("main", create_agent_session)  # returns AgentSession per thread
thread = await service.open_thread(
    "main", thread_id="my-thread", cwd=Path.cwd()
)
first = await CodingSession.from_service(service, thread.thread_id)
second = await CodingSession.from_service(service, thread.thread_id)
try:
    receipt = await first.prompt("Inspect the project", request_id="shared-1")
    print((await second.wait(receipt.request_id)).status)
finally:
    await first.aclose()
    await second.aclose()
    await service.aclose()
    journal.close()
```

Configure shared Agents, including their checkpoint extension, in the trusted
factory before execution starts. Attaching a facade does not modify a live Agent's
hooks or tools. Both facades refer to the same execution owner and history. Closing a shared
facade does not close the service or interrupt its runs. The host closes the
service and factory-owned resources when the runtime itself shuts down. Services
are bound to one event loop. HTTP/SSE transport and terminal interfaces can consume
this API separately.

The factory receives a `ServiceThread`, not just a thread ID. Use
`thread.thread_id` for per-thread Agent/checkpoint identity and `thread.cwd` when
the application chooses to bind a workspace to the project's canonical host
directory. A required project root should be rejected when it is missing. The
stored `cwd` is immutable and does not grant workspace or tool permissions;
factories still decide which resources and permissions to provide.


## Remote Conversations

Use the generic [`AgentSessionClient`](service-http.md#thread-bound-client)
when an application or TUI connects to a separate Agent service. It represents
one remote thread and does not construct an Agent or own backend resources.
There is no parallel coding-specific HTTP client. Coding policies, tools, models
and workspace grants remain configured by the host.
