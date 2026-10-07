# Embedded Agent Service

`AgentService` owns Agent executions independently of clients observing them.
A TUI or application can submit an input, disconnect its watcher, and attach
again while the run continues. No HTTP framework is required for this API.
The optional [HTTP/SSE adapter](service-http.md) exposes it to separate processes.
[Agent channels](channels.md) provide an application boundary for authorization,
commands, preprocessing, and presentation over the same service.

## Service, Agent Session, And Coding Session

| Component | Responsibility | Lifetime |
| --- | --- | --- |
| `AgentSession` | Supplies an Agent and its live dependencies: checkpoint/task stores, inbox, scope factory, and resource cleanup callback. | Created by the trusted host factory for one service thread. |
| `AgentService` | Owns executions across threads, persists admissions, and coordinates observation, interruption, steering, recovery, and shutdown. | Remains alive while clients attach or detach. |
| [`CodingSession`](coding-session.md) | Provides the application's API for one coding conversation: prompt, stream, history, observation, and run controls. | Creates an embedded service or attaches to a host-owned service. |

The service resolves each thread's `AgentSession` through its registered factory.
A `CodingSession` then delegates execution to that service using its thread ID.
Multiple coding facades can observe the same thread. They share the existing
checkpoint history and admission journal.

Both execution APIs use `prompt()`. On the service, callers supply `thread_id`
and a stable `request_id`; a coding session already identifies its thread and
can generate the request ID when one is omitted. Both return an admission receipt
once the input is recorded and work is scheduled. Use `wait()` for the settled
receipt, `watch()` for snapshot and future events, or `CodingSession.stream()`
for a finite iterator of one run's events.

## Prompt And Observe

The host registers a factory that creates a separate Agent for each thread.
`AgentSession` supplies its live dependencies; `SQLiteServiceStore` records
admission identities independently of conversation checkpoints.

```python
import asyncio

import msgflux as mf
import msgflux.nn as nn
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.runtime import AgentService, AgentSession, SQLiteServiceStore


async def main():
    journal = SQLiteServiceStore()  # :memory: for this example
    service = AgentService(store=journal)

    def create_session(thread_id):
        model = mf.Model.chat_completion("openai/gpt-6-luna")
        agent = nn.Agent(
            name="main",
            model=model,
            checkpoint_store=InMemoryCheckpointStore(),
            config={"stream": True},
        )
        return AgentSession(agent, on_close=model.aclose)

    service.register("main", create_session)
    thread = await service.open_thread("main")
    try:
        async with service.watch(thread.thread_id) as observer:
            print(observer.snapshot.messages)
            receipt = await service.prompt(
                thread.thread_id,
                "Explain this project briefly.",
                request_id="explain-project-1",
            )
            async for event in observer:
                if event.type == "message.delta":
                    print(event.data["delta"], end="", flush=True)
                elif event.type in {"run.end", "run.error", "run.paused"}:
                    if event.run_id == receipt.run_id and len(event.source_path) == 1:
                        break
        settled = await service.wait(thread.thread_id, receipt.request_id)
        print(settled.status)
    finally:
        await service.aclose()
        journal.close()


asyncio.run(main())
```

Set `OPENAI_API_KEY` before running. The watcher captures a snapshot and subscribes
to future events through `Agent.watch()`. The service consumes the producer's
`stream_events()` internally, so observation does not own execution. Child Agents
may emit their own terminal events; the example filters by run and root source.

The example uses memory-backed stores and does not survive a process restart.
Use a file-backed journal and persistent checkpoint/inbox/task stores when the
application needs persistence. Nothing is written to the home directory by default.

## Admission And Concurrency

`prompt()` returns after its admission record is committed and its worker is
scheduled. The returned receipt initially has status `accepted`; `receipt()` or
`wait()` returns the current state. A thread can have one foreground execution
at a time, while different threads run concurrently.

A repeated `(thread_id, request_id)` with the same input returns the same run.
Changing the prompt under the same identity raises `ServiceConflictError`.
Use a stable client-generated request ID for retries; correlation IDs are not
implicitly idempotency keys. Claims and finalization compare a journal revision
and owner, so a late finalization cannot replace a newer attempt.

`open_thread()` records its logical binding without constructing an Agent. The
factory runs when submission or observation first needs the session. Factory
failure must release resources it acquired; successful sessions remain owned
by the service until shutdown. Registering the same mutable Agent for different
threads is rejected.

## Disconnect, Wait, And Interrupt

Closing a watcher or cancelling `wait()` leaves execution running. A later
`watch()` starts with the current snapshot and future events. Live deltas are
process-local; this is not a persistent event replay cursor.

```python
receipt = await service.prompt(thread_id, "Run the tests", request_id="tests-1")
# The client can detach while the service remains alive.
snapshot = await service.snapshot(thread_id)
await service.interrupt(thread_id, receipt.run_id)
settled = await service.wait(thread_id, receipt.request_id)
```

This example requests cooperative cancellation of one explicit run. It does not
interrupt another thread or an old run merely because it was previously selected.
`interrupt()` returns `False` if that run has no local worker.

## Steering And Dependencies

```python
await service.steer(thread_id, receipt.run_id, "Use the existing test runner")
```

Steering publishes to the selected execution's existing AgentInbox. The Agent
places the message at its normal model-request boundary. This API does not
implement a follow-up queue or dedupe retries of steering messages.

For a workspace, create one binding in the host factory and reuse it in the
scope factory:

```python
from msgflux.runtime import AgentWorkspace, AgentSession


def create_session(thread_id):
    workspace = AgentWorkspace.local(".")
    agent = make_agent(thread_id)  # host-defined Agent factory
    return AgentSession(
        agent,
        scope_factory=lambda scope: scope.with_overrides(workspace=workspace),
        on_close=workspace.aclose,
    )
```

The base scope inherits the Agent's configured workspace. The scope factory
can supply current workspace, permissions and principal. It
preserves the thread and service-owned run identity. `checkpoint_store`,
`task_store` and `agent_inbox` can also be supplied to `AgentSession`. A checkpoint
store must agree with one already configured on the Agent. The API does not
serialize factories, resources, credentials, or grants.

## Run Identity And Host Bindings

`receipt_for_run(thread_id, run_id)` locates the admission associated with a saved
execution. `session(thread_id)` resolves the factory's `AgentSession` for trusted
in-process integrations; domain facades such as [CodingSession](coding-session.md)
use it to share dependencies without duplicating runtime ownership. Applications
can normally use prompt/watch without accessing live dependencies.

## Reopen And Recover

Use `SQLiteServiceStore("service.sqlite3")` to retain bindings and admissions.
Re-register trusted factories in a new service before opening an existing thread.
Persistent Agent checkpoints are a separate requirement; a journal alone cannot
reconstruct a started execution.

```python
receipt = service.receipt(thread_id, request_id)
# The host has inspected/reconciled any uncertain effects and established that
# the old worker stopped; clients cannot manufacture that evidence.
receipt = await service.resume(thread_id, request_id, worker_stopped=True)
```

An admission still `accepted` can start without replaying an earlier execution.
An attempt marked `running` after a crash requires quiescence and a checkpoint.
Without a checkpoint its outcome stays uncertain, so the service refuses to
resend the original prompt. A paused or failed run resumes under its existing
run ID and Agent's workspace, approvals and command-receipt validation. Only the
latest unfinished checkpoint can resume.

A terminal checkpoint written before the journal settled can reconcile its
receipt without another model call. These are separate transactions, not a
global transaction across checkpoint, task and admission stores. Store fencing
does not prevent a surviving old process from performing external effects.

`resume_checkpoint(thread_id, run_id, worker_stopped=...)` recovers by durable run
identity. If that checkpoint predates the admission journal, the host must first
establish quiescence and supply `worker_stopped=True`. The service validates the
checkpoint and records its existing identity before recovery; the original prompt
is not reconstructed. A terminal checkpoint settles without another model call.

## Shutdown And Ownership

`aclose()` stops admission, requests cooperative cancellation, joins foreground
workers and invokes session `on_close` callbacks. Cancelling its waiter leaves
shutdown running; another `aclose()` waits for the same operation. The service
borrows stores and does not close them automatically.

Close callbacks release factory-owned resources and must account for delegated
work using those resources. Non-cooperative code can delay shutdown. The service
is bound to one event loop; its journal operations are local synchronous SQLite
transactions. Network serving and daemon startup are separate integrations.
