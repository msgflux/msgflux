# Embedded Agent Service

`AgentService` owns Agent executions independently of clients observing them.
A TUI or application can submit an input, disconnect its watcher, and attach
again while the run continues. No HTTP framework is required for this API.
The optional [HTTP/SSE adapter](service-http.md) exposes it to separate processes.

## Service, Agent Session, And Coding Session

| Component | Responsibility | Lifetime |
| --- | --- | --- |
| `AgentSession` | Supplies an Agent and its live dependencies: checkpoint/task stores, inbox, scope factory, and resource cleanup callback. | Created by the trusted host factory for one service thread. |
| `AgentService` | Owns executions across threads, persists admissions, and coordinates observation, interruption, steering, recovery, and shutdown. | Remains alive while clients attach or detach. |
| [`CodingSession`](coding-session.md) | Provides the application's API for one coding conversation: prompt, stream, history, observation, and run controls. | Creates an embedded service or attaches to a host-owned service. |

The service resolves each thread's `AgentSession` through its registered factory.
The factory receives the immutable `ServiceThread` binding, including its
`thread_id`, `agent_id`, and optional project `cwd`.
A `CodingSession` then delegates execution to that service using its thread ID.
Multiple coding facades can observe the same thread. They share the existing
checkpoint history and admission journal.

Both execution APIs use `prompt()`. On the service, callers supply `thread_id`
and a stable `request_id`; a coding session already identifies its thread and
can generate the request ID when one is omitted. Both return an admission receipt
once the input is recorded and work is scheduled. Use `wait()` for the settled
receipt, `watch()` for snapshot and future events, or `CodingSession.stream()`
for a finite iterator of one run's events. The `CodingSession` facade exposes
asynchronous `receipt()`, `runs()`, `latest_run()`, and `inspect_run()` queries so local and
service-backed sessions have the same interface. Run listings contain service
`RunSummary` records; checkpoint state remains host-local through its synchronous
`saved_state()` method. `AgentService.receipt()` itself remains synchronous for
trusted in-process host code. `CodingSession.snapshot()` and `watch()` expose
portable `SnapshotRecord` and `EventRecord` values, including when the facade is
attached to a local service. The lower-level `AgentService.snapshot()` and
`watch()` continue to use native `ThreadSnapshot` and `ExecutionEvent` values.
`CodingSession.stream()` remains a local convenience and yields `EventRecord`;
there is no corresponding stream method on the remote session client.

For operational recovery, `await service.inspect_run(thread_id, run_id)` returns
a `RunInspection` with the optional admission `receipt`, `checkpoint_status` and
`checkpoint_revision`, whether a `local_worker` is present, whether
`requires_quiescence` is set, `approval_phase`, `approval_request_count`,
`background_tasks` summaries, and human-readable `reasons`. Each task summary
contains `task_id`, `tool_name`, `status`, `updated_at`, and `error`. These
summaries are selected from the TaskStore for the thread and tasks whose
`root_run_id` or `parent_run_id` matches the inspected run. They omit task
results, progress metadata, arguments, and ownership claims. This inspection is
observational: it excludes raw checkpoint state, prompts, and owner IDs, and it
does not provide a `safe_to_resume` decision. `requires_quiescence` means the
trusted host still needs to establish that the old worker stopped; the
inspection itself is not that assertion.

Task summaries read persisted TaskStore state, independently of
`watch().snapshot.background_tasks`. A `queued` or `running` status after
restart does not prove a worker is alive, and a completed foreground checkpoint
does not imply its detached background tasks have completed. Reconcile task
ownership through the task recovery workflow before taking action.

The admission journal and checkpoint store are separate. Their statuses or
revisions can disagree, and inspection reads them at different times rather than
as one atomic snapshot. Treat `reasons` as diagnostic evidence, then reconcile
the underlying state and re-inspect before taking an action. Approval phase and
pending count are summaries only; use the host-authorized approval review flow
to inspect and decide individual requests. `approval_request_count` counts
requests in the saved batch, including requests with recorded decisions.
The `background_tasks` field is a tuple of lightweight `TaskSummary` records,
not a claim that those tasks are currently executing. Use the task store and
[background task recovery workflow](tools/background-tasks.md#inspecting-and-coordinating-recovery)
for durable task inspection and recovery.

## Prompt And Observe

The host registers a factory that creates a separate Agent for each thread.
`AgentSession` supplies its live dependencies; `SQLiteServiceStore` records
admission identities independently of conversation checkpoints.

```python
import asyncio
from pathlib import Path

import msgflux as mf
import msgflux.nn as nn
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.runtime import AgentService, AgentSession, SQLiteServiceStore


async def main():
    journal = SQLiteServiceStore()  # :memory: for this example
    service = AgentService(store=journal)

    def create_session(thread):
        model = mf.Model.chat_completion("openai/gpt-6-luna")
        agent = nn.Agent(
            name="main",
            model=model,
            checkpoint_store=InMemoryCheckpointStore(),
            config={"stream": True},
        )
        return AgentSession(agent, on_close=model.aclose)

    service.register("main", create_session)
    thread = await service.open_thread("main", cwd=Path.cwd())
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

Admission also checks the latest checkpoint before scheduling new work. If a
workspace command receipt has an uncertain outcome, `prompt()` raises
`ServiceRecoveryRequiredError` with the reconciliation reason and does not
admit or start another run. Recovery checks report the same error when a command
receipt still needs host reconciliation. Reconcile the command through the
trusted workspace host, then retry admission or recovery. An approval request
raised during a run is different: the run settles as `paused` and can be reviewed
and resumed through the approval flow.

`open_thread()` records its logical binding without constructing an Agent. Its
optional `cwd` must be an absolute path to an existing directory; it is resolved to an
absolute canonical host path and becomes part of the immutable thread binding.
Reopening an existing thread without `cwd` returns its stored binding, while a
different explicit `cwd` conflicts. The path identifies a project location; it
does not grant filesystem or process permissions. Those remain factory-owned.
The factory runs when submission or a cold snapshot first needs the session.
Factory failure must release resources it acquired; successful sessions remain
loaded until the host releases them or shuts down the service. Registering the
same mutable Agent for different threads is rejected.

Existing SQLite journals gain the optional `cwd` column automatically. Older
threads keep `cwd=None`; opening them does not infer a workspace from the
frontend's current directory. A coding factory requiring a root must reject
such a binding. Open a new thread with an explicit root to start a new coding
conversation there.

## Disconnect, Wait, And Interrupt

Closing a watcher or cancelling `wait()` leaves execution running. Attaching a
watcher briefly loads the thread binding to capture its snapshot and subscribe
atomically, then drops its references to the Agent and stores. The watcher does
not pin those resources, so the host can release an idle binding while the
watcher remains open. The consumer can drain events already queued, and later
events published for that thread reach the same watcher after a fresh binding
loads. Live deltas are process-local; they are not a persistent event replay
cursor. A new snapshot query for an unloaded thread still calls the registered
factory to load its trusted dependencies.

`AgentService.aclose()` first stops admissions and drains its foreground workers,
then detaches only watchers attached through that service. Closing their queues
preserves already published events, so consumers can drain those events and
then reach the end of the iterator. Standalone watchers created directly from
an Agent or EventHub are not owned by the service and remain open.

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


def create_session(thread):
    if thread.cwd is None:
        raise ValueError("This coding service requires a project cwd")
    workspace = AgentWorkspace.local(thread.cwd)
    agent = make_agent(thread.thread_id)  # host-defined Agent factory
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

The service does not construct a local workspace from `cwd` automatically.
The factory can also use that host project directory with an existing Docker
backend and `AgentWorkspace.open()`. Tools still receive `AgentWorkspace` through
the same dependency injection, regardless of the chosen backend.

This thread-opening API identifies a directory on the service host; it does not
accept backend selection, connection credentials, container options, or an
exclusively remote filesystem path. Per-thread backend profiles and remote
locations require a future persisted workspace configuration. A host changing
its factory's backend must explicitly address existing thread and checkpoint
bindings rather than silently moving a conversation to another environment.

## Run Identity And Host Bindings

`receipt_for_run(thread_id, run_id)` locates the admission associated with a saved
execution. Applications can normally use prompt/watch without accessing live
dependencies. A trusted host that needs the live `AgentSession` acquires a lease:

```python
lease = await service.acquire_session(thread_id)
try:
    session = lease.session
    print(session.namespace)
finally:
    await lease.aclose()
```

The lease pins that loaded binding until `aclose()`. Do not keep the session or
its Agent, stores, inbox, or workspace after releasing the lease. A
service-backed [CodingSession](coding-session.md) facade stores the service and
thread identity and does not pin an Agent; its ordinary prompt, watch, history,
and recovery operations remain available across a release.

`await service.release_session(thread_id)` closes and evicts an idle loaded
binding, returning `True`; an already unloaded thread returns `False`. The
service raises `ServiceBusyError` while a foreground worker, lease, or
delegated task is using the binding. An attached watcher does not pin the
binding after its initial snapshot and subscription are captured. Cleanup
failure quarantines the binding and raises `ServiceRecoveryRequiredError` with
the cleanup cause, so a new factory result cannot overlap an incompletely
closed generation. Release preserves the durable thread binding, admissions,
checkpoints, tasks, approvals, and history; the next operation loads a fresh
Agent from the registered factory. It does not delete a thread or reset its run
IDs.

Host cleanup callbacks are not repeated automatically after failure, because
their effects may already have happened. A successful callback is also not
repeated when closing another owned store fails. The host must resolve a
quarantined binding's failed cleanup before replacing its resources.

Release is a host lifecycle operation, not a client action. There is no idle
timeout or LRU cap in this API; the host can explicitly release threads when
they are no longer in use. `AgentService.aclose()` still shuts down all loaded
bindings. The service borrows its admission journal and leaves it open.

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

If command-receipt validation finds an uncertain command, recovery raises
`ServiceRecoveryRequiredError` with the concrete reason and schedules no worker.
The host must reconcile that command before retrying recovery; elapsed time alone
does not establish its outcome.

A terminal checkpoint written before the journal settled can reconcile its
receipt without another model call. These are separate transactions, not a
global transaction across checkpoint, task and admission stores. Store fencing
does not prevent a surviving old process from performing external effects.

`resume_checkpoint(thread_id, run_id, worker_stopped=...)` recovers by durable run
identity. If that checkpoint predates the admission journal, the host must first
establish quiescence and supply `worker_stopped=True`. The service validates the
checkpoint and records its existing identity before recovery; the original prompt
is not reconstructed. A terminal checkpoint settles without another model call.

Call `inspect_run()` before explicit recovery to see the observed admission,
checkpoint and outstanding reasons. The host still decides whether it has
reconciled the effects and can assert worker quiescence; inspection cannot make
that decision on the host's behalf.

## Shutdown And Ownership

`aclose()` stops admission, requests cooperative cancellation, joins foreground
workers and invokes session `on_close` callbacks. Cancelling its waiter leaves
shutdown running; another `aclose()` waits for the same operation. The service
borrows stores and does not close them automatically.

Close callbacks release factory-owned resources and must account for delegated
work using those resources. Non-cooperative code can delay shutdown. The service
is bound to one event loop; its journal operations are local synchronous SQLite
transactions. Network serving and daemon startup are separate integrations.
