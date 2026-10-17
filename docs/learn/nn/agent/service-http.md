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
authorized clients. This example reads provider credentials from a local
managed account bundle.

```python
import asyncio
import os

import uvicorn
import msgflux as mf
from msgflux.accounts import Accounts
from msgflux.coding import CodingCheckpointExtension
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.nn import Agent
from msgflux.runtime import AgentSession, AgentWorkspace
from msgflux.runtime.service import AgentService, SessionCachePolicy, SQLiteServiceStore
from msgflux.runtime.service.http import create_service_app
from msgflux.tools.builtin import ReadFileTool


async def main():
    journal = SQLiteServiceStore()
    accounts = Accounts()
    service = AgentService(
        store=journal,
        default_account="default",
        cache_policy=SessionCachePolicy(max_loaded=64, idle_timeout=300.0),
    )

    def create_session(thread):
        if thread.cwd is None:
            raise ValueError("This coding service requires a project cwd")
        model = mf.Model.chat_completion(
            "openai/gpt-6-luna",
            reasoning_effort="medium",
            credential_resolver=accounts.resolver("openai"),
        )
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
to use that path and which workspace and maximum permissions to grant. A path
alone grants no tool access. Reopening the thread without a path returns its
stored binding, and supplying a different path conflicts. Clients cannot replace
model settings or expand the workspace's permission ceiling through the HTTP
payload.

The host selects the account for new runs through `AgentService.default_account`.
Coding services created from a local manifest read this from `defaults.account`,
which defaults to `"default"`. The manifest's optional `defaults.accounts_dir`
sets the account bundle root; a relative path is resolved beside the manifest,
and an omitted path uses `~/.msgflux/accounts/`. In the Python example above,
`Accounts()` uses that same default root. Pass `Accounts(root=...)` when a host
needs a different root.
For a custom `AgentSession` factory, also attach an `Accounts.resolver(provider)`
to each model that should use managed credentials; `default_account` selects the
run identity and does not create or load provider credentials by itself.

An authenticated client can choose an account for a new prompt:

```python
receipt = await client.prompt(
    thread.thread_id,
    "Review the current changes",
    request_id="review-17",
    account_name="work",
)
```

The service records that selection with the admission before scheduling the
Agent. Omitting `account_name` selects the service default for a new admission.
Replaying a request ID with no account reuses its original admission; supplying
a different explicit account for that ID returns a conflict. Resume always uses
the account already stored for the run, even after the service default changes.
Secrets never travel in the prompt request. Missing credentials fail for the
selected account and provider without falling back to environment keys or a
different account. Older admissions without a recorded account remain unknown;
managed authentication cannot safely resume them under today's default, so
start a new execution with an explicit account.

An authenticated service owner can read and update the selected policy for an
existing thread:

```python
from msgflux.runtime import PermissionSet

policy = await session.workspace_policy()
updated = await session.update_workspace_policy(
    permissions=PermissionSet({"filesystem.read", "filesystem.list"}),
    approval_policy="on-request",
    expected_revision=policy.revision,
)
print(updated.permissions, updated.approval_policy, updated.revision)

read_only = await session.update_workspace_policy(permissions="read-only")
full = await session.update_workspace_policy(permissions="full-access")
```

`permissions` can be an exact `PermissionSet`, the `read-only` or `full-access`
preset, or omitted to leave the current selection unchanged. The policy is
persisted with an incrementing revision and audit history. Concurrent updates
can supply `expected_revision`; a stale revision returns a conflict. The
selected permissions are always intersected with the workspace's host-configured
ceiling and any parent execution permissions. `full-access` therefore does not
expand the host grant. Approval mode selects the workspace's review behavior; it
never selects a reviewer or bypasses permission checks. A read-only child scope
remains limited after a parent policy update. Policy changes are visible at live
permission checks and do not rewrite prior decisions or grant authority to an
already approved proposal.

Changing a managed service thread from `on-request` to `never` may resume a
pending approval batch only when the current service instance owns the paused
run (or the local daemon safely adopts its unexecuted approval batch) and validates the complete batch against current tool code, invocation,
workspace identity, and permissions. Expired, claimed, uncertain, changed, or
otherwise unverifiable approvals remain paused. The
`workspace.policy_updated` event reports a resumed run ID or a `resume_error`
explaining why recovery did not proceed. A freshly constructed
`AgentWorkspace.local()` has a new identity; durable approval recovery across
process restarts requires reconnecting to the same backend workspace identity.
See [durable runtime resources](resources.md) for the managed Agent setup.

Workspace approvals support Bash in foreground and background. The approval
binds both the command arguments and the selected dispatch mode. The runtime
consumes an approved background invocation before admitting its durable task;
pending, denied, expired or changed invocations create no task and execute no
command. A dispatch hook cannot reuse a background approval to run the same
call in foreground.

Approval consumption and task admission use separate journals. Process loss
between them leaves a consumed approval requiring explicit host reconciliation;
recovery never automatically repeats the command. Prepared file changes must
execute in foreground so their reviewed transaction remains bound to the effect.
Detached and call-as-response tools remain unsupported by approval policies.

This adapter runs in a single process around one service. Local daemon discovery/startup is available through the [local process API](service-local.md).
Coordination across independent server workers remains a separate integration. Do not run multiple Uvicorn workers against one in-memory service.

## Review Tool Approvals

The host may expose review operations for sessions where it configures an
`approval_reviewer`. The reviewer identity and approval policy stay on the
server. The HTTP bearer token authenticates the service owner; the request body
cannot choose a reviewer or grant permissions. Keep the execution principal in
the run scope distinct from the reviewer identity:

```python
from msgflux.runtime import AgentApprovals, SQLiteApprovalStore
from msgflux.runtime.context import ExecutionScope

approval_store = SQLiteApprovalStore("approvals.sqlite3")
policy = AgentApprovals(
    approval_store,
    tools={"write_file": "v3"},
    policy_version="workspace-policy-7",
)

def create_session(thread):
    agent = make_agent_for_thread(thread, approvals=policy)

    def scope_for_run(scope: ExecutionScope) -> ExecutionScope:
        # The execution principal describes the agent run, not its reviewer.
        return scope.with_overrides(principal="coding-agent")

    return AgentSession(
        agent,
        scope_factory=scope_for_run,
        approval_reviewer="alice@example.test",
    )
```

`make_agent_for_thread` represents the host's normal Agent construction and
must attach the same approval store and policy used by the runtime. A session
without `approval_reviewer` receives 403 for these routes. Review responses
contain request and tool metadata plus a verified file diff when available;
they omit raw arguments and checkpoint contents.

The client first lists review requests, decides against the revision it
inspected, then resumes the paused run explicitly:

```python
reviews = await session.approval_reviews(run_id)
for review in reviews:
    print(review.request_id, review.tool_name, review.status, review.diff)
    decided = await session.decide_approval(
        run_id,
        review.request_id,
        approved=True,
        expected_revision=review.revision,
    )
    print("Recorded:", decided.status)

# A decision records the human choice. It does not execute tools or resume work.
receipt = await session.resume(run_id)
```

An expired or stale request returns HTTP 409 and must be reviewed again. A
decision does not itself resume execution or claim exactly-once behavior; normal
service recovery checks still govern `resume()`.

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

Closing a client or its watcher only releases observation resources; it never
requests server-side session release. After the server captures a watcher's
initial snapshot and subscription, that watcher does not pin the loaded
AgentSession. The trusted host can call `AgentService.release_session()` while
remote watchers remain attached. They can drain already queued events, and
events published for the logical thread after a fresh factory load continue to
reach them. Release has no HTTP endpoint, so remote observers cannot close
resources used by other clients.

```python
saved = await client.runs(thread_id)
for run in saved:
    print(run.run_id, run.status, run.updated_at)
```

This lists resumable checkpoint identities without loading saved state into the
client. Pass a selected identity to the explicit resume operation when the
service's recovery rules allow it.

Use `await client.inspect_run(thread_id, run_id)` for a portable `RunInspection`
before recovery. It reports the admission receipt, checkpoint status and revision,
local-worker presence, approval batch phase and request count, background task
summaries, and diagnostic reasons. Its `background_tasks` tuple contains
`TaskSummary` values (`task_id`, `tool_name`, `status`, `updated_at`, `error`)
selected from the TaskStore for this thread and tasks associated by `root_run_id`
or `parent_run_id`. The summaries exclude results, progress/metadata, arguments,
and claim guarantees. If no admission record exists, including for a checkpoint
created before the admission journal, `receipt` is `None`.

```python
inspection = await client.inspect_run(thread_id, run_id)
print(inspection.checkpoint_status, inspection.receipt)
for reason in inspection.reasons:
    print(reason)
for task in inspection.background_tasks:
    print(task.task_id, task.tool_name, task.status, task.error)
```

This query does not start work, import a checkpoint, claim ownership or assert
that replay is safe. `requires_quiescence=True` means the trusted host must
establish that the old worker stopped; remote clients cannot supply that
assertion. The journal and checkpoint are separate stores, so inspection is not
an atomic snapshot across them. Approval summaries do not authorize review or
decisions. `approval_request_count` counts requests in the saved batch, including
requests with recorded decisions. Background task summaries are persisted
TaskStore observations, independent from live task projections in a watch
snapshot. After restart, `queued` or `running` does not prove a task worker is
alive; a completed foreground checkpoint may still have background tasks running
or completed. Use task recovery to inspect ownership and determine whether a
supported recovery claim is available. Raw
checkpoints, invocation arguments, task results, progress metadata, and owner IDs
are excluded.

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

The initial snapshot and subscription are captured together by `Agent.watch()`;
the service only holds a session lease during that attachment. The watcher then
observes by logical thread ID, independent of which Agent generation is loaded.
It can drain buffered events across an idle release and continue receiving
events after reload. Reconnection starts with current history and live state,
followed by future events. Old token deltas are not a durable replay log.
`Last-Event-ID` is rejected with
422; the adapter does not advertise resumable delta cursors.

Each observer has a buffer limit of 1024 events by default. The host can configure
`event_buffer_limit` when creating the app, including `None` for an unbounded
buffer. Overflow emits an SSE error asking the client to reconnect, closes that
observer, and leaves the Agent running. The client raises `AgentServiceHTTPError`;
open another watcher to obtain a fresh snapshot. There are no heartbeat frames or
automatic reconnect loops in this increment.

## Retrieve A Recorded Answer

```python
latest = await client.thread_result(thread_id)
print(latest.run_id, latest.status)

recorded = await client.run_result(thread_id, latest.run_id)
if recorded.status == "completed":
    print(recorded.output)
```

`thread_result()` selects the most recently admitted execution in that thread,
using the service admission journal. It does not use checkpoint update order,
and never substitutes an older completed answer when the selected run is still
processing. `run_result()` selects an explicit run. The embedded `AgentService`
provides the same methods.

Both queries return immediately with a `RunResult`: `thread_id`, `run_id`,
`status`, `output`, and `error`. Only completed runs expose final assistant text;
other statuses have `output=None`. Events and conversation history are excluded.
An unknown thread or run, or a thread with no admitted execution, returns HTTP
404. Reading a result never admits, resumes or interrupts work.

The CLI exposes these queries as `threads result THREAD_ID` and
`runs result THREAD_ID RUN_ID`. Its optional `--wait` fixes the selected run and
waits on the client side; the HTTP result endpoints themselves do not wait.
See [Retrieve A Thread Or Run Result](service-local.md#retrieve-a-thread-or-run-result)
for output and recovery behavior.

## Run Controls And Recovery

An upstream stream failure records a failed run and preserves received partial
history when checkpoint persistence is available. `run_result()` exposes no
completed output for that run. Reusing its `request_id` returns the same receipt,
including after service restart; it does not automatically repeat completed
tool effects.

Checkpoint and admission writes are separate failure boundaries. A checkpoint
write failure must not produce a completed result. If the checkpoint succeeds
but the service cannot persist the terminal receipt, client `wait()` requires
recovery instead of reporting success. Inspect both records before recovery:

```python
inspection = await client.inspect_run(thread_id, run_id)
print(inspection.receipt.status, inspection.checkpoint_status)
for reason in inspection.reasons:
    print(reason)
```

The inspection explains disagreement between the admission and checkpoint. It
does not authorize replay or prove whether an external effect occurred. After
restoring storage access, the trusted host must reconcile the execution using
the existing recovery controls.

```python
await client.steer(thread_id, receipt.run_id, "Focus on integration tests")
interrupted = await client.interrupt(thread_id, receipt.run_id)
resumed = await client.resume_checkpoint(thread_id, receipt.run_id)
```

Steering returns the published notification as a JSON dictionary and is not a
follow-up queue. Interruption targets an explicit run and returns a boolean.
An unfinished run already paused or failed in this service can resume through its
normal recovery checks. The HTTP resume route does not itself approve a tool
invocation; use the explicit review and decision operations described above.
An uncertain workspace command blocks admission or recovery with HTTP 409 and
error code `recovery_required`; the HTTP client raises
`ServiceRecoveryRequiredError` with the host's reason. No new worker is started.
Reconcile the command through the trusted host before retrying. An approval
request raised by a run remains a normal paused run: review and record a decision,
then resume it explicitly.

HTTP clients cannot assert `worker_stopped` or import uncertain executions after
a process restart. The local daemon can establish quiescence for
[paused, unexecuted approval batches](service-local.md#recover-approvals-after-a-crash)
using registered owner locks. Other recovery requires the trusted host to
establish old-worker quiescence and use the Python recovery API first. Persistent reconnection requires a file-backed
admission journal and checkpoint stores, plus persistent inbox/task stores when
those features must survive restart. The HTTP adapter reuses these stores and
creates no second conversation history.

## Compact A Paused Or Failed Run

Use manual compaction when the latest run is paused or failed and no worker is
still executing it. The call waits for compaction and returns a
`CompactionResult`; it does not admit a new run or resume the existing one.

```python
from msgflux.runtime.service.http import AgentSessionClient

session = await AgentSessionClient.open(client, thread_id=thread_id)
result = await session.compact()
print(result.run_id, result.status, result.reason)

# Select the same latest run explicitly when its identity is already known.
result = await client.compact(thread_id, run_id=result.run_id)
```

The second call returns `already_compacted` when no new completed turn exists.
`CodingSession.compact()` and embedded `AgentService.compact(thread_id)` have
the same result contract. Normal checkpoint recovery remains a separate call.

The authenticated HTTP equivalent is:

```http
POST /v1/threads/{thread_id}/compact
Authorization: Bearer <token>
Content-Type: application/json

{}
```

The body may contain `run_id`; it never accepts messages or history. The service
reserves the thread during maintenance, so prompt, resume and workspace-policy
updates cannot race with it. Other threads remain available. An active worker or
competing maintenance returns 409. A non-latest explicit run also returns 409;
a completed or interrupted run returns 422.

Pending approvals and run status remain unchanged. An uncertain tool effect or
missing old-worker quiescence returns a recovery-required conflict. After a local
daemon restart, its registered owner lease must prove the old worker has stopped;
embedded hosts without that evidence must reconcile ownership first. Compaction
does not bypass the approval decision or execute the pending tool. See
[manual compaction](compaction.md#compact-a-saved-execution-manually) for checkpoint,
cancellation and skipped-result semantics.

## Browser Clients And CORS

CORS is disabled by default. An embedded host can allow explicit frontend origins:

```python
from msgflux.runtime.service.http import ServiceHTTPConfig, create_service_app

app = create_service_app(
    service,
    token="your-private-service-token",
    http_config=ServiceHTTPConfig(
        cors_allow_origins=("http://localhost:3000", "https://app.example.com"),
    ),
)
```

Use exact HTTP(S) origins with no trailing slash or path. Wildcard domains such
as `https://*.example.com` and explicit `null` entries are not supported. A local
daemon reads the same policy from `[http]` in its service TOML:

```toml
[http]
cors_allow_origins = ["http://localhost:3000", "https://app.example.com"]
```

To allow every browser origin, use `*` as the only entry:

```toml
[http]
cors_allow_origins = ["*"]
```

The SDK equivalent is `ServiceHTTPConfig(cors_allow_origins=("*",))`.
This includes opaque browser origins such as `null`. The wildcard cannot be mixed
with other entries. Actual API requests still require the bearer token, and
cookie credentials remain disabled.

The CLI can update an existing service file without starting a daemon:

```bash
msgflux config cors ls --config ./service.toml
msgflux config cors add http://localhost:3000 --config ./service.toml
msgflux config cors remove http://localhost:3000 --config ./service.toml
msgflux config cors add '*' --config ./service.toml --json
```

The final command requires an empty origin list; it never silently replaces
existing explicit origins. Add/remove are idempotent and report whether restart
is required. Omit `--config` to use the default service file. These commands
preserve unrelated sections, comments and file permissions; they never import
extensions or construct agents.

Restart the daemon after editing its policy. `msgflux config validate` validates
origins offline, without importing agent factories or constructing models.
`msgflux service inspect --json` exposes the running policy under `http`; it
does not substitute the current file contents for the loaded snapshot.

Preflight `OPTIONS` requests are handled before bearer authentication. Actual
API calls, including SSE, still require `Authorization: Bearer <token>`:

```javascript
const response = await fetch(`${endpoint}/v1/health`, {
  headers: { Authorization: `Bearer ${token}` },
});
const health = await response.json();
```

The allowed methods are GET, POST, PUT and OPTIONS. Browser requests may send
Authorization, Content-Type, Last-Event-ID and W3C tracing headers. Allowed-origin
responses include CORS headers for API errors as well as successes, so a frontend
can read a 401 or 422 instead of receiving an opaque browser error. Disallowed
origins receive no access grant; CORS is browser policy, not server authentication.

For watch streams, use a streaming `fetch()` client that supplies the bearer
header. Native browser `EventSource` cannot set that header; tokens are not
accepted in query parameters. The CORS middleware sends the SSE response headers
immediately and does not buffer the event stream.

## Inspect The Loaded HTTP Policy

`GET /v1/service/inspection` returns an authenticated snapshot of the running
instance's HTTP configuration. The SDK exposes the same typed result:

```python
inspection = await client.inspect_service()
print(inspection.instance_id)
print(inspection.http.cors_allow_origins)
```

Here `client` is an existing `AgentServiceClient`. The call reads the app's
loaded configuration without opening threads, constructing models or rereading
a service file. After an edit, this result changes only when the server restarts.
The local `service inspect` CLI verifies the instance against its discovery
record before displaying this policy. The health endpoint remains a lightweight
identity probe.

## HTTP Telemetry

The service extra includes Litestar's OpenTelemetry plugin. HTTP instrumentation
uses the host's global tracer and meter providers, including the tracer already
configured by `msgtrace-sdk`. It does not create another exporter or shut down
shared providers when an app closes.

Enable the existing tracing pipeline before starting the service:

```bash
export MSGTRACE_TELEMETRY_ENABLED=true
export MSGTRACE_EXPORTER=otlp
export MSGTRACE_OTLP_ENDPOINT=http://localhost:4318/v1/traces
export MSGTRACE_SERVICE_NAME=msgflux-service
msgflux service serve
```

For development, `MSGTRACE_EXPORTER=console` prints spans in the service process.
A detached daemon writes its process output to its existing daemon log. Restart
an existing daemon after changing its telemetry environment. Embedded hosts can
also configure their own global OpenTelemetry providers before calling
`create_service_app()`. Without a configured provider, instrumentation is a no-op.

Each request creates one server span, including bearer failures, preflight,
validation errors and server errors. Incoming W3C `traceparent` context is
respected. Resolved span names use templates such as
`GET /v1/threads/{thread_id}/snapshot`; unknown paths use only the method.
Spans include `msgflux.service.instance_id` and, when available from a path or
response receipt, `msgflux.agent_id`, `msgflux.thread_id`, `msgflux.run_id` and
`msgflux.request_id`. IDs are span attributes, not metric labels.

The adapter does not capture request or response bodies. Header capture is off
by default; if enabled through standard OpenTelemetry environment settings,
authorization, cookies and token/API-key headers are redacted. Existing Agent,
Tool and Model telemetry has its own capture settings; see
[Telemetry](../../telemetry.md).

Watch spans last for the SSE connection and end on disconnect. Internal ASGI
send/receive spans are disabled, avoiding one span per token or event. Closing a
watch connection does not cancel the run. An admitted execution runs in its own
context and does not inherit the HTTP request span, event sink, authorization
ContextVars or baggage. Instead, each claimed attempt creates a root
`msgflux.service.execute` span with a
[Span Link](https://opentelemetry.io/docs/concepts/signals/traces/#span-links) to
its initiating prompt or resume request. Agent, Tool and Model spans created
during execution are children within this attempt's trace. This also works for
embedded `AgentService` calls with an active caller span, without installing the
HTTP extra.

An attempt span lasts through execution and journal settlement. Its attributes
include `msgflux.agent_id`, `msgflux.thread_id`, `msgflux.run_id`,
`msgflux.request_id`, `msgflux.admission.revision` and `msgflux.run.status`.
Completion sets OpenTelemetry status `OK`; failure sets `ERROR`. Approval pauses
and explicit interruptions are expected control flow and leave the status unset.
When available, `msgflux.run.stop_reason` identifies the pause/interruption
exception class; failures use `error.type`. These service spans never record
exception messages, stack traces, prompts or tool results. Existing child span
capture settings remain independent.

For example, a run that pauses for approval and then resumes has two independent
attempt spans linked to their respective requests:

```text
POST /v1/threads/{thread_id}/prompt
  └─ link → msgflux.service.execute [status=paused]
              └─ Model request
POST /v1/threads/{thread_id}/runs/{run_id}/resume
  └─ link → msgflux.service.execute [status=completed]
              ├─ Approved tool execution
              └─ Model request
```

Both attempts retain the same durable run/request IDs but have distinct span
and trace IDs. Retrying an already admitted prompt does not create an extra
execution span. A new attempt is instrumented only after the service claims it;
reconciling an already completed checkpoint does not invent an execution span.
Links are captured from the live scheduling caller and are not persisted in
checkpoints. A resume after server restart links to the new resume request;
use the durable IDs to correlate it with attempts exported before the restart.
An automatic policy resume links to its live scheduling span when one exists.
Disconnecting a watcher or closing its HTTP span never ends an attempt span.
Closing the service also leaves shared OpenTelemetry providers owned by the host.

The opt-in integration check uses the existing Codex account, a real TCP service
and a Docker OpenTelemetry Collector. It checks concurrent threads, approval
pause/resume, model/tool span ancestry, HTTP links and watcher disconnection:

```bash
MSGFLUX_LIVE_SERVICE_TELEMETRY=1 \
MSGFLUX_LIVE_SERVICE_ACCOUNT=work \
uv run --extra service pytest -q \
  tests/integration/test_service_execution_telemetry_e2e.py
```

This invokes the configured subscription model and requires Docker plus the
`otel/opentelemetry-collector-contrib:0.163.1` image. The test removes its collector
container and closes the owned service, stores and telemetry provider.

The native instrumentation records HTTP duration, active requests and available
body sizes when the host supplies a metric provider. `msgtrace-sdk` configures
tracing; it does not automatically configure a metrics exporter. For example,
an embedded host can set up metrics before creating the app:

```python
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

metric_provider = MeterProvider(
    metric_readers=[
        PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint="http://localhost:4318/v1/metrics")
        )
    ]
)
metrics.set_meter_provider(metric_provider)
app = create_service_app(service, token=token)
```

This sends HTTP metrics to the collector using one host-owned provider. The host
must shut it down after all apps using it have stopped. No collector or new
provider is started by the HTTP adapter. The plugin follows the current
[Litestar OpenTelemetry integration](https://docs.litestar.dev/latest/usage/metrics/opentelemetry.html).

## Endpoints And Errors

All routes require `Authorization: Bearer <token>`.

| Method | Route | Result |
| --- | --- | --- |
| GET | `/v1/health` | Authenticated runtime identity, protocol version and optional default agent |
| GET | `/v1/service/inspection` | Runtime identity and loaded HTTP policy |
| POST | `/v1/shutdown` | Graceful shutdown request when the host enables its shutdown hook |
| GET | `/v1/agents` | Registered agent IDs |
| GET / POST | `/v1/threads` | List bindings / open a thread |
| GET | `/v1/threads/{thread_id}/snapshot` | Portable thread snapshot |
| GET | `/v1/threads/{thread_id}/runs` | Saved run metadata summaries |
| GET | `/v1/threads/{thread_id}/result` | Status and final answer of the most recently admitted run |
| GET | `/v1/threads/{thread_id}/runs/{run_id}/result` | Status and final answer of a specified run |
| GET | `/v1/threads/{thread_id}/runs/{run_id}/inspection` | Recorded run state and recovery reasons |
| POST | `/v1/threads/{thread_id}/prompt` | Admission receipt |
| POST | `/v1/threads/{thread_id}/compact` | Completed or skipped manual compaction |
| GET | `/v1/threads/{thread_id}/requests/{request_id}` | Current receipt |
| GET | `/v1/threads/{thread_id}/watch` | Initial snapshot and continuous SSE events |
| POST | `/v1/threads/{thread_id}/runs/{run_id}/interrupt` | Interruption result |
| POST | `/v1/threads/{thread_id}/runs/{run_id}/steer` | Published notification |
| POST | `/v1/threads/{thread_id}/runs/{run_id}/resume` | Recovery receipt |
| GET | `/v1/threads/{thread_id}/runs/{run_id}/approvals` | Verified approval reviews |
| POST | `/v1/threads/{thread_id}/runs/{run_id}/approvals/{request_id}/decision` | Recorded decision |

`await client.open_thread()` can omit the agent ID when the server configures a
`default_agent`. `await client.health()` exposes that preference as optional
metadata; older health responses decode it as `None`. For an existing thread,
omitting the ID preserves its original binding. Without a configured default,
a new thread needs an explicit ID. Authentication and workspace permissions
remain governed by the existing service policies.

Thread-open bodies contain `agent_id` and may include `thread_id` and an absolute
`cwd`. Prompt bodies contain `prompt` and `request_id`; steer bodies contain
`content`; resume bodies are empty JSON objects. Unknown fields are rejected.
The native API is separate from any future Chat Completions compatibility adapter.

## Graceful Server Shutdown

A host may opt into an authenticated management route by passing an
`on_shutdown` callback to `create_service_app()`. The route is absent (404) when
no callback is configured. Requests must include the runtime's current
`expected_instance_id`; stale identities receive 409 and do not invoke the
callback. The callback runs after the 202 response is sent:

```python
from msgflux.runtime.service.http import AgentServiceClient

health = await client.health()
accepted = await client.shutdown(expected_instance_id=health.instance_id)
```

This is a server-owner capability for managing that process. The shared bearer
credential does not provide per-user tenancy or authorization boundaries. The
local service runner enables this route with its private runtime token and
requests ordinary Uvicorn graceful shutdown, allowing the service lifespan to
drain work it owns.

Receipt inspection is observational by default. `AgentSessionClient.wait()`
uses the same receipt endpoint with `require_waitable=true` to check that an
accepted/running attempt has a worker in this service instance. Otherwise it
raises `ServiceRecoveryRequiredError`, as the embedded session does. This check
does not assert that an old worker has stopped and does not authorize replay.
Use ordinary `receipt()` to inspect an uncertain attempt; recovery still needs
trusted host reconciliation.

Errors have the JSON shape `{"code": "...", "message": "..."}`. Missing or invalid
authentication returns 401, unknown resources return 404, conflicts/busy threads
and recovery requirements return 409, review access without a configured reviewer
returns 403, and invalid payloads return 422. Unexpected
server errors return a generic 500 message and are logged on the host. Model
failures settle their run receipts and appear as `run.error` events, rather than
turning accepted prompts into transport validation errors.

For an uncertain workspace command, recovery responses use
`{"code": "recovery_required", "message": "<host reason>"}`. The HTTP client
maps this response to `ServiceRecoveryRequiredError`; inspect the reason, have
the trusted host reconcile the command, then retry. A run waiting for an ordinary
tool approval instead settles with status `paused` and emits `run.paused`.

Built-in coding workspaces also retain identity by thread in the durable local
workspace registry. A backend or resource ID change, configuration revision
change, root path change, or replaced directory is a recovery condition. These
identity failures return HTTP 409 with code `recovery_required`. A workspace
factory load failure includes thread context and a specific mismatch reason in
the response message; a direct `WorkspaceIdentityError` also maps to this code.
Generic `PermissionError` responses remain neutral and do not include exception
text. `WorkspaceIdentityError` uses fixed public messages for known mismatch
reasons; any underlying exception remains in the server-side cause chain. The
service never adopts a replacement directory or escalates permissions
automatically. Resolve the mismatch through the trusted host before retrying.

```python
from msgflux.runtime import ServiceRecoveryRequiredError

try:
    receipt = await client.prompt(thread_id, "Review the project")
except ServiceRecoveryRequiredError as exc:
    print("Recovery required:", str(exc))
```

The SDK raises `ServiceRecoveryRequiredError` for a `recovery_required` response.
Remote clients can inspect its message; the HTTP response does not transmit the
server's Python exception cause.

## Ownership And Coding Sessions

`create_service_app()` borrows the service by default. Set `close_service=True`
when application shutdown owns service shutdown; borrowed stores still remain
host-owned. The host may release an idle loaded thread binding with
`await service.release_session(thread_id)`; active workers, session leases, and
delegated work prevent release. The service also applies its configured cache
policy, which defaults to at most 64 loaded bindings and a 300-second idle
timeout. Watchers do not block eviction after their initial snapshot and
subscription. A remote client cannot configure this policy or request release;
disconnecting only closes its observation connection. Durable thread and run
records remain available for a later factory-created Agent. A cold snapshot
still resolves dependencies through the registered factory; durable readers
that avoid constructing an Agent are a separate future change. Resource
callbacks must drain any delegated work they own before releasing its
model/workspace resources.

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
        answer = await session.result(receipt.run_id)
        print("Final answer:", answer.output)
        print("Saved run summaries:", await session.runs())
    finally:
        await client.aclose()


asyncio.run(main())
```

Omitting `agent_id` in `AgentSessionClient.open()` selects the service's configured
default agent; pass `agent_id="main"` explicitly when that binding is intended.
The optional `interactive` flag becomes part of the durable thread binding.
Omitting it preserves an existing thread's mode or uses `true` for a new thread.
An explicit conflicting mode is rejected.

The stable request ID lets a caller safely check admission after a network
uncertainty without silently submitting a second prompt. To reconnect after a
watch disconnect, open another `session.watch()` context and replace the local
view with its new `observer.snapshot`; missed event deltas are not replayed.
`wait()` polls the receipt and does not observe events or cancel execution if
its caller is cancelled. Use `cancel(run_id)` for an explicit interruption.
`runs()` and `latest_run()` expose only service-provided run summaries, not
checkpoint contents or configuration. `await session.result()` selects the most
recently admitted execution; `await session.result(run_id)` selects that run.
Both return `RunResult` immediately, with no implicit wait. The embedded
`CodingSession` exposes the same `result()` contract. The remote facade has no
automatic reconnect loop. `await session.inspect_run(run_id)` returns the same inspection
record as the embedded CodingSession. `session.aclose()` is an optional no-op because the facade
borrows its client and owns no background tasks; a UI can close its active watch
context on shutdown, while the host closes the shared HTTP client.
