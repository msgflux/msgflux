# Runtime

Runtime is the layer used to identify, resume, interrupt, and feed an agent
while it is executing.

The core pieces are:

| Piece | Purpose |
|-------|---------|
| `ExecutionScope` | Identifies the active execution with `thread_id`, `run_id`, and `namespace`. |
| `AbortSignal` | Carries local cancellation requests to the active runtime before safe interruption points. |
| `CheckpointStore` | Persists the agent snapshot so a run can resume. |
| `TaskStore` | Persists background task records, activity, outputs, and routing metadata. |
| `AgentInbox` | Holds pending messages, notifications, and control signals for the agent loop. |
| `AgentInboxStore` | Optional persistence boundary for the inbox. Without one, the inbox is in memory. |

## Execution Scope

Use `ExecutionScope` when you need stable runtime identity.

```python
import msgflux as mf
import msgflux.nn as nn

agent = nn.Agent(
    name="incident_analyst",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
)

scope = mf.ExecutionScope(
    thread_id="warehouse_incident_42",
    run_id="initial_analysis",
)

incident_log = """
09:02 - Scanner A stopped sending inventory updates.
09:07 - Orders continued to reserve stock from the last known snapshot.
09:18 - Operations restarted Scanner A; queued updates began arriving.
09:23 - Two orders were found with overlapping reservations for SKU-1842.
09:31 - New reservations were paused for SKU-1842.
"""

result = agent(
    "Identify the likely failure sequence, customer impact, and next actions "
    f"from this incident log:\n{incident_log}",
    scope=scope,
)
```

- `thread_id`: identifies the conversation thread. In a chat UI, this is the
  conversation id. In a workflow, it is the root workflow id. Every execution
  that should share history and durable context should keep the same
  `thread_id`.
- `namespace`: identifies the component that owns runtime state. For agents,
  msgFlux uses the agent module name as the effective namespace.
- `run_id`: identifies one resumable execution inside that thread. For a root
  agent this usually means one turn, command, or workflow step. For a
  background subagent, it is the task id. Reusing the same `run_id` means "try
  to resume this execution"; using a new `run_id` means "start new work in the
  same conversation". A subagent uses its own `thread_id`; parent/root lineage
  is carried separately by `parent_run_id` and `root_run_id`.

If no scope is passed, msgFlux generates runtime identifiers:

```text
thread_id = generated thd_<uuid>
namespace = default_namespace
run_id = generated run_<uuid>
```

These generated IDs are convenient local fallbacks. They are correct for
one-off calls, but they are not enough for recovery after a process restart. If
you need durability, provide the same `thread_id` and `run_id` again when
re-dispatching the work.

Resolution prefers explicit values, then existing message state, then inherited
runtime context, and only then generates a fallback. Omit an ID when you want
msgFlux to inherit it from the current context; pass an ID when you want to
force a specific execution identity.

## Checkpointing

Use a checkpoint store when a run should resume after a pause, interruption,
process restart, or tool-driven continuation. `ExecutionScope` provides the
checkpoint identity; `CheckpointStore` persists the execution state associated
with that identity.

You can bind the store directly to the agent:

```python
import msgflux as mf
import msgflux.nn as nn

checkpoint_store = mf.Store.checkpoint(
    "sqlite",
    path=".msgflux/checkpoints.sqlite3",
)

agent = nn.Agent(
    name="incident_analyst",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    checkpoint_store=checkpoint_store,
)

scope = mf.ExecutionScope(
    thread_id="warehouse_incident_42",
    run_id="initial_analysis",
)

incident_log = """
09:02 - Scanner A stopped sending inventory updates.
09:07 - Orders continued to reserve stock from the last known snapshot.
09:18 - Operations restarted Scanner A; queued updates began arriving.
09:23 - Two orders were found with overlapping reservations for SKU-1842.
09:31 - New reservations were paused for SKU-1842.
"""

result = agent(
    "Identify the likely failure sequence, customer impact, and next actions "
    f"from this incident log:\n{incident_log}",
    scope=scope,
)
```

Alternatively, provide the same store through `execution_context(...)`. The
task and identity values remain the same; the context supplies the scope to the
agent call:

```python
import msgflux as mf
import msgflux.nn as nn

checkpoint_store = mf.Store.checkpoint(
    "sqlite",
    path=".msgflux/checkpoints.sqlite3",
)

agent = nn.Agent(
    name="incident_analyst",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
)

scope = mf.ExecutionScope(
    thread_id="warehouse_incident_42",
    run_id="initial_analysis",
)

incident_log = """
09:02 - Scanner A stopped sending inventory updates.
09:07 - Orders continued to reserve stock from the last known snapshot.
09:18 - Operations restarted Scanner A; queued updates began arriving.
09:23 - Two orders were found with overlapping reservations for SKU-1842.
09:31 - New reservations were paused for SKU-1842.
"""

with mf.execution_context(scope=scope, checkpoint_store=checkpoint_store):
    result = agent(
        "Identify the likely failure sequence, customer impact, and next actions "
        f"from this incident log:\n{incident_log}"
    )
```

`ExecutionScope` carries identity; it does not store runtime resources. The
context manager propagates both the scope and resources such as
`checkpoint_store`, `task_store`, and `agent_inbox` to nested runtime calls.

`Agent(checkpoint_store=...)` and
`execution_context(checkpoint_store=...)` accept the same `CheckpointStore`
abstraction. If both are provided, the store bound directly to the agent takes
precedence over the store inherited from the execution context.

Revisioned stores keep a `_checkpoint` envelope with a schema version,
monotonic revision, branch identity, active head item, and extension state.
Commits may provide `expected_revision`; stale writers are rejected atomically
and their state and event are not persisted. Existing snapshots without this
envelope remain readable and acquire revision `0` on their next revisioned
commit. Forks start a new root branch and retain the source namespace, run,
branch, and head in `_checkpoint.fork_of` for provenance.

Inside the call, the agent resolves the effective scope first. It then uses the
active checkpoint store to load or save state under the effective
`(namespace, thread_id, run_id)` key.

??? tip "Available checkpoint stores"

    - `mf.Store.checkpoint("in_memory")`
    - `mf.Store.checkpoint("sqlite", path=".msgflux/checkpoints.sqlite3")`

When you call an agent with a `scope.run_id`, msgFlux first checks whether a
checkpoint already exists for `(namespace, thread_id, run_id)`.

Resume behavior:

- `running`: resumed from the saved snapshot.
- `paused`: resumed from the saved snapshot.
- `failed`: resumed from the saved snapshot. This is the primary recovery path
  after a provider, tool, process, or infrastructure failure.
- `completed`: not resumed.
- `interrupted`: not resumed.

On resume, the new task input is ignored and the saved interaction timeline
continues from the checkpointed state. `vars` is deliberately not part of
`ChatMessages` or the checkpoint: the current call supplies it as an ordinary
dictionary. This is intentional—the retry restores the same execution instead
of adding another user message. Use the same `thread_id` with a new `run_id`
when you want to continue the conversation with fresh input.

### Interaction timeline

`ChatMessages` persists one provider-neutral timeline. Messages, reasoning,
tool calls, tool results, and turn lifecycle events are all ordered items in
that timeline; there is no second copy of turn inputs, assistant output, vars,
or response type.

Each model response may also annotate its final generated item with model audit
metadata produced by the LM: `provider`, `model_id`, `api_mode`, and
`reasoning_effort` when it was used. The same item retains the minimal usage
counters `input_tokens`, `output_tokens`, and `cached_input_tokens` when the
provider reports them. Derived totals, cache percentages, costs, and the raw
provider payload are not checkpointed. These metadata annotations are for
inspection and are not sent back to the model provider.

Turn events are `start`, `pause`, `resume`, `complete`, `fail`, and
`interrupt`. The `messages.turns` property is a calculated view of those
events, not additional persisted state. This means a failed or paused turn can
resume without duplicating its messages.

An unfinished `ModelStreamResponse` temporarily owns the `ChatMessages`
instance supplied to that run. Finish or abort the stream before starting a
second run with the same object. msgFlux rejects overlapping use instead of
allowing an older stream finalizer to overwrite newer messages. If application
code mutates the history directly while the stream is open, the completed
stream can still be saved to its own checkpoint, but it will not replace the
newer in-memory timeline.

Every occurrence has a stable `item_id`, even when two items have identical
content. Use that identity to create an append-only branch at an exact history
boundary. For example, fork immediately after the first completed turn:

```python
state = checkpoint_store.load_state(namespace, thread_id, run_id)
first_complete = next(
    item
    for item in state["messages"]["items"]
    if item.get("type") == "turn" and item.get("event") == "complete"
)

forked = checkpoint_store.fork_run(
    namespace,
    source_thread_id=thread_id,
    source_run_id=run_id,
    target_thread_id=f"{thread_id}_review",
    target_run_id=f"{run_id}_review",
    at_item_id=first_complete["item_id"],
    position="at",
    status="paused",
)
```

Use `position="before"` to exclude the selected item itself. The store rejects
a boundary inside an active turn or between a tool call and its output. History
alternatives use explicit forks. Conversation compaction also remains
append-only: it records a complete model-visible view at a completed-turn
boundary without rewriting existing items. See
[Conversation Compaction](compaction.md) for configuration and replay behavior.

For background subagents, the task id is used as the subagent `run_id`. Reusing
that task id resumes or continues the same subagent. Creating a new task id
starts a separate subagent execution with its own conversation identity.

The checkpoint store can also be used directly when you need to inspect or
manage durable runs outside the agent loop. The lookup key is always
`(namespace, thread_id, run_id)`. For an agent, `namespace` is normally the
agent name:

```python
namespace = "incident_analyst"
thread_id = "warehouse_incident_42"
run_id = "initial_analysis"

state = checkpoint_store.load_state(namespace, thread_id, run_id)
print(state["status"] if state else "missing")
```

List recent runs for a thread:

```python
runs = checkpoint_store.list_runs(namespace, thread_id, limit=10)
for run in runs:
    print(run["run_id"], run["status"], run["updated_at"])
```

Find runs that may still need recovery:

```python
incomplete = checkpoint_store.find_incomplete_runs(namespace, thread_id)
```

Load the newest checkpointed run in a thread. This is useful when the caller
has a `thread_id` but did not persist the latest `run_id` separately:

```python
latest = checkpoint_store.load_latest_run(namespace, thread_id)
```

Fork a complete checkpoint into a new thread/run. Omitting `at_item_id` copies
the whole state while preserving the original run:

```python
state = checkpoint_store.load_state(
    namespace,
    "warehouse_incident_42",
    "initial_analysis",
)
second_turn_start = next(
    item
    for item in state["messages"]["items"]
    if item.get("type") == "turn"
    and item.get("event") == "start"
    and item.get("index") == 1
)

forked = checkpoint_store.fork_run(
    namespace,
    source_thread_id="warehouse_incident_42",
    source_run_id="initial_analysis",
    target_thread_id="warehouse_incident_42_review",
    target_run_id="initial_analysis_review",
    status="paused",
)
```

To fork a prefix instead, pass the stable `item_id` and choose whether that
occurrence is included:

```python
forked = checkpoint_store.fork_run(
    namespace,
    source_thread_id="warehouse_incident_42",
    source_run_id="initial_analysis",
    target_thread_id="warehouse_incident_42_review",
    target_run_id="before_second_turn",
    at_item_id=second_turn_start["item_id"],
    position="before",
)
```

Delete a single run when it is no longer needed:

```python
deleted = checkpoint_store.delete_run(namespace, thread_id, run_id)
```

Clear a broader set of checkpoints:

```python
removed = checkpoint_store.clear(namespace=namespace, thread_id=thread_id)
```

Stores also expose low-level event methods for append-only audit entries:

```python
checkpoint_store.append_event(
    namespace,
    thread_id,
    run_id,
    {"type": "operator_note", "message": "Reviewed by support lead."},
)

events = checkpoint_store.load_events(namespace, thread_id, run_id)
```

## Agent Inbox

`Agent` creates a memory-backed inbox by default:

```python
agent = nn.Agent(
    name="policy_assistant",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
)

agent.agent_inbox.store
# InMemoryAgentInboxStore(...)
```

When you instantiate `AgentInbox` directly, pass a store. Direct inbox creation
without a store raises an error, because the inbox needs a persistence boundary
to queue and drain notifications. Use an explicit store when pending messages
and control signals should survive process restarts or be shared by inbox
handles created in different places:

```python
inbox_store = mf.Store.agent_inbox(
    "sqlite",
    path=".msgflux/inbox.sqlite3",
)
agent_inbox = mf.AgentInbox(store=inbox_store)

agent = nn.Agent(
    name="policy_assistant",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    agent_inbox=agent_inbox,
)
```

You can also provide the inbox through runtime context instead of binding it to
the agent instance:

```python
scope = mf.ExecutionScope(
    thread_id="refund_conversation_42",
    run_id="refund_summary_01",
)
agent_inbox.bind_scope(scope, namespace="policy_assistant")

with mf.execution_context(scope=scope, agent_inbox=agent_inbox):
    agent("Summarize this policy: Returns are accepted within 30 days.")
```

Use a stable `thread_id` for any workflow that expects inbox delivery across
multiple turns, tools, or background tasks. If no scope is provided, msgFlux
generates fallback `thread_id` and `run_id` values for local execution. Those
generated identifiers are valid runtime keys, but another producer cannot
reliably target the same inbox unless it uses the same scope.

??? tip "Available inbox stores"

    - `mf.Store.agent_inbox("in_memory")`
    - `mf.Store.agent_inbox("sqlite", path=".msgflux/inbox.sqlite3")`

You can also instantiate concrete classes directly, but the `Store` factory is
the preferred public interface for application code.

Bind an inbox to a runtime identity when you want to write to the same pending
message queue that an agent will drain:

```python
scope = mf.ExecutionScope(thread_id="refund_conversation_42", run_id="refund_summary_01")

agent_inbox = mf.AgentInbox(store=inbox_store)
agent_inbox.bind_scope(scope, namespace="policy_assistant")

agent("Summarize this policy: Returns are accepted within 30 days.", scope=scope)
```

Use `fork(...)` to create another handle over the same store with a different
runtime key. This is useful when a root agent launches child work but you still
want a shared store:

```python
child_inbox = agent_inbox.fork(
    owner="research_agent",
    namespace="research_agent",
    run_id="task_123",
)
```

### Inspecting And Rendering Inbox Items

`peek()` reads pending notifications without removing them:

```python
pending = agent_inbox.peek()
```

`drain()` reads and clears the pending notifications for the current inbox key.
The key includes the agent namespace and `thread_id`, so notifications for one
conversation are not drained by another conversation:

```python
notifications = agent_inbox.drain()
```

If you used `peek()` and processed only some items, acknowledge them explicitly
by id:

```python
agent_inbox.ack([notification.notification_id for notification in notifications])
```

`render_messages(...)` converts inbox items into provider-ready chat messages.
System notifications become a `system` message, while incoming user messages
become a `user` message:

```python
messages = agent_inbox.render_messages(notifications)
```

`render(...)` is a convenience wrapper: it returns `None` for an empty list, one
message dict for a single rendered message, or a list when multiple messages are
needed:

```python
rendered = agent_inbox.render(notifications)
```

### Sending Messages While The Agent Is Running

To feed a running agent, write an incoming user message to the same inbox. The
agent drains the inbox before each provider call and after tool calls, before
the next provider call.

```python
inbox_store = mf.Store.agent_inbox("sqlite", path=".msgflux/inbox.sqlite3")
agent_inbox = mf.AgentInbox(store=inbox_store)
scope = mf.ExecutionScope(thread_id="refund_conversation_42", run_id="refund_summary_01")
agent_inbox.bind_scope(scope, namespace="policy_assistant")

agent = nn.Agent(
    name="policy_assistant",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    agent_inbox=agent_inbox,
)

# In one thread/task:
agent(
    "Draft a two-sentence reply using this policy: Returns are accepted within 30 days.",
    scope=scope,
)

# In another thread/task while the agent is still processing:
agent_inbox.user_message("Keep the reply under 50 words.")
```

The model receives the message as a synthetic user block:

```xml
<incoming_user_message>
Keep the reply under 50 words.
</incoming_user_message>
```

If the writer does not have the `agent` object, create another inbox with the
same store and execution key:

```python
store = mf.Store.agent_inbox("sqlite", path=".msgflux/inbox.sqlite3")
scope = mf.ExecutionScope(thread_id="refund_conversation_42", run_id="refund_summary_01")

external_inbox = mf.AgentInbox(
    store=store,
    namespace="policy_assistant",
    thread_id=scope.thread_id,
    run_id=scope.run_id,
)

external_inbox.user_message("Ask for the latest invoice number before deciding.")
```

If the pending user messages become stale, clear only those messages while
preserving runtime notifications and control signals:

```python
removed = external_inbox.clear_user_messages()
print(f"Removed {removed} pending user message(s).")
```

To attach metadata to a new user message, use the dedicated method:

```python
external_inbox.user_message(
    "Keep the reply under 50 words.",
    metadata={"origin": "chat-ui"},
)
```

### Control Messages

Control messages interrupt execution at safe provider boundaries.

```python
agent_inbox.pause(reason="Wait for user approval.")
agent_inbox.interrupt(reason="Operator interrupted the run.")
```

Behavior:

- `pause` raises `TaskPauseRequestedError` and checkpoints the run as `paused`
  when a checkpoint store is configured.
- `interrupt` raises `TaskInterruptRequestedError` and checkpoints the run as
  `interrupted` when a checkpoint store is configured.
- Unknown control commands remain normal system notifications.

For a persistent writer:

```python
external_inbox.pause(reason="Need human review before continuing.")
```

### Recoverable Inbox Delivery

Inbox delivery uses a short lease while an Agent transforms and persists a
notification. A claimed notification is acknowledged after the Agent
checkpoint succeeds. If a hook fails or the process stops before that point,
the lease expires and another execution can claim the notification again.

You can use the explicit operations when integrating a worker or external
consumer:

```python
claimed = agent_inbox.claim(lease_seconds=30)
try:
    # Process the notifications and persist the resulting Agent state.
    agent_inbox.ack(item.notification_id for item in claimed)
except BaseException:
    agent_inbox.release()
    raise
```

Claims coordinate separate inbox views and SQLite connections. Acknowledging
a notification is safe to repeat by its notification ID; external tools still
eed their own idempotency keys because the framework cannot make an external
side effect exactly once across a process crash.

### System Notifications

Non-user inbox items are delivered as compact system notifications:

```python
agent_inbox.publish(
    {
        "source": "policy",
        "status": "policy_update",
        "metadata": {"policy": "Returns are accepted within 30 days."},
    }
)
```

The model receives:

```xml
<notification source="policy" status="policy_update" policy="Returns are accepted within 30 days."/>
```

Use `user_message(...)` for new user turns. Use a machine-friendly source such
as `policy`, `task`, or `operator` for state that is not a direct user request.

## Live authority

`ExecutionScope` carries optional `principal` identity and immutable
`PermissionSet` grants. These belong to the generic runtime, not to the model's
messages or Agent variables.

```python
from msgflux.runtime import ExecutionScope, PermissionSet, execution_context

scope = ExecutionScope(
    principal="user:42",
    permissions=PermissionSet(["filesystem.read"]),
)
with execution_context(scope=scope):
    # Nested modules inherit filesystem.read, but cannot add filesystem.write.
    result = system(input_data)
```

This example grants an exact capability at the trusted application entry point.
An explicit child PermissionSet is intersected with the parent; an empty set
removes all grants. Nested executions cannot change principal. Omitted child
permissions inherit the parent's grants. Concurrent root executions are isolated.

`scope.to_dict()` serializes execution identity only, excluding principal and
grants. Restoring a checkpoint never restores authority: the application must
supply current grants on resume. Capability names have no wildcard semantics.
These grants are authorization metadata, not an operating-system sandbox.

## Approval journal (experimental)

`Store.approval(...)` records host-created approval requests and decisions. The
store alone does not pause or execute tools. To connect it to Agent checkpoints,
use [Agent approvals](#agent-approvals-experimental) below. Neither API adds
capabilities to the caller's live authority.

Available providers are `in_memory` (process-local) and `sqlite` (persistent,
including independent worker processes). Both use the same transition rules.

```python
import time
from uuid import uuid4

from msgflux.data.stores import Store
from msgflux.runtime import ApprovalBinding, ExecutionScope, PermissionSet, execution_context

store = Store.approval("sqlite", path=".msgflux/approvals.sqlite3")
request_id = uuid4().hex
binding = ApprovalBinding.from_call(
    namespace="catalog", thread_id="thread:42", run_id="run:1",
    principal="user:42", tool_call_id="call:1", tool_name="update_catalog",
    tool_revision="implementation:v1", policy_version="policy:v1",
    arguments={"sku": "ABC", "quantity": 3},
    resources={"catalog_id": "warehouse:1"},
    required_permissions=("catalog.write",),
)

try:
    requested = store.request(
        binding, request_id=request_id, expires_at=time.time() + 300,
    )
    pending = store.pending("catalog", "thread:42", "run:1")

    # Only after authenticating the reviewer and receiving their actual decision:
    decided = store.decide(
        "catalog", request_id, approved=True, decided_by="reviewer:7",
    )

    # The host must recompute the binding from the current invocation and policy.
    # This example retains the same binding because neither has changed.
    with execution_context(scope=ExecutionScope(
        namespace="catalog", thread_id="thread:42", run_id="run:1",
        principal="user:42", permissions=PermissionSet(["catalog.write"]),
    )):
        receipt = store.consume(request_id, binding=binding)

    audit = store.events("catalog", request_id)
finally:
    store.close()
```

This example creates a five-minute request, records a host-authenticated decision,
and consumes it once. It deliberately performs no external action. The journal
contains the `pending`, `approved`, and `consumed` revisions. A second consumption
raises `ApprovalConflictError`, even from another process. `approved=False`
records a terminal denial. Repeating the same request ID and binding/deadline or
the same decision/reviewer is idempotent; conflicting retries are rejected.

### Binding and authority

Bindings include execution identity, principal, tool-call identity, a host-owned
implementation revision, policy version, required capabilities, and SHA-256
digests of canonical public arguments and resource constraints. Any change needs
a new request; a request cannot be reused in another run or namespace. Argument
objects must contain JSON values with string keys: custom objects, tuples, and
non-finite numbers are rejected. Dictionary order does not change the digest.

The journal does not store original arguments, resource values, injected runtime
inputs, or a copy of the live grants. Keep invocation data in appropriately
protected application state when it is needed for review or resumption. Digests
are not encryption and may reveal low-entropy values through guessing; restrict
database access and avoid sensitive text in identifiers.

`consume` compares the supplied binding with the stored request and checks the
live principal, namespace, thread/run, and required capabilities. Approval alone
never widens authority. The host must recompute current arguments, resource
constraints and policy/implementation versions, authenticate reviewers, authorize
access to the journal, and enforce actual resource or sandbox restrictions.
Do not expose `decide` directly as a model tool or accept reviewer identity from
an unauthenticated request.

### Expiration, recovery, and async calls

Deadlines use absolute Unix seconds. `get`, `pending`, decision and consumption
operations record expiration when they encounter an elapsed pending/approved
request; no timer or background sweeper is installed. A recorded expiration
cannot be reversed by a clock rollback. The host owns clock correctness.
`ApprovalExpiredError` is a subclass of `ApprovalConflictError`.

SQLite commits the current record and its append-only audit revision in one
transaction. Reopening the database preserves decisions and used requests.
`pending` is a polling view, not a gap-free, atomic watcher snapshot; audit
revisions are per request, not cursors for the global execution stream. There is
no automatic journal retention or deletion policy in this API.

All operations have async counterparts: `arequest`, `aget`, `apending`, `adecide`,
`aconsume`, `aevents`, and `aclose`. They run storage operations in worker threads
and preserve execution context. For example, within the same live scope:

```python
receipt = await store.aconsume(request_id, binding=current_binding)
```

Use this **instead of** synchronous consumption for that request. It performs
the same binding and live-authority checks and does not execute the tool.

!!! warning "Consumption is not exactly-once execution"
    Cancelling an async wait does not undo an already committed transaction.
    A crash after consumption but before an external action leaves the request
    consumed. Do not automatically execute or retry an action merely because a
    record says `consumed`: external idempotency or reconciliation is still
    required. This API neither coordinates an Agent checkpoint transaction nor
    deduplicates every invocation across different approval request IDs.

## Agent approvals (experimental)

Pass `AgentApprovals` to an Agent to require host approval for named tools. The
`tools` mapping contains tool names and host-owned implementation revisions;
`policy_version` identifies your current approval policy. Increment these
versions when the implementation or policy changes. They are not inferred from
Python source code.

The following example assumes `model` is your configured chat-completion model.
The demonstration tool returns a string; it performs no external write.

```python
from msgflux.data.stores import Store
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.nn import Agent
from msgflux.runtime import AgentApprovals, ExecutionScope, PermissionSet
from msgflux.tools.config import tool_config


@tool_config(required_permissions=["catalog.write"], retry=False)
def publish(sku: str) -> str:
    """Publish a catalog entry."""
    return f"Published {sku}"


checkpoints = Store.checkpoint("sqlite", path=".msgflux/checkpoints.sqlite3")
journal = Store.approval("sqlite", path=".msgflux/approvals.sqlite3")
agent = Agent(
    name="publisher", model=model, tools=[publish], checkpoint_store=checkpoints,
    approvals=AgentApprovals(
        store=journal, tools={"publish": "implementation:v1"},
        policy_version="policy:v1", ttl_seconds=300,
    ),
)
scope = ExecutionScope(
    namespace="publisher", thread_id="catalog:42", run_id="publication:1",
    principal="user:42", permissions=PermissionSet(["catalog.write"]),
)

try:
    result = agent("Publish SKU ABC", scope=scope)
except TaskPauseRequestedError:
    requests = journal.pending("publisher", "catalog:42", "publication:1")
    # Present these requests through your authenticated host UI.
```

When the model requests `publish`, the Agent checkpoints the pending tool-call
batch and raises `TaskPauseRequestedError`. No tool in that batch runs yet,
including siblings that do not require approval. The checkpoint retains the
original public arguments and call IDs; the journal retains their digests.
Protect both stores according to their contents.

After the host has authenticated the reviewer, authorized their access to the
request and received an actual decision, it can record that decision:

```python
agent.decide_approval(
    request_id, approved=reviewer_approved, decided_by=authenticated_reviewer_id,
)
result = agent("", scope=scope)
```

Use the `request_id` returned in `requests` or a watcher snapshot. The second
call resumes the **same** namespace, thread and run; its message is ignored.
Pending calls are replayed before requesting another model response and without
duplicating the original call history. Repeated resumes while decisions remain
pending simply pause again. The model may request another protected call later,
so the host should handle subsequent pauses too.

An approval is consumed at foreground executor entry, after checking the final
public arguments and dispatch plan. Hooks cannot alter a call and reuse its old
approval. Declared capabilities and the live principal are checked again;
checkpoint restoration never restores grants. Supply live runtime inputs again
on resume, as for other Agent checkpoints.

### Async execution and observation

Use `agent.acall(...)` and `agent.adecide_approval(...)` for async applications.
They share the same approval state machine. `stream_events(...)` yields
`tool.approval_required` and `run.paused` before ending with
`TaskPauseRequestedError`; catch the
exception around the async iteration. The event identifies the request and tool
call without including its arguments.

Reconnect after a pause, including after recreating the Agent with the same
SQLite stores and policy:

```python
async with agent.watch("catalog:42") as watcher:
    requests = watcher.snapshot.approvals
    # Render request IDs, status, deadline and tool identity in the host UI.
```

`snapshot.approvals` contains journal records referenced by the latest run's
pending batch. They may be pending, decided, expired, or consumed-but-unsettled.
`decide_approval` emits `tool.approval_resolved` to live watchers in this process;
direct `journal.decide` only updates storage. A decision never automatically
restarts the Agent. Cross-process live events and an atomic snapshot spanning
the checkpoint and journal databases are not provided; reconnect or poll storage
to refresh external decisions.

### Denial, timeout, and recovery

Denied or expired approvals become blocked tool observations on the next resume;
the model can continue without executing those calls. Deadline checks are lazy:
there is no timer that resumes a paused Agent automatically. Removing a pending
rule or changing its binding leaves the run paused for host reconciliation.

Before dispatch, the Agent atomically checkpoints the batch as `executing`.
Only a checkpoint containing the results clears that marker. A restart that
finds `executing`, or an already consumed approval without results, **does not
retry any tool in the batch**. The host must inspect external effects and
reconcile the run using the host API below. Starting a
new run is not a safe substitute unless the host has established that replaying
the action is safe. Existing tool retry settings still apply within a single
invocation; use explicit idempotency where external effects require it.

An `executing` batch raises `ApprovalReconciliationRequiredError`, a subclass
of `TaskPauseRequestedError`, without changing the checkpoint. The original
worker may still be active: wait for it before treating the state as a crash.
This prevents a competing resume from invalidating that worker's commit.

!!! warning "Supported boundary"
    This integration requires atomic checkpoints and canonical foreground
    ToolLibrary calls, including canonical Chat Completions and Responses tool
    responses. Detached/background approval dispatch, flow-control DSL tools,
    provider-hosted effects and resource-scoped sandbox policies are unsupported.
    Nested protected calls without their own approved batch are blocked.
    The Agent policy uses an empty resource binding; public arguments and host
    policy versions provide its current binding boundary.

    Policies, context injectors, dispatchers and raw Python implementations are
    trusted host code. Agent approval rules do not protect arbitrary direct
    Python or standalone ToolLibrary calls outside that Agent execution.
    Do not expose the decision method as a model tool. Keep the policy and stores
    configured throughout the run; they are live host dependencies, not objects
    reconstructed from the checkpoint.

### Host reconciliation

Stop the old worker and verify external effects before reconciling. The runtime
cannot prove worker quiescence or undo an external write. These methods are
host-only: authenticate the operator and authorize access to the run yourself.
Never expose them as model tools.

```python
state = agent.inspect_approval_batch("catalog:42", "publication:1")
receipt = agent.reconcile_approval_batch(
    "catalog:42", "publication:1",
    expected_revision=state["_checkpoint"]["revision"],
    decision_id="incident:123", decided_by=authenticated_reviewer_id,
    reason="Verified the catalog entry in the external system",
    worker_stopped=True,
    results={"call_123": "Published ABC; verified by the operator"},
)
result = agent("", scope=scope)
```

Replace `call_123` with the pending call ID. Supply confirmed text observations
for **every** call in `state["runtime"]["extensions"]["pending_approvals"]["intents"]`,
including unprotected siblings. This atomically appends outputs, clears the
pending batch, saves a receipt and appends `approval.reconciled` to checkpoint
events. Resume requests the model without executing those tools again. This
initial API accepts text observations, not runtime commands or artifact objects.

To stop instead, omit `results` and pass `abandon=True`. The run becomes
`interrupted`; observations explicitly report unconfirmed effects, not success.
This does not undo effects or authorize replay in another run.

An identical `decision_id` and payload returns the original receipt; a conflicting
reuse or stale revision raises `CheckpointConflictError`. Concurrent callers may
retry the identical decision after a conflict. Revision checks fence old
checkpoint writes, **not** external tool execution. Receipts include the reason
and confirmed results: protect checkpoint access accordingly. Approval journal
records remain unchanged, preserving their original execution evidence.

`ainspect_approval_batch` and `areconcile_approval_batch` are async mirrors. A
cancelled await may leave an already-started storage transaction committed;
retry the same decision ID to discover its outcome safely.

## Abort Signal

`AbortSignal` is local runtime cancellation for the currently active process.
It is useful for UI and CLI controls such as pressing `Esc` while a model is
generating. It is carried by `ExecutionScope` and exposed through
`get_execution_context().get("abort_signal")`.

```python
abort_signal = mf.AbortSignal()
scope = mf.ExecutionScope(
    thread_id="refund_conversation_42",
    run_id="refund_summary_01",
    abort_signal=abort_signal,
)

# From another UI/CLI control path:
abort_signal.abort("User pressed Esc.")
```

Providers observe the signal before output starts. After the first model token
or tool call is produced, that model response is treated as committed; abort is
then observed only at the next safe runtime boundary, such as before executing
tools or before a later model call. When an abort reaches `Agent`, msgFlux
converts it into the durable interrupt semantics: open tool calls are closed
with synthetic interrupted outputs, and the checkpoint/task status becomes
`interrupted`. The canonical timeline retains that status for audit. If the
timeline is later converted to Responses input, the corresponding
`function_call_output` uses the protocol's `incomplete` wire status.


## Revisioned checkpoints

Checkpoint providers that support atomic commits expose a monotonically increasing
revision. A runtime checkpoint can carry `branch_id`, `head_item_id`, and durable
extension state alongside the message snapshot. Writers pass the revision they
loaded as `expected_revision`; a competing writer raises
`CheckpointConflictError` instead of overwriting newer state. Legacy providers
continue to support ordinary `save_state` calls, but do not provide this CAS
guarantee.

`AgentRun.durable_state()` preserves budgets, extension state, run lineage, and the
active branch so extensions can resume without allocating a new run or resetting
limits.

```python
from msgflux.data.stores import InMemoryCheckpointStore

store = InMemoryCheckpointStore()
first = store.commit_state(
    "agent", "thread", "run", {"status": "running"},
    expected_revision=0, event={"event_type": "checkpoint"},
)
second = store.commit_state(
    "agent", "thread", "run", {"status": "completed"},
    expected_revision=first.revision, event={"event_type": "checkpoint"},
)
```

This example atomically writes each snapshot and its event. Reusing
`first.revision` after the second commit raises `CheckpointConflictError`.
Invalid metadata or failed payload preparation does not publish a new revision.
The Agent uses this operation automatically with the built-in stores; direct
`save_state()` remains a legacy, unconditional snapshot operation.

Forks record `fork_of` provenance, reset the destination revision, and retarget
runtime identity to the destination run. A preserved scope tree retains its
active branch; an ordinary history starts at `root`. Resuming the fork writes
to the target run, never to its source. `load_latest_run()` selects the latest
updated **run**, not a context branch; the active branch lives in run metadata.

## Context scopes

Context scopes are nested conversation branches within the same execution. They
keep the same thread, run, and budgets. The built-in tools return a transition
command; the Agent applies it only after every tool call and output in the
current batch has settled. Opening a scope records the parent prefix and starts
a child branch. Closing returns to the parent and copies a summary into it.
Closing an already closed or root scope is idempotent, while closing a non-active
name raises a conflict.

Register the built-in tools when the model should decide when to enter and leave
a scope:

```python
from msgflux.tools.builtin import close_context_scope, open_context_scope
from msgflux.nn import Agent

agent = Agent(
    name="investigator",
    model=model,
    tools=[open_context_scope, close_context_scope],
)
```

The tools emit a command mapping tagged as `context_scope_transition`. Only tools
registered with `tool_kind="context_scope"` may request transitions, and their
calls must be exclusive within a batch. The Agent applies the command after every call
and output in the current batch has settled, so a scope change cannot split a
partially completed tool batch. Applications can apply the same command directly:

```python
from msgflux.runtime import ContextScopeCommand, ContextScopeController

controller = ContextScopeController()
controller.apply_command(
    messages,
    ContextScopeCommand(action="open", name="research", summary="Research context"),
)
# The Agent continues on the research branch.
controller.apply_command(
    messages,
    ContextScopeCommand(
        action="close", name="research", summary="Research completed"
    ),
)
```

Summaries are recorded as assistant messages in the parent branch. The original
call and output remain paired in the canonical timeline; closed branches retain
their full snapshot in `ChatMessages.metadata` for recovery and inspection.
The metadata records lineage, active head, and scope revisions while the message
events remain append-only. Checkpoints therefore restore the active branch without
allocating a new `ExecutionScope`, thread, run, or budget.
