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

### Images and described conversation messages

`user_message()` also accepts a non-empty list of canonical text/image blocks.
Use `ChatBlock.image()` with an HTTP(S) URL or an image data URI:

```python
from msgflux.utils.chat import ChatBlock

external_inbox.user_message([
    ChatBlock.text("Please inspect this screenshot."),
    ChatBlock.image("https://example.com/screenshot.png", detail="low"),
])
```

This publishes an ordinary incoming user message containing both text and an
image. Image-only messages are supported too. The inbox stores and forwards
blocks; it does not download URLs, read host paths or decode images. To publish
local bytes already obtained through authorized access, the publisher can use
`Image(image_bytes)()` from `msgflux.data.types` to prepare an image block.
Only text and images are supported here, not audio, video or arbitrary provider
payloads. Actual image support depends on the selected model/provider.

Use `message()` for content that should arrive as role `user` without claiming
to be a new message from the human user:

```python
external_inbox.message(
    [ChatBlock.image("https://example.com/chart.png")],
    description="Image produced by the referenced tool call.",
    source="render_chart",
    ref="call_123",
)
```

The resulting message contains a short `incoming_message` text wrapper with
the source, reference and description, followed by the image and closing text.
It does not use `incoming_user_message`. Source and reference are also retained
in history metadata. These labels document provenance; they do not authenticate
publishers or grant authority. Descriptions and text are escaped, and role is
always `user`, never a publisher-controlled system role.

Inside a tool, the existing notification handle can publish the same content:

```python
from msgflux.tools import Hidden, ToolLibraryHandle
from msgflux.tools.config import tool_config

@tool_config(runtime_inputs=["handle"], retry=False)
def show_chart(*, handle: Hidden[ToolLibraryHandle]) -> str:
    """Attach a chart to the next model turn."""
    handle.get_notification().message(
        [ChatBlock.image("https://example.com/chart.png")],
        description="Chart attached by this tool call.",
    )
    return "The image follows as a user-role message."
```

The handle supplies its source and reference: a tool-call ID for foreground
calls, or the existing task reference for background tools. Foreground delivery
follows the tool results, before the next provider request. Multiple described
or multimodal conversation messages retain their relative inbox order; system
notifications retain their existing separate rendering. Background publication
is delivered at the next inbox drain, not synchronized with a future task result.
`clear_user_messages()` removes user-origin text/images but preserves described
messages and runtime signals.

Memory and SQLite stores use the existing claim/receipt/ack mechanism for these
messages. Checkpoints preserve the image blocks; URL references are not immutable
snapshots and data URIs increase storage size. This increment does not add an
artifact store or make publication atomic with external tool effects. It also
does not delegate to a vision model: the main agent/application must explicitly
choose delegation when needed. No native multimodal tool-output support is
required, but multimodal user-message support still is.

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

### Choosing when to ask for approval

Approval is host policy, not a property of `BashTool` or its native transport.
For new runs that should execute without confirmation prompts, omit `approvals`
or pass `approvals=None`:

```python
from msgflux.nn import Agent
from msgflux.tools.builtin import BashTool

agent = Agent(
    name="workspace", model=model, tools=[BashTool()],
    checkpoint_store=checkpoints, approvals=None,
)
```

This disables approval prompts, not sandboxing or authorization. Supply an
`ExecutionScope` with an authorized `ExecutionEnvironment`, `process.execute`,
and the necessary resource grants. There is no implicit wildcard/full-access
grant. If your application's UI calls this mode “full access”, define its live
grants separately. For selective approval, configure `AgentApprovals.tools` with
only the names that need confirmation; omit `bash` to let it execute
without a prompt. An empty mapping is not accepted; use `approvals=None` instead.
An unprotected sibling still waits when another call in its batch needs approval.

### Overriding approvals per invocation

`forward`/`aforward` accept `approvals` as a keyword-only runtime override, also
available through `agent(...)`, `agent.acall(...)` and `agent.stream_events(...)`.
Omitting it uses the configured default; passing `None` explicitly disables
prompts for new batches. Passing an `AgentApprovals` instance replaces the default
for that invocation only, including when the constructor default is `None`.

```python
from msgflux.runtime import AgentApprovals

policy = AgentApprovals(
    store=journal, tools={"bash": "implementation:v1"},
    policy_version="interactive:v1",
)

# The workspace agent above has approvals=None by default.
try:
    result = await agent.acall("Run the checks", scope=scope, approvals=policy)
except TaskPauseRequestedError:
    # Display requests; wait for an authenticated user's decision.
    pass
```

This invocation requires approval for Bash without changing `agent.approvals`.
Use `approvals=None` explicitly for an invocation without prompts, or omit the
keyword to use the constructor default. The policy is held in execution-local
context, not shared mutable Agent state or model inputs. Concurrent calls may
use different policies. The override is not saved as a live object in checkpoints.

Pass the policy again when observing, deciding and resuming a runtime-only policy:

```python
async with agent.watch(scope.thread_id, approvals=policy) as watcher:
    requests = watcher.snapshot.approvals
    # Render these records and obtain a decision outside this example.

await agent.adecide_approval(
    request_id, approved=reviewer_approved, decided_by=authenticated_reviewer_id,
    approvals=policy,
)
result = await agent.acall("", scope=scope, approvals=policy)
```

The second example assumes the host has selected a request and authenticated the
reviewer. Synchronous applications use `decide_approval(..., approvals=policy)`
and `agent(..., approvals=policy)`. Watchers retain the policy selected when they
are created; changing the mode in the UI does not reconfigure an existing watcher.

The override is selected at invocation entry, not by mutating a running loop.
To change modes during a conversation, serialize the UI's operations and pass the
new policy on the next invocation or safe resume boundary. Permissions remain
separate: supply their current values in the live `ExecutionScope` on each call.
Do not remove a policy to bypass an already-pending approval. Pending requests
retain their binding and must be resolved under the original policy; uncertain
executing batches require host reconciliation.

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

### Connecting a TUI

Run event consumption in an async task so the UI stays responsive. For example,
with the approval-enabled `agent`, `journal`, and `scope` above:

```python
import asyncio

from msgflux.exceptions import TaskPauseRequestedError
from msgflux.runtime.approvals.agent import ApprovalReconciliationRequiredError


async def run_turn(message):
    try:
        async for event in agent.stream_events(message, scope=scope):
            render_event(event)
    except ApprovalReconciliationRequiredError:
        show_recovery_required()
    except TaskPauseRequestedError:
        requests = await asyncio.to_thread(
            journal.pending, scope.namespace, scope.thread_id, scope.run_id,
        )
        show_approval_requests(requests)


async def answer_request(request_id, approved, reviewer_id):
    await agent.adecide_approval(
        request_id, approved=approved, decided_by=reviewer_id,
    )
    await run_turn("")
```

`render_event`, `show_recovery_required`, and `show_approval_requests` are your
UI functions. Schedule `run_turn(prompt)` as a task in the TUI event loop. Call
`answer_request` only after that task has ended, from an explicit authenticated
user action; serialize submissions so only one worker resumes a run at a time.
Keep the same scope and re-supply live dependencies. A decision alone does not
resume execution. If other requests remain pending, the resumed call pauses again.

The approval event and journal intentionally omit raw arguments. To display the
exact command batch, the authorized host can read the checkpoint through
`await agent.ainspect_approval_batch(scope.thread_id, scope.run_id)` and correlate
its pending intents by tool-call ID. Treat those arguments as untrusted display
data, including terminal escape sequences. Use `agent.watch(...)` and its snapshot
when reconnecting; live events alone are not a durable approval queue. Other
extensions can also pause an Agent, so an empty approval list is not permission
to automatically resume it.

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
    The Agent policy binds declared static resource requirements and, when an
    environment is present, its workspace ID and isolation requirements. Public
    arguments and host policy/tool versions remain part of the binding. It does
    not infer every resource accessed by arbitrary Python or a shell command.

    Policies, context injectors, dispatchers and raw Python implementations are
    trusted host code. Agent approval rules do not protect arbitrary direct
    Python or standalone ToolLibrary calls outside that Agent execution.
    Do not expose the decision method as a model tool. Keep the policy and stores
    configured throughout the run; they are live host dependencies, not objects
    reconstructed from the checkpoint.

### Host reconciliation

Recovery is covered by an offline process-level conformance gate: independent
workers pause for approval, restart, perform a test effect, die before saving its
result, refuse automatic replay, reconcile and reconnect a commit observer.
The suite also checks legacy checkpoint upgrades and transaction rollback after
abrupt process loss. These tests do not imply exactly-once external execution or
power-loss guarantees for the underlying filesystem.

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

Cancellation has two distinct cases. An exception raised **inside** a SQLite
transaction rolls back before the connection is reused. Cancelling an async
caller while a worker thread is committing does not stop that thread: the write
may still succeed. After an uncertain outcome, inspect committed state and use
the original decision ID/revision instead of assuming rollback or replaying a
tool. A stale checkpoint revision must fail, not overwrite the winner.

## Workspaces and execution environments (experimental)

`ExecutionEnvironment` supplies live execution dependencies: a virtual filesystem
and an optional `ProcessExecutor`. `ExecutionScope.permissions` remains the only
source of grants. Child executions inherit the same environment and may narrow
permissions, but cannot replace the environment. Checkpoint identity serialization
omits both environment and authority; supply them again on resume.

### Virtual files and live resource grants

`InMemoryWorkspace` implements `WorkspaceFilesystem` without accessing the host
filesystem. Its initial files are supplied by trusted host code. Runtime operations
require exact resource/action grants and an active scope bound to that filesystem.

```python
from msgflux.runtime import (
    ExecutionEnvironment, ExecutionScope, InMemoryWorkspace, PermissionSet,
    execution_context,
)

filesystem = InMemoryWorkspace(
    "project-42", {"/workspace/report.txt": b"Quarterly report"},
)
environment = ExecutionEnvironment(filesystem)
scope = ExecutionScope(
    principal="user:42", environment=environment,
    permissions=PermissionSet(resources=[
        filesystem.permission("/workspace/report.txt", "filesystem.read"),
        filesystem.permission("/workspace/output.txt", "filesystem.write"),
        filesystem.permission("/workspace", "filesystem.list"),
    ]),
)

with execution_context(scope=scope):
    text = filesystem.read_text("/workspace/report.txt")
    filesystem.write_text("/workspace/output.txt", text.upper())
    names = filesystem.listdir("/workspace")
```

This reads one authorized file, writes another, and lists the directory. A write
grant does not imply read access. A directory-list grant exposes child names, not
child contents or recursive access. Resource IDs include the workspace ID and
canonical virtual path; keep workspace IDs unique and stable within your host.

Operations are `read_bytes`/`read_text`, `write_bytes`/`write_text`, `listdir`,
`mkdir` and `unlink`, with async counterparts prefixed by `a`. Grants use
`filesystem.read`, `filesystem.write`, `filesystem.list`, `filesystem.mkdir` and
`filesystem.delete`, respectively. Writes create or replace a file in an existing
parent directory. `mkdir` creates one directory; `unlink` deletes files only.
The host's initial file map creates the required parent directories.

Paths are absolute POSIX paths inside the workspace, independent of the host OS.
Dot/repeated-separator aliases are canonicalized; `..`, backslashes, control
characters and leading `//` are rejected. There are no symlinks, host mounts or
implicit path-containment grants. Use `filesystem.permission(...)` to construct
the same resource identity used by operations. Grants are checked on each
operation, including when an injected handle is reused under a narrower scope.

### Injecting a filesystem into tools

Declare runtime inputs explicitly, so they remain outside the model-facing schema:

```python
from msgflux.nn import ToolLibrary
from msgflux.tools.config import tool_config

@tool_config(runtime_inputs=["filesystem"], retry=False)
async def read_file(path: str, *, filesystem) -> str:
    """Read an authorized file in the virtual workspace."""
    return await filesystem.aread_text(path)

tools = ToolLibrary("files", [read_file])
with execution_context(scope=scope):
    text = await tools.arun("read_file", {"path": "/workspace/report.txt"})
```

This is an application-defined example, not a shipped `read_file` builtin. The
same tool can be passed to an Agent invoked with this live scope. The runtime
supplies `filesystem`; model arguments and `vars` cannot supply a substitute.
Missing bindings fail before tool execution. The VFS checks the actual `path`
when called and raises `PermissionError` on denial. It also checks an already
aborted scope before attempting an operation. Cancelling an async await cannot
undo a write that already completed in the backend's worker thread.

For tools accessing a fixed resource, `@tool_config(required_resources=[...])`
adds mandatory preflight checks to the canonical ToolLibrary boundary and local/
MCP adapters. Construct requirements using `ResourcePermission(resource, action)`
or `filesystem.permission(...)`. `required_permissions` remains independent:
declaring both requires both grants. Static requirements neither authorize a
dynamic path by themselves nor stop arbitrary Python from using host APIs.

### Process executors

`ProcessExecutor` is an abstract, host-supplied backend. **No shell or OS sandbox
backend is included.** An environment without one refuses process execution:

```python
from dataclasses import replace
from msgflux.runtime import ProcessRequest

process_scope = replace(scope, permissions=PermissionSet(["process.execute"]))
with execution_context(scope=process_scope):
    try:
        await environment.arun(ProcessRequest(("bash", "-lc", "pwd")))
    except PermissionError:
        pass  # No executor configured; nothing was launched on the host.
```

The builtin `bash` declares `runtime_inputs=["environment"]` and calls
`environment.arun(...)`. It receives the same workspace as `read_file` and does
not fall back to `subprocess` when the backend cannot use that workspace.

Before calling a backend, the environment checks `process.execute`, declared
`SandboxCapabilities`, workspace compatibility and cancellation. Default
`SandboxRequirements` require filesystem, network, process and resource-limit
mechanisms. The trusted host owns any explicit relaxation. A backend must enforce
the requested policy, the passed live resource grants, virtual cwd, timeout and
output limit. It must not inherit the host's filesystem, credentials or environment
implicitly. Cancellation must terminate/reap its children before cleanup returns.

`ProcessRequest` carries explicit argv, a virtual cwd, timeout and output byte
limit; `ProcessResult` carries return code and byte stdout/stderr. The runtime
requests cancellation on timeout and rejects oversized returned output, but the
backend must bound capture while running. Declaring capabilities is a contract,
not proof of OS isolation. The current tests use a fake backend, never real bash.

!!! warning "Current boundaries"
    The memory VFS is process-local and has no persistence, quotas or artifact
    resolver. Files are not checkpointed. Filesystem/network enforcement for real
    processes, mounts/materialization, synchronization, shell emulation and durable
    workspace versions remain future backend work. A VFS is not a Python sandbox.

    Host tools, extensions and backend implementations remain trusted code.
    Resource IDs are exact opaque names, not wildcard policies. Network resource
    interpretation belongs to a future enforcing backend. Approval bindings include
    static requirements, workspace identity and isolation mechanisms, but do not
    pin file contents or inspect shell commands. Change host policy/tool revisions
    when those implementations or their security meaning change.

### Ready-to-use workspace tools

`ReadFileTool` and `BashTool` use the same live dependencies described above. Add
them explicitly to a ToolLibrary or an Agent; they are not enabled automatically.

```python
from msgflux.nn import ToolLibrary
from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    PermissionSet,
    execution_context,
)
from msgflux.tools.builtin import BashTool, ReadFileTool

workspace = InMemoryWorkspace("report", {"/report.txt": b"Quarterly report"})
scope = ExecutionScope(
    environment=ExecutionEnvironment(workspace),
    permissions=PermissionSet(
        resources=[workspace.permission("/report.txt", "filesystem.read")],
    ),
)
tools = ToolLibrary("workspace_tools", [ReadFileTool(), BashTool()])

with execution_context(scope=scope):
    text = tools.run("read", {"path": "/report.txt"})
    assert text == "Quarterly report"
```

This example reads only the authorized virtual file, not a host path. The async
equivalent is `await tools.arun("read", {"path": "/report.txt"})` inside
the same execution context. For an Agent, pass `tools=[ReadFileTool(), BashTool()]`
and supply the live scope to its call. `filesystem` and `environment` are injected
runtime inputs, excluded from model schemas; arguments cannot replace them.

`ReadFileTool` is exposed as `read(path, offset=None, limit=None)`. It accepts an absolute
virtual path and returns strict UTF-8 text by default. There is no separate
`read_file` Python function or shared default instance.

For host-configured instructions, use a separate `ReadFileTool` instance for each
agent. Guidance is optional and is not shared between instances:

```python
from msgflux.tools.builtin import ReadFileTool

reader = ReadFileTool()
reader.tool_config["usage_guidance"] = (
    "Use this tool for text files. For image interpretation, explicitly "
    "delegate to an available vision agent."
)
tools = ToolLibrary("reader", [reader])
```

This example provides guidance appropriate to a text-only main agent when a
vision agent is available. It does not delegate automatically. For an agent whose
model accepts image messages, enable image publication explicitly:

```python
class ProjectReader(ReadFileTool):
    """Read project files and optionally attach images."""

    tool_config = {
        **ReadFileTool.tool_config,
        "usage_guidance": "Read only files relevant to the current task.",
    }

reader = ProjectReader(supports_vision=True)
```

The constructor copies `tool_config` to the instance and appends an instruction
explaining that images arrive in a subsequent user-role message linked to the
tool call. Existing `tool_config["usage_guidance"]` is preserved; neither the
class configuration nor other instances are modified. `usage_guidance` is not
a constructor argument. Configure it before adding the instance to a library.
The vision flag is host configuration, not a model argument;
it does not change the model's capabilities. The default is `supports_vision=False`.

PNG, JPEG, GIF and WebP paths are recognized by their filename MIME type. Image
bytes are read through the same authorized VFS, encoded with the existing `Image`
helper and published through `handle.get_notification().message(...)`. No host
path is opened by the encoder. This is format routing, not image integrity or
provider compatibility validation. Other image formats, images with vision
disabled, or publication without an inbox raise errors. Standalone ToolLibrary
calls normally have their own inbox; an Agent is needed to consume that inbox
into a model conversation. The result is only a short publication confirmation,
not base64 text. In foreground Agent execution, the attachment follows the tool
results before the next model request.

Both tools define explicit `annotations` containing only model-visible inputs
and the return type. Runtime dependencies remain in `runtime_inputs`, not that
mapping. Forged public arguments that collide with injected dependencies are
rejected before execution. Their UI labels are `Read` and `Bash`.

### Reading a window and configuring the working directory

```python
reader = ReadFileTool(cwd="/project")
shell = BashTool(cwd="/project")
tools = ToolLibrary("workspace_tools", [reader, shell])

with execution_context(scope=scope):
    snippet = tools.run("read", {"path": "src/main.py", "offset": 20, "limit": 40})
```

The example reads up to 40 lines starting at line 20 of `/project/src/main.py`;
`scope` must authorize that exact virtual file. The result is only the selected
text, preserving LF/CRLF line endings, without a metadata wrapper. `offset` is
1-based and defaults to 1. `limit` defaults to 2000 lines and is capped at 2000.
Both must be positive integers when supplied. An offset past EOF is an error;
an empty file at offset 1 returns an empty string. Multiple ranges are not exposed.

Only the selected text is decoded as UTF-8 and checked against the 1,000,000-byte
output ceiling. A small window of a larger file is allowed. A selected window
that exceeds the byte ceiling is rejected rather than silently cutting a line.
Image reads reject explicit offset/limit and retain their existing size ceiling.

`WorkspaceFilesystem.read_lines`/`aread_lines` authorize the same `filesystem.read`
resource as `read_bytes`. Backends can implement `_read_lines` for bounded I/O;
the compatibility implementation reads the source bytes once, scans newline
positions and slices only the requested window. It does not split or decode the
whole file, but does not promise bounded backend I/O for legacy implementations.
Backends must enforce storage quotas and coherent reads as appropriate.

`cwd` is constructor configuration, never a model argument or the host process's
working directory. Relative read paths are resolved under it; absolute virtual
paths remain allowed when authorized. Traversal (`..`) is still rejected. This
cwd is not a security root: permissions and the sandbox define accessible paths.
When changing constructor configuration for an approval-protected tool, update
its host-owned implementation revision. A generic Agent resource container is
not introduced; live resources still come from `ExecutionScope.environment`.

`BashTool` is exposed as `bash(command, timeout_ms=None)` through function calling.
Output budgets are internal execution controls, not model arguments. It requests
`bash --noprofile --norc -c <command>` from the environment's executor. The entire
command is one argv element; shell syntax is intentionally interpreted by Bash
inside that executor. The executor must provide Bash, enforce the live resource
grants and isolation requirements, and prevent inherited host environment or
startup hooks such as `BASH_ENV`. `process.execute` alone does not grant file or
network access. No command inspection here establishes which resources it uses.

The host ceilings are 30 seconds per command and 1,000,000 combined output bytes
per batch. The requested timeout may reduce the deadline, never increase it. A
single command and a batch both return `ShellResult`, a `msgspec.Struct` from
`msgflux.tools.shell`. Its `results` tuple contains `ShellCommandResult` records
with `status` (`exited`, `timed_out`, or `not_executed`), `stdout`, `stderr`, and
`returncode` (only set for an exited process). Function calling serializes this
canonical result as JSON. Commands in a batch execute in order,
in independent Bash processes with the same initial virtual cwd; `cd` and shell
variables do not persist between commands. After the output budget is exhausted,
remaining commands are not started and receive an explanatory error result.
Invalid UTF-8 in process output is
replaced for display. Automatic retries are disabled to avoid duplicating shell
effects. Cancellation, cleanup and bounded output capture follow the executor
contract above; this tool does not add durable process execution or live stdout
deltas.

`allow_background=True` injects only `run_in_background`, not a timeout. The
tool's `timeout_ms` limits each process execution; the separate `TaskWaitTool.timeout`
only limits how long a caller waits for a background task result, without changing
the process deadline. Background execution does not bypass the shell's time cap.
For example, configure `shell.tool_config["allow_background"] = True` before
building the ToolLibrary. Existing approval restrictions for foreground tools
still apply.

An asyncio subprocess backend should disconnect stdin, enforce output limits
during capture, and terminate and reap the process tree on timeout/cancellation.
Collecting all output with `communicate()` and truncating afterward does not
provide bounded memory. These responsibilities belong to the configured isolated
executor, not a subprocess fallback inside the tool.

### Native local shell in OpenAI Responses

`BashTool()` declares the provider-independent `shell` tool kind. The model owns
the binding, enabled by default through `native_tools=True`. With the OpenAI
Responses provider, its adapter emits
`{"type": "shell", "environment": {"type": "local"}}`. The model's
`shell_call.action.commands` become the same canonical tool arguments used by
the tool library; permissions, policies, approvals and the execution environment
remain in that path. Results return as `shell_call_output`, not a JSON-encoded
`function_call_output`. This is local execution by your configured backend, not
an OpenAI-hosted container. See the [OpenAI shell contract](https://developers.openai.com/api/docs/guides/tools-shell).

The tool contains no OpenAI configuration or wire-format results. For example:

```python
from msgflux.models.providers.openai import OpenAIChatCompletion

model = OpenAIChatCompletion(model_id="your-shell-capable-model", native_tools=False)
```

This model uses function calling, including for `BashTool()`. Other providers and
Chat Completions also retain function calling unless they implement a native
adapter. A native shell can have any logical tool name, but only one shell tool
can be bound per request, and it cannot use deferred loading. Use function mode
for those configurations. Explicit selection of that native tool
is translated to `tool_choice={"type": "shell"}`. The host must select a model
that supports the shell tool; the runtime does not silently retry unsupported
native requests through function calling.

When a shell schema includes `run_in_background` or other inputs not representable
by the native shell protocol, the adapter selects function calling at request
construction. This preserves those controls instead of silently dropping them;
it is not a retry after a provider error.

For custom implementations, `tool_kind="shell"` is a contract, not merely a
display category: the implementation must accept a command batch through
`command`, optional `timeout_ms`, and return `ShellResult`.
Use another kind for tools with a different interface. Provider wire fields such
as `max_output_length` stay in versioned transport metadata and the provider
continuation, not in tool arguments or the public function schema.

Streaming accumulates only completed shell-call items; it never executes partial
command text. Native calls/results survive history and checkpoint serialization,
and pending approvals retain versioned codec metadata and the logical name on
resume. This metadata is not a tool argument, is not sent to the provider, and
does not grant execution authority. Unknown codec versions are rejected before
replaying an approval. Routing is local to each request, including streaming.
Experimental shell approval snapshots from before versioned transport metadata
are not automatically migrated; do not resume them as a different tool protocol.
For host reconciliation of a native shell batch, the confirmed text value must
be canonical JSON, for example
`{"results": [{"status": "exited", "returncode": 0, "stdout": "ok", "stderr": ""}]}`,
with one result per command. The provider adapter converts it to Responses.
Invalid results are rejected rather than assigned an invented success status.
The current executor contract cannot recover partial captured output when its
await times out, so that timeout observation has empty stdout. Abort/cancellation
still propagates; it is not converted into successful completion or retried.

!!! warning "No default shell execution"
    The example intentionally has neither `process.execute` nor a process
    executor: `bash` will refuse to run. To enable it, the host must supply a
    trusted `ProcessExecutor` through `ExecutionEnvironment` and grant the needed
    permissions. This release supplies the integration contract, not a concrete
    sandbox backend, and never falls back to the host shell. Approval integration
    is opt-in through `AgentApprovals`; resource grants do not substitute for
    user confirmation when your application requires it.

## Durable commit observation (experimental)

`agent.watch_commits(thread_id, run_id)` observes checkpoint transactions, not
model deltas. It works across SQLite connections and process restarts. Memory
storage only survives for the lifetime of its store instance. A run must have
at least one new-format atomic commit before it can be observed.

```python
from contextlib import aclosing
from dataclasses import asdict

async with aclosing(agent.watch_commits("catalog:42", "publication:1")) as pages:
    async for page in pages:
        if page.snapshot is not None:
            render_snapshot(page.snapshot)
        for event in page.events:
            render_transition(event.event_id, event.data)
        save_cursor(asdict(page.cursor))
```

The first page contains an atomic snapshot and its cursor, with no old events.
Subsequent pages contain only committed transitions. Persist the cursor **after**
processing the entire page. Renderers should deduplicate by `event_id` when
replaying a page after a consumer crash. Each event also carries its own cursor
for consumers that acknowledge individual events.

```python
from msgflux.data.stores import CheckpointCursor

cursor = CheckpointCursor(**load_cursor())
async with aclosing(agent.watch_commits(
    "catalog:42", "publication:1", after=cursor, limit=100, poll_interval=0.2,
)) as pages:
    async for page in pages:
        for event in page.events:
            render_transition(event.event_id, event.data)
        save_cursor(asdict(page.cursor))
```

This resumes strictly after the supplied cursor, without replacing the consumer's
existing snapshot. `CheckpointStore.read_commits(...)` and `aread_commits(...)`
offer the same API as individual reads; an empty events page means caught up.
The watcher polls until closed, including after terminal status. Closing it or
cancelling observation never sends cancellation to the producer.

Every new `commit_state` transaction stores one durable transition alongside its
snapshot, using the supplied event or a generic `checkpoint` event. Stable event
IDs combine a random stream incarnation and the checkpoint revision. Cursors
also bind namespace, thread and run. Deletion/recreation invalidates old cursors.
Missing history, unknown streams and invalid cursors raise `CheckpointCursorError`;
do not silently reset to a new snapshot when replay continuity matters.

Pages contain at most `limit` events (1–1000); slow readers leave their backlog in
storage instead of building a background queue. This bounds event count, not
payload bytes or the initial snapshot size. Events remain until the run is
deleted; this release does not implement retention/compaction of commit history.

!!! warning "Separate observation contracts"
    This is a run-scoped commit feed, not a thread-wide event bus. It does not
    replay token/reasoning deltas, `tool.start`, or every live `watch()` event.
    Approval decisions written to the separate approval journal are not magically
    part of a checkpoint transaction; reconciliation is, and emits a durable
    `approval.reconciled` transition. Read the journal to refresh decisions.

    `save_state`, `save_with_event`, `append_event`, and `load_events` retain their
    legacy behavior and are outside this feed's atomicity contract. Do not mix
    legacy writes with a run observed through commit cursors. Existing history is
    not backfilled; a legacy run gains a stream on its next atomic commit. Forks
    receive an independent stream on their first atomic commit.

    Cursor possession grants no access. The host must authorize observation and
    protect snapshots and event payloads, which may contain application data.

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
