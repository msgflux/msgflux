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
