# Runtime

Runtime is the layer used to identify, resume, interrupt, and feed an agent
while it is executing.

The core pieces are:

| Piece | Purpose |
|-------|---------|
| `ExecutionScope` | Identifies the active execution with `thread_id`, `run_id`, and `namespace`. |
| `AbortSignal` | Carries local cancellation requests to the active runtime before safe interruption points. |
| `checkpointer` | Persists the agent snapshot so a run can resume. |
| `TaskStore` | Persists background task records, activity, outputs, and routing metadata. |
| `AgentInbox` | Holds pending messages, notifications, and control signals for the agent loop. |
| `AgentInboxStore` | Optional persistence boundary for the inbox. Without one, the inbox is in memory. |

## Execution Scope

Use `ExecutionScope` when you need stable runtime identity.

```python
import msgflux as mf
import msgflux.nn as nn

agent = nn.Agent(
    name="support_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
)

scope = mf.ExecutionScope(
    thread_id="customer_42",
    run_id="ticket_9001",
)

result = agent("Investigate this ticket.", scope=scope)
```

You can also bind the same scope as ambient runtime context. This is useful
when nested calls should inherit the same thread/run identity without passing
`scope=...` through every call:

```python
with mf.execution_context(scope=scope):
    result = agent("Investigate this ticket.")
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
  same conversation".

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

Runtime resources can be attached to the same context. For example, this makes
the checkpointer available to any agent call made inside the block:

```python
checkpointer = mf.Store.checkpoint(
    "sqlite",
    path=".msgflux/checkpoints.sqlite3",
)

with mf.execution_context(scope=scope, checkpoint_store=checkpointer):
    result = agent("Investigate this ticket.")
```

`ExecutionScope` carries identity. Runtime resources such as `checkpoint_store`,
`task_store`, and `agent_inbox` are passed to `execution_context(...)` or to the
agent/runtime objects that own them. When an agent runs inside this context, it
reads the active scope for `thread_id`, `run_id`, and `namespace`, then reads
the active `checkpoint_store` from the same runtime context. The checkpointer is
therefore applied to the scoped execution without being stored inside the
`ExecutionScope` object itself.

## Abort Signal

`AbortSignal` is local runtime cancellation for the currently active process.
It is useful for UI and CLI controls such as pressing `Esc` while a model is
generating. It is carried by `ExecutionScope` and exposed through
`get_execution_context().get("abort_signal")`.

```python
abort_signal = mf.AbortSignal()
scope = mf.ExecutionScope(
    thread_id="customer_42",
    run_id="ticket_9001",
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
`interrupted`.

## Checkpointing

Use a checkpointer when a run should resume after pause, interrupt, process restart,
or tool-driven continuation.

```python
checkpointer = mf.Store.checkpoint(
    "sqlite",
    path=".msgflux/checkpoints.sqlite3",
)

agent = nn.Agent(
    name="support_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    checkpointer=checkpointer,
)

scope = mf.ExecutionScope(
    thread_id="customer_42",
    run_id="ticket_9001",
)

agent("Investigate this ticket.", scope=scope)
```

If you do not want to attach the checkpointer to the agent instance, provide it
through runtime context together with the execution scope:

```python
agent = nn.Agent(
    name="support_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
)

with mf.execution_context(scope=scope, checkpoint_store=checkpointer):
    agent("Investigate this ticket.")
```

Inside the call, the agent resolves the effective scope first. It then uses the
active checkpointer to load or save state under the effective
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

On resume, the new task input is ignored and the saved messages/vars continue
from the checkpointed state. This is intentional: the retry is restoring the
same execution, not adding a new user message. Use the same `thread_id` with a
new `run_id` when you want to continue the conversation with fresh input.

For background subagents, the task id is used as the subagent `run_id`. Reusing
that task id resumes or continues the same subagent. Creating a new task id
starts a separate subagent within the same thread.

The checkpointer can also be used directly when you need to inspect or manage
durable runs outside the agent loop. The lookup key is always
`(namespace, thread_id, run_id)`. For an agent, `namespace` is normally the
agent name:

```python
namespace = "support_agent"
thread_id = "customer_42"
run_id = "ticket_9001"

state = checkpointer.load_state(namespace, thread_id, run_id)
print(state["status"] if state else "missing")
```

List recent runs for a thread:

```python
runs = checkpointer.list_runs(namespace, thread_id, limit=10)
for run in runs:
    print(run["run_id"], run["status"], run["updated_at"])
```

Find runs that may still need recovery:

```python
incomplete = checkpointer.find_incomplete_runs(namespace, thread_id)
```

Load the newest checkpointed run in a thread. This is useful when the caller
has a `thread_id` but did not persist the latest `run_id` separately:

```python
latest = checkpointer.load_latest_run(namespace, thread_id)
```

Fork a checkpoint into a new thread/run. This copies the checkpoint state while
preserving the original run:

```python
forked = checkpointer.fork_run(
    namespace,
    source_thread_id="customer_42",
    source_run_id="ticket_9001",
    target_thread_id="customer_42_review",
    target_run_id="ticket_9001_review",
    status="paused",
)
```

Delete a single run when it is no longer needed:

```python
deleted = checkpointer.delete_run(namespace, thread_id, run_id)
```

Clear a broader set of checkpoints:

```python
removed = checkpointer.clear(namespace=namespace, thread_id=thread_id)
```

Stores also expose low-level event methods for append-only audit entries:

```python
checkpointer.append_event(
    namespace,
    thread_id,
    run_id,
    {"type": "operator_note", "message": "Reviewed by support lead."},
)

events = checkpointer.load_events(namespace, thread_id, run_id)
```

## Agent Inbox

`Agent` creates a memory-backed inbox by default:

```python
agent = nn.Agent(
    name="support_agent",
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
    name="support_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    agent_inbox=agent_inbox,
)
```

You can also provide the inbox through runtime context instead of binding it to
the agent instance:

```python
scope = mf.ExecutionScope(thread_id="customer_42", run_id="ticket_9001")
agent_inbox.bind_scope(scope, namespace="support_agent")

with mf.execution_context(scope=scope, agent_inbox=agent_inbox):
    agent("Investigate this ticket.")
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
scope = mf.ExecutionScope(thread_id="customer_42", run_id="ticket_9001")

agent_inbox = mf.AgentInbox(store=inbox_store)
agent_inbox.bind_scope(scope, namespace="support_agent")

agent("Investigate this ticket.", scope=scope)
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
scope = mf.ExecutionScope(thread_id="customer_42", run_id="ticket_9001")
agent_inbox.bind_scope(scope, namespace="support_agent")

agent = nn.Agent(
    name="support_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    agent_inbox=agent_inbox,
)

# In one thread/task:
agent("Work on the ticket until finished.", scope=scope)

# In another thread/task while the agent is still processing:
agent_inbox.user_message("The user added that the payment already cleared.")
```

The model receives the message as a synthetic user block:

```xml
<incoming_user_message>
The user added that the payment already cleared.
</incoming_user_message>
```

If the writer does not have the `agent` object, create another inbox with the
same store and execution key:

```python
store = mf.Store.agent_inbox("sqlite", path=".msgflux/inbox.sqlite3")
scope = mf.ExecutionScope(thread_id="customer_42", run_id="ticket_9001")

external_inbox = mf.AgentInbox(
    store=store,
    namespace="support_agent",
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

You can also publish directly:

```python
external_inbox.publish(
    {
        "source": "incoming_user_message",
        "hint": "Use a shorter answer.",
        "metadata": {"origin": "chat-ui"},
    }
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
  when a checkpointer is configured.
- `interrupt` raises `TaskInterruptRequestedError` and checkpoints the run as
  `interrupted` when a checkpointer is configured.
- Unknown control commands remain normal notifications and are shown to the
  model as `system_note`.

For a persistent writer:

```python
external_inbox.pause(reason="Need human review before continuing.")
```

### System Notifications

Non-user inbox items are delivered as `system_note`:

```python
agent_inbox.publish(
    {
        "source": "system_note",
        "status": "policy_update",
        "hint": "Use the enterprise refund policy for this answer.",
    }
)
```

The model receives:

```xml
<system_note>
<notification>
source: system_note
status: policy_update
hint: Use the enterprise refund policy for this answer.
</notification>
</system_note>
```

Use `incoming_user_message` for new user turns. Use `system_note` or another
system-like source for runtime hints, progress, policy updates, or operator
notes that should not be treated as a direct user request.
