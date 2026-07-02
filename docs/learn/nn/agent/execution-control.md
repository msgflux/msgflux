# Execution Control

Execution control is the runtime layer used to identify, resume, interrupt, and
feed an agent while it is executing.

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

Available checkpoint stores:

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

## Background Tasks And `task_message`

When a tool is allowed to run in the background, msgFlux creates a task record
and returns a task id to the model. The task record lives in the task store and
tracks lifecycle state such as queued, running, paused, completed, failed, and
interrupted.

For normal background tools, the task id is mostly an operational handle:

- `task_status(task_id=...)` reads lifecycle and progress.
- `task_output(task_id=...)` reads the final result once available.
- `task_wait(task_id=...)` waits for completion.
- `task_interrupt(task_id=...)` requests interruption.

Some background tools also support messages after dispatch. The built-in
`AgentTool` does this for subagents. In that case, `task_message` is the way
for the root model to send another message to the same running or resumable
task:

```python
task_message(
    task_id="ab12cd34",
    message="The user clarified that the payment already cleared.",
)
```

The task metadata records enough routing information to reconstruct the call:

- which tool owns the task
- which child agent or tool target was selected
- the checkpoint namespace for that child execution
- the parent/root run lineage
- the `thread_id` shared with the root conversation
- the task id used as the child `run_id`

For an agent task, `task_message` re-dispatches the same tool with the saved
routing parameters and a scope like:

```text
thread_id = original root thread
run_id = task_id
parent_run_id = root run that launched the task
root_run_id = root run of the whole execution tree
```

That means the child agent can recover through the normal checkpoint rule:
same `(namespace, thread_id, run_id)` resumes a non-terminal checkpoint. If the
child had completed and you want a new independent subagent conversation, call
the `agent` tool again so a new task id/run id is created. If you want to keep
talking to the same subagent task, use `task_message` with the existing
`task_id`.

## Persisting The Inbox

`AgentInbox` is in-memory by default:

```python
inbox = mf.AgentInbox()
```

Pass a store when pending messages and control signals should survive process
restarts or be written by another runtime component:

```python
inbox_store = mf.Store.agent_inbox(
    "sqlite",
    path=".msgflux/inbox.sqlite3",
)

agent = nn.Agent(
    name="support_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    agent_inbox=mf.AgentInbox(store=inbox_store),
)
```

Available inbox stores:

- `mf.Store.agent_inbox("in_memory")`
- `mf.Store.agent_inbox("sqlite", path=".msgflux/inbox.sqlite3")`

You can also instantiate concrete classes directly, but the `Store` factory is
the preferred public interface for application code.

## Full Persistent Runtime

Use both stores when you want resumable execution and durable inbox delivery.

```python
checkpointer = mf.Store.checkpoint(
    "sqlite",
    path=".msgflux/checkpoints.sqlite3",
)
inbox_store = mf.Store.agent_inbox(
    "sqlite",
    path=".msgflux/inbox.sqlite3",
)

agent = nn.Agent(
    name="support_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
    checkpointer=checkpointer,
    agent_inbox=mf.AgentInbox(store=inbox_store),
)

scope = mf.ExecutionScope(
    thread_id="customer_42",
    run_id="ticket_9001",
)

agent("Investigate this ticket.", scope=scope)
```

When the agent starts, it binds the inbox to the active `ExecutionScope` using
the agent module name as namespace. For the example above, pending inbox data is
stored under:

```python
namespace = "support_agent"
thread_id = "customer_42"
run_id = "ticket_9001"
```

For tests and local shared-memory scenarios, use `InMemoryAgentInboxStore`:

```python
store = mf.Store.agent_inbox("in_memory")
agent = nn.Agent(
    name="support_agent",
    model=model,
    agent_inbox=mf.AgentInbox(store=store),
)
```

## Sending Messages While The Agent Is Running

To feed a running agent, write an incoming user message to the same inbox. The
agent drains the inbox before each provider call and after tool calls, before
the next provider call.

```python
scope = mf.ExecutionScope(thread_id="customer_42", run_id="ticket_9001")

# In one thread/task:
agent("Work on the ticket until finished.", scope=scope)

# In another thread/task while the agent is still processing:
agent.agent_inbox.user_message("The user added that the payment already cleared.")
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

external_inbox = mf.AgentInbox(
    store=store,
    namespace="support_agent",
    thread_id="customer_42",
    run_id="ticket_9001",
)

external_inbox.user_message("Ask for the latest invoice number before deciding.")
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

## Control Messages

Control messages interrupt execution at safe provider boundaries.

```python
agent.agent_inbox.pause(reason="Wait for user approval.")
agent.agent_inbox.interrupt(reason="Operator interrupted the run.")
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

## System Notifications

Non-user inbox items are delivered as `system_note`:

```python
agent.agent_inbox.publish(
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
