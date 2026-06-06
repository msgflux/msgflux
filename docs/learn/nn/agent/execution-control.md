# Execution Control

Execution control is the runtime layer used to identify, resume, interrupt, and
feed an agent while it is executing.

The core pieces are:

| Piece | Purpose |
|-------|---------|
| `ExecutionScope` | Identifies the active execution with `session_id`, `run_id`, and `namespace`. |
| `checkpointer` | Persists the agent snapshot so a run can resume. |
| `AgentInbox` | Holds pending messages, notifications, and control signals for the agent loop. |
| `AgentInboxStore` | Optional persistence boundary for the inbox. Without one, the inbox is in memory. |

## Execution Scope

Use `ExecutionScope` when you need stable cross-session identity.

```python
import msgflux as mf
import msgflux.nn as nn

agent = nn.Agent(
    name="support_agent",
    model=mf.Model.chat_completion("openai/gpt-4.1-mini"),
)

scope = mf.ExecutionScope(
    session_id="customer_42",
    run_id="ticket_9001",
)

result = agent("Investigate this ticket.", scope=scope)
```

- `session_id`: identifies the conversation, user session, or chat thread.
- `run_id`: identifies the current resumable execution inside that session.

If no scope is passed, msgFlux still runs with a default session.

## Checkpointing

Use a checkpointer when a run should resume after pause, stop, process restart,
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
    session_id="customer_42",
    run_id="ticket_9001",
)

agent("Investigate this ticket.", scope=scope)
```

Available checkpoint stores:

- `mf.Store.checkpoint("in_memory")`
- `mf.Store.checkpoint("sqlite", path=".msgflux/checkpoints.sqlite3")`

When you call an agent with a `scope.run_id`, msgFlux first checks whether a
checkpoint already exists for `(namespace, session_id, run_id)`.

Resume behavior:

- `running`: resumed from the saved snapshot.
- `paused`: resumed from the saved snapshot.
- `failed`: resumed from the saved snapshot. This is the primary recovery path
  after a provider, tool, process, or infrastructure failure.
- `completed`: not resumed.
- `stopped`: not resumed.

On resume, the new task input is ignored and the saved messages/vars continue
from the checkpointed state. Use a new `run_id` when you want a fresh execution.

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
    session_id="customer_42",
    run_id="ticket_9001",
)

agent("Investigate this ticket.", scope=scope)
```

When the agent starts, it binds the inbox to the active `ExecutionScope` using
the agent module name as namespace. For the example above, pending inbox data is
stored under:

```python
namespace = "support_agent"
session_id = "customer_42"
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
scope = mf.ExecutionScope(session_id="customer_42", run_id="ticket_9001")

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
    session_id="customer_42",
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
agent.agent_inbox.stop(reason="Operator stopped the run.")
agent.agent_inbox.cancel(reason="User cancelled the request.")
```

Behavior:

- `pause` raises `TaskPauseRequestedError` and checkpoints the run as `paused`
  when a checkpointer is configured.
- `stop` and `cancel` raise `TaskStopRequestedError` and checkpoint the run as
  `stopped` when a checkpointer is configured.
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
