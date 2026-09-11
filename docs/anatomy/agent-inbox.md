# Agent Inbox

This page records the notification primitive used by `Agent`.

Implementation lives under `src/msgflux/runtime/agent_inbox/` and
`src/msgflux/runtime/context.py`.

The goal is to make runtime notifications pluggable, durable, and
easy to inject into model execution without hard-coding every new signal inside
the core tool loop.

## Goals

- provide one runtime primitive for agent-visible notifications
- let tasks, checkpoints, budget tracking, and registry updates publish through
  the same contract
- avoid adding ad hoc message-building logic across the codebase
- keep delivery centralized in `Agent`
- prepare a clean path for future hooks and streamed runtime events

## Primitives

The current design uses two core objects:

- `AgentNotification`: one structured runtime notification
- `AgentInbox`: the queue/store responsible for publish, dedupe, coalescing,
  and delivery

Recommended shape:

```python
@dataclass
class AgentNotification:
    notification_id: str
    source: str
    ref: str | None = None
    status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    dedupe_key: str | None = None
    created_at: str | None = None
```

Notes:

- `source` identifies the origin, for example `task`, `context_budget`,
  `checkpoint`, or `tool_registry`
- `ref` points to the concrete object, for example `task_id` or `run_id`
- `status` stays simple and machine-friendly
- `metadata` carries extra fields without changing the base contract
- `dedupe_key` allows coalescing repeated updates

## Inbox API

The current `AgentInbox` stays small:

- `publish(notification)`
- `publish_many(notifications)`
- `peek()`
- `drain()`
- `ack(notification_ids)`

The inbox is a runtime primitive, not just a renderer.

That means it is responsible for:

- holding pending notifications
- deduplicating noisy updates
- deciding what is delivered together in the next injection
- exposing a single batch to `Agent`

## Scope Binding

An inbox can be created before it is attached to an agent or execution scope.
In that unbound state it uses explicit fallback identifiers:

```text
namespace = default_namespace
thread_id = generated thd_<uuid>
run_id = generated run_<uuid>
```

When an inbox is owned by an agent, the agent resolves a scoped inbox view
before model execution. The view shares the configured store, but owns its
namespace, `thread_id`, and `run_id`. A concurrent run or nested agent can
therefore resolve another view without rebinding the inbox already being used
by its parent. The namespace usually becomes the agent module name, while
`thread_id` and `run_id` come from the active `ExecutionScope` or inherited
runtime context.

`thread_id` is the conversation identity for the agent that owns the inbox.
`run_id` is narrower: it identifies one resumable execution inside that
conversation. A background subagent uses its own `thread_id` and `run_id`;
`parent_run_id` and `root_run_id` preserve its relationship with the root
execution.

## Delivery Model

`Agent` remains the only component that injects notifications into model input.

The recommended delivery point is `_prepare_model_execution()`, not
`_prepare_inputs()`.

This matters because `_prepare_model_execution()` runs both:

- on the first turn
- before each new provider call during tool loops and flow-control loops

That gives one safe boundary for delivery without scattering injection logic
through the rest of the execution path.

The flow is:

```text
runtime producer publishes AgentNotification
  -> AgentInbox stores it
  -> Agent._prepare_model_execution() drains pending notifications
  -> Agent renders one synthetic message
  -> provider call receives that message with the rest of the history
```

More concretely:

```text
background tool / checkpoint / hook
              |
              v
     AgentNotification(...)
              |
              v
      AgentInbox.publish()
              |
              v
   Agent._prepare_model_execution()
              |
              v
     AgentInbox.drain() + render_messages()
              |
              v
<notification source="task" ref="abcd1234" status="completed"/>
              |
              v
         provider call
```

## Rendering

The renderer converts a drained batch into one or more synthetic messages.

Runtime notifications are delivered as `role="system"` messages containing one
or more compact `<notification .../>` tags. They are operational context, not
user speech.

Incoming user messages are delivered separately as `role="user"` messages
because they represent new user input rather than runtime context.

When a drain contains both runtime notifications and incoming user messages, the
message order is:

1. `system`: runtime notifications
2. `user`: incoming user message batch

That lets the model receive operational state before the new user turn.

Example:

```xml
<notification source="task" ref="abcd1234" status="completed" tool="long_sum"/>
<notification source="context_budget" ref="run_1" status="warning" usage_percent="92"/>
```

The corresponding ChatML message is:

```python
{
    "role": "system",
    "content": "<notification source=\"task\" .../>",
}
```

An incoming user message renders separately:

```python
{
    "role": "user",
    "content": (
        "<incoming_user_message>\n"
        "Please adjust the answer.\n"
        "</incoming_user_message>"
    ),
}
```

Why group runtime notifications in one message:

- it reduces message noise
- it keeps the system-visible content grouped
- it allows multiple runtime sources to deliver together

The notification has no instruction field. Behavioral guidance belongs in
agent instructions or tool guidance; the event carries only state.

## Context Propagation

`AgentInbox` should be propagated through `execution_context`, alongside:

- `thread_id`
- `run_id`
- `parent_run_id`
- `root_run_id`
- `checkpoint_store`

That lets runtime producers publish without importing `Agent` or returning
special payloads through normal outputs.

Example direction:

```python
with execution_context(
    thread_id=...,
    run_id=...,
    checkpoint_store=...,
    agent_inbox=inbox,
):
    ...
```

## Producers

Any runtime component should be able to publish directly to the inbox through
the current execution context or through an explicitly injected
`ToolLibraryHandle`.

Current and planned producers include:

- background task completion/failure
- task progress updates
- checkpoint restore/replay signals
- context budget warnings
- tool registry changes made through injected `ToolLibraryHandle` values

The important rule is:

- producers publish structured notifications
- producers do not build synthetic messages

`task_message` uses this same identity model. When the root sends a message to
a background agent task, the dispatcher uses task metadata to recover the child
agent namespace, `thread_id`, and `run_id`. The child can then continue from its
own checkpointed execution state while the root receives task notifications
through the root inbox.

## Coalescing

Coalescing is necessary for progress-like updates.

For example, one background task may emit five progress updates before the
agent reaches the next provider boundary. The inbox should be able to keep only
the latest relevant update by `dedupe_key`.

That allows simple producers:

```python
inbox.publish(
    AgentNotification(
        source="task_progress",
        ref=task_id,
        status="update",
        dedupe_key=f"task_progress:{task_id}",
        metadata={"step": 3, "of": 5},
    )
)
```

Without requiring a more complex progress-specific protocol in the task API.

## Persistence

The inbox should support pluggable storage, because notifications may need to
survive process restart and replay.

This means `AgentInbox` is designed as a real store boundary. `Agent` creates a
memory-backed inbox for local use when no inbox is provided. Direct
`AgentInbox` instances need an explicit store and fail fast without one instead
of silently creating hidden state.

## Hooks

The inbox should be easy to extend through the hook system.

The cleanest way to do that is to expose stable methods on the inbox itself,
then allow hooks to observe or transform around those methods.

Examples of useful hook points:

- before `publish`
- after `publish`
- before `drain`
- after `drain`
- before render
- after render

This avoids editing core `Agent` code every time a new notification source is
introduced.

## Relation To Future Events

This primitive should not be confused with future streamed runtime events.

The planned separation is:

- runtime events: everything that can happen in the system
- agent inbox notifications: the curated subset delivered to the model

That separation keeps the future event system broad without forcing the model
to see every internal event directly.
