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
    hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    dedupe_key: str | None = None
    created_at: str | None = None
    role: Literal["system", "user"] | None = "system"
```

Notes:

- `source` identifies the origin, for example `task`, `context_budget`,
  `checkpoint`, or `tool_registry`
- `role` controls delivery: `system` for runtime context, `user` for external
  user input, and `None` for `control` events that never reach the model
- `ref` points to the concrete object, for example `task_id` or `run_id`
- `status` stays simple and machine-friendly
- `hint` is the short instruction the model can act on
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

When an inbox is owned by an agent, the agent binds it to the effective
execution scope before model execution. In practice, that means the namespace
usually becomes the agent module name, while `thread_id` and `run_id` come
from the active `ExecutionScope` or inherited runtime context.

`thread_id` is the conversation identity. It should stay stable while the root
agent and its child tasks are working on the same user conversation or workflow.
`run_id` is narrower: it identifies one resumable execution inside that
conversation. The root agent and each background subagent may share a
`thread_id`, but they should have different `run_id` values.

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
<notifications>...</notifications>
              |
              v
         provider call
```

## Rendering

The renderer converts a drained batch into one or more synthetic messages.

Runtime notifications are delivered as `role="system"` messages containing a
compact `<notifications>` batch. They are operational context, not user speech.

Delivery is independent from `source`. Trusted inbox publishers can set
`role="user"`, but tool notification handles do not expose that option and
therefore cannot manufacture user speech.

Incoming user messages are delivered separately as `role="user"` messages. They
are not wrapped in `<notifications>`, because they represent new user input rather
than runtime context.

When a drain contains both runtime notifications and incoming user messages, the
message order is:

1. `system`: runtime notifications
2. `user`: incoming user message batch

That lets the model receive operational state before the new user turn.

Example:

```xml
<notifications>
task_id=abcd1234 status=completed tool=long_sum
source=context_budget ref=run_1 status=warning usage_percent=92 | be concise and avoid repeating prior context
</notifications>
```

The corresponding ChatML message is:

```python
{
    "role": "system",
    "content": "<notifications>...</notifications>",
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

Why group runtime notifications:

- it reduces message noise
- it keeps the system-visible content grouped
- it allows multiple runtime sources to deliver together

Current tradeoff: the XML-like envelope is explicit but verbose. If token
pressure becomes a problem, the renderer is the right place to simplify the
format without changing producers or persistence.

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
a background agent task, the dispatcher uses the task metadata to recover the
child agent namespace and run id, then delivers the message under the same
`thread_id`. The child agent can then continue from its checkpointed execution
state while the root receives task notifications through the inbox.

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

## Current Direction

The current direction is:

- introduce `AgentNotification`
- introduce `AgentInbox`
- propagate the inbox through `execution_context`
- inject notifications from `_prepare_model_execution()`
- let runtime producers publish structured notifications directly
- keep rendering centralized in one place
- persist notifications in history by default
