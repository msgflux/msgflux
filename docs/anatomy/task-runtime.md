# Task Runtime

`Task Runtime` is the runtime contract for background tools in msgFlux.

It exists to support three behaviors that do not fit well inside a plain
tool-call result:

- work that continues after the current model turn ends
- progress updates while that work is still running
- result delivery through either polling or passive notification

This page describes the internal shape for `background task`, `progress`, and
`notification` support. Checkpointing and background agent resume are covered in
[Checkpoints And Replay](checkpoints-and-replay.md). A general event stream is
still outside this primitive.

## Mental Model

The runtime should separate execution from task state.

```text
tool call
  -> ToolLibrary
  -> TaskStore.create(...)
  -> background runner
     -> task status updates
     -> optional progress updates
     -> final result or error
  -> AgentInbox.publish(...)
```

The important shift is that a background tool no longer writes directly to a
`Future` registry only. It writes to a durable task record first, and the
future becomes an implementation detail of the runner.

## Core Primitives

The first version should introduce three runtime primitives.

### `TaskStore`

`TaskStore` is the source of truth for task state.

It owns:

- task creation
- task status transitions
- progress snapshots
- result and error persistence
- lookup by `task_id`
- listing tasks by scope

The minimum task shape should be:

```text
task_id
status
tool_name
created_at
updated_at
completed_at
result
error
progress
metadata
```

`progress` should be a structured object, not a string:

```text
stage
message
current
total
percent
```

`percent` stays optional. Some tasks only have stage-based progress.

### `TaskHandle`

`TaskHandle` is the controlled object exposed through `handle.get_task()` inside
a background tool.

It exists so the tool can report progress without mutating the store directly.

The core methods are:

- `set_running(...)`
- `update_progress(...)`
- `complete(result)`
- `fail(error)`

This keeps background progress cooperative. The runtime does not try to infer
progress from stdout, logs, or provider output.

### `AgentInbox`

`AgentInbox` is the delivery layer for task updates.

It should not own task state. It only reacts to task events and decides how
they are delivered.

The first version only needs:

- `publish(task_notification)`
- `peek()`
- `drain()`

## Two Delivery Paths

Background tasks should support both delivery paths from the start.

### Active Path

The active path is explicit polling through tools:

- `task_status(task_id)` returns compact non-null state
- `task_interrupt(task_id)` requests a cooperative interrupt
- `task_wait(task_id)` blocks until the task reaches a terminal state or times out
- `task_output(task_id)` returns only the final output
- `task_list()` returns one compact line per task visible in the current scope
- `task_activity(task_id)` returns compact newline-separated activity for tasks with the
  `activity` capability

This path is canonical. A passive notification should never be the only way to
observe task state.

### Passive Path

The passive path is automatic delivery when a task changes state.

The runtime materializes that delivery as a synthetic system message containing
a compact `<notifications>` batch.

That choice keeps notifications compatible with the existing message-driven
agent loop without inventing a second message protocol.

## Status Model

The initial status machine should stay small:

```text
queued -> running -> completed
queued -> running -> failed
queued -> running -> interrupted
```

The first implementation does not need retries, child tasks, or checkpoint
resume in the state machine.

## Execution Flow

The background path should look like this:

```text
model emits tool call
  -> ToolLibrary resolves tool
  -> if tool_config.background:
       -> TaskStore.create(status=queued)
       -> ToolLibrary returns task_id immediately
       -> runner executes tool in background
          -> TaskHandle.set_running(...)
          -> tool body updates progress
          -> TaskHandle.complete(...) or fail(...)
          -> AgentInbox publishes completion/failure
```

End-to-end shape:

```text
model tool call
    |
    v
ToolLibrary.forward(...)
    |
    +--> TaskStore.create(status=queued)
    |
    +--> return task_id immediately
    |
    `--> Executor.submit(background runner)
             |
             v
      TaskHandle.set_running/update_progress
             |
             +--> TaskStore updates task record
             |
             `--> AgentInbox.publish(
                     source="task" | "task_progress",
                     ref=task_id,
                     status=...
                 )

later, on the next provider boundary:

Agent._prepare_model_execution()
    |
    +--> AgentInbox.drain()
    +--> render <notifications>...</notifications>
    `--> provider call sees the notification
```

This means the immediate tool result becomes a lightweight dispatch response,
not the business result itself.

## Tool Config Shape

The first implementation can stay close to the current tool config model while
keeping runtime-only handles out of the model schema.

Recommended public configuration:

- `background=True`
- `allow_background=True`
- `background_capabilities=["activity", "message"]`
- `handle={...}` for exact runtime-only handle access
- `mf.Hidden` when a parameter should be excluded from the model-facing schema

`background=True` means the developer has chosen background execution for every
call. The model never receives a choice parameter for that tool.

`allow_background=True` means the model can choose per call. The runtime adds a
reserved boolean argument named `run_in_background` to the tool schema. When the
model sets it to `true`, `ToolLibrary` removes that argument from the call
payload and dispatches the tool through the background task runtime. When the
argument is `false` or `null`, the tool runs normally. Manual callers may also
omit the argument, which is treated the same as `false`.

The important detail is that `Hidden` only removes the parameter from the
model-facing schema. `ToolLibrary` remains responsible for building the actual
runtime kwargs.

An injected `ToolHandle` exposes only the configured domains and actions without
exposing the whole `ToolLibrary` object.

For background tools, `handle.task` exposes configured operations such as
`progress` and `interrupt_check`. `handle.notifications.publish(...)` emits
agent-visible status updates bound to the current `task_id`.

When the background source is an `Agent`, it receives `activity` and `message`
capabilities by default. The task runtime uses the union of capabilities across
active background sources to install
`task_activity(task_id=...)` and `task_message(task_id=..., message=...)`.
Source identity remains separate from those controls: a future `bash` source can
declare `activity` without inheriting agent messaging or checkpoint resume.

## Why Both `task_status` And `task_output`

These tools solve different problems.

- `task_status` is for orchestration and polling logic
- `task_interrupt` is for cooperative interruption
- `task_wait` is for synchronous orchestration when the caller wants to pause
- `task_output` is for consuming the final payload
- `task_activity` is for compact recent activity from a task with `activity`

If `task_output` is the only reader, the runtime will collapse state, progress,
and final result into one interface. That makes notifications and collaboration
harder later.

When it is available, `task_activity` fills the gap between `task_status` and
`task_output`. It gives the model a compact view of what happened inside a
background task without dumping raw tool outputs into the prompt.

`task_interrupt` should be treated as cooperative by default. It can interrupt a task
immediately only when the background future has not started yet. Otherwise the
task observes the interrupt request at the next checkpoint.

## Notification Policy

The runtime should be conservative about progress notifications.

By default:

- always notify on `completed`
- always notify on `failed`
- do not notify every progress update

## Implementation Boundaries

The code should stay split by responsibility:

```text
src/msgflux/tasks/store.py
  -> TaskRecord
  -> TaskProgress
  -> TaskActivity
  -> TaskStore
  -> TaskActivityRecorder
  -> TaskHandle

src/msgflux/runtime/agent_inbox/
  -> AgentNotification
  -> AgentInbox
  -> ToolNotificationHandle

src/msgflux/tools/handles.py
  -> ToolHandle facades
  -> internal ToolLibraryHandle

src/msgflux/nn/modules/tool.py
  -> background dispatch integration
  -> hidden-handle task tool registration
  -> Hidden runtime parameter resolution

src/msgflux/nn/modules/agent.py
  -> notification delivery into model execution
```

`ToolLibrary` remains the execution boundary for tools. It should not absorb
all task and notification logic directly into one class.

## Out Of Scope

The first implementation should explicitly avoid these features:

- checkpointing
- agent resume/reopen
- multi-agent coordination
- free-form event streaming

Those can build on top of this runtime later, but they should not shape the
first task contract.

## Why This Shape Matters

If background work is modeled only as `Future -> string result`, msgFlux gets
dispatch but not orchestration.

If notifications are modeled only as ad-hoc callbacks, msgFlux gets delivery
but not a clean message contract.

This runtime shape keeps the boundaries clear:

- `ToolLibrary` dispatches work
- `TaskStore` owns task state
- `AgentInbox` owns delivery
- `Agent` consumes notifications as messages

That gives msgFlux a base for polling, passive delivery, progress reporting,
and later coordination features without redesigning the first contract.
