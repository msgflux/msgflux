# Checkpoints And Replay

This page records the checkpoint and replay contract for msgFlux.

It covers the durability model that should align:

- `Agent`
- background agent tasks
- `Inline`
- parallel `Functional` helpers such as `bcast_gather`

It is a design page, not an API tutorial. The agent, task-runtime, and store
pieces already use this identity model, while some integrations described here
remain architectural direction so the durability surface stays coherent as it
expands.

## Goal

msgFlux needs a durability model that answers three questions consistently:

- how does an execution identify itself across restarts?
- how does the runtime know whether a call is fresh or a resume?
- how do child executions inherit identity without colliding with each other?

The core decision is to make execution identity explicit and stable.

## Identity Model

The durability model should use four concepts:

- `namespace`: which component owns the checkpoint
- `thread_id`: stable conversation or workflow identity
- `run_id`: stable identity for one resumable execution
- `attempt`: retry counter for observability and policy

Only the first three belong to the checkpoint key.

```text
checkpoint key = (namespace, thread_id, run_id)
```

`attempt` stays in metadata and events. It should not be part of the key.

That keeps replay simple:

- retrying the same execution keeps the same `run_id`
- loading the checkpoint uses the same key every time
- the event log still shows which attempt produced each transition

## Why `thread_id` And `run_id` Are Different

`thread_id` is the stable conversation identity.

Examples:

- one user conversation
- one root workflow execution
- one root agent thread

All runs that should see the same long-lived conversation context belong under
the same `thread_id`. Starting a new `thread_id` means starting a separate
conversation or workflow lineage.

`run_id` is the identity of a specific resumable execution inside that thread.

Examples:

- one root agent turn
- one background subagent task
- one inline pipeline run
- one worker inside a parallel step

This separation matters because several executions can belong to the same
thread at once. For example, a root agent run can launch multiple background
subagents. They all share the root `thread_id`, but each subagent gets its own
`run_id` so it can checkpoint and resume independently.

## Default Identity

Runtime identity should be explicit in durable production paths, but the
runtime still needs stable fallback values for local use, tests, and components
that are created before an agent binds a concrete scope.

The generated fallback identifiers are:

```text
thread_id = generated thd_<uuid>
namespace = default_namespace
run_id = generated run_<uuid>
```

These values are convenient local fallbacks, not recommended durable IDs. A
host that needs recovery after process restart should provide a stable
`thread_id` and `run_id` before dispatching work.

Most agent executions do not keep `default_namespace` for long. When an
`Agent` prepares execution, it overrides the effective namespace with the
agent's module name. The default namespace mainly covers runtime primitives
such as an unbound `AgentInbox`.

Agent checkpoint resume still depends on receiving the same `thread_id` and
`run_id`; using a fresh generated run id starts a separate execution instead of
resuming an older one.

## Resume Rule

The runtime should not expose a public `checkpoint_action` or `resume=True`
flag as the main decision point.

The decision should come from the checkpoint store itself:

- if no checkpoint exists for `(namespace, thread_id, run_id)`, start fresh
- if a non-terminal checkpoint exists, resume from it
- if a terminal checkpoint exists, do not resume it automatically

This keeps the caller API smaller and avoids ambiguous booleans.

In practice, the supervisor or host only needs to re-dispatch the same logical
unit of work with the same `thread_id` and `run_id`. The module decides
whether that becomes a fresh execution or a resume by looking at the store.

The practical rule is:

- failure recovery: keep the same `thread_id` and `run_id`
- next message in the same conversation: keep `thread_id`, use a new `run_id`
- new conversation: use a new `thread_id`

That prevents completed runs from being accidentally reused while still making
crash recovery deterministic.

## Checkpoint Store Boundary

Checkpointing and task tracking should stay separate.

### `CheckpointStore`

`CheckpointStore` owns resumable execution state.

It should persist:

- full execution snapshot
- current status
- append-only events
- metadata such as `attempt`, `parent_run_id`, and timestamps

### `TaskStore`

`TaskStore` owns async task lifecycle.

It should persist:

- `task_id`
- task status
- result or error
- progress
- task metadata

The important split is:

- `TaskStore` answers "is this background task alive, done, or failed?"
- `CheckpointStore` answers "how far did this module get?"

This is what allows a task to be re-dispatched after a crash without losing
the module's internal progress.

## Checkpoint Shape

The durable model should use full snapshots, not diffs.

Minimum shape:

```text
status
snapshot
metadata
```

Typical metadata:

```text
attempt
parent_run_id
root_run_id
scope_id
item_key
created_at
updated_at
```

Status should stay small and execution-oriented:

- `running`
- `completed`
- `failed`
- `interrupted`

The exact set can evolve, but the model should stay about checkpoint semantics,
not task queue semantics.

## `ChatMessages` And Thread Persistence

`ChatMessages` is the natural snapshot container for durable agents.

It should carry:

- `thread_id`
- turn records
- message history
- serialized assistant/tool history

That gives the agent a concrete way to:

- save after each tool-loop boundary
- recover the exact chat state later
- continue from the last durable boundary instead of rebuilding context

## Context Propagation

Execution context should live outside `Agent` and outside `Module`, in a shared
context module.

The context should be entered at orchestration boundaries such as:

- `Agent.forward(...)`
- `Inline(...)`
- background task runners
- supervised parallel workers

Recommended context fields:

- `thread_id`
- `run_id`
- `parent_run_id`
- `root_run_id`
- `checkpoint_store`

### Precedence

Context resolution should follow this order:

1. explicit call argument
2. component-local configured value
3. inherited runtime context
4. generated fallback

This is what allows nested execution to inherit by default while still letting
the caller override identity explicitly.

## Root Agent

The root agent should run under a stable `thread_id`.

Its `run_id` should identify the current logical turn or execution.

Example:

```text
namespace = agent:research_root
thread_id = user_42
run_id = run_f13a2b
```

If the process crashes and the host re-dispatches that same run:

- same `thread_id`
- same `run_id`

the agent checks the store and resumes from the last saved boundary.

## Background Subagent

A background subagent should inherit the parent's `thread_id`.

Its `run_id` should be the `task_id`.

Example:

```text
namespace = agent:research_worker
thread_id = user_42
run_id = task_ab12cd34
parent_run_id = run_f13a2b
root_run_id = run_f13a2b
```

This keeps two things true at once:

- the worker belongs to the same thread as the root
- the worker is resumable by its own task identity

If the process collapses and the task is re-dispatched, the worker enters with
the same `(namespace, thread_id, run_id)` and resumes automatically if a
checkpoint exists.

## When A Subagent Needs More Input

The task does not need a special `blocked` status to represent "waiting for a
decision from the root".

Instead, the worker can finish the current task with a structured output such
as:

```python
{
    "kind": "needs_input",
    "question": "Should I preserve compatibility with the legacy parser?",
    "checkpoint": {
        "thread_id": "user_42",
        "run_id": "task_ab12cd34",
    },
}
```

That keeps the responsibilities clean:

- the subagent stops cleanly
- the root decides whether to continue it
- continuation uses the same checkpoint identity

For runtime-managed background agents, the root usually does not need to build
that continuation call by hand. It can call `task_message(task_id=..., message=...)`.
The task store metadata points back to the tool, selected child agent, thread,
checkpoint namespace, and child run id. The dispatcher reconstructs the tool
call and passes the new message while preserving:

- the same `thread_id`
- the task id as child `run_id`
- the original parent/root lineage
- the same checkpoint store

This makes `task_message` a routing operation, not a second task lookup API.
The resumed child still relies on the checkpoint store to decide whether it is
recovering an interrupted run or starting from the current task state.

## Inline And Nested Inline

`Inline` should use the same durability key:

```text
(namespace, thread_id, run_id)
```

A child inline should inherit the parent's `thread_id` by default.

It should get its own `run_id`, while also recording:

- `parent_run_id`
- `root_run_id`

Example:

```text
parent inline:
  thread_id = user_42
  run_id = run_parent

child inline:
  thread_id = user_42
  run_id = run_child
  parent_run_id = run_parent
  root_run_id = run_parent
```

This keeps identity tree-shaped without collapsing parent and child into the
same checkpoint stream.

## Parallel Functional Workers

Parallel execution needs stable child identities so one failed worker can be
retried or resumed without waiting for the others.

The recommended composition is:

```text
child_run_id = <parent_run_id>:<scope_id>:<item_key>
```

### `scope_id`

`scope_id` answers:

- where in the flow did this child come from?

Examples:

- `s2`
- `s4.body.s1`
- `translate_batch`

### `item_key`

`item_key` answers:

- which child inside that scope is this?

Examples:

- `w0`
- `customer_42`
- `translator`

Together they avoid collisions:

- `scope_id` distinguishes different parallel sites
- `item_key` distinguishes siblings in the same site

### Example

```text
parent run_id = run_a1b2c3

workers:
run_a1b2c3:s2:w0
run_a1b2c3:s2:w1
run_a1b2c3:s2:w2
run_a1b2c3:s2:w3
```

If only `w1` fails:

- the supervisor increments `attempt` for `w1`
- re-dispatches only `run_a1b2c3:s2:w1`
- the worker checks its checkpoint and resumes if possible

The other workers continue normally.

## Why `key_fn(item)` Can Exist

For data-driven parallelism, index-based identities are sometimes too weak.

If item order can change between retries or restarts, replay by index becomes
fragile. A caller-supplied key function can provide a stable `item_key`.

Example:

```text
item_key = customer_42
```

instead of:

```text
item_key = 3
```

This is optional. Fixed-order worker lists can still use generated keys like
`w0`, `w1`, and so on.

## Persistence Provider Selection

Provider selection should stay outside runtime modules.

The host or bootstrap layer should decide which checkpoint backend to use and
inject that store into the components that need it.

Examples of future configuration names:

- `MSGFLUX_CHECKPOINT_PROVIDER`
- `MSGFLUX_CHECKPOINT_PATH`

The important rule is that `ToolLibrary`, `Agent`, and `Inline` should receive
stores, not own provider selection themselves.

## Summary

The durability model should stay simple:

- use `(namespace, thread_id, run_id)` as the checkpoint identity
- keep `attempt` out of the key
- derive resume from the presence of a checkpoint, not from a public boolean
- keep `TaskStore` and `CheckpointStore` separate
- use `task_id` as the `run_id` for background subagents
- derive stable child run ids in parallel execution from `scope_id` and `item_key`

That contract is enough to support:

- agent replay
- background subagent recovery
- nested inline inheritance
- one-for-one worker replay in parallel execution

without adding a large public surface too early.
