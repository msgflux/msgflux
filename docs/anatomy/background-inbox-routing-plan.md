# Background Inbox Routing Without Per-Task Retention

Status: design proposal plus a focused resume-routing fix. This PR does not
remove either dispatcher map.

## Problem and measured boundary

`BackgroundTaskDispatcher` keeps a `task_id -> AgentInbox` map. Dispatch adds
one entry for each background agent; completion removes its future but not its
inbox. The map is cleared only with `ToolLibrary.clear()`. A separate map keeps
the checkpoint-store object used at dispatch. This is useful for later messages
and resume, but a long-lived library retains one inbox object per historical
agent task even when the task record and inbox contents live in stores.

An offline profile used one library, one synthetic child model and 100
sequential completed `AgentTool` tasks. `tracemalloc` started after setup; each
sample followed GC. With an in-memory task store, retained Python allocations
were about 878 KiB when the root did not drain completion notifications and
507 KiB when it did. With a SQLite task store and drained notifications, the
result was about 234 KiB. In all three runs the dispatcher retained 100 inbox
entries and 100 checkpoint-store entries, while completed futures returned to
zero. These are workload-specific allocations, not RSS measurements or proof
that the whole difference belongs to the dispatcher map. In particular, the
in-memory task store intentionally keeps task records and activity in RAM.

The desired invariant is **retained memory proportional to active work and
pending messages, not all historical task IDs**, without giving up late
messages, recovery or bucket-scoped notification routing.

## Current routes

| Operation | Current destination | Why a map currently matters |
| --- | --- | --- |
| Initial background agent dispatch | A fork of the execution-context inbox, scoped to child namespace, thread and task ID | The fork is put in `_task_inboxes` and passed into the worker. |
| `task_message` while running | `handle.get_task_inbox(task_id)` | Without the map it currently returns `unsupported`; the task store does not expose an inbox publisher. |
| `task_message` after completion/pause/failure | `resume_agent_task()` uses the mapped inbox, or forks the current root inbox if none is mapped | The checkpoint route is stored in task metadata, but the original inbox store binding is not. |
| Task completion status | The root inbox captured by the worker | This is **not** the child inbox used for messages to the running agent. |
| Captured tool in a bucket | `ToolBucketHandle.with_runtime()` receives the worker-scoped inbox | The bucket/tool name describes the source, not the destination. |

The persisted task metadata already contains child checkpoint namespace,
thread ID and current checkpoint run ID. On a resumed completed task,
`checkpoint_run_id` changes. Before the fix in this PR, the resume path reused
the cached inbox without changing its run ID. A controlled run returned
`task_message: delivered` while the old run had one pending message and the
current run had none. The fix re-forks a mismatched cached inbox from its
original store, and a regression test holds the resumed model call open while
a second message is published. This is a correctness fix, not the proposed
replacement for the map.

## Proposed contract

Persist a **logical inbox route** with each background agent task: a store
binding identifier plus `(namespace, thread_id, run_id)`. Keep separate routes
for (1) messages to the child and (2) status notifications to the parent.
The route must not contain a Python object or machine-specific SQLite path.
The application/runtime must resolve its store binding on recovery, as it
already supplies checkpoint and task stores. Failure to resolve the binding
must be explicit; publishing to a convenient default store could lose a message
or deliver it to the wrong conversation.

`task_message` would load the task record, resolve its child route and publish
through a short-lived `AgentInbox` view sharing the configured store. The
worker can keep its own live view until its future settles. Publishing must
continue through `AgentInbox` normalization, dedupe and claim semantics, not
write directly to the store from each caller. A completed task may acquire a
new child `run_id` during resume; update the persisted route at that transition
before exposing it to concurrent messages. A conditional task-state update or
equivalent serialization is needed to keep `task_message` from racing with
resume.

Bucket tools should receive a destination publisher in their execution-scoped
handle. The captured tool or bucket contributes source metadata and call ID,
but does not infer where to publish. Completion notifications use the parent
route even when a nested bucket was executing with a child route. This keeps
the two directions distinct and avoids a global registry of bucket instances.

The default in-memory inbox and task stores remain valid for one process, but
they cannot provide crash recovery. A durable configuration needs persistent
task, checkpoint and inbox stores, plus a stable way to rebind their logical
identifiers when the runtime is reconstructed. No default filesystem write or
automatic worker replay is proposed here.

## Delivery order and review gates

1. Add route metadata and a resolver/publisher abstraction without removing
   existing maps. Validate direct and bucket-captured agent tasks, separate
   parent/child destinations, concurrent runs and resume to a new run ID.
2. Change `task_message` to resolve the route from the task record, then keep
   only active worker views. Verify it works after reconstructing the library
   with SQLite stores and the same logical store bindings. Preserve a clear
   error when a running task cannot be found after a process failure; status
   alone does not prove a worker still exists.
3. Remove the historical inbox map after before/after same-process profiling.
   Treat `_task_checkpoint_stores` separately: it also retains references, but
   replacing it requires an equally explicit checkpoint-store binding.

Tests must cover initial dispatch, `task_message` during a running child,
completed/paused/failed resume, repeated resume, bucket capture, nested agents,
parallel task IDs, notification dedupe and `notification.drain`, SQLite restart,
missing/wrong store bindings and concurrent publish/resume. In particular,
assert that a message after resume reaches the **new** run, and that a child
message never appears in the parent inbox merely because its tool was captured
by a bucket. Benchmark allocations with one reused library and both drained
and undrained parent notifications; report task-store growth separately.

This design does not authorize expiring task IDs, deleting checkpoints or
silently dropping late messages. A retention/garbage-collection policy remains
a separate product decision.
