# Runtime v1 contracts

Status: incremental implementation. This RFC separates implemented foundations
from follow-up work. Foreground Agent approvals are implemented; host isolation
and exactly-once external effects are not promised.

## Boundaries

The runtime serves heterogeneous AI systems. Agent, Transcriber, classifiers,
retrievers and the future inline DSL consume the same execution identity and
authority; conversational history is not the generic execution state.

| Contract | Owner | Persistence |
| --- | --- | --- |
| Module definition | Application composition | Configuration/state_dict; no active authority |
| Execution identity | ExecutionScope | Namespace, thread/run and lineage |
| Live authority | Application entry point | Never restored from a checkpoint |
| Execution progress | CheckpointStore | Versioned state and committed events |
| Conversation branches | Conversation runtime | Domain-specific history and head |
| Presentation | Event observers and output renderers | Raw deltas may be ephemeral |

Python implementations, extensions and dispatchers are trusted application code.
These contracts guard runtime-mediated invocations, not arbitrary Python code
running in the same process. Tool arguments, model messages and checkpoint data
are not sources of authority.

## Implementation sequence

1. `docs/runtime-v1-contracts`: this RFC and review boundaries.
2. `feat/runtime-permissions`: immutable PermissionSet and ExecutionScope live
   authority, restrictive nested inheritance, identity-only serialization.
   Files: runtime/permissions.py, runtime/context.py, runtime exports and tests.
3. `feat/tool-permission-boundary`: required_permissions compiled into immutable
   ToolDefinition metadata; mandatory checks before runtime argument injection,
   before dispatch and before executor entry. Denials are terminal blocked
   outcomes with audit-safe events. Files: tools/config.py, tool/definitions.py,
   tool/execution_runtime.py, runtime/background.py, runtime/events.py, tests,
   and existing tools/runtime learning pages.

The tool milestone also preserves contextvars when LocalTool sends synchronous
Python implementations to an async executor thread. Without that propagation,
nested tool calls would lose their caller's live authority and execution identity.
Standalone local/MCP adapters check their configuration as defense in depth;
raw Python callables and arbitrary custom executor code remain trusted host code.

Risks: permissions leaking across concurrent runs, explicit child scopes
escalating authority, metadata silently dropped by compiler/adapters, captured
or background tools bypassing checks, and checkpoint restore reviving authority.
Tests must cover sync/async, canonical/direct calls, sibling independence,
background denial before spawn, context nesting/concurrency and serialization.
Run focused tests, repository Ruff checks, offline pytest and MkDocs.
Pre-existing user files, including the design-only tool permissions plan, are
excluded from commits. These branches remain local until explicitly submitted.

## Authority contract

Permissions are exact capability names, not resource patterns or wildcard
expressions. Empty authority denies tools with declared requirements. Existing
tools without requirements retain their behavior; they are not automatically
sandboxed. The trusted host grants capabilities at a root execution boundary.
Nested contexts inherit the parent's authority or its intersection with explicit
child grants. A child cannot switch principal or widen authority.

ExecutionScope.to_dict() serializes identity only. Resuming work requires live
grants from the application again. Neither run metadata nor an approval audit
record can grant permission by being deserialized. Missing permissions fail
closed independently of optional policies or hooks. Policies may further deny.

Permission checks are not sandbox enforcement. A sandbox driver must eventually
declare supported isolation mechanisms, enforce resource limits and reject
unsupported restrictions. In-process tools have ambient host access regardless
of the capability labels declared here. Provider-hosted tools require separate
gating at request compilation; local tool checks cannot intercept remote effects.

## Approval contract: follow-up

### Implemented prerequisite: durable decision storage

Branch: `feat/durable-approval-store`. This implements the storage prerequisite before
exposing an Agent approval policy. The API is experimental and host-operated;
it does not suspend tools, grant capabilities, or attach to watch snapshots yet.

Implementation order and affected files:

1. `runtime/approvals/`: immutable call bindings and records, shared transition
   rules and typed conflicts. Bind namespace/thread/run/principal, call and tool
   identity, tool revision, policy version, public argument and resource digests,
   required capabilities, and an absolute expiration time. Never persist raw
   arguments or injected runtime values in this journal.
2. `runtime/approvals/providers/`: memory and SQLite implementations sharing
   those rules; atomic record + append-only audit transitions and one-use
   consumption. SQLite must arbitrate independent connections/processes.
3. Runtime exports and `data/stores/{types,store,__init__}.py`: use the existing
   typed factory/registry pattern, without new dependencies.
4. `tests/test_approval_store.py`: shared provider conformance, strict binding,
   expiry, decision conflicts, live-authority revalidation, concurrency,
   restart and rollback. Validate async wrappers against the same transitions.
5. Existing `docs/learn/nn/agent/runtime.md`: executable host-side example and
   explicit security/lifecycle limitations; no navigation edits are needed.

Risks: approving a changed invocation, replaying a consumed decision, reviving
expired authority, returning a mutable record, and committing state without its
audit event. Clock and database integrity belong to the trusted host. Digests
bind data but are not encryption. Consumption commits before external effects;
a crash after consumption must not automatically retry an external action.
The Agent integration below connects pending calls, checkpoint pause, decision
routing, deadline checks on resume, and watcher snapshots to this journal.

Approval requests must bind principal, run, tool identity, canonical public
arguments, resource constraints, expiry and an application policy version.
Decisions are auditable, one-use decisions are consumed atomically, and resume
revalidates authority and sandbox constraints. A changed invocation requires a
new decision. Model text cannot approve a request. No require_approval result
will be advertised until durable pause, decision, timeout and resume are wired
end-to-end, including watcher snapshots.

## State and event contract: follow-up

Checkpoint state is authoritative only after commit. Its schema version is
independent from provider wire schemas. Domain history, branch heads and artifact
references remain distinct from generic progress and from live authority.
Artifacts referenced durably need an authorized persistent resolver and explicit
retention; the current in-memory registry is not that resolver.

Durable events need stable IDs, a sequence scoped to a documented stream, and a
cursor associated with the committed snapshot. Reconnection obtains a snapshot
and subsequent events without a race. Ephemeral deltas must not pretend to be
replayable. Slow-consumer policies must bound buffering and report gaps instead
of silently losing durable transitions. Observer detachment and execution
cancellation are different operations.

The current live event hub and revisioned checkpoints do not yet constitute a
durable event replay service. Storage adapters should share conformance tests
before that API is stabilized. The inline DSL should use these same boundaries
instead of implementing another persistence or event engine.

## Agent approval integration plan

Branch `feat/agent-approval-resume` connects host-configured approval rules to
canonical tool-call batches. Files: runtime/approvals Agent policy and execution
guard; nn/modules/agent approval mixin, core, model runtime and lifecycle;
nn/modules/tool/execution_runtime.py; runtime/events.py and event_hub.py;
runtime exports; tests/test_agent_approvals.py; existing runtime learning page.
The shared Module event boundary also classifies cooperative pauses as
`run.paused`, rather than `run.error`; EventHub settles the paused projection.

Order: (1) batch preflight and immutable call binding; (2) checkpoint pending
intents before suspension, replay pending calls before a new model request;
(3) final-plan validation and consume at foreground executor entry; (4) pending
watcher snapshot and approval events; (5) sync/async and SQLite recovery tests.

The entire batch waits before any tool executes. Denied/expired requests become
blocked observations; changed bindings or consumed approvals without committed
results remain paused for host reconciliation. Approval removal on a pending
run fails closed. Capabilities remain mandatory. This initial integration
supports canonical local foreground tool calls, not flow-control DSL execution,
provider-hosted effects or detached/background approval dispatch. This Agent
policy uses an empty resource binding; argument-aware resource policies and
OS sandbox enforcement remain separate follow-up work.

Before dispatch, an atomic checkpoint changes the batch to `executing`. A resume
observing that marker cannot repeat any sibling, even one without an approval.
Only a checkpoint containing the batch results removes the pending marker.
This deliberately chooses manual reconciliation over automatic retry when the
process dies after claiming the batch. No background timeout timer is installed.
Competing resumes exit without writing a paused checkpoint over the claimed
batch, so a still-active worker can commit its results normally.

Risks/tests: duplicate model calls or history on resume; partial batch effects;
checkpoint/journal failures; altered public arguments after hooks; stale policy
versions or grants; competing resumes; expiration; denial; process restart;
live stream events and pending snapshots without exposing arguments. Keep
exactly-once external execution explicitly out of scope. Run offline pytest,
Ruff and MkDocs; preserve unrelated user files and do not publish branches.

## Release gates

## Recovery and durable observation implementation plan

1. `feat/agent-approval-reconciliation`: host inspection and revision-checked
   reconciliation of an uncertain whole batch. Record confirmed text results or
   abandon the batch, never retry tools. Require explicit worker quiescence,
   reviewer identity, reason and an idempotency key. Keep receipts and an audit
   event in the same checkpoint transaction. Files: agent approvals mixin,
   runtime approvals reconciliation helper, runtime learning page and tests.
2. Durable observation: add a run-scoped cursor and atomic snapshot/read API to
   checkpoint adapters, followed by a polling async observer. Persisted checkpoint
   transitions are distinct from ephemeral model deltas and the live EventHub.
   Files: store contracts, memory/SQLite adapters, Agent lifecycle, docs and
   shared adapter tests. Missing/deleted streams and invalid cursors fail loudly.

Risks: a live worker continuing external effects after reconciliation (the host
must stop it; CAS only fences checkpoint writes), repeated/conflicting decisions,
partial batch results, concurrent checkpoints, event gaps on reconnect and
unbounded observer buffers. Test sync/async recovery, SQLite reopen, stale
revisions, duplicate decisions, rollback, snapshot/cursor races and cancellation.
Run focused tests, offline suite, Ruff and MkDocs. No external publishing.

### Release checklist

- Explicit stable/experimental/internal API classification and migration policy.
- Mixed-model reference application exercising shared runtime contracts.
- Permission conformance across supported invocation paths.
- Crash/reconnect and process-boundary tests for advertised durability guarantees.
- No exactly-once promise for external tool effects without reconciliation.
