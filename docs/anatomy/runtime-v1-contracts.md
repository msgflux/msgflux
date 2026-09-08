# Runtime v1 contracts

Status: incremental implementation. This RFC separates implemented foundations
from follow-up work; it does not claim host isolation or durable approval support.

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

## Release gates

- Explicit stable/experimental/internal API classification and migration policy.
- Mixed-model reference application exercising shared runtime contracts.
- Permission conformance across supported invocation paths.
- Crash/reconnect and process-boundary tests for advertised durability guarantees.
- No exactly-once promise for external tool effects without reconciliation.
