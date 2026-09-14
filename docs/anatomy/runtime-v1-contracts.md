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

### Workspace identity and write guarantees

The first workspace-contract increment uses immutable `msgspec.Struct`
descriptors (`runtime/workspace_contracts.py`). `WorkspaceIdentity` binds a
backend resource, generation and configuration revision independently of the
logical workspace name. The host verifies these values when opening/reconnecting;
they carry neither credentials nor grants. Default filesystem identities have
instance-local generations. Persistent backends must supply verified identities
to retain approval compatibility across reconnection, never across replacement.

Both `PreparedFileChange` and Agent approval resource bindings include this
identity. Legacy proposals remain decodable but execution without a matching
identity fails closed. Their old approvals must not be silently migrated.

The strict compare/exchange operation remains unchanged in its guarantee.
`checked_replace` exposes a separately selected cooperative contract for future
local backends. Capabilities distinguish replacement from comparison and are
declarations by trusted adapters, not proof of enforcement. WorkspaceEditor
defaults to atomic comparison and binds the chosen guarantee into the proposal
and approval. Permissions and cancellation checks still run at the filesystem
boundary; adapter implementations recheck them after acquiring their locks.

This increment does not introduce a vendor factory/session lifecycle, OS
sandbox, network mediation, local filesystem or overlay. It also does not make
approval consumption and filesystem changes one transaction. Tests cover
replacement under the same workspace name, identity serialization, legacy
proposal rejection, cooperative opt-in and unchanged atomic behavior.

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

The live event hub remains ephemeral. The run-scoped commit feed implemented
below provides durable checkpoint transitions, not replay of every live event.
Storage adapters share conformance tests before that API is stabilized. The
inline DSL should use these same boundaries instead of implementing another
persistence or event engine.

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
policy initially used an empty resource binding. The resource-security increment
binds static requirements plus workspace identity and isolation mechanisms;
arbitrary argument-aware resource policies and OS sandbox enforcement remain
separate follow-up work.

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

The committed feed intentionally uses a separate `checkpoint_commits` table
(and memory collection), rather than changing the legacy `load_events` shape.
Each atomic commit appends one transition in the state transaction. The checkpoint
envelope gains a stream incarnation; revisions provide its sequence. Snapshot
reads share a lock/SQLite read transaction with cursor validation. Legacy writes
are explicitly outside this contract. Agent `watch_commits` polls bounded pages,
without tying observer lifetime to producer cancellation or a background queue.

## Release gates

### Resource security foundation plan

Branch `feat/runtime-resource-security` is the first security-boundary increment,
not an OS sandbox implementation. Add immutable exact resource/action grants to
PermissionSet (intersection-only delegation, no checkpoint restoration), static
required_resources declarations to tool config/definitions and mandatory checks
at existing permission boundaries. Resource IDs are host-owned opaque identifiers:
no inferred path containment, symlink resolution, DNS matching or wildcards.
This avoids making filesystem/network enforcement claims from string matching.

Define an independent SandboxRequirements/SandboxCapabilities contract that
rejects unsupported isolation mechanisms before a host launches a backend.
Do not expose a pretend sandbox provider or route arbitrary Python through one.
Actual backend enforcement, argument-to-resource resolution and dynamic approval
resource bindings need subsequent reviewed integrations.

Following the workspace discussion, this increment also introduces
runtime/workspace.py (authorized VFS interface and memory backend),
runtime/environment.py (host-owned ExecutionEnvironment and a process backend
contract), and explicit `filesystem`/`environment` runtime-input bindings.
ExecutionScope carries the environment only as a live reference; child scopes
cannot replace it and checkpoint serialization omits it. Permissions remain in
the scope rather than being copied into the environment.

The VFS maps canonical absolute POSIX paths to workspace-qualified resource IDs.
Read/write/list/mkdir/delete require exact operation grants. There are no host
mounts, symlinks, path traversal, shell implementation or durable file storage in
this increment. A process executor must explicitly accept the same workspace,
receive a snapshot of live authority and enforce the requested policy; absent
or incapable executors fail before execution. No subprocess fallback is supplied.
Workspace backend implementations and arbitrary Python tool code remain trusted.
Tests also cover dynamic paths, forged context parameters, workspace separation,
sync/async injection, nested authority, cancellation, backend preflight and
absence of environment/authority from serialized execution identity.

The async filesystem injection test exposed that `tool_config` wraps coroutine
functions in synchronous wrappers, preventing LocalTool from awaiting them.
Preserve coroutine-function identity in the decorator and keep regression
coverage for both sync and async ToolLibrary entry points.

Order/files: runtime/permissions.py and new runtime/isolation.py; tool config,
definitions and mandatory execution boundary; runtime exports; shared security
tests and runtime learning page. Tests cover validation, denied-by-default grants,
restrictive inheritance, serialization, sync/async denial, transformed definitions,
and unsupported isolation requirements. Reuse capability checks, never make an
optional policy the sole authorization boundary. Run offline pytest, Ruff and
MkDocs. Preserve user files and do not publish this increment.

### Shared workspace changes and approval previews

Native patch increment (`feat/openai-apply-patch`): adapt the OpenAI Agents SDK
V4A text parser into `tools/patch.py`, preserve its MIT notice, replace parser
dataclasses with msgspec.Struct, and reject ignored trailing file/envelope data.
Add shared create/transform preparation in WorkspaceEditor and an ApplyPatchTool
subclass using WorkspaceChangeTool, constructor cwd, public-only annotations and
compact outcomes. Add `models/tool_adapters/openai_patch.py` to the explicit codec
registry and OpenAI provider. Reuse request-local routing, completed stream items,
portable history, approval previews and reconciliation without provider branches
in Agent. No remote execution, subprocess patch command, SDK dependency or
multi-file atomicity. Native calls represent one file each; ordinary function
transport remains available through native_tools=False.

Order/tests: pure parser create/update/anchors/EOF/CRLF/conflicts/malformed tails;
workspace create-only/update-existing/delete authorization and CAS; native schema,
renamed tools, mixed catalogs, decoding, failed output, streaming dedup, portable
history/interruption, approval SQLite restart and host reconciliation. Document
the model/native selection and host review policy in the runtime learning page.
Run focused/full offline tests, durability gate, Ruff and MkDocs. Preserve local
user changes; no commit or push of this increment without a new request.

Agent integration increment (`feat/agent-workspace-edit-tools`): add a shared
workspace-change tool contract and builtin WriteTool/EditTool with public-only
annotations, constructor cwd and compact JSON results. Extend AgentApprovals
preparation/binding with serialized prepared changes, preserving argument-free
journals. Carry the approved proposal through execution-local context after the
existing guard consumes the approval; never consume it twice. Expose host-only
preview inspection from the Agent checkpoint; events carry identifiers, not file
contents. Reuse existing watcher approval records to locate previews. Update
tool exports, runtime docs and offline tests. Validate sync/async, SQLite restart,
changed cwd/arguments/files, denial, changed permissions, empty/missing files,
ambiguous edits, concurrent dispatch, schemas, full access and no preview leakage
into model history. Run the durability gate, full pytest, Ruff and MkDocs.
Approval remains host policy (`AgentApprovals`); omitted/None means no prompts,
without removing resource authorization. Native apply_patch is the next branch.

Increment order: (1) `runtime/workspace.py` gains an opt-in atomic compare/exchange
contract, implemented under the InMemoryWorkspace lock; (2)
`runtime/workspace_changes.py` defines immutable msgspec prepared changes, text
preparation and unified diff previews; (3) reuse ApprovalBinding/ApprovalStore for
host-operated approval and single-use consumption; (4) offline tests and examples
in the existing runtime learning guide. No new journal format or implicit grants.
The prepared change is serializable separately from the argument-free journal;
its exact contents are bound by digest. The host persists it with its checkpoint
and authenticates preview readers/reviewers. Do not broadcast file contents.

Risks/tests: stale files, create races, empty versus missing files, deletion,
ambiguous text matches, changed previews, wrong workspace/principal, revoked
permissions, expired/denied/reused approval, unsupported atomic backends,
concurrent writers, sync/async and JSON round trips. Comparison and mutation must
be atomic per file; approval consumption and filesystem mutation are not one
transaction, and uncertain consumption is never automatically retried. Preserve
unrelated user files. Validate focused tests, durability gate, full pytest, Ruff
and MkDocs. Follow-up branches add model-facing write/edit tools and provider-owned
V4A apply_patch transport using this backend, then Agent preview automation; this
increment exposes the shared host API without changing Agent journal behavior.

### Provider-owned tool transport refactor

Public shell schema reduction: keep command and timeout_ms only; output byte
budgets remain internal to ProcessRequest/execution, while Responses
max_output_length stays in transport metadata. Update builtin tests and docs for
the host's shorter read/bash names without rewriting legacy history names. Test
schema size, native decoding/replay, rejection of obsolete model arguments and
unchanged internal output enforcement. Do not migrate pending approval bindings
implicitly when changing the tool implementation revision.

Workspace tool ergonomics increment: explicit public annotations and UI
labels in tools/builtin/workspace.py; offset/limit line reads backed by an
authorized WorkspaceFilesystem.read_lines hook; cwd configured in tool
constructors, never taken from the host process. Update tool/provider tests and
runtime learning docs. Keep one shell process deadline (timeout_ms); background
selection only injects run_in_background. No ranges, new Agent resource container
or host subprocess fallback in this increment. Test backend authorization,
bounded selection, UTF-8/newlines, invalid limits, image paging rejection,
hidden schemas, cwd propagation, background schema and native continuation.
Select function transport when the shell schema includes runtime selectors or
other arguments not representable by the native shell protocol; never silently
drop run_in_background. Injected dependencies stay outside public annotations;
forged collisions are rejected by ToolExecutionPlan before dispatch.

Follow-up: per-invocation approval override in Agent forward/aforward, with an
omitted-value sentinel and execution-local ContextVar binding. Update approval
dispatch/replay, decisions and watcher loading without mutating the constructor
default or persisting live policy/store objects. Reserve the argument from task
inputs. Validate sync/async, streaming, pause/resume, explicit None, concurrency,
watch snapshots, invalid values and default restoration; document TUI usage in
the runtime learning guide. Existing pending bindings must not be bypassed.

Order/files: canonical `tools/shell.py` msgspec results; provider-independent
BashTool arguments/results and shell kind; model-owned adapters in
`models/tool_transport.py` and `models/tool_adapters/openai_shell.py`; request-local
route resolution in OpenAICompatibleChatCompletion; generic transport metadata in
ToolCallAggregator, approval checkpoints and reconciliation; ChatMessages native
projection; tests and runtime learning documentation. Bindings are selected by
logical capability rather than tool name. Model configuration owns native versus
function representation; tools contain no provider switch or wire output fields.
Versioned metadata must preserve routing and continuation but never carry live
execution authority. Reject unbound native calls and unknown codec versions.
Tests cover canonical results, renamed tools, mixed/concurrent catalogs,
sync/async/streaming, disabled model-native mode, approvals, reconciliation,
interruption and serialization. Run full offline pytest, Ruff and MkDocs.
No automatic provider fallback, sandbox backend or command policy implementation
is introduced. Existing uncommitted work and unrelated user files are preserved.

Implemented boundary: `tool_kind="shell"` selects a provider-owned adapter;
`BashTool` accepts commands and a millisecond deadline and returns canonical
`ShellResult` records. Request-local routes carry logical names separately from
provider declarations, are removed before HTTP dispatch and are represented in
cache identity through the original catalog. Approval snapshots retain a codec
identifier/version plus continuation options, never an executor or permission
grant. The codec registry is explicit application code, not checkpoint-selected
imports. History uses the same codecs for interruption and portable projection;
internal metadata is stripped on provider replay. Unversioned experimental shell
approval metadata is deliberately rejected rather than guessed.

### Tool configuration and native shell increment

Order: (1) reader instance-owned tool_config and BashTool class in
tools/builtin/workspace.py, exports/tests; (2) native-binding propagation through
the existing tool compiler and Responses catalog, shell call aggregation and
history/streaming; (3) approval reconstruction and continuation fidelity;
(4) offline integration tests and runtime learning docs. Use the public local
shell contract (shell_call/shell_call_output), not hosted execution. Reuse the
execution environment and permission boundary. Preserve current uncommitted
work and user files; no commit/push requested. Risks: shared class config,
ambiguous native-tool routing, interpreting incomplete streamed commands,
loss of native identity on resume, timeout/output caps, and bypassing permissions.
Validate focused/full offline pytest, Ruff and MkDocs.

Security extension plan (not a default tool behavior): implement a reusable
before_dispatch policy after canonical argument preparation, applicable to native
and function calls. Its host-owned rule set can block, allow or require approval;
approval requirements must bind the normalized command batch and policy version.
Inspect shell syntax using an explicitly selected parser, not substring denylist
claims. Reject unsupported constructs under restrictive policies. Separate cwd
validation from relative paths inside shell syntax; shell expansions, scripts and
interpreters prevent complete path/effect inference. Filesystem/network/process
enforcement stays in the sandbox. Tests must cover substitutions, pipes, redirects,
interpreters, alternate executable paths, nested/background dispatch and resume
under changed policy. Start with explicit executable/construct allow rules and
host confirmation, and clearly document what analysis cannot establish.
Proposed follow-up files: `runtime/shell_policy.py` for immutable host rules,
`nn/modules/tool/extensions.py` for policy registration, approval binding code
for the policy revision, `tests/test_shell_policy.py` for parser/dispatch cases,
and the runtime learning page for examples and limitations. Select the parser
and supported shell subset before implementation; do not add a dependency or
promise complete shell analysis in this increment.

### Inbox conversation content increment

Follow-up: unify the builtin reader in `tools/builtin/workspace.py` as
`ReadFileTool`, removing the duplicate coroutine/export. Add host-only
`supports_vision=False`; concatenate visual delivery guidance with existing
instance/class guidance. Read image bytes through the authorized VFS and publish
through the tool notification handle, never host paths. Reuse Image encoding and
MIME helpers, dispatch supported image extensions explicitly, and reject disabled
vision or missing inbox. Update builtin exports, runtime docs and reader tests;
cover sync/async paths, guidance preservation, hidden inputs, denied reads,
disabled vision, missing inbox and Agent trajectory/checkpoint ordering.

Extend `runtime/agent_inbox/inbox.py` with described conversation messages and
text/image content, preserving legacy user-message rendering and system signals.
Reuse canonical ChatBlock dictionaries; validate before publishing, never resolve
host paths or download images. Extend `agent_inbox/handles.py` so tools can publish
conversation content with the handle's provenance. Test memory/SQLite roundtrips,
claim/release/receipt recovery, multimodal delivery after tool outputs, validation,
escaping, and verbose rendering. Update the runtime learning page. Risks include
role/provenance confusion, losing image blocks during serialization, reordering
messages and duplicate delivery. No automatic vision delegation or provider tool
output formats are introduced. Existing uncommitted workspace-tool work is kept
separate; no commit or publication in this increment.

The requested guidance follow-up uses `ReadFileTool.tool_config["usage_guidance"]`
in the pending workspace-tool increment, reusing the tool configuration API.
Avoid global guidance mutation and test independent instances and both execution
paths. Mark injected resources Hidden, retaining their runtime bindings. The
unification follow-up above replaces the function entry point and adds image
publication; ranged reading remains separate work.

### Workspace builtin tools increment

Branch `feat/workspace-builtin-tools` depends on `feat/runtime-resource-security`.
Implementation order: add `tools/builtin/workspace.py`, export `ReadFileTool` and
`BashTool` from `tools/builtin/__init__.py`, cover invocation and denial in
`tests/test_workspace_builtin_tools.py`, then document both on the existing
runtime learning page. Reuse runtime-input injection, VFS authorization and
ExecutionEnvironment process preflight; no separate permission or shell runner.
No new data contracts are needed; future contracts should prefer msgspec.Struct.

Risks/tests: forged runtime inputs, missing authority or executors, traversal,
invalid UTF-8 and oversized files, process errors, and duplicate shell effects
from retries. Test sync/async ToolLibrary entry points with an in-memory workspace
and a fake executor only. Disable automatic retries. Keep process limits out of
model-controlled arguments. The read limit bounds returned content, not backend
read allocation. Real isolation, resource-policy enforcement and bounded process
capture remain backend responsibilities; this increment does not ship a sandbox.
Run focused tests, full offline pytest, Ruff and MkDocs.

### Durability conformance gate implementation

Branch `test/runtime-durability-conformance` extends the existing offline suite.
Order/files: (1) shared memory/SQLite fixtures in tests/conftest.py, consumed by
the existing approval and observation suites; (2) tests/test_durability_conformance.py
for legacy upgrade, fork cursor isolation and cancellation during a storage
transaction; (3) tests/test_durability_processes.py for spawned independent
workers, abrupt process loss during a SQLite commit and the Agent approval /
external-effect / crash / reconciliation / observer reconnect scenario;
(4) contributor gate commands and runtime recovery guidance. Fix only production
defects demonstrated by these tests, with regression coverage.

Tests use temporary databases and deterministic fake models, never live APIs.
Process synchronization uses bounded barriers/events and every worker is joined
or terminated on failure. Risks: flaky timing, orphan workers, SQLite locks,
mistaking cancelled awaits for rolled-back writes, and implicitly claiming
exactly-once external effects. Run focused suites repeatedly, full offline pytest,
Ruff and MkDocs. Existing user changes remain untouched. No publication in this
step. New adapters must run the shared checkpoint/journal gates appropriate to
the capabilities they advertise; persistent adapters additionally need crash and
independent-worker tests equivalent to the SQLite cases.

Fault injection exposed open SQLite transactions after `CancelledError`: the
explicit transaction guards caught only `Exception`. Extend rollback to
`BaseException` in commit/fork/save-with-event, immediately re-raising the original
exception. Test transaction closure, unchanged snapshots/events, absent fork
targets and subsequent connection reuse. This is not cancellation-driven rollback
of an already committed worker-thread operation.

- Explicit stable/experimental/internal API classification and migration policy.
- Mixed-model reference application exercising shared runtime contracts.
- Permission conformance across supported invocation paths.
- Crash/reconnect and process-boundary tests for advertised durability guarantees.
- No exactly-once promise for external tool effects without reconciliation.
