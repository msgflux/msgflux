# Workspace reconnection and recovery plan

Status: approved follow-up to merged PR #201. Stage 1 is committed as
`3b25780a` on `feat/workspace-reconnection`. Stage 2 is implemented on
`feat/agent-task-recovery`; stage 3 remains planned. The developer authorized
continuing the stages on dependent branches before review. Merge order remains
stage 1, then stage 2, then stage 3. This specification is outside PR #201.

Stage 1 validation includes independent-process crash/restart, concurrent first
registration, approval recovery with changed file/grants/policy/tool revisions,
real Docker reconnection and live OpenAI native-tool approval recovery. Resource
replacement is explicit and does not revoke live bindings: the host must drain
commands and close old bindings first. No task coordinator or command receipt
implementation is included in this first stage.

## Objective

An application that restarts should recover the same workspace, inspect durable
Agent/task state, and continue only when it can establish the execution outcome
and current authority. Start with local files and the existing Docker executor;
new sandbox and remote backends are outside this plan.

The application must not need a context manager. Keep explicit opening,
reconnection and closing, with the application owning the binding and Agents
borrowing it.

## Existing behavior to reuse

- `WorkspaceIdentity` already separates backend, resource ID, generation and
  configuration revision. It is a resource check, never a permission grant.
- `WorkspaceBackend.reconnect()` already exists. Local/Docker implementations
  currently depend on the original backend instance's in-memory registry.
- `SQLiteCheckpointStore` commits conversation/runtime state with revision checks.
- `SQLiteTaskStore` retains task records, results, activity, messages and worker
  leases. `TaskHandle` validates ownership during task state transitions.
- `BackgroundTaskDispatcher` can explicitly recover an expired Agent task from
  its checkpoint or durable initial input. It can reconcile a completed checkpoint
  into a task result without another model call.
- SQLite Agent inboxes retain messages and cursor state across process restarts.
- Approval reconciliation already records idempotent host decisions against an
  expected checkpoint revision and requires confirmation that the old worker
  stopped. Reuse this path for uncertain protected execution.

Keep these stores as their current sources of truth. Do not introduce another
queue, worker lease system, approval journal or complete copy of task state.

## Recovery guarantees

| Operation | Intended guarantee |
| --- | --- |
| Reconnect workspace | Verify and bind the previously registered resource and compatible configuration. Never silently create or replace it. |
| Observe task | Read committed state/results/messages even when the original dispatcher no longer exists. |
| Resume Agent | Reconstruct trusted live dependencies and use existing checkpoint/task recovery after ownership and uncertain effects are resolved. |
| Reconcile command | Inspect the backend resource and committed execution receipts; retain uncertainty if evidence is insufficient. |
| Attach to running command | A separate executor capability. Ordinary workspace/task reconnection does not promise attachment to a surviving command or recovery of its pipes. |

Lease expiry is permission to attempt a conditional task claim, not proof that
an old process stopped. Store ownership checks fence writes to that store; they
do not fence arbitrary filesystem or network effects of a surviving command.

## Proposed initialization order

1. Load the application's trusted configuration and open persistent stores.
2. Inspect the thread/run checkpoint and task records without dispatching tools.
3. Resolve its workspace reference through the host-selected backend/registry.
4. Verify resource generation, root identity, backend configuration and schema.
5. Reconnect a new binding and construct `AgentWorkspace` with freshly supplied
   permissions, cwd and requirements. Revalidate cwd against the live resource.
6. Classify pending approvals, tasks and command receipts. Establish quiescence
   wherever external work may still exist; do not infer it from lease expiry.
7. Claim recovery ownership through existing conditional store operations and
   re-read the state that authorized recovery after claiming.
8. Reconcile committed terminal results or uncertain batches, then resume the
   Agent at a safe checkpoint boundary. Rehydrate inbox/event observation from
   durable state rather than expecting the old in-process futures to exist.

If a dependency is missing or incompatible, expose the reason and leave execution
blocked. Reading the saved history remains possible without executing tools.

## PR 1: Persistent resource identity and explicit reconnection

### Design

Add a small workspace resource registry with a SQLite implementation. The host
supplies it to a backend; persistence remains opt-in for core users. The registry
lives in host-controlled application state, outside the model-writable workspace.
Core code does not invent a default directory or store credentials. The harness
chooses a stable registry path when it integrates this API.

Persist one versioned record per registered backend resource using immutable
`msgspec.Struct` records. Minimum fields:

- Schema version and explicit stable backend kind, independent of Python module
  names and code relocation.
- Existing `WorkspaceIdentity` fields, with random generation created on first
  registration and retained until explicit replacement.
- Host-selected absolute root and its validated device/inode fingerprint for
  the POSIX local backend.
- A deterministic configuration fingerprint containing the settings relevant
  to filesystem/process behavior and approval compatibility. For Docker, include
  the selected daemon identity/endpoint and pinned image identity, mount policy,
  user, network policy and limits. Do not include credential material.
- Record revision for conditional updates and explicit retirement/replacement.

Use a SQLite transaction to register once under concurrent initial opens. A
registry entry is not proof that the current root is unchanged: reopen the root
with existing descriptor-relative, no-symlink checks and compare its fingerprint.
Do not automatically adopt a new root, rotate a generation or overwrite a record
on reconnect. A missing/replaced root or different configuration requires explicit
host registration of a new generation; old approvals remain invalid.

Device/inode checks detect ordinary directory replacement, not every conceivable
inode reuse or copied-filesystem scenario. Document this local-backend limit.
Do not claim protection against a malicious host or power-loss durability from
these checks. A stricter filesystem identity scheme can be evaluated separately
if required; a UUID derived only from a pathname or workspace ID is insufficient.

Add `AgentWorkspace.reconnect(backend, workspace_id, identity, ...)` using the same
construction and failure-cleanup helper as `open()`. It requires live permissions
from the host; identity/checkpoint records never restore grants. Reconnect creates
an independently closable binding rather than reopening a failed binding object.
Keep existing ephemeral operation possible without a registry.

Persist only the workspace reference/identity and safe runtime metadata in a
versioned checkpoint extension. Never serialize the backend, live workspace,
locks, credentials or a permission set as authority. Reuse the identity field
already captured by approval bindings instead of introducing another identity.

### Affected files

- `runtime/workspace/contracts.py`: versioned resource/reference records.
- New `runtime/workspace/registry.py`: minimal registry contract and SQLite
  implementation, following existing SQLite connection/transaction conventions.
- `runtime/workspace/local.py`, `docker_executor.py`, `backend.py`: optional
  registry registration/verification and backend configuration identity.
- `runtime/workspace/api.py`: explicit reconnect and common binding constructor.
- `nn/modules/agent/lifecycle.py` and existing checkpoint extension plumbing:
  record/check workspace references at durable boundaries.
- `runtime/__init__.py`: public exports only where needed.
- `tests/test_workspace_local.py`, `test_workspace_backend.py`,
  `test_workspace_lifecycle.py`; new `test_workspace_reconnection.py`.
- `docs/learn/nn/agent/runtime.md` and
  `docs/anatomy/workspace-failure-recovery.md`.

### Required tests and acceptance

- Spawned process A opens/registers, writes and exits abruptly; independent B
  reconnects to the same identity and content with freshly supplied permissions.
- Concurrent first registration yields one resource generation.
- Different root/configuration, replaced root, missing registry entry and unknown
  schema all fail without implicit registration or model execution.
- Reconnect construction failure releases only its acquired binding.
- A valid pending approval can resume against the exact restored identity;
  changed policy/tool revision, reduced grants or changed file content are handled
  by existing validation rather than silently accepted.
- Resource identity survives package/class renaming through stable backend kinds.
- Existing process-local and in-memory semantics remain explicit and tested.

## PR 2: Coordinated Agent/task recovery using existing stores

Depends on PR 1.

Add one host-facing inspection/recovery entry point that composes existing
reconnection, task lease and approval reconciliation operations. Separate a
read-only inspection report from actions that claim, reconcile or resume work.
Use small versioned `msgspec.Struct` reports; avoid a universal context object.

Persist workspace identity in background dispatch metadata alongside the already
persisted checkpoint/inbox routing IDs. Validate it when recovering tasks. Reuse
`resume_agent_task`, `recover_agent_task` and `reconcile_agent_task`; avoid a second
scheduler. Missing stores/dependencies produce a blocked report, never a new run.

Classification should distinguish: completed result available; active valid
owner; expired owner with safe checkpoint; uncertain execution needing host
reconciliation; incompatible dependency. Recover terminal results first so a
completed model call is not repeated. Recovery of interrupted model generation
may need a new provider request; do not promise recovery of an unfinished stream.

Workspace identity persistence does not make local comparison transactional
across processes. Preserve `cooperative_compare` limits. For recovery with possible
old writers, require verified quiescence and inspect the latest revision before
writes. If coordinated multi-writer editing becomes a requirement, design its
locking separately and do not upgrade advertised guarantees prematurely.

### Affected files

- `runtime/background.py`, `task_leases.py` only if a concrete change is necessary.
- New `runtime/recovery.py` or a similarly small host coordinator module.
- `tasks/protocol.py`, `tasks/providers/sqlite.py`, `tasks/handle.py` only for
  necessary atomic query/claim additions; do not duplicate their existing APIs.
- `runtime/approvals/reconciliation.py` and Agent approval entry points only for
  composing existing validated decisions.
- `tests/test_terminal_task_recovery.py`, `test_durability_processes.py`,
  `test_task_store_conformance.py`; new coordinator recovery tests.
- Public runtime/recovery examples under `docs/learn/nn/agent/`.

### Required tests and acceptance

- Crash before initial checkpoint, mid-Agent turn, after terminal checkpoint and
  before task completion; preserve the existing terminal result/zero-model-call
  regression test.
- Two independent recovering hosts contend: only one wins a claim; losing host
  does not dispatch. Old owners cannot publish results after ownership changes.
- Valid active lease is observed without taking over. Expired lease alone does
  not authorize replay of uncertain commands or consumed approvals.
- Inbox messages/cursors and native call/output pairs survive recovery without
  duplicating completed results or granting authority from persisted metadata.
- Inspection performs no tool calls and can report incompatible dependencies.

## PR 3: Command receipts and orphan reconciliation for existing executors

Depends on PR 2. This stage improves evidence and cleanup; live pipe attachment
and a persistent local worker service are outside its scope.

Persist an execution intent before launch, linked to the existing tool call/run
and task ID when backgrounded. Use checkpoint extensions for foreground calls
and existing task activity/metadata for background work rather than creating a
parallel task store. Define which record is canonical and how duplicate updates
are reconciled; checkpoint and task databases are not a shared transaction.

Record a stable execution ID, workspace identity, owner, backend resource
reference, launch state and terminal receipt. Keep output bounded, reference
artifacts when necessary, and define retention/cleanup. Do not persist inherited
process environment or authentication data. Commands themselves may contain
sensitive input and follow the conversation/task storage policy.

For Docker, derive a unique container name from the precommitted execution ID,
label it with its ownership/reference, then persist and verify the daemon-returned
container ID. After restart inspect the exact container on the configured daemon:
collect a verifiable outcome, keep observing, or explicitly terminate/reconcile.
Name/labels help discovery but are not authority by themselves. The current
`--log-driver=none` and immediate removal prevent durable output recovery; make
bounded log retention and cleanup policy explicit before claiming output recovery.
Capture the terminal receipt before routine removal where possible. Retain
uncertainty across launch/receipt/removal gaps instead of rerunning commands.

For local execution, PID alone is insufficient because it can be reused. Store
available host boot/process-start identity for inspection, but never claim the
new process owns the previous subprocess's pipes or can recover an absent exit
status. When identity/outcome cannot be established, require host reconciliation.
A surviving command requires explicit termination/quiescence before replacement
work can cause conflicting external effects.

### Affected files

- `runtime/workspace/environment.py`: execution identity/receipt plumbing.
- `runtime/workspace/local_executor.py`, `docker_executor.py`,
  `process_capture.py`: lifecycle evidence, cleanup and backend inspection.
- `runtime/background.py`, task activity and checkpoint extension plumbing.
- Existing approval reconciliation APIs for protected uncertain batches.
- New spawned-process executor recovery tests; existing local/Docker integration
  tests and `docs/anatomy/workspace-failure-recovery.md`.

### Required tests and acceptance

- Abrupt exits before launch, after backend creation, during execution, after
  command completion, after receipt persistence and during cleanup.
- Docker controller restart finds only the exact owned container, preserves a
  recorded result and can reconcile cleanup; daemon unavailable means blocked.
- Reused/mismatched local PID or unrelated container is never killed/adopted.
- Unknown outcome remains explicit and is never automatically retried.
- No unbounded logs, automatic image pulls or leftover workers from the tests.

## Verification for every implementation PR

Run Ruff checks, relevant focused tests, and the CONTRIBUTING.md durability gate.
Extend shared store conformance tests if their contracts change. Use spawned
processes and abrupt exits, deterministic synchronization and bounded cleanup;
caught exceptions alone do not test restart behavior. Run real local/Docker
integration for the affected execution path and strict MkDocs for public docs.

Run the complete offline suite before each PR. Live model integration is useful
for recovered history/native tool compatibility after deterministic crash tests,
but paid calls are not needed to prove lease contention or resource identity.
Report skipped/unsupported backend checks explicitly.

## Scope and risks

- No bwrap, SSH, OAuth changes, mandatory context manager or general daemon.
- SQLite/file durability under sudden machine power loss remains a separate
  storage/platform guarantee; process restart tests do not establish it.
- Exactly-once external effects are not promised. An intent commit and a command
  launch cannot be made one transaction across arbitrary operating-system effects.
- Existing stores can disagree after a crash; classify and reconcile using
  committed evidence and revisions rather than choosing whichever looks newer.
- No automatic authority restoration or automatic replacement of resources.
- A persistent process supervisor/attach API is a later proposal only if the
  product requires local commands to continue with recoverable output after
  their controller dies.
