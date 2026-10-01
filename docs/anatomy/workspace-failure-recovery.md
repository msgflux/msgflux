# Workspace failure and recovery

This describes the current implementation, not a promise of automatic recovery.
Workspace lifecycle and permissions are implemented under runtime/workspace/.
Agent approvals and durable execution remain in their existing runtime/Agent
components. Permission grants and live bindings are never restored from a model
message or checkpoint.

## Ordinary failures

| Failure | Current behavior |
| --- | --- |
| Workspace construction fails after backend opening | `AgentWorkspace.open()` releases the acquired binding, preserves the original error and adds a note if cleanup also fails. A backend must clean up its own partial opening failures. |
| A tool fails, times out or is cancelled | The executor attempts to terminate and reap its owned process group. Docker also attempts to force-remove its named container. File effects already performed remain. |
| Owner closes the workspace successfully | Subsequent mediated operations on the owner and borrowed cwd views are rejected. Closing a borrowed wrapper/view does not close its owner's binding. |
| Binding release fails | Binding enters `release_failed`; further access is blocked. Repeated close does not silently retry an uncertain release. Host reconciliation is required. |

The Agent borrows the workspace. The application owns its lifecycle and should
close it during ordinary shutdown, after draining/cancelling active operations.
`aclose()` is not a command cancellation method and does not terminate active
remote work. Internal cleanup remains automatic for each executor operation;
removing a mandatory user context manager does not remove that cleanup.

## Abrupt host/process death

SIGKILL, interpreter crashes and machine failure cannot run Python `finally`
blocks. Local processes may outlive the host agent. Docker containers may also
remain active; the daemon is independent of the agent process. Owned containers
carry the label `msgflux.executor=ephemeral`, but no automatic reaper or durable
process lease is implemented. The host must inspect and reconcile them.

Filesystem writes are real. Completed effects are not rolled back by checkpoint
restoration, permission denial or approval expiration. The same applies to other
external effects initiated by commands.

## Checkpoints and approval uncertainty

Persistent checkpoint and approval stores retain their committed records; the
in-memory implementations disappear with the process. Storage durability under
power loss depends on the adapter/platform and is not established by the existing
process-death tests.

Before a protected tool batch starts executing, its pending calls and decisions
can be resumed only with compatible live workspace identity, grants, tool/policy
versions and persistent checkpoint/journal records. The host records a decision
and explicitly resumes; approval never grants additional permissions.

A crash can happen after an approval is consumed but before a result is recorded.
Consumption and an external command/file effect are not a single transaction.
An executing/consumed batch with an unknown result is therefore not automatically
reexecuted. The host must ensure its old worker is stopped, inspect actual effects
and reconcile the batch. Exactly-once external execution is not guaranteed.

## Workspace reconnection

Workspace handles, permissions and bindings are not serialized in checkpoints.
A restarted application reconstructs its live dependencies and supplies current
permissions. A checkpoint stores a versioned workspace reference, including its
identity and cwd, and validates a matching live workspace before resuming a
nonterminal run. Reading saved history does not reconnect or authorize tools.

Local and Docker backends optionally use `SQLiteWorkspaceRegistry` outside the
model-writable project. Registration retains a resource generation across
processes; reconnect verifies the existing registered root, device/inode
fingerprint and configuration before creating an independent binding. It never
registers a missing resource or silently adopts a replacement. Configuration
identity uses stable backend names, independent of Python package paths.

The local fingerprint detects ordinary root replacement; it cannot establish
protection against every inode-reuse or copied-filesystem scenario. Docker's
persistent configuration also checks a pinned image digest, daemon socket
identity, user, isolation policy and limits. Replacing the socket invalidates
reconnection. Persistent registry state is not authority or credential storage.

Without a registry, local/Docker identities still belong to the backend instance.
The in-memory backend remains process-local. Old approvals cannot authorize a
new resource just because its pathname or workspace ID matches.

## Remaining execution recovery limitations

Workspace reconnection does not attach to a running command. Local subprocess
handles and dispatcher futures remain process-local. Docker still creates and
removes ephemeral containers per command and does not retain durable output.
An abrupt failure can leave external work alive or its outcome unrecorded.

Existing task store leases, committed terminal checkpoints and approval
reconciliation remain the available recovery primitives. Lease expiry does not
prove the old worker stopped. Reconnect files, establish quiescence, inspect
committed state and actual effects, then use explicit recovery/reconciliation.
There is no implicit command replay, automatic reaper or exactly-once guarantee.

The next stages in the workspace reconnection plan compose these primitives into
host recovery and add command receipts/orphan reconciliation. They are separate
from persistent resource reconnection.
