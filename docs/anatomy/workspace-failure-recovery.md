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

## Current reconnection limitation

Workspace handles, permissions and bindings are not serialized in checkpoints.
A restarted application must construct its live dependencies again. Restoring
conversation history alone does not reconnect a backend or authorize a command.

The current local/Docker backend resource registry belongs to a backend instance;
it is not persisted across processes. The in-memory backend is also process-local.
A new process does not automatically recover the old workspace generation, even
when given the same local directory. Pending reviewed changes require matching
resource identity and will be blocked when that identity cannot be reproduced.
There is no implicit adoption of a new resource under an old approval.

Persistent resource identity/reconnection and host-level worker/container
reconciliation are subsequent improvements, separate from file organization.
