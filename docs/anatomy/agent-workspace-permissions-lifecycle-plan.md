# AgentWorkspace: permissions and explicit lifecycle

## Accepted direction

AgentWorkspace owns its filesystem, executor, cwd, live binding and immutable
workspace permissions. ExecutionScope can restrict that authority but cannot
expand it. Agent approvals remain a policy of the Agent and never grant access.
Backend bindings must be usable without a mandatory context manager.

## Implementation order and ownership

1. Define the permission ceiling and resolve effective permissions centrally.
   Keep unrelated application capabilities in ExecutionScope; the workspace
   ceiling applies to filesystem/process capabilities and workspace resources.
   Broad filesystem capabilities may authorize exact resources of this same
   workspace when a scope narrows access. None and an empty permission set must
   remain distinct at invocation boundaries.
2. Add explicit asynchronous opening and idempotent closing to AgentWorkspace.
   Opening owns the acquired binding; wrapping an existing environment borrows it.
   Cwd views share resource lifetime and cannot gain access by switching handles.
   Agent always borrows the dependency. Partial opening failures release resources.
3. Migrate advanced callers from scope grants to workspace grants. Update examples
   to configure workspace once and pass it to Agent without execution_context.
4. Validate real builtin tools, scope narrowing/escalation, approvals/resume,
   background calls, closure and error cleanup. Repeat the bounded live OpenAI
   read/patch/read/Bash integration in a temporary directory.

## Affected files

src/msgflux/runtime/workspace_api.py, context.py, workspace.py, workspace_local.py;
src/msgflux/nn/modules/agent/lifecycle.py; relevant advanced backend call sites,
scripts/validate_agent_runtime.py, workspace/runtime tests and Docker integration
examples; docs/learn/nn/agent/runtime.md. Reuse existing PermissionSet,
WorkspaceBinding, environment and backend implementations.

## Risks and validation

Do not let a generous scope or a sibling cwd handle bypass workspace authority.
Preserve exact resource identity, cancellation, nested scope narrowing and live
binding checks. Approval decisions and checkpoints restore neither grants nor
connections. Background tools must finish before closing their shared workspace.
No expansion into new sandbox backends or credential/login work.

Run Ruff, focused Agent/workspace integration, offline durability gate, full
offline suite and MkDocs. Docker tests require an available local daemon/image;
report whether actually exercised. Keep changes reviewable in separate commits
for permission ownership and explicit lifecycle if possible. Leave user plan.md
and credentials untouched.

## Implemented API

- `AgentWorkspace.permissions` is the immutable workspace authority ceiling.
  `local(..., permissions=...)` configures a local workspace; advanced constructors
  and `open()` deny workspace operations when permissions are omitted.
- `workspace.permission(path, action)` constructs an exact resource descriptor
  with the same cwd mapping. Describing a grant does not grant access.
- `await AgentWorkspace.open(backend, id, permissions=...)` acquires and owns its
  binding; `await workspace.aclose()` releases it. Wrappers and cwd views borrow
  the binding. Closing the owner invalidates all views; Agent never closes it.
- `process.workspace` explicitly authorizes the whole workspace mount and resolves
  to the live root resource for Docker executor validation. File tool grants are
  independent. Scope restrictions and approvals cannot expand workspace authority.
- Application capabilities/resources outside the workspace remain scope-owned.

## Validation results

- Full offline suite: 3,663 passed, 33 skipped.
- Offline runtime durability gate: 148 passed.
- Public API, lifecycle and permission-intersection tests: 32 passed.
- Local process tests: 8 passed; runtime Agent playground: 53 passed.
- Real Docker combined suite: 13 passed, including open/file/command/close
  integration through the public workspace API without a context manager.
- Live OpenAI Agent local: passed; live OpenAI Agent Docker: passed. Both used
  gpt-6-luna, medium reasoning, Responses streaming and actual read/patch/read/Bash
  calls on a temporary directory, verifying file effects and final sentinels.
- Ruff lint/format, diff check, and strict MkDocs build passed.
- Fixed a test readiness race: PID-file creation could be observed before its
  content was written. The test fixture now publishes the PID atomically.
- The old opt-in offload provider suite remains skipped without its own opt-in.
  No claim is made that every live provider integration was executed.
