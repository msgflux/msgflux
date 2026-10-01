# Workspace package organization

## Scope

Group workspace filesystem, binding, execution driver and executors under
src/msgflux/runtime/workspace/. Preserve public exports from msgflux and
msgflux.runtime; update internal module imports and module-string references.
No compatibility forwarding modules are required before v1.

## Files and order

1. workspace.py -> workspace/filesystem.py; workspace_api.py -> workspace/api.py.
2. workspace_backend/contracts/changes/local.py -> workspace/backend/contracts/
   changes/local.py; environment.py -> workspace/environment.py.
3. local_executor.py, docker_executor.py, process_capture.py -> workspace/.
4. Rewrite imports in runtime, tools, Agent, scripts and tests together. Review
   dynamic imports and monkeypatch targets. Keep permissions, isolation, abort
   and generic tool-result shell capture at runtime level.
5. Validate imports, focused workspace/lifecycle/permission tests, full offline
   suite and real local/Docker integrations; Ruff and strict MkDocs.

## Risks

Avoid circular imports between environment, bindings, filesystem and facade.
Workspace handles remain live, not checkpoint-serialized. Builtin filesystem
identity defaults include their Python class module, so relocation changes the
backend identity label; stale proposals/approvals must fail validation rather
than silently reuse a decision for a changed identity. Resource-id syntax and
root public API are unchanged. Keep the existing failure/reconciliation rules.

## Validation

- Public root and internal package import smoke check passed.
- Workspace API/lifecycle/permission tests: 32 passed.
- Full offline suite after relocation: 3,663 passed, 33 skipped.
- Real Docker executor/public workspace suite: 13 passed.
- Live OpenAI Agent flows passed after relocation for both local and Docker,
  using actual read/patch/read/Bash calls and sentinel verification.
- Ruff lint/format and diff checks passed.
- Current crash/recovery behavior is documented separately in
  workspace-failure-recovery.md; the move adds no automatic reaper/reconnect.
- Strict MkDocs build passed after adding the recovery documentation.
