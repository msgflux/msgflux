# AgentWorkspace: simplify ownership and authority

Reference: developer plan.md (left untouched as a local reference).

## Implementation order

1. Baseline committed as c729c680. Scope holds only workspace; migrate scopes,
   lifecycle and runtime consumers. ExecutionEnvironment stays an internal driver.
2. Workspace owns a public editor boundary using relative paths and exact prepared
   proposals. Move prepare/apply there; mutations use explicit editor calls.
3. Resolve live identity, binding and cancellation through one authority helper;
   backends keep resource checks and executor keeps process checks. Local broad
   grants are configured on LocalWorkspace rather than a duplicate subclass.
4. Type public I/O contracts, rename supports_execution and migrate docs/examples.
5. Integration validation: actual Agent/ToolLibrary foreground, background,
   approvals/resume, cwd and permissions; bounded live OpenAI run in a temporary
   project. No credential content or model-generated commands on repository files.

## Files

runtime/context.py, environment.py, workspace.py, workspace_local.py,
workspace_api.py, workspace_changes.py, approvals/agent.py; agent/lifecycle.py;
nn/extensions/workspace.py; builtin workspace tools, patch tools and shared
workspace_changes.py; relevant tests and docs/learn runtime/tool pages.

## Risks and gates

No compatibility aliases for removed scope environment or facade prepare methods.
Preserve nested authority intersection, abort, live binding identity, bounded
reads/process output, approved-proposal identity and checkpoint restart behavior.
Keep current synchronous/asynchronous API and read-only command denial.
Ruff, focused integration tests, offline durability gate, full offline suite,
MkDocs, and bounded live provider integration. Commit related implementation after
review and validation; do not stage developer plan.md or credentials.

## Completed validation

- Full offline suite: 3,642 passed, 32 skipped.
- Runtime durability gate: 148 passed.
- Focused Agent runtime playground: 53 passed.
- Ruff check and format check passed; MkDocs build passed.
- Live integration exposed a separate native-tool history defect: Agent retained
  native outputs but projected their calls to function_call. Track that fix in a
  separate commit and repeat the read/patch/read/Bash flow after correcting it.
