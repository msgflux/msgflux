# AgentWorkspace: simplify ownership and authority

Reference: developer plan.md (left untouched as a local reference).

## Implementation order

1. Commit the previous implementation as c729c680. Make scope hold only workspace;
   migrate scopes,
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

- Final full offline suite: 3,646 passed, 32 skipped.
- Runtime durability gate: 148 passed.
- Focused Agent runtime playground: 53 passed.
- Ruff check and format check passed; MkDocs build passed.
- Live integration exposed a separate native-tool history defect: Agent retained
  native outputs but projected their calls to function_call. Fixed separately in
  b5ad0f1e, covering streamed and nonstreamed history for patch and shell.
  The repeated live test passed
  with OpenAI gpt-6-luna, medium reasoning, Responses streaming: read, apply_patch,
  read, Bash and final sentinel answer. Credentials were read from the authorized
  dotenv path, never copied to this worktree or committed.

## Additional integration fix

The live flow justified a focused change to agent/conversation.py and
tests/nn/test_event_streaming.py. Native call types now survive trajectory
filtering and participate in call-id deduplication, preserving matched native
call/output pairs. The focused event-streaming/native-tool suite passed 107 tests.
The opt-in tests/integration/test_agent_workspace_live.py bounds provider calls
and operates only on a temporary project.
