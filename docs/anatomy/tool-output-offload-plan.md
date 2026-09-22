# Pluggable tool output offload

## Order and affected files

1. Change `runtime/resources.py` to accept `initialize(extra_dirs=(...))`, with
   only `threads` and `tool-results` as defaults. Validate all extra directory
   names before creating anything. Allow single safe directory components, not
   arbitrary paths. Preserve `plans_path` as a location helper without creation.
   Update layout tests and the resources learning page.
2. Add a generic `transform_tool_output` lifecycle boundary to the shared sync/
   async ToolLibrary executor after ordinary `after_tool` handlers and before
   `tool.end`. It takes/returns `AfterTool`. Unlike best-effort observational
   hooks, failures clear the result and report a bounded processing error; never
   restore an oversized original. Cancellation keeps existing propagation.
   This boundary contains no size, file, serialization or offload policy.
3. Add `ToolOutputOffloadExtension` using this boundary and the abstract
   `ToolResultStore`. Opt in on a ToolLibrary; default execution is unchanged.
   Preserve small results by identity. Serialize dict/list JSON incrementally,
   including slicing large string values before msgspec encoding. Plain text
   stays UTF-8 text, including subagent text returned as a tool result. Return a
   JSON-compatible descriptor and bounded text preview for large outputs.
4. Export lazily from `nn.extensions`, document the lifecycle boundary and usage,
   and add focused sync/async, events, checkpoint, serialization and failure tests.

## Scope, risks and validation

Initial supported top-level results are str, dict and list. Structured containers
must contain JSON primitives with string object keys. Reject cycles, excessive
depth and unsupported nested values explicitly; never stringify unknown objects.
The subsequent `shell-offload-plan.md` adds typed ShellResult offload and transport
preservation. Other provider-native typed results and binary/media results remain
untouched: replacing their types with a generic descriptor would break contracts. This is
not a universal output-size bound. Tool-local allocations, ordinary hooks,
custom tool-emitted updates and the Agent's main streaming output remain separate.

The encoder probes only up to the inline threshold plus a bounded encoded chunk,
then streams into storage. Preview truncation respects UTF-8 boundaries; a JSON
preview is a text excerpt, not necessarily a valid standalone JSON document.
Async lifecycle handlers use the existing worker-thread hook implementation.
Cancellation can leave an unreferenced completed artifact if storage is already
in flight; no cancellation response may imply rollback of tool side effects.

Resource reads remain explicitly authorized host integration. Do not add a tool
that lets a model browse all shared result IDs. No new prompt, AGENTDIR variable,
implicit root mount, automatic retry or per-tool model argument. No commits.

Tests: optional directories and invalid paths; text/JSON round trips; thresholds,
Unicode/escaping/cycles/depth; unchanged small/unsupported top-level objects;
storage failure without original output leakage; sync/async execution; library
extension removal; event and checkpoint payloads containing the descriptor. Run
focused tests, full offline pytest, Ruff and MkDocs.
