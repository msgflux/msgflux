# Workspace navigation tools

## Files and order

1. Add `workspace_query.py` with shared bounded traversal plus `LsTool`,
   `GlobTool`, and `GrepTool` using the authorized workspace APIs.
2. Add `tests/test_workspace_query.py` for both workspace backends, schemas,
   permissions, ignore rules, async calls, and resource budgets.

The parent change will wire exports, optional dependencies, and user-facing
documentation.

## Risks and validation

Traversal must never follow entries classified as `other`, must stay below
depth/node/result/file-byte budgets, and must fail closed on denied or unstable
I/O. Glob matching is virtual-path matching (`*` does not cross `/`); grep
uses a bounded regex engine and reads only prefixes. `.gitignore` inheritance
starts at the selected search root, so no unauthorized ancestor is inspected.
