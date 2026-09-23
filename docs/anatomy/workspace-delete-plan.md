# Workspace deletion tool

Expose the existing text-deletion editor through `DeleteTool`, reusing
`WorkspaceChangeTool` so approval previews, checkpoint recovery, exact-content
comparison and live permissions follow the write/edit path without new guards.

Implementation order: add the tool and builtin export; cover deletion on both
backends and approval/restart; document usage in the runtime guide. Files:
`tools/builtin/workspace.py`, `tools/builtin/__init__.py`,
`tests/test_workspace_delete.py`, `tests/test_agent_workspace_edits.py`, and
`docs/learn/nn/agent/runtime.md`.

Initial scope is one UTF-8 file, no recursive directory deletion. Binary deletion
would require a separate review representation and remains a design decision.
Risks: changed contents between review and application, denied deletion grants,
and accidentally widening deletion to directories. Tests must reject each case
and verify durable approval previews. No host files are deleted by validation;
local-backend tests use temporary workspaces.
