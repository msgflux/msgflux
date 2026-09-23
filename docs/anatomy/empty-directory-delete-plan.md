# Empty-directory deletion plan

This change adds deletion of a single empty directory to the workspace change
review flow. Recursive deletion is intentionally out of scope for this increment.

## Contract

`DeleteTool` continues to delete UTF-8 files and gains an explicit
`target_kind=empty_directory` preparation path. `ApplyPatchTool` remains
file-only. A prepared directory change contains an opaque, serialized
directory identity token in addition to the workspace identity and canonical
path; applying it rejects a directory that was replaced after approval.

Preparation requires live `list` and `delete` permission on the directory,
rejects the workspace root, symlinks, non-directories, and non-empty
directories. Application rechecks identity and emptiness immediately before a
checked `rmdir`. The in-memory backend performs this check and removal under
one lock. The local backend coordinates its own operations and uses descriptor-
relative, no-follow POSIX calls, but explicitly does not claim atomicity
against unrelated external writers.

## Implementation order

1. Add the checked empty-directory operation to the workspace abstraction and
   both backends.
2. Extend `PreparedFileChange` and `WorkspaceEditor` with a directory-target
   preparation/apply path while preserving file defaults and serialization.
3. Route only `DeleteTool` to the new path and add backend/serialization tests.
4. Exercise SQLite approval/checkpoint restart and stale directory rejection in
   `tests/test_agent_workspace_edits.py`; update public runtime and builtin docs.

Affected implementation files: `runtime/workspace.py`,
`runtime/workspace_local.py`, `runtime/workspace_changes.py`, and
`tools/builtin/workspace.py`. Focused backend tests live in
`tests/test_workspace_empty_directory.py`. Run the offline durability gate,
full offline tests, Ruff, package build and strict MkDocs validation.

## Risks and v1 boundaries

Directory identity is backend-defined and must be validated again at apply
time. A local external writer can still create an entry between inspection and
`rmdir`; the operation then fails rather than deleting recursively. There is
no rollback promise for local filesystem errors. Recursive deletion,
symlink-following, binary-tree deletion, and multi-directory transactions are
not part of this increment.
