# Workspace enumeration and bounded reads

Plan for the workspace navigation contract:

1. Add a serializable `WorkspaceEntry` descriptor and bounded `scandir` and
   `read_prefix` methods to the abstract filesystem API.
2. Implement both methods in the in-memory backend and in the POSIX backend
   using descriptor-relative, no-follow operations with live authorization
   rechecks under the local lock.
3. Add contract tests for grants, binding lifecycle, traversal, limits, large
   files, and unsafe local filesystem entries.

Risks are path substitution, symlink or mount traversal, unbounded directory
or file reads, and stale authorization after waiting for a backend lock. The
tests cover both supported backends and verify that limits fail closed rather
than silently truncating results.
