# Durable runtime resources

## Agreed contract

Use a configurable application root with central `tool-results/` and
`threads/<thread-id>/checkpoint.sqlite`. `plans/`, `skills/` and other directories
are opt-in through `initialize(extra_dirs=...)`, not mandatory defaults.
Resource identities are independent of
host paths. Checkpoints retain the association between calls and result references;
do not introduce a second per-thread call index. Shared storage does not grant
agents access. No `$AGENTDIR` or implicit home-directory writes on Agent creation.

## Reviewable delivery order

1. **Storage foundation (this increment).** Add `runtime/resources.py` for the
   local layout and existing SQLite checkpoint factory; `runtime/tool_results.py`
   for `msgspec.Struct` references, an abstract store and a local incremental
   implementation. Export through `runtime/__init__.py`. Use UUID4 as existing
   runtime IDs do (Python 3.11 compatibility, no new UUID dependency); keep SHA256
   separate from identity. UUID prevents collisions, not content deduplication.
   Stage writes, sync files/directories and publish atomically on POSIX. Readers
   use byte ranges/chunks; verification streams the checksum. Per-result size
   limits stop writes; failures must never return a usable partial reference.
2. **Tool offload integration.** Apply an opt-in policy before tool-end emission,
   transport and checkpoint serialization, with compact previews and references.
   Add explicit authorized resource reads for tools/UI, including recovery after
   relocation. Small outputs and structured/native outputs need explicit policies.
   Do not implement this as a hook that silently falls back to the enormous
   original output when storage fails. Add aggregate quota/admission control
   before exposing untrusted producers or automatic retention cleanup.
3. **Durable delegation.** Keep children in the parent's thread and inherited
   checkpoint store, isolated by namespace/run ID, as AgentTool already does.
   Do not create a new thread or SQLite file for each child. Verify child identity
   and parent linkage before dispatch; restore completed or interrupted children
   without blindly repeating delegation. Integrate with approval journals and
   task recovery. No unconfigured agent is silently switched to disk storage.
4. **Streaming capture and portability.** Bounded subprocess capture and previews;
   then export/import closure over child trajectories and referenced resources.
   Preserve immutable result identities and verify imported bytes. Shared mutable
   plans need explicit concurrent-edit semantics, not an implicit writable mount.

## Risks and tests for increment 1

Files: new runtime modules, `tests/test_tool_result_store.py`,
`tests/test_runtime_resources.py`, existing runtime exports, a new `docs/learn`
page and its navigation entry. Preserve unrelated local edits and earlier event
buffer work. No commits without a new explicit request.

Cover chunk generators, byte ranges, invalid/traversal identifiers, corruption,
missing files, writer exceptions, size limits, concurrent creation, immutable
publication, restart and copying a closed store to another root. Store references
in a real SQLite checkpoint and restore them after relocation. Test subprocess
death before publication: partial staging data may remain, but cannot be resolved
as a complete result. No automatic deletion or TTL: old trajectories may still
reference content. Filesystem fsync guarantees depend on the OS/device; tests do
not prove power-loss safety on every filesystem.

The local store is host/runtime infrastructure on a trusted directory, not an OS
sandbox or a new model filesystem permission. Explicitly reject symlink/special
result entries. Do not expose the storage root to agents. Reference integrity is
not authorization. Existing ArtifactRegistry remains the presentation renderer;
do not make it load full durable results into memory automatically.

Run focused tests, durability gate, full offline pytest, Ruff and MkDocs.
