# V1 workspace hardening

Three reviewable increments, in dependency order:

1. Bound image and editor reads before allocation. Files: workspace tools,
   `WorkspaceEditor`, `ExecutionEnvironment`, focused tests and public runtime
   docs. Host-configured edit budget must cover old/new text, apply-time reads
   and prepared proposals restored from checkpoints. Image reads use bounded
   prefixes and validate vision options before reading. Test sparse real files
   and allocation peaks, sync/async, approvals, and oversized UTF-8 inputs.
2. Add total result-store quota and explicit offline garbage collection. Files:
   result store, resources, new storage tests, resources documentation. Serialize
   writers/maintenance across processes with a POSIX advisory lock; account for
   published and interrupted staging content. Cleanup defaults to dry-run and
   needs a complete host-owned reference inventory with quiescent writers and
   checkpoints. Never evict referenced results to make room. Test process races,
   failures, referenced preservation and actual filesystem space accounting.
3. Add a Docker-backed streaming executor for LocalWorkspace, plus offline
   contract and opt-in real Docker integration tests. Deny network, restrict
   mounts to the exact host-bound workspace, drop capabilities, use non-root,
   read-only image root, PID/memory/CPU limits, and explicit whole-workspace
   process grants. No inference from individual file permissions. Reuse
   `drain_subprocess`; container cleanup must survive cancellation and failures.
   Containers/images/daemon are trusted host infrastructure, not protection
   against hostile kernel exploits or privileged daemon operators. No image
   pull without explicit host action. Validate real writes, host path denial,
   network denial, quotas, timeouts and cancellation using temporary data.

Documentation: runtime and resources guides, dependency setup and public exports.
Validation: focused tests, offline durability gate, complete offline suite, Ruff,
strict MkDocs, wheel/sdist, real filesystem/process/Docker integration checks.
Preserve unrelated local user changes. No automatic deletion in the user's
existing resource directories and no real provider credentials are needed for
deterministic OS/storage integration tests.
