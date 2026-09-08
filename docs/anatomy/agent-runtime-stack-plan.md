# Agent Runtime Implementation Stack

Base: `fix/agent-runtime-stability` (`0a653690`). Implement and merge in the
following order. Each branch contains its own regression tests and public docs.
Existing uncommitted work in the original checkout is excluded.

| Order | Branch | Deliverable and affected areas |
| --- | --- | --- |
| 1 | `fix/agent-terminal-integrity` | Agent core/lifecycle/conversation/model runtime and Module streams: settle external cancellation, preserve direct tool outcomes, share native/flow feedback, finalize streaming hooks and wrapped responses. Tests: durable Agent, tool control, event streaming. |
| 2 | `fix/agent-inbox-delivery` | Inbox stores and Agent conversation: atomic queue operations, recoverable claims, acknowledge only persisted delivery, deduplication. Tests: hook/checkpoint failures, competing views and SQLite connections, recovery. |
| 3 | `feat/agent-checkpoint-revisions` | Checkpoint stores and run state: schema version, revision-checked atomic commits, explicit branch/head and durable extension state foundations. Tests: round trips, conflicting writers, legacy snapshots, forks. |
| 4 | `feat/agent-tool-turn-extension` | Extensions and shared loop policy: terminal tool-turn budget, last-round context notice, persisted counters and structured stop reason; no repeated finalization requests. Tests: sync/async, native/flow, resume. |
| 5 | `feat/agent-incremental-output` | Output streaming contract and artifact-reference renderer: bounded marker buffering, immutable registered artifacts, canonical/rendered separation. Tests: arbitrary chunk boundaries, escapes, malformed/missing references, cancellation and event parity. |
| 6 | `feat/agent-context-scopes` | Conversation branch controller and builtin scope tools: inherit prefix, exclusive safe transitions, nested return points, summary import and idempotent close, revision checks. Tests: open/close/nesting/recovery, call-output pairing, budget inheritance, compaction and inbox interaction. |
| 7 | `fix/agent-inbox-leases` | Validation follow-up: selectively release rejected notifications without releasing retained receipts in the same lease; regression tests for both stores. |
| 8 | `fix/agent-checkpoint-atomicity` | Validation follow-up: shared revision metadata preparation, fail before publishing invalid state/events, retain fork provenance and resume under the target identity. |
| 9 | `fix/agent-output-finalization` | Validation follow-up: non-streaming output envelopes, sync callbacks returning awaitables, and propagation of finalizer failures to stream consumers. |

## Implementation rules

- Keep the public Agent/Module composition and existing provider-neutral tool
  contracts. Prefer shared decisions over another native/flow loop.
- Keep stored model output canonical. Rendering must not expand checkpoint text.
- Preserve immutable content payload deduplication. Branch lineage and active
  position are explicit metadata, not inferred from update timestamps.
- Make context transitions at settled tool-batch boundaries. Never move an
  unfinished tool call to another branch. Scope changes do not reset run budgets.
- Keep policy in extensions and generic execution/finalization in the runtime.
- Document behavioral changes in existing `docs/learn/nn/agent/` pages and link
  new pages from navigation when needed. Avoid unrelated documentation edits.

## Risks and validation

Primary risks are dropped/duplicated inbox delivery, replayed external effects,
partial branch transitions, stale checkpoint writers, streaming resource leaks,
and differences between synchronous/asynchronous execution. Fault-injection
tests must cover state boundaries, not just individual helper return values.

Run focused tests and Ruff per commit, then the complete local suite and MkDocs
on the integrated stack. Network-backed integration tests require credentials;
report any environment limitation explicitly. Do not claim exactly-once
external effects: recovery must retain evidence and avoid silently replaying
unknown in-flight effects. Durable distributed event replay and a general
external-effect reconciliation engine remain separate extensions of this work;
the concrete scope is the six deliverables above and their validation fixes.

## Local validation

The integrated stack is developed in `/tmp/msgflux-runtime-stack-R2bOK7` to leave
the original checkout's uncommitted files untouched. Branches and commits are
local; no push or PR submission is part of this handoff.

Validation commands use `uv run` with the existing project environment and
`PYTHONPATH=src` to test this worktree. Run the offline suite with
`pytest -q --ignore=tests/integration`; the integration directory includes live
provider tests that construct clients during collection and require credentials.
Also run repository-wide `ruff check`, `ruff format --check`, and `mkdocs build`.
