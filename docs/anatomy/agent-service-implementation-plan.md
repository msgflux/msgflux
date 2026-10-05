# Embedded Agent service: implementation increment 1

## Scope and order

The developer approved the shared runtime/channel plan and asked to start its
incremental implementation. The first increment introduces execution ownership
and durable admission independently of Textual, HTTP, and social adapters.

1. Add immutable msgspec records and a SQLite service admission journal.
   Existing Agent checkpoints cannot acknowledge an input before execution;
   the journal stores identity, normalized input and attempt state only.
2. Add AgentService/AgentSession: per-thread factories, submit/receipt, shielded
   wait, snapshot/watch, targeted interruption, inbox steering and explicit
   recovery. The service consumes Agent.stream_events as the producer.
3. Export through runtime/__init__.py without importing NN at module load.
4. Exercise real Agents with deterministic models, independent SQLite connections
   and abrupt process exits. Document the public Python API before moving to the
   CodingSession/TUI consumer changes of increment 2.

## Files

- src/msgflux/runtime/service_records.py: serializable identities/errors.
- src/msgflux/runtime/service_store.py: binding and admission state, conditional
  claims and revision/owner-fenced transitions.
- src/msgflux/runtime/service.py: live dependency bindings and execution owner.
- src/msgflux/runtime/__init__.py: exports only.
- tests/test_service_store.py and tests/test_agent_service.py: contracts,
  integration and processes.
- docs/learn/nn/agent/service.md and mkdocs.yml: API usage and guarantees.

## Recovery boundaries

An accepted record may start once after restart. A running record without a
checkpoint stays uncertain; never resend its prompt blindly. An old worker must
be known stopped before another host reclaims its attempt. Fencing service-store
writes is not fencing arbitrary filesystem/network effects. Resume preserves
run identity and existing Agent approvals, workspace and command-receipt checks.
A committed terminal checkpoint can settle the journal without a model call.

Use revision CAS as well as owner identity: a stale finish must fail even if the
same service owner has resumed the run. Paused admissions remain thread-busy.
Failures with resumable checkpoints require explicit recovery, not implicit retry.

## Resource ownership and risks

Clients own watchers/waiters, not workers. Worker tasks start with fresh
ContextVars to avoid inheriting another client/Agent's event sink or task handle.
The host supplies scopes and dependencies; factories cannot share mutable Agents
across threads or replace service-owned run identity. Borrowed stores remain
host-owned. A factory close callback must drain resources/delegated work it owns.
No automatic home writes, credential persistence or network dependency.

The first service is bound to one event loop. Local SQLite admission operations
are short synchronous transactions, like existing local snapshot readers; this
is not a throughput guarantee. Shutdown is cooperative and may wait for
non-cooperative extension code. Cancellation of a shutdown wait leaves shutdown
running, so it cannot close resources beneath a still-running worker.

## Required verification

Ruff, focused service/store tests, coding/session and event-stream regressions,
the existing offline durability gate, and strict MkDocs. Process tests use spawn,
bounded synchronization, SQLite files and abrupt exits; cleanup must leave no
workers behind. A small real-provider check can validate detached observation
without replacing deterministic tests. No HTTP/daemon/TUI migration in increment 1.
