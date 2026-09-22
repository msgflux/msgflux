# Large-output memory budgets

## First increment: measure retention

Add `scripts/benchmark_event_memory.py` and focused tests in
`tests/test_event_memory_benchmark.py`. Exercise the real delivery buffer with
unique payloads, bounded and unlimited queues, and one oversized event. Measure
live hub projections separately, with and without a retained reconnect snapshot.
Document invocation and limitations in the existing event-streaming guide.

Use `tracemalloc` retained/peak Python allocation measurements, elapsed time and
explicit lifecycle cleanup. Do not assert machine-specific memory thresholds in
CI. Test scenario counts, overflow and snapshot semantics instead. Synthetic
payloads must be allocated inside measurement, not shared across all events;
otherwise the benchmark hides payload retention. This is not an RSS benchmark,
provider benchmark, token estimate or production default calibration.

The initial 20,000 × 1,024-byte run identified duplicate retention when a live
projection is joined for a reconnect snapshot. Include a narrow optimization in
`runtime/event_hub.py`: consolidate homogeneous text/byte chunks after joining
under the existing hub lock. Future deltas append normally; old immutable
snapshots remain unchanged. Repeated snapshots without new deltas reuse the
joined object. Test text/bytes, subsequent deltas, mixed payload compatibility
and prior snapshot immutability in `tests/test_event_hub.py`. This reduces
duplicate allocations, but does not cap text size or free fragments still owned
by another consumer's queue. No public API or checkpoint format changes.

## Following increments, in dependency order

Initial local measurement (CPython 3.12, 20,000 events × 1,024 ASCII bytes):
snapshot retained allocations fell from 41,964,053 to 20,491,181 bytes after
consolidation. Peak allocation remained about 42 MB because joining temporarily
requires old and new text together. The unlimited delivery queue retained about
27 MB; the 256-event overflow policy peaked at about 353 KB, but delivered an
explicit incomplete-stream error. One 20 MB event was still accepted by that
count limit. These are synthetic Python-allocation observations, not RSS limits,
production sizing recommendations or evidence that overflow is lossless.

1. Define a backend-neutral output-store contract: bounded chunk writes, opaque
   references, ranged reads, ownership, expiry and explicit storage quota failure.
   Integrate with workspace resources and permissions, not arbitrary host paths.
   Reference metadata must survive checkpoints; storage availability after restart
   must be explicit. The current in-memory ArtifactRegistry is not a disk spool.
2. Add a ToolLibrary output policy at the shared outcome boundary, before
   `tool.end`, transport encoding and checkpoint serialization. Keep small results
   unchanged. For large supported results, produce a bounded preview and a
   reference to complete stored content. UI and model budgets are independent;
   don't silently cut JSON, images, approval diffs or provider-native structures.
   Apply equally to sync, async, background, return-direct and native transports.
3. Add incremental process capture to the executor layer before enabling real
   Bash execution: concurrently drain stdout/stderr with bounded reads, spool
   without collecting complete strings, enforce aggregate storage limits, retain
   bounded previews, and clean up children/pipes on abort, timeout or failure.
   Backend isolation and approval requirements remain separate from buffering.
4. Bound live projection previews with explicit truncation metadata and provide
   ranged retrieval for UIs. Preserve canonical model/checkpoint content and
   avoid repeatedly serializing full accumulated output on each delta.

Each increment needs its own implementation plan naming concrete public APIs,
affected files, compatibility decisions, tests and `docs/learn` examples before
changing behavior. In particular, count-limited queues must not be advertised as
byte-limited queues or as a total runtime memory bound.

## Calibration and risks

Measure ASCII, multilingual text, enormous single lines, binary/structured
results, concurrent tools, slow UI consumers, reconnects and disk-full failures.
Record emitted bytes, retained bytes, preview bytes, spool bytes and serialized
wire bytes separately. A low rendered-line limit does not bound UI transport.
Avoid dynamic system-prompt changes for per-call output bookkeeping: references
belong in results, with stable guidance, preserving the reusable prompt prefix.

No automatic host file writes, new shell executor, output truncation default or
change to model-visible tool parameters is introduced by the measurement step.

## Reference implementations inspected (2026-09-20)

- [Pi Bash](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/bash.ts)
  uses a 2,000-line / 50 KiB returned tail and throttles progress snapshots at
  100 ms. Its
  [output accumulator](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/tools/output-accumulator.ts)
  keeps a rolling tail and spills full output to a temporary file. Incoming
  chunk size still affects transient allocation. These constants are reference
  points, not adopted msgflux defaults or a total UI/session memory guarantee.
- [Tau tools](https://github.com/huggingface/tau/blob/main/src/tau_coding/tools.py)
  also truncate returned Bash text and expose a full-output file, but capture via
  `communicate()` before truncation. Its builtin tool adapter discards
  `on_update`. This illustrates why bounded returned text does not establish
  bounded subprocess memory or incremental UI delivery.

For msgflux, keep the executor's capture chunk size, in-memory preview budget,
stored-output quota and UI update frequency separate. Native process timeouts,
background execution, approval and cancellation must use existing runtime
contracts, not duplicate model-facing arguments. Verify both behavior and peak
memory before choosing defaults; lines alone cannot bound a huge single line.
