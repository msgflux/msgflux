# Bounded event delivery

## Scope and order

1. Add a shared private, thread-safe event buffer for direct streams and watchers.
   Bound accepted pending events before scheduling loop callbacks; coalesce wakes.
2. Expose optional `event_buffer_limit` on Module.stream_events, EventHub.watch and
   Agent.watch. None preserves unlimited delivery; positive integers opt in.
3. Overflow clears the incomplete queue and raises EventBufferOverflowError to
   the consumer once. Producers never block or receive this delivery error.
   A watcher unsubscribes without cancelling the Agent. A direct stream follows
   its existing finally/cancellation path when its consumer observes the error.
4. Add unit/race/lifecycle and Agent tests, then update the existing event-streaming
   learning page. No checkpoint/event schema changes and no automatic retry.

## Files and risks

Affected: runtime/event_buffer.py, events.py, event_hub.py, exceptions.py,
nn/modules/module.py, nn/modules/agent/lifecycle.py, focused tests and docs/learn.
Reuse one buffer to avoid divergent overflow/closure behavior. Lock ordering,
lost wakes, closed-loop publication, exception retention, thread publishers and
full-buffer close need coverage. Watchers and streams each own independent bounds.
The bound is event count, not bytes or total runtime memory: payloads, model
buffers, live projections and checkpoints remain separate concerns. Accepted
events may already correspond to completed effects; overflow is not rollback.

## Validation and follow-ups

Run buffer/hub/event-streaming/playground tests, durability gate, full offline
pytest, Ruff and MkDocs. Deterministic tests use barriers/events, not timing sleeps.
Keep Bash execution, output spooling, byte accounting, backend isolation and
Resource loader out of this increment. Do not enable a shell executor implicitly.
