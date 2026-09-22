# Workspace output completion

## Scope and order

1. Extend the process execution contract with optional incremental delivery and
   bounded, awaited chunk consumption. Preserve existing buffered executors.
   Connect a per-call shell capture policy through the existing tool context
   extension mechanism; no model-visible output-limit arguments.
2. Capture stdout/stderr in private temporary files, serialize the complete shell
   batch incrementally to the result store, and return only previews/references
   for large results. Keep authorization and cancellation in ExecutionEnvironment.
3. Verify the reference contract across events, native transport, function tools
   and checkpoints. Test relocation of a closed resource layout and abrupt process
   exits around result publication/checkpoint persistence.
4. Run the offline integration/durability gates, bounded opt-in live provider
   scenarios, formatting and documentation builds.

## Affected areas

- `runtime/environment.py`: incremental executor contract and enforcement.
- Process capture helpers: bounded consumption and temporary-file lifecycle.
- `nn/extensions/tool_output.py`, tool context registry and builtin workspace
  tools: opt-in capture integration using hidden runtime inputs.
- `tests/test_*output*`, new portability/process tests, opt-in live integration
  tests: full retrieval, previews, cancellation, failure, restart and transport.
- `docs/learn/nn/agent/resources.md` and workspace/tool extension guides: usage,
  executor migration, memory guarantees and safe relocation procedure.

## Risks and boundaries

- Buffered legacy executors cannot acquire bounded capture retrospectively.
- Process adapters remain responsible for real sandbox enforcement and child
  cleanup; a subprocess helper must not claim filesystem/network isolation.
- Storage publication must precede emitting/persisting a reference. Interrupted
  calls may leave unreferenced published content, never partially published data.
- Temporary capture requires a disk budget and cleanup on error/cancellation;
  slow storage must not create an unbounded chunk queue.
- Relocation tests use closed SQLite stores, including all referenced shared
  tool-results. Concurrent export/garbage collection is outside this increment.
- Real requests use synthetic fixtures, explicit opt-in and bounded calls; never
  send repository contents or credentials as model inputs.

## Review boundaries

Keep capture/contract implementation, durability tests and live validation in
separate reviewable units. No commits or PRs are requested in this turn.

## Integration findings

Live OpenRouter exposed Chat Completions tool-call deltas without repeated `id`
or function name. Correct the shared parser and add an offline regression before
rerunning the provider matrix. Also preserve structured result references in
function-output history metadata (as already done for native shell); serialized
wire strings are not a sufficient UI/checkpoint reference contract.

The failed live stream also exposed delayed HTTP generator cleanup after parser
failure. Close provider streams explicitly before reporting completion/failure,
and propagate close through the chat transport. Cover parser failures and early
consumer close with deterministic tests; do not rely on garbage collection.

## Validation and reproduction

The deterministic suites include real, fixed Python subprocesses, quota/sink
failures, repeated cancellation, forced termination, split UTF-8, independent
concurrent captures, portable closed bundles and abrupt spawned-process deaths.
The subprocess fixture never executes model-supplied commands and declares no
filesystem/network isolation. Capture uses two temporary files per batch, with
byte ranges separating command output.

Run the offline suite independently from credential-bearing integration tests:

```bash
uv run --no-sync pytest -q --ignore=tests/integration
```

The opt-in live matrix covers OpenAI Responses native shell, OpenAI Chat
Completions function tools, and OpenRouter function tools. It uses synthetic
outputs from a fake executor, streams both the first run and the continuation
to completion, checks durable references and verifies checkpoint completion.
It does not launch shell commands or transmit repository contents.

```bash
uv run --no-sync python - <<'PY'
import os
import msgflux as mf
import pytest

mf.load_dotenv()
os.environ["MSGFLUX_LIVE_OFFLOAD"] = "1"
raise SystemExit(pytest.main([
    "-q", "tests/integration/test_tool_offload_e2e.py",
]))
PY
```

This command makes paid requests. The test module itself neither loads `.env`
nor enables other live suites. Model choices are explicit in its case table.

A controlled tracemalloc run observed 841,461 bytes of peak Python allocations
for roughly 1 MiB of output and 904,310 bytes for roughly 8 MiB. The regression
test allows allocator/scheduling variation while rejecting a retained whole
large output. This is not RSS, subprocess memory, a platform-wide guarantee or
a statistical performance benchmark. Legacy buffered executors remain buffered.
