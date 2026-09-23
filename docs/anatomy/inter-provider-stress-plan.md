# Inter-provider Agent stress and release benchmarks

Implement as a separate reviewable increment on top of the current Agent runtime.
It is opt-in because live runs consume provider quota and may incur charges.

1. Inspect provider registration, public model construction, credential lookup,
   Agent event streaming, tool dispatch, checkpoints, and existing benchmark
   helpers. Identify which requested vendors have first-class providers and
   which use the OpenAI-compatible transport only in this validation harness.
2. Add `tests/integration/test_live_agent_provider_matrix.py` with explicit
   opt-in, six-vendor configuration, bounded output/tokens/turns, live
   streaming/tool/checkpoint cases, and actionable failure classification.
   Never log credentials or complete model content by default.
3. Add `scripts/benchmark_agent_release.py` and focused offline tests. Record
   reproducible JSON measurements for latency, event counts/bytes, peak Python
   memory, and checkpoint growth. Compare with a versioned, host-selectable
   baseline using tolerances rather than enforcing one machine's timings in CI.
4. Document exact commands, model/endpoint overrides, expected costs, and
   limitations under `docs/learn/nn/agent/`. Public integration-only settings
   must not silently become production provider APIs.
5. Validate formatting/lint, focused offline tests, strict MkDocs, then run a
   bounded real matrix for configured vendors. Record model IDs, provider/API
   mode, case status, and coarse timing without publishing secrets or payloads.

Risks: model capabilities differ; some vendors have no first-class provider;
rate limits or remote faults may be transient; output can contain sensitive
data; benchmark wall times vary across machines. Keep credentials at request
time, cap calls and output, and distinguish transport/provider failure from a
local invariant violation. Do not make live API calls during default pytest.

Live validation exposed a Chat Completions parser defect: a stream delta can
contain reasoning and tool-call fragments together, but the parser previously
skipped the latter. Add a focused parser regression to the same increment and
process all populated fields in both sync and async paths. This is required
for the provider matrix to test the intended tool-call invariant reliably.
