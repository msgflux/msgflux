# Changelog

## [1.0.0rc2] - 2026-10-15

Second release candidate for Coding and AgentService.

### Service Operation
- Added service inspection and stop commands, thread interruption, stop and
  resume commands, and run inspection (#242).
- Added agent activity listing, effective agent configuration inspection, and
  service configuration validation (#243, #244).

### Durability And Diagnostics
- Preserve non-authentication Codex HTTP errors and their diagnostics (#245).
- Preserve interrupted status and cancellation reasons for aborted background
  tasks instead of reporting them as failed (#247).
- Preserve structured task results, including workspace execution results,
  when persisting and reopening SQLite task stores (#248).
- Preserve partial assistant text and commentary when provider streams fail,
  including failed-run history, without executing incomplete tool calls (#249).
- Added controlled integration coverage for interrupted provider streams,
  checkpoint write failures, receipt write failures, thread isolation, and
  duplicate-request safety (#249).
- Extended live service stress tests to use saved accounts (#246).

### Installation
- Renamed the workspace tool extra to `coding` and validated installed-package
  Coding and AgentService workflows (#245).
- Install with `pip install "msgflux[coding,service]==1.0.0rc2"` or
  `uv tool install "msgflux[coding,service]==1.0.0rc2"`.
- Requires Python 3.11 or newer; CI covers Python 3.11 through 3.14.
- Recovery of uncertain external effects still requires an explicit trusted-host
  decision. This release does not introduce automatic replay or exactly-once
  guarantees for external operations.

## [1.0.0rc1] - 2026-10-14

First release candidate for the 1.0 API, focused on Coding and AgentService.

### Coding And AgentService
- Added `CodingSession`, a coding harness with provider-aware workspace tools,
  configurable agents and subagents, reasoning settings, and a default system prompt.
- Added `AgentService` with durable prompt admission, per-thread workspace and
  account bindings, recorded results, approvals, and explicit recovery.
- Added the optional HTTP/SSE service adapter and session client, supporting
  observation, reconnection, interruption, and frontend-independent execution.
- Added the `msgflux` CLI for initialization, account login, service status and
  restart, and noninteractive coding prompts with JSON or streamed NDJSON output.
- Added named account bundles for API keys and Codex OAuth, including managed
  token refresh and account isolation across concurrent executions.
- Consolidated workspace injection, local and Docker execution, bounded reads,
  approval previews, background tasks, and durable subagent histories.

### Recovery And Validation
- Preserve workspace identity and approval bindings across service restarts.
- Recover lost prompt-admission responses by request ID without resubmission.
- Await streamed response classification on every tool round so subsequent tool
  calls execute and receive matching outputs before final completion.
- Preserve concrete recovery diagnostics and avoid automatic replay of uncertain
  external effects; previous-worker quiescence remains a trusted-host decision.
- Validate coding, service, accounts, approvals, cleanup, and recovery with
  offline integration and independent-process failure tests.

### Compatibility And Installation
- Requires Python 3.11 or newer; CI covers Python 3.11 through 3.14.
- The 1.0 coding/service configuration uses agent entries and named accounts.
  Earlier experimental configuration formats are not retained.
- Install the local/HTTP service dependencies with
  `pip install "msgflux[service]==1.0.0rc1"` or
  `uv add "msgflux[service]==1.0.0rc1"`.
- This is a prerelease for adoption testing. Interruption is cooperative and
  does not undo completed effects; independent background tasks require their
  own inspection and interruption.

## [0.5.1] - 2026-05-29

### CI / Maintenance
- Updated documentation publishing to run on non-prerelease GitHub releases while keeping pull requests as build validation only (#116).
- Enabled PR labeling and release draft automation for forked pull requests, and simplified the release draft template (#117).

## [0.5.0] - 2026-05-29

### Feat
- Added opt-in input validation for signatures (#82).
- Added method hooks beyond `forward` (#87).
- Added `extra_body` passthrough extensions for OpenAI-compatible chat completions (#88).
- Added Open-Meteo weather retriever support and the builtin `Weather` tool (#97).
- Added tool display names and usage guidance metadata (#98).
- Added the Ceramic web retriever (#99).
- Added the SearXNG web retriever (#101).
- Added agent prompt warmup (#105).
- Added `msgspec.Struct` tool parameter support (#110).
- Added Agent Skills runtime support (#111).

### Docs
- Corrected retriever section numbering in `docs/learn/nn/agent/tools.md` (#80).
- Added the Builtin Tools section to `tools.md` (#81).

### Refactor
- Renamed the WebSearch steering parameter to `goal` (#100).
- Clarified tool display-name and usage-guidance examples and aligned `ToolLibrary.get_tool_display_names()` with a fallback to the registered tool name (#114).

### CI / Maintenance
- Updated pre-commit hooks (#86).
- Bumped `packaging` from 25.0 to 26.2 (#92).
- Bumped `openai` from 2.32.0 to 2.38.0 (#106).
- Bumped `bm25s` from 0.3.4 to 0.3.9 (#107).
- Bumped `ruff` from 0.15.11 to 0.15.14 (#108).
- Bumped `pymdown-extensions` from 10.21.2 to 10.21.3 (#109).
- Bumped `pytest-asyncio` from 1.3.0 to 1.4.0 (#113).

## [0.4.1] - 2026-04-22

- Fixes: restored uv.lock file for CI compatibility (#78).

## [0.4.0] - 2026-04-22

- Features: added new web retrievers (Arxiv, Brave, Exa, LinkUp, SerpAPI, Tavily) and model integrations (Brave, Exa), plus configurable WebSearch tool (#71).
- Web Retriever: added support for Wikipedia highlighting in web retriever section (#73).
- Providers: added OpenAI-compatible `extra_body` support for chat completions (#72).
- Maintenance: removed uv.lock file (#74).

## [0.3.0] - 2026-04-20

- Features: added OpenAI prompt cache retention support (#52), token logprobs metadata for chat completions (#53), and the `LLMAsVerifier` workflow with docs and examples (#54).
- Fixes: mapped OpenRouter `reasoning_max_tokens` to `extra_body.reasoning.max_tokens` and enforced the `reasoning_effort` conflict rule (#57).
- Docs: updated chat completion and agent reasoning docs to reflect the OpenRouter-only reasoning budget behavior (#56, #57).
- Maintenance: updated pre-commit hooks (#55).

## [0.2.0] - 2026-04-17

- Features: added `provider/model-id` string shorthand support to `Agent`, `Speaker`, and `Transcriber` (#26, #27, #28), plus runtime tool controls for `Agent` (#30).
- Fixes: fixed registry naming for generic callables (#29).
- Maintenance: updated release automation and cleaned up the release script (#36, #49).

## [0.1.1] - 2026-04-15

- Fixed message copying in the OpenAI provider (#47).
- Fixed isolation of injected messages for agent tools (#46).
- Updated CI and dependency tooling to keep release checks passing (#45, #44, #38, #37, #35, #34, #33, #32).

## [0.1.0] - 2026-04-09

## [0.1.0a31] - 2026-04-09
