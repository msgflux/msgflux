# Changelog

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
