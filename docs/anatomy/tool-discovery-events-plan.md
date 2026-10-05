# Tool discovery execution events

## Scope and order

1. Extend `runtime/events.py` with `tools.updated`. Portable search already
   emits tool lifecycle events. Emit loading changes from `ToolLibrary.load_tools`
   after successful mutation, only for newly loaded tools.
2. Add a Responses search event adapter in `models/tool_search_events.py`.
   Observe real search call/output items, pair their identifiers, normalize
   arguments and return compact loaded names without duplicating schemas.
3. Extend `_private/response.py`, `models/openai_compatible.py`, `nn/events.py`
   and the Module event consumer to carry provider events through the existing
   response event journal. Support ordinary and streamed responses, inherited
   by OpenAI Codex. Keep search items in canonical history unchanged.
4. Add focused tests for portable loading, provider pairing and event order,
   duplicate protocol items, failures, sync/async paths and runtime forwarding.
5. Update `docs/learn/nn/agent/event-streaming.md` with payloads and usage.

## Public behavior

Provider search emits `tool.start`, `tool.update` (completed arguments),
`tools.updated`, and `tool.end`. Shared fields are `tool_name=tool_search`,
`tool_call_id`, `arguments`, `execution`, `provider`, and `api_mode`.
`tools.updated` reports `loaded_tools` and origin. Portable changes include
`catalog_id`; provider changes refer to native schema discovery, not mutation
of the portable catalog. No request schemas or cache strategy change.

## Risks and verification

Never dispatch provider search as a local tool or fabricate schemas. Provider
items can repeat or omit an added event; keep lifecycle pairing idempotent and
close pending searches on transport failure. Avoid exposing full schemas in
UI events. Preserve raw native history for replay. Validate relevant model,
response, event stream and library tests, Ruff and strict MkDocs. This PR starts
from upstream main; the Coding branch can rebase after merge and consume these
core events without bringing TUI changes into the PR.
