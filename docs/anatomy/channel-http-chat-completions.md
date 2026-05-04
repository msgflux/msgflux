# HTTP Chat Completions Channel

The HTTP channel is the server boundary that exposes registered msgFlux agents as
OpenAI-compatible chat models.

This page documents the internal contract of the HTTP/chat-completions path only.
It does not cover social adapters, Telegram, Slack, or provider internals.

## Files In The Boundary

The channel is split into a small set of files with clear ownership:

- `src/msgflux/channels/registry.py` owns registration, settings, defaults,
  auth, authorization, rate limits, hooks, and readiness.
- `src/msgflux/channels/http/app.py` owns FastAPI integration, HTTP routes,
  body limits, timeouts, error mapping, correlation headers, CORS, OpenTelemetry,
  and lifespan wiring.
- `src/msgflux/channels/http/openai.py` owns OpenAI-compatible request decoding,
  `AgentRun` construction, agent invocation, post-processing, and response/SSE
  encoding.
- `src/msgflux/channels/http/schemas.py` owns msgspec structs for the external
  OpenAI-compatible payloads.
- `src/msgflux/channels/http/msgspec.py` owns FastAPI response/route adapters so
  msgspec can be used without forcing the rest of the application to speak
  Pydantic.

This keeps the HTTP protocol boundary separate from agent execution and provider
execution.

## High-Level Flow

Non-streaming requests follow this path:

```text
POST /v1/chat/completions
  -> app._handle_chat_completions(...)
     -> read and validate request body
     -> decode ChatCompletionRequest
     -> openai.create_chat_completion(...)
        -> build ChannelContext(channel="http")
        -> request_start hooks
        -> auth / authorize / rate limit
        -> registry.get_agent(request.model)
        -> prepare AgentRun
        -> agent.acall(..., stream=False)
        -> post processors
        -> ChatCompletionResponse
     -> JSON response
```

Streaming requests follow the same control path until the agent call:

```text
POST /v1/chat/completions stream=true
  -> app._handle_chat_completions(...)
     -> openai.create_chat_completion_stream(...)
        -> build ChannelContext(channel="http")
        -> request_start hooks
        -> auth / authorize / rate limit
        -> prepare AgentRun with stream=True
        -> agent.acall(..., stream=True)
        -> post processors
        -> SSE role chunk
        -> content/reasoning chunks
        -> finish chunk
        -> data: [DONE]
```

The HTTP app intentionally does not know how to run an agent. It only validates
transport concerns, chooses JSON vs SSE, and delegates execution to
`http/openai.py`.

## Registry As The Application Boundary

`ChannelRegistry` is the application-level composition object. It is not an HTTP
router, but the HTTP channel depends on it for every application decision:

- which agents exist
- which metadata `/agents` exposes
- what global settings apply
- what defaults should be applied to runs
- how auth and authorization work
- which rate limits exist
- which lifecycle and observability hooks run
- how exceptions should be mapped

The important internal types are:

```text
ChannelContext
  channel: "http"
  agent_name: request.model
  request_id: resolved request id
  request: ChatCompletionRequest
  state: mutable boundary metadata

AgentRun
  messages
  vars
  stream
  model_preference
  tool_filter
  kwargs
  policies
```

`ChannelContext` is boundary metadata. `AgentRun` is the normalized agent input.
Keeping those separate avoids overloading `vars` with transport details.

## Request Metadata

`resolve_request_metadata(...)` centralizes IDs used by logs, hooks, responses,
and error payloads.

Resolution order:

- `request_id` comes from explicit metadata, `X-Request-ID`, or a generated id.
- `correlation_id` comes from explicit metadata, `X-Correlation-ID`, or
  `request_id`.
- `traceparent` is propagated when present.

`app.py` mirrors these values back as response headers. Error payloads also carry
`request_id` and `correlation_id`, so clients can join server logs with failed
requests.

## Request Decoding

`ChatCompletionRequest` is a msgspec struct with `forbid_unknown_fields=False`.
That is intentional.

The channel only consumes a small OpenAI-compatible subset:

- `model`
- `messages`
- `stream`
- `run_config`
- `stream_options`

Unknown fields are accepted so OpenAI SDK clients can send extra fields without
breaking the server. Unsupported fields are not automatically forwarded to the
agent. Forwarding is explicit through `run_config`, defaults, or pre-processors.

## Preparing `AgentRun`

`prepare_agent_run(...)` merges three sources:

```text
registry defaults
  + request.run_config
  + pre_processor updates
  -> AgentRun
```

The base mapping is:

- `request.messages` -> `run.messages`
- `request.stream` -> `run.stream`
- `run_config.vars` merged over default `vars`
- `run_config.model_preference` over default `model_preference`
- `run_config.tool_filter` over default `tool_filter`
- default `kwargs` copied into `run.kwargs`
- default policies copied into `run.policies`

Pre-processors run last and may return:

- `None`, meaning no change
- a full `AgentRun`
- a mapping of run fields

This design makes global defaults cheap, request overrides explicit, and
application-specific mutation possible without coupling the HTTP adapter to an
agent implementation.

## Auth, Authorization, And Rate Limits

Auth runs before the agent is fetched and before pre-processing side effects
matter.

The order is:

```text
registry.auth_handler()
  -> registry.authorizers(agent_name)
  -> registry.check_rate_limits(...)
```

Auth may return any principal object. That object is stored in:

- `context.state["principal"]`
- `context.state["auth"]`

Authorizers can either reject by returning `False` or enrich `context.state` by
returning a mapping.

Rate limits run after auth so bucket policies such as `api_key`, `client`, and
`tenant` can use authenticated identity when available. This is why rate limiting
belongs in the registry rather than inside the FastAPI route function.

## Agent Invocation Contract

The HTTP channel calls agents with keyword arguments:

```python
await agent.acall(
    messages=run.messages,
    vars=run.vars,
    model_preference=run.model_preference,
    tool_filter=run.tool_filter,
    stream=False,  # or True for SSE
    **run.kwargs,
)
```

The channel does not call provider models directly. It treats the registered
object as the agent boundary. That preserves all existing `nn.Agent` behavior,
including system prompts, tools, signatures, model gateways, and internal
streaming support.

## Response Mapping

For non-streaming responses, `_extract_message_content(...)` maps common output
shapes into OpenAI-compatible assistant content.

The important branches are:

- `ModelResponse`-like objects are consumed when they expose `consume()`.
- mappings can expose `answer`, `response`, `content`, or `text`.
- mappings can expose `reasoning` or `reasoning_content`.
- all other values are stringified.

The response is always encoded as:

```text
ChatCompletionResponse
  object = "chat.completion"
  choices[0].message.content
  choices[0].message.reasoning_content, when present
```

`usage` is currently `None` because the channel does not assume every agent or
provider exposes token accounting in the same shape.

## Streaming Contract

The stream path emits OpenAI-compatible SSE frames:

```text
data: {role chunk}\n\n
data: {reasoning/content chunk}\n\n
data: {finish chunk}\n\n
data: [DONE]\n\n
```

If the agent returns `ModelStreamResponse`, the channel consumes content and
reasoning concurrently. Reasoning chunks are mapped to `delta.reasoning_content`;
content chunks are mapped to `delta.content`.

If the agent returns a final non-streaming value even though `stream=True`, the
channel still emits a valid SSE response by converting that final value into one
or two chunks.

This fallback matters because not every agent implementation can stream.
Clients still receive a stream-shaped response.

## Timeouts And First Chunk Handling

HTTP timeouts are enforced at two levels:

- non-streaming calls use `_with_timeout(...)` around the whole completion
- streaming calls wait for the first chunk with `_first_stream_chunk(...)`, then
  wrap the remaining iterator with `_with_stream_timeout(...)`

The first-chunk step is important. It lets setup/auth/agent-start failures become
normal JSON error responses before FastAPI starts an SSE response. Once the first
chunk is yielded, later timeout failures must be represented as SSE error chunks.

## Error Mapping

Errors pass through registry-level custom handlers before falling back to built-in
HTTP error shapes.

The built-in mapping covers:

- invalid request body -> `400 invalid_request`
- `UnauthorizedError` -> `401 unauthorized`
- `ForbiddenError` -> `403 forbidden`
- `AgentNotFoundError` -> `404 agent_not_found`
- `PayloadTooLargeError` -> `413 payload_too_large`
- `RateLimitExceededError` -> `429 rate_limit_exceeded`
- `RequestTimeoutError` -> `504 request_timeout`
- `ChannelError` -> `400 channel_error`

Custom error handlers let applications keep a stable external error contract
without changing agent code or FastAPI route code.

## Lifecycle And Readiness

The FastAPI lifespan updates registry readiness:

```text
starting
  -> startup hooks
  -> ready
  -> shutdown hooks
  -> stopped
```

If startup fails, readiness becomes `startup_failed`. `/ready` returns `503` when
not ready and `200` when ready.

This gives deployers a clear health/readiness distinction:

- `/health` only says the process is alive
- `/ready` says startup hooks completed and traffic can be accepted

## Why This Is Separate From Provider Chat Completion

This channel exposes an OpenAI-compatible HTTP API, but it is not the OpenAI
provider adapter.

The provider adapter translates msgFlux model calls into upstream OpenAI API
requests. This HTTP channel translates external client calls into msgFlux agent
runs.

The two boundaries are intentionally independent:

```text
OpenAI SDK client
  -> msgFlux HTTP channel
  -> registered Agent
  -> any model/provider supported by the Agent
```

That is why a client can call `model="support"` through `/v1/chat/completions`
even when the underlying agent uses a different provider, tools, model gateway,
or custom runtime policy.
