# ⇄ Channels

Channels expose msgFlux modules through external interfaces. The first channel
is the HTTP server, which exposes registered `Agent` instances through an
OpenAI-compatible Chat Completions endpoint.

This means any OpenAI SDK client can call specialized msgFlux agents as if they
were regular chat models. The request `model` selects the registered agent name.

## ✦₊⁺ Overview

The channel layer is the external boundary for msgFlux modules. In practice,
it lets you:

- expose agents over HTTP through an OpenAI-compatible Chat Completions API
- discover registered agents through `/agents`, including description, tags, and capabilities
- control execution with `run_config`, global defaults, and per-agent defaults
- adapt incoming requests with pre-processors
- adapt outgoing responses with post-processors
- protect production servers with auth, authorization, rate limits, request limits, and timeouts
- integrate with OpenTelemetry through the official FastAPI instrumentation adapter
- propagate `X-Request-ID`, `X-Correlation-ID`, and `traceparent` at the HTTP boundary
- add lifecycle and observability hooks without coupling this logic to the Agent
- receive social webhooks through Telegram and Slack adapters

The default channel is the HTTP server, but the design is meant to support
other external interfaces as they are added.

## 1. **Quick Start**

The HTTP server depends on FastAPI and Uvicorn:

```bash
uv pip install "msgflux[server,openai]"
```

If you want to run the example without cloning the repository, you can start
directly from a remote Python file:

```bash
uv run --with 'msgflux[server,openai]' msgflux server https://raw.githubusercontent.com/msgflux/msgflux/main/examples/server_streaming_agent.py --trust-remote-code --host 127.0.0.1
```

Or start from a local file:

```bash
uv run --with 'msgflux[server,openai]' msgflux server server_streaming_agent.py --host 127.0.0.1
```

Use `--port` to override the default `8010`:

```bash
uv run --with 'msgflux[server,openai]' msgflux server server_streaming_agent.py --host 127.0.0.1 --port 9000
```

!!! warning "Remote execution requires explicit trust"
    Remote targets are blocked by default. Pass `--trust-remote-code` to allow
    download and execution. The server logs an `INFO` message when downloading
    the file and when saving it locally.

!!! tip "Default address"

    The default server address is:

    ```text
    http://127.0.0.1:8010/v1
    ```

Call a registered agent through the Chat Completions endpoint:

```bash
curl -N http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "support",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "Write a short status update for a delayed order."
      }
    ],
    "run_config": {
      "vars": {
        "tenant": "acme",
        "tier": "standard"
      }
    }
  }'
```

## 2. **Registry**

Create a Python file that exports a `ChannelRegistry`. The server loads that
object from the CLI target.

Save the following example as `server_streaming_agent.py`:

```python
import msgflux as mf
import msgflux.nn as nn

mf.load_dotenv()

registry = mf.ChannelRegistry()
registry.settings(
    title="msgFlux Support Channel",
    subtitle="Streaming order and billing agents.",
    description="OpenAI-compatible support server with per-agent rate limits.",
)
registry.rate_limit(
    name="support-ip-minute",
    agent="support",
    requests=60,
    window_s=60,
    by="ip",
)
registry.rate_limit(
    name="billing-api-key-minute",
    agent="billing",
    requests=30,
    window_s=60,
    by="api_key",
)

@registry.agent(name="support")
class SupportAgent(nn.Agent):
    """Customer support agent for order and delivery questions."""

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = "You are a concise customer support specialist."

@registry.agent(
    name="billing",
    tags=["billing", "payments"],
    capabilities={"streaming": True, "tools": True},
)
class BillingAgent(nn.Agent):
    """Billing support agent for invoices and payment questions."""

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = "You are a concise billing support specialist."
```

Once the file is saved, run it with:

```bash
uv run --with 'msgflux[server,openai]' msgflux server server_streaming_agent.py --host 127.0.0.1
```

!!! tip "Agent registration"

    `registry.agent` accepts either an `Agent` instance or an `Agent` class.
    When you pass a class, the registry instantiates it with no arguments and
    stores the resulting instance.

    If you do not pass `name=`, the registry uses the agent name exposed by the
    object. For `nn.Agent` subclasses, that usually comes from the class name
    via AutoParams. Pass `name="..."` only when you want to override the
    registered name.

    `description` comes from the agent itself, normally from the class docstring
    captured by AutoParams. Use the registry decorator for channel/discovery
    metadata that does not belong inside the Agent, such as `tags` and
    `capabilities`.

    The same decorator also works with instances:

    ```python
    registry.agent(SupportAgent())
    ```

!!! tip "Custom registry object"

    If your registry object has a different name, pass it explicitly as
    `module.py:object_name`.

After registration, `/agents` returns registered agents with metadata:

```bash
curl -s http://127.0.0.1:8010/agents
```

Use `model: "billing"` for the billing agent and `model: "support"` for the
support agent. The request examples below show each one in context.

## 3. **HTTP API**

| Name | Method | Description |
|---|---|---|
| `/v1/chat/completions` | `POST` | OpenAI-compatible endpoint that runs a registered agent. |
| `/health` | `GET` | Liveness check for the server process. |
| `/ready` | `GET` | Readiness check that flips after startup hooks complete. |
| `/agents` | `GET` | Lists registered agents with description, tags, and capabilities. |

The supported request fields are:

| Field | Description |
|---|---|
| `model` | Registered agent name, such as `support` or `billing`. |
| `messages` | OpenAI-compatible chat messages. |
| `stream` | Enables OpenAI-compatible SSE streaming when `true`. |
| `run_config` | msgFlux extension for runtime execution config. |

`run_config` currently supports:

| Key | Description |
|---|---|
| `vars` | Runtime variables passed to the Agent. |
| `model_preference` | Optional model preference forwarded to the Agent. |
| `tool_filter` | Optional tool filtering metadata forwarded to the Agent. |

The request is converted into an `AgentRun` before the Agent executes:

| HTTP input | Runtime field | Agent call |
|---|---|---|
| `messages` | `run.messages` | `Agent.acall(messages=...)` |
| `stream` | `run.stream` | `Agent.acall(stream=...)` |
| `run_config.vars` | `run.vars` | `Agent.acall(vars=...)` |
| `run_config.model_preference` | `run.model_preference` | `Agent.acall(model_preference=...)` |
| `run_config.tool_filter` | `run.tool_filter` | `Agent.acall(tool_filter=...)` |

`AgentRun` also has an internal `run.kwargs` field. It is not accepted from the
HTTP request. Use pre-processors to populate it when you need Agent runtime
inputs that are not part of the OpenAI Chat Completions shape, such as `task`,
`task_context`, and `task_multimodal`.

The endpoint does not persist sessions. Each request should contain the
messages required for that turn.

The HTTP boundary propagates request metadata:

| Header | Behavior |
|---|---|
| `X-Request-ID` | Used as the channel request id when provided; otherwise generated. |
| `X-Correlation-ID` | Used for cross-service correlation; defaults to request id. |
| `traceparent` | Preserved in context and response headers for trace propagation. |

Responses include `X-Request-ID` and `X-Correlation-ID`. Error payloads include
the same ids under `error.request_id` and `error.correlation_id`.

## 4. **Streaming and Tools**

This is the most complete end-to-end example because it combines a real tool,
streaming output, and request-level tool control.

Save the following as `server_streaming_agent.py`:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.tools.builtin import WebFetch, WebSearch

registry = mf.ChannelRegistry()

@registry.agent(name="support")
class SupportAgent(nn.Agent):
    model = "openai/gpt-4.1-mini"
    tools = [
        WebFetch,
        WebSearch("retriever/wikipedia"),
    ]
```

Run it with `uv run --with` and include the `wikipedia` package:

```bash
uv run --with 'msgflux[server,openai]' --with wikipedia msgflux server server_streaming_agent.py --host 127.0.0.1
```

Stream a response while allowing only the web search tool:

```bash
curl -N http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "support",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "What can you tell me about the Python programming language?"
      }
    ],
    "run_config": {
      "vars": {
        "tenant": "acme"
      },
      "tool_filter": {
        "allow": ["web_search"]
      }
    }
  }'
```

Block all tools for the same agent:

```bash
curl -N http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "support",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "Answer from model knowledge only."
      }
    ],
    "run_config": {
      "tool_filter": {
        "block": "*"
      }
    }
  }'
```

## 5. **Model Preference**

Use `run_config.model_preference` when the Agent uses a `ModelGateway`. This
lets the caller select a named model deployment without changing the public
agent name in `model`.

```python
import msgflux as mf
import msgflux.nn as nn

gateway = mf.ModelGateway([
    {
        "model_name": "fast",
        "model": mf.Model.chat_completion("openai/gpt-4.1-mini"),
    },
    {
        "model_name": "quality",
        "model": mf.Model.chat_completion("openai/gpt-5.2"),
    },
])

registry = mf.ChannelRegistry()


@registry.agent(name="assistant")
class AssistantAgent(nn.Agent):
    model = gateway
```

Select the gateway model per request:

```bash
curl -s http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "assistant",
    "messages": [
      {
        "role": "user",
        "content": "Summarize this contract clause."
      }
    ],
    "run_config": {
      "model_preference": "quality"
    }
  }'
```

With the msgFlux client provider:

```python
model = mf.Model.chat_completion(
    "msgflux/assistant",
)
response = await model.acall(
    [{"role": "user", "content": "Analyze this carefully."}],
    run_config={"model_preference": "quality"},
)
```

## 6. **Pre-processing**

Use `registry.pre()` to mutate the `AgentRun` before `Agent.acall`. A
pre-processor can:

- rewrite `run.messages`
- add or replace `run.vars`
- override `run.stream`
- set `run.model_preference` or `run.tool_filter`
- add internal `run.kwargs` such as `task`, `task_context`, or `task_multimodal`

Use `@registry.pre("support")` to target a single agent, or `@registry.pre()`
to run the processor for every registered agent. Pre-processors can mutate the
`run` object in place, return `None`, or return a mapping with replacement
fields:

```python
@registry.pre("support")
def enforce_tool_policy(_request, _context, run):
    tier = (run.vars or {}).get("tier", "basic")
    mode = (run.vars or {}).get("mode", "default")

    # Strict mode: model-only answer, no tool access.
    if mode == "internal_only":
        run.tool_filter = {"block": "*"}
        return run

    # Cost/risk policy by tenant tier.
    if tier == "basic":
        run.tool_filter = {"block": ["web_fetch", "web_search"]}
    return run
```

Returning a mapping replaces the provided fields. Mutate `run` in place when
you want to merge with the current execution state.

### 6.1 Convert Messages to Task

Use a pre-processor to reshape `messages` into a single `task` when you want
to work with a task-oriented input instead of the raw chat history.

```python
def last_user_text(messages):
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""


@registry.pre("support")
def messages_to_task(_request, _context, run):
    task = last_user_text(run.messages)

    # Clear the original chat history and pass only the derived task.
    run.messages = None
    run.kwargs["task"] = task
    return run
```

This produces a call equivalent to:

```python
await agent.acall(
    messages=None,
    task="...",
)
```

### 6.2 Convert OpenAI Image Content to `task_multimodal`

Use a pre-processor to keep the chat input, store the image URL in `vars`, and
route image-aware work through a dedicated tool.

```python
import msgflux as mf

@mf.tool_config(inject_vars=True)
def inspect_image(**kwargs) -> str:
    vars = kwargs.get("vars", {})
    image_url = vars.get("image_url")
    return f"Inspecting image at {image_url}"
```

```python
import msgflux as mf
import msgflux.nn as nn

@registry.agent(name="vision")
class VisionAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    tools = [inspect_image]
    system_message = "Describe images clearly and concisely."


def split_openai_content(content):
    if isinstance(content, str):
        return content, []

    text_parts = []
    image_urls = []

    for block in content or []:
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "image_url":
            image = block.get("image_url")
            if isinstance(image, dict):
                image_urls.append(image.get("url"))
            elif isinstance(image, str):
                image_urls.append(image)

    return "\n".join(text_parts).strip(), [url for url in image_urls if url]


@registry.pre("vision")
def openai_images_to_task_multimodal(_request, _context, run):
    last_user = next(
        (
            message
            for message in reversed(run.messages)
            if message.get("role") == "user"
        ),
        {},
    )
    task, image_urls = split_openai_content(last_user.get("content"))

    run.messages = None
    run.vars["image_url"] = image_urls[0] if image_urls else None
    run.kwargs["task"] = task or "Describe the image."
    run.kwargs["task_context"] = (
        "The user attached an image. Call `inspect_image` before answering."
    )
    run.kwargs["task_multimodal"] = {"image": image_urls}
    return run
```

With `inject_vars=True`, the tool receives `kwargs["vars"]` and can read the
image URL directly from the execution context.

## 7. **Reasoning**

Reasoning can reach the HTTP response through two paths:

- provider-level reasoning (for example, Groq `gpt-oss`) mapped to
  `message.reasoning_content`
- schema-level reasoning (for example, `ChainOfThought` and `ReAct`) extracted
  and also mapped to `message.reasoning_content`

### 7.1 Provider Reasoning (Best Streaming Support)

Provider reasoning has the most complete streaming support in channels because
it emits incremental reasoning tokens through SSE chunks:

- `delta.reasoning_content` for reasoning tokens
- `delta.content` for answer tokens

!!! warning "Current streaming limitation with tool decisions"
    Today, reasoning streaming is only released after the reasoning phase is
    finalized. The channel cannot safely forward a raw `ModelStreamResponse`
    to the client while the model may still decide between normal content and a
    tool call. That decision is only known after the post-reasoning token
    phase, which is why reasoning delivery is gated at that point.

```python
import msgflux as mf
import msgflux.nn as nn

registry = mf.ChannelRegistry()

@registry.agent(name="groq_reasoning")
class GroqReasoningAgent(nn.Agent):
    model = mf.Model.chat_completion(
        "groq/openai/gpt-oss-120b",
        reasoning_effort="low",
    )
    config = {"reasoning_in_response": True}
```

```bash
curl -sS -N http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "groq_reasoning",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Solve: (37 * 9) - 15"}
    ]
  }'
```

### 7.2 ChainOfThought (Non-streaming)

`ChainOfThought` uses a generation schema. In non-streaming mode, channels
return:

- `message.content` with `final_answer`
- `message.reasoning_content` with the CoT reasoning trace

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import ChainOfThought

registry = mf.ChannelRegistry()

@registry.agent(name="cot_solver")
class CoTSolver(nn.Agent):
    model = "openai/gpt-4.1-mini"
    generation_schema = ChainOfThought
    config = {"reasoning_in_response": True}
```

```bash
curl -sS http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cot_solver",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Solve: 8x + 7 = -23"}
    ]
  }'
```

### 7.3 ReAct + Tools (Non-streaming)

`ReAct` is also schema-based and follows the same non-streaming mapping.
Because `generation_schema` is not stream-compatible, keep `stream=false`.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.generation.reasoning import ReAct
from msgflux.tools.builtin import WebFetch

registry = mf.ChannelRegistry()

@registry.agent(name="openai_react")
class OpenAIReActAgent(nn.Agent):
    model = "openai/gpt-4.1-mini"
    generation_schema = ReAct
    tools = [WebFetch]
    config = {"reasoning_in_response": True}
```

```bash
curl -sS http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai_react",
    "stream": false,
    "messages": [
      {
        "role": "user",
        "content": "Use python.org and tell me the current Python release."
      }
    ]
  }'
```

!!! info "Reasoning mapping notes"
    For schema-based reasoning, channels expose reasoning in
    `message.reasoning_content`; schema-specific fields like `reasoning`,
    `thought`, and `paths` are extracted out of `message.content`.

    Post-processors can override or enrich `reasoning_content` by returning a
    mapping with `reasoning` or `reasoning_content`.

## 8. **Post-processing**

Use `registry.post()` to transform the Agent output before the HTTP response is
encoded. Post-processors receive `(output, context, run)`.

### 8.1 Non-streaming output shaping

Return `None` to keep the original output. Return a new value to replace it.
For non-streaming responses, a mapping can control the final assistant message.

A practical pattern is to centralize output policy (what to expose to users)
in post-processing:

```python
@registry.post("support")
def normalize_support_output(output, context, run):
    expose_reasoning = bool((run.vars or {}).get("debug_reasoning", False))

    # `output` may already be a dict (for example from reasoning-enabled runs).
    if isinstance(output, dict):
        answer = str(output.get("answer") or output.get("response") or "")
        reasoning = output.get("reasoning") or output.get("reasoning_content")
    else:
        answer = str(output)
        reasoning = None

    payload = {"answer": answer.strip()}
    if expose_reasoning and reasoning:
        payload["reasoning"] = str(reasoning)
    return payload
```

The HTTP adapter maps these keys:

- `answer` or `response` -> `message.content`
- `reasoning` or `reasoning_content` -> `message.reasoning_content`

This lets you standardize reasoning output in one place, even when upstream
providers or generation schemas use different internal field names.

!!! warning "Do not stringify stream responses"
    Converting a `ModelStreamResponse` to `str` (or any fully buffered value)
    forces buffering and removes incremental token delivery.

!!! info "Practical rule"
    Use post-processing to reshape **final** non-streaming payloads.
    For streaming requests, treat `ModelStreamResponse` as passthrough.

## 9. **Production Customization**

After the basic server is running, use the registry to configure the HTTP
boundary in one place.

### 9.1 Server Settings

Use `registry.settings(...)` for server-wide HTTP behavior:

```python
registry = mf.ChannelRegistry()

registry.settings(
    title="Support Agents API",
    subtitle="Support and billing agents.",
    description="OpenAI-compatible support and billing agents.",
    max_request_bytes=2 * 1024 * 1024,
    request_timeout_s=30,
    social_debounce_s=0.5,
    social_dedup_ttl_s=300,
    social_rate_limit_message="Too many requests. Try again later.",
    social_unauthorized_message=None,
    social_forbidden_message=None,
    enable_docs=False,
    disable_chat_completions=False,
    cors=True,
    allowed_origins=["https://app.example.com"],
    enable_otel=True,
)
```

`title` and `description` customize the FastAPI/OpenAPI metadata. `title` and
`subtitle` are also returned by `/`. Set `enable_docs=False` to disable
`/docs`, `/redoc`, and `/openapi.json`.

Set `disable_chat_completions=True` when the server should expose only social
webhook routes and not `/v1/chat/completions`. At least one social adapter must
be registered when this is enabled.

`social_debounce_s` buffers multiple social messages for the same `session_id`
for a short window before starting the Agent run. Each new message renews the
timer. This is useful for chat apps where users often send a thought across
multiple short messages.

`social_dedup_ttl_s` keeps recently seen social message ids in a dedupe store.
This protects Slack, Telegram, and future social adapters from platform retries
that resend the same event. Set it to `0` or `None` to disable dedupe.

`social_rate_limit_message`, `social_unauthorized_message`, and
`social_forbidden_message` control optional user-facing social error responses.
By default, rate limits send a short retry-later message, while unauthorized and
forbidden events are dropped silently.

`enable_otel=True` instruments the FastAPI app through the official
`opentelemetry-instrumentation-fastapi` adapter. The package is included in the
`server` extra.

### 9.2 Defaults

Defaults can be global or agent-specific. Request `run_config` still wins over
defaults, and pre-processors continue to run last:

```python
registry.defaults(
    vars={"tenant": "default"},
    model_preference="fast",
    tool_filter={"block": "*"},
)

registry.defaults(
    "support",
    vars={"tenant": "support"},
    tool_filter={"allow": ["search_docs"]},
)
```

Use defaults for policies the server owns. Use request `run_config` when the
caller is allowed to choose the value per request.

### 9.3 Authentication

Use `@registry.auth` to identify the caller. It runs before authorization, rate
limits, and agent execution.

Return a mapping to store the authenticated principal in `context.state`, return
`False` for `401 Unauthorized`, or raise a `ChannelError` subclass directly.

Use `context.channel` when one registry serves more than one boundary. HTTP chat
completions use `context.channel == "http"`. Social adapters use values such as
`context.channel == "social:telegram"`.

```python
@registry.auth
def authenticate(http_request, request, context):
    if context.channel != "http":
        return None

    authorization = http_request.headers.get("authorization")
    if authorization == "Bearer secret":
        return {"api_key": "secret", "tenant": "acme"}

    # Optional fallback for non-OpenAI-compatible clients.
    if http_request.headers.get("x-api-key") == "secret":
        return {"api_key": "secret", "tenant": "acme"}

    return False
```

When using an OpenAI-compatible SDK, pass the key as the client `api_key`. The
SDK sends it as `Authorization: Bearer <api_key>`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8010/v1",
    api_key="secret",
)
```

For raw HTTP requests, send the same header:

```bash
curl http://127.0.0.1:8010/v1/chat/completions \
  -H "Authorization: Bearer secret" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "support",
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

### 9.4 Authorization

Use `@registry.authorize` after authentication when the server needs to enforce
ACL rules, tenant isolation, plan checks, or per-agent access.

Return `False` for `403 Forbidden`. Return a mapping to add derived authorization
state to `context.state`.

```python
@registry.authorize(agent="support")
def authorize_support(request, context):
    principal = context.state["principal"]
    tenant = request.run_config.get("vars", {}).get("tenant")
    if principal["tenant"] != tenant:
        return False
```

Use `auth` to answer "who is calling?" and `authorize` to answer "is this caller
allowed to do this?".

`authorize` receives the same `context.channel`, so one authorizer can branch
between HTTP requests and social messages when needed.

### 9.5 Rate Limits

Rate limits are configured at the boundary and apply to HTTP and social channel
traffic. The default store is in-memory and per Python process, which is useful
for local protection and single-worker deployments. For multi-worker or
distributed deployments, plug a shared store such as Redis, a gateway, or your
billing system.

```python
registry.rate_limit(name="api-key-minute", requests=60, window_s=60, by="api_key")
registry.rate_limit(name="client-minute", requests=300, window_s=60, by="client")
registry.rate_limit(name="tenant-minute", requests=600, window_s=60, by="tenant")
registry.rate_limit(name="service-minute", requests=2000, window_s=60, by="service")
```

`by` accepts:

| Key | Scope |
|---|---|
| `"api_key"` | One bucket per authenticated principal key, `Authorization: Bearer ...`, or `X-API-Key`. |
| `"client"` | One bucket per API key when present, otherwise per IP. |
| `"ip"` | One bucket per client IP. |
| `"tenant"` | One bucket per tenant from auth principal or `run_config.vars`. |
| `"service"` | One global bucket shared by all traffic. |
| callable | Custom bucket key. |

Use `agent="..."` to apply a policy only to one registered agent:

```python
registry.rate_limit(
    name="billing-tenant-minute",
    agent="billing",
    requests=30,
    window_s=60,
    by="tenant",
)
```

For custom bucketing, pass a callable to `by`:

```python
def key_by_model(request, context):
    return request.model


registry.rate_limit(name="model-minute", requests=20, window_s=60, by=key_by_model)
```

When using a distributed store, give every policy a stable `name`. The registry
resolves the bucket identity from `by=...` and sends a `RateLimitBucket` to the
store:

```python
from msgflux.channels import RateLimitDecision


class RedisRateLimitStore:
    def __init__(self, redis):
        self.redis = redis

    async def hit(self, bucket):
        current = await self.redis.incr(bucket.key)
        if current == 1:
            await self.redis.expire(bucket.key, int(bucket.policy.window_s))

        ttl = await self.redis.ttl(bucket.key)
        retry_after_s = ttl if current > bucket.policy.requests else None
        return RateLimitDecision(
            allowed=current <= bucket.policy.requests,
            remaining=max(0, bucket.policy.requests - current),
            retry_after_s=retry_after_s,
            reset_after_s=ttl,
        )


registry.rate_limit_store(RedisRateLimitStore(redis))
registry.rate_limit(
    name="api-key-minute",
    requests=60,
    window_s=60,
    by="api_key",
)
```

Custom stores may also be plain callables and can return `RateLimitDecision`, a
mapping with `allowed`/`retry_after_s`, a boolean, or `None` to allow the
request.

Built-in rate limit errors return `429` with a `Retry-After` header.

### 9.6 Server Hooks

Use lifecycle hooks for startup validation, model warmup, DB/cache setup, and
resource cleanup. `/ready` flips to ready only after startup hooks complete.

```python
@registry.startup
async def warmup(app):
    ...


@registry.shutdown
async def cleanup(app):
    ...
```

Use request hooks for boundary observability without coupling that logic to the
agent implementation:

```python
@registry.on_request_start
def on_start(request, context):
    ...


@registry.on_request_end
def on_end(request, context, run, response, error):
    ...


@registry.on_stream_chunk
def on_chunk(chunk, context, run):
    ...
```

Hook usage by boundary:

| Hook | Best use |
| --- | --- |
| `startup` | Validate env, warm model clients, open DB/cache connections. |
| `shutdown` | Flush telemetry, close DB/cache clients, stop background work. |
| `on_request_start` | Start traces, log sanitized request metadata, initialize per-run state. |
| `on_request_end` | Emit success/error metrics, persist audit records, inspect final `run` and `error`. |
| `on_stream_chunk` | Observe HTTP streaming chunks without coupling to the Agent. |

For social channels, `context.channel` is `social:{name}` and
`context.state["social_context"]` contains the `SocialContext`. Any callable that
receives the channel `context` can send intermediate social messages:

```python
@registry.on_request_start
async def social_start(request, context):
    if context.channel.startswith("social:"):
        await context.state["social_context"].send("Working...")


@registry.post("support")
async def progress_after_model(output, context, run):
    if context.channel == "social:slack":
        await context.state["social_context"].send("Formatting the answer...")
    return output
```

Prefer hooks for observability and coarse progress signals. Prefer
pre-processors for input shaping, such as converting social attachments into
multimodal `messages`. Prefer post-processors for final response formatting or
last-mile notifications.

The HTTP boundary also propagates request metadata. Send `X-Correlation-ID` when
you need one id shared across multiple services; otherwise it defaults to the
request id.

### 9.7 Error Mapping

Error handlers let the server map domain exceptions to predictable HTTP
payloads:

```python
@registry.error_handler(ValueError)
def map_value_error(exc):
    return {
        "message": str(exc),
        "code": "provider_error",
        "type": "agent_error",
        "status_code": 502,
        "headers": {"X-Provider": "internal"},
    }
```

Error payloads include `error.request_id` and `error.correlation_id`, and the
same values are returned as response headers.


## 10. **Social Channels**

Social channels use webhook adapters to convert platform events into normal
agent calls. The first supported adapter is Telegram. See
[Telegram Social Channel](telegram.md) for local tunneling, webhook setup,
routing, and runtime metadata.

## 11. **Serving Custom Modules**

Although channels are designed for `nn.Agent`, today the registry does not
enforce a strict concrete type. In practice, what matters is keeping the same
runtime call contract (`messages`, `stream`, `vars`, `model_preference`,
`tool_filter`, and compatible return shapes).

This allows a composed module that keeps the Agent-like interface but adds
domain logic before delegating to an internal agent. A common case is mixed
text/audio input: detect audio in `vars` (or parse it from `messages`),
transcribe it, and append the transcript to the task context before calling
the underlying agent.

```python
import msgflux as mf
import msgflux.nn as nn

class VoiceSupport(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = nn.Transcriber("openai/whisper-1")
        self.agent = nn.Agent(
            model="openai/gpt-4.1-mini",
            system_message="You are a concise support specialist."
        )

    async def aforward(self, messages, stream=False, vars=None, **kwargs):
        vars = vars or {}
        audio_path = vars.get("audio_path")
        if audio_path:
            transcript = await self.transcriber.acall(audio_path)
            messages = list(messages) + [
                {"role": "user", "content": f"Audio transcript: {transcript}"}
            ]
        return await self.agent.acall(messages=messages, stream=stream, vars=vars, **kwargs)
```

With this pattern, you can keep `messages` empty and pass runtime payload via
`vars`:

```bash
curl -sS http://127.0.0.1:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voice_support",
    "messages": [],
    "run_config": {
      "vars": {
        "audio_path": "./samples/ticket.wav"
      }
    }
  }'
```

!!! info "Why not pre-processing?"
    Pre-processors are intentionally sync and lightweight. They are ideal for
    small request reshaping, not async I/O or external orchestration. For that,
    prefer a composed module with the same Agent-compatible call contract.

## 12. **See Also**

- [Agent](../nn/agent/index.md) - Core module that channels expose over HTTP
- [Message](../nn/message.md) - Structured message passing
- [Model Gateway](../nn/agent/model-gateway.md) - Multi-model routing for `model_preference`
