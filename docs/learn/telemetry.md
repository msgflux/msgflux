# Telemetry

msgFlux integrates with **msgtrace-sdk** — a lightweight wrapper around [OpenTelemetry](https://opentelemetry.io/) from the `msg*` library family — to provide production-grade observability for your AI systems.

All modules, agents, and tools are automatically instrumented. You can also add custom instrumentation to your own code using `Spans`.

---

## ✦₊⁺ Overview

The telemetry pipeline works at two levels:

- **Automatic** — modules, agents, tools, functional operations, and model HTTP requests emit spans with no extra code.
- **Manual** — use `Spans.instrument()` / `Spans.ainstrument()` to trace your own functions.

Telemetry is **disabled by default** and has zero overhead when turned off.

```python
from msgflux import Spans
```

---

## 1. **Enabling Telemetry**

Set the environment variable before running your application:

```bash
export MSGTRACE_TELEMETRY_ENABLED=true
export MSGTRACE_EXPORTER=console
```

Or configure it programmatically at startup:

```python
from msgflux.telemetry.config import configure_msgtrace

configure_msgtrace(enabled=True, exporter="console")
```

---

## 2. **Environment Variables**

### msgtrace-sdk (transport & exporter)

These variables control how traces are collected and exported.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MSGTRACE_TELEMETRY_ENABLED` | `bool` | `false` | Master switch — enable/disable all telemetry |
| `MSGTRACE_EXPORTER` | `str` | `"otlp"` | Exporter backend: `"console"` or `"otlp"` |
| `MSGTRACE_OTLP_ENDPOINT` | `str` | `"http://localhost:8000/api/v1/traces/export"` | Full OTLP HTTP trace endpoint |
| `MSGTRACE_SERVICE_NAME` | `str` | `"msgtrace-app"` | Service name shown in your tracing backend |
| `MSGTRACE_SAMPLING_RATIO` | `str` | `None` | Reserved setting; the current SDK does not apply it |
| `MSGTRACE_CAPTURE_PLATFORM` | `bool` | `true` | Attach OS/platform metadata to spans |
| `MSGTRACE_MAX_RETRIES` | `int` | `3` | Reserved setting; the current SDK does not apply it |

These are the defaults used by the installed msgtrace SDK at export time.
`configure_msgtrace()` applies the values you pass to it; it does not apply
unspecified defaults from `MsgTraceSettings` to the exporter.

### msgflux (what to capture)

Fine-grained control over the data included in spans.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MSGFLUX_TELEMETRY_CAPTURE_TOOL_CALL_RESPONSES` | `bool` | `true` | Include tool return values in spans |
| `MSGFLUX_TELEMETRY_CAPTURE_MODEL_OUTPUT` | `bool` | `true` | Include generated text and tool calls in model spans |
| `MSGFLUX_TELEMETRY_CAPTURE_AGENT_PREPARE_MODEL_EXECUTION` | `bool` | `false` | Capture agent state, system prompt and tool schemas before each LM call |
| `MSGFLUX_TELEMETRY_CAPTURE_STATE_DICT` | `bool` | `false` | Attach the full `state_dict()` of a module to its span |

---

## 3. **Console Exporter (development)**

The console exporter prints spans to stdout — useful during local development.

```bash
export MSGTRACE_TELEMETRY_ENABLED=true
export MSGTRACE_EXPORTER=console
```

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Spans

model = mf.ChatCompletion("openai/gpt-4.1-mini")
agent = nn.Agent("MyAgent", model)
result = agent("What is the capital of France?")
# Span output will be printed to the console
```

---

## 4. **OTLP Exporter (production)**

Send traces to any OpenTelemetry-compatible backend (Jaeger, Tempo, Honeycomb, Datadog, etc.):

```bash
export MSGTRACE_TELEMETRY_ENABLED=true
export MSGTRACE_EXPORTER=otlp
export MSGTRACE_OTLP_ENDPOINT=http://localhost:4318/v1/traces
export MSGTRACE_SERVICE_NAME=my-ai-app
```

### Quick start with Jaeger

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

Then open `http://localhost:16686` to browse traces.

Set `MSGTRACE_OTLP_ENDPOINT` to the full OTLP HTTP traces URL, including
`/v1/traces`. The msgtrace SDK passes this value directly to its HTTP exporter.

---

## 5. **Instrumenting Your Own Code**

Use `Spans.instrument()` to add tracing to any function without changing its signature.

### Sync functions

```python
from msgflux import Spans

@Spans.instrument()
def fetch_documents(query: str) -> list[str]:
    # This function now emits a span automatically
    ...
```

### Async functions

```python
@Spans.ainstrument()
async def embed_and_store(texts: list[str]) -> None:
    ...
```

### Custom span attributes

Pass arbitrary key/value pairs to attach metadata to the span:

```python
@Spans.ainstrument(attributes={"pipeline.stage": "retrieval", "index": "products"})
async def retrieve(query: str) -> list[str]:
    ...
```

### Context manager (manual span)

For finer control, use the context manager API directly:

```python
from msgflux import Spans
from opentelemetry.trace import Status, StatusCode

with Spans.init_flow("my-pipeline") as span:
    try:
        result = run_pipeline()
        span.set_status(Status(StatusCode.OK))
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        raise
```

Async version:

```python
async with Spans.ainit_flow("my-pipeline") as span:
    result = await run_pipeline_async()
    span.set_status(Status(StatusCode.OK))
```

---

## 6. **Automatic Instrumentation**

### Modules and Agents

Every call to a `Module` subclass automatically creates a span. When the module is the entry point (no parent span), a **flow** span is created; nested modules get **module** spans.

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.ChatCompletion("openai/gpt-4.1-mini")
agent = nn.Agent("Summarizer", model)

# Emits: flow > module(Summarizer) > model call
result = agent("Summarize this document...")
```

Each span records:

- Module name and type
- Execution status (`OK` / `ERROR`)
- Exception details on failure
- Full `state_dict()` when `MSGFLUX_TELEMETRY_CAPTURE_STATE_DICT=true`

### Model requests

Calls through the shared model HTTP transport emit one OpenTelemetry client span
per logical request. The span includes retries and, for streamed responses, stays
open until the stream finishes or is closed. This also applies when a model is
called directly, outside an agent:

```python
import msgflux as mf

model = mf.Model.chat_completion("openai/gpt-4.1-mini")
response = model("Summarize this document", stream=False)
# Emits a client span named "chat gpt-4.1-mini".
```

Ollama's default native `/api/chat` mode also uses the shared HTTP transport.
It emits the same GenAI span attributes, including structured JSON output,
tool calls, finish reasons, and token usage. Both sync and async calls,
including streams, are covered. OpenAI Responses API calls use this transport
too. Anthropic's native Messages API also uses it and records text, tool use,
stop reason, and token usage, including cache reads and writes.

The span records the GenAI operation, provider, requested model, available
response model and ID, finish reasons, and token usage using `gen_ai.*`
attributes. Generated text and tool calls are recorded in `gen_ai.output.messages`
as a JSON array of assistant messages with text or `tool_call` parts. Each tool
call part includes its name, ID when available, and parsed arguments. Streaming
text and tool call arguments are assembled before the span ends. Chat Completions
stop reasons and Responses API terminal status are recorded in
`gen_ai.response.finish_reasons`; Responses API also records
`gen_ai.response.status`. Request parameters such as temperature are included
when present. When the request sends `reasoning_effort` or `reasoning.effort`,
the span records the exact value in `gen_ai.request.reasoning.level` (for
example, `low`). HTTP failures mark the span as an error.

Ollama's native `/api/chat` uses `think` instead of `reasoning_effort`. Named
levels such as `think="high"` use `gen_ai.request.reasoning.level`. Boolean
`think` values are recorded as `ollama.request.think`, preserving whether
thinking was enabled or disabled without treating a boolean as a level.

Anthropic adaptive thinking uses `output_config.effort` as the requested
reasoning level. A disabled thinking request records `none`; a fixed thinking
budget records `anthropic.request.thinking.budget_tokens`. The common request
attributes also include supported limits, sampling settings, stop sequences,
and whether the request is streamed. Anthropic's native `stop_reason` is kept
in `gen_ai.response.finish_reasons`.

Model output can contain sensitive data and increase span size. Set
`MSGFLUX_TELEMETRY_CAPTURE_MODEL_OUTPUT=false` to omit generated text and tool
call arguments while retaining stop reasons and token usage. Prompts,
authorization headers, and request bodies are not added to model spans.
Streaming token usage is recorded
when the provider includes it in a stream event.

For example, `gen_ai.operation.name=chat`, `gen_ai.provider.name=openai`,
`gen_ai.request.model=gpt-4.1-mini`, and `gen_ai.usage.input_tokens=42` can be
queried in an OpenTelemetry collector or trace backend. Other model operations
such as embeddings use the corresponding operation name.

In Jaeger, open a `chat <model>` span and expand its attributes to see
`gen_ai.output.messages` and `gen_ai.response.finish_reasons`.

### Tools

`LocalTool` and `MCPTool` emit spans with:

- Tool name, description, and type
- Tool call ID (for correlation with the LM call)
- Input arguments (JSON-encoded)
- Execution type (`local` or `remote`)
- Protocol (`mcp` for MCP tools)
- Return value when `MSGFLUX_TELEMETRY_CAPTURE_TOOL_CALL_RESPONSES=true`

### Functional API

Fan-out operations in `msgflux.nn.functional` emit a span around the full gather:

| Function | Description |
|----------|-------------|
| `map_gather` / `amap_gather` | Map over args and gather results |
| `scatter_gather` / `ascatter_gather` | Scatter inputs and gather outputs |
| `bcast_gather` / `abcast_gather` | Broadcast and gather |

These spans record `msgflux.functional.task_count` and
`msgflux.functional.failed_tasks`; synchronous helpers also record
`msgflux.functional.timeout_seconds` when a timeout is supplied. Child spans
created by the dispatched work are nested under the fan-out span when execution
context is propagated. Wait and detached dispatch helpers do not emit their own
spans; the work they invoke keeps its existing module, tool, and model spans.

---

## 7. **Programmatic Configuration**

You can configure everything at runtime instead of using environment variables:

```python
from msgflux.telemetry.config import configure_msgtrace

configure_msgtrace(
    enabled=True,
    exporter="otlp",
    otlp_endpoint="http://otel-collector:4318/v1/traces",
    service_name="my-ai-app",
    capture_platform=True,
)
```

!!! note
    Call `configure_msgtrace()` before creating any modules or agents to ensure all spans are captured correctly.

---

## 8. **Sampling**

The current `msgtrace-sdk` stores `MSGTRACE_SAMPLING_RATIO` but does not attach
it to the OpenTelemetry tracer provider. Until the SDK supports it, configure
sampling in the tracer provider used by your application.

---

## 9. **Reducing Span Payload Size**

For high-throughput systems, disable verbose captures to keep span sizes small:

```bash
# Disable tool response capture
export MSGFLUX_TELEMETRY_CAPTURE_TOOL_CALL_RESPONSES=false

# Disable generated model text capture
export MSGFLUX_TELEMETRY_CAPTURE_MODEL_OUTPUT=false

# Disable full agent state capture
export MSGFLUX_TELEMETRY_CAPTURE_AGENT_PREPARE_MODEL_EXECUTION=false

# Disable module state dict capture (already off by default)
export MSGFLUX_TELEMETRY_CAPTURE_STATE_DICT=false
```
