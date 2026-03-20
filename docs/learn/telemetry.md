# Telemetry

msgFlux integrates with **msgtrace-sdk** — a lightweight wrapper around [OpenTelemetry](https://opentelemetry.io/) from the `msg*` library family — to provide production-grade observability for your AI systems.

All modules, agents, and tools are automatically instrumented. You can also add custom instrumentation to your own code using `Spans`.

---

## ✦₊⁺ Overview

The telemetry pipeline works at two levels:

- **Automatic** — every `Module`, `Agent`, `Tool`, and functional operation emits spans with no extra code.
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
```

Or configure it programmatically at startup:

```python
from msgflux.telemetry.config import configure_msgtrace

configure_msgtrace(enabled=True)
```

---

## 2. **Environment Variables**

### msgtrace-sdk (transport & exporter)

These variables control how traces are collected and exported.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MSGTRACE_TELEMETRY_ENABLED` | `bool` | `false` | Master switch — enable/disable all telemetry |
| `MSGTRACE_EXPORTER` | `str` | `"console"` | Exporter backend: `"console"` or `"otlp"` |
| `MSGTRACE_OTLP_ENDPOINT` | `str` | `"http://localhost:4318"` | OTLP collector endpoint (gRPC/HTTP) |
| `MSGTRACE_SERVICE_NAME` | `str` | `"msgflux"` | Service name shown in your tracing backend |
| `MSGTRACE_SAMPLING_RATIO` | `str` | `None` | Sampling ratio, e.g. `"0.5"` for 50% |
| `MSGTRACE_CAPTURE_PLATFORM` | `bool` | `true` | Attach OS/platform metadata to spans |
| `MSGTRACE_MAX_RETRIES` | `int` | `3` | Max retries on export failure |

### msgflux (what to capture)

Fine-grained control over the data included in spans.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MSGFLUX_TELEMETRY_CAPTURE_TOOL_CALL_RESPONSES` | `bool` | `true` | Include tool return values in spans |
| `MSGFLUX_TELEMETRY_CAPTURE_AGENT_PREPARE_MODEL_EXECUTION` | `bool` | `false` | Capture agent state, system prompt and tool schemas before each LM call |
| `MSGFLUX_TELEMETRY_CAPTURE_STATE_DICT` | `bool` | `false` | Attach the full `state_dict()` of a module to its span |

---

## 3. **Console Exporter (development)**

The default exporter prints spans to stdout — useful during local development.

```bash
export MSGTRACE_TELEMETRY_ENABLED=true
# MSGTRACE_EXPORTER defaults to "console"
```

```python
from msgflux import Spans
from msgflux.nn import Agent, LM

agent = Agent(lm=LM("gpt-4o-mini"), name="MyAgent")
result = agent("What is the capital of France?")
# Span output will be printed to the console
```

---

## 4. **OTLP Exporter (production)**

Send traces to any OpenTelemetry-compatible backend (Jaeger, Tempo, Honeycomb, Datadog, etc.):

```bash
export MSGTRACE_TELEMETRY_ENABLED=true
export MSGTRACE_EXPORTER=otlp
export MSGTRACE_OTLP_ENDPOINT=http://localhost:4318
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
from msgflux.nn import Agent, LM

agent = Agent(lm=LM("gpt-4o-mini"), name="Summarizer")

# Emits: flow > module(Summarizer) > model call
result = agent("Summarize this document...")
```

Each span records:

- Module name and type
- Execution status (`OK` / `ERROR`)
- Exception details on failure
- Full `state_dict()` when `MSGFLUX_TELEMETRY_CAPTURE_STATE_DICT=true`

### Tools

`LocalTool` and `MCPTool` emit spans with:

- Tool name, description, and type
- Tool call ID (for correlation with the LM call)
- Input arguments (JSON-encoded)
- Execution type (`local` or `remote`)
- Protocol (`mcp` for MCP tools)
- Return value when `MSGFLUX_TELEMETRY_CAPTURE_TOOL_CALL_RESPONSES=true`

### Functional API

All operations in `msgflux.nn.functional` are automatically traced:

| Function | Description |
|----------|-------------|
| `map_gather` / `amap_gather` | Map over args and gather results |
| `scatter_gather` / `ascatter_gather` | Scatter inputs and gather outputs |
| `bcast_gather` | Broadcast and gather |
| `Inline` | DSL workflow execution |
| `spawn` | Background execution |

---

## 7. **Programmatic Configuration**

You can configure everything at runtime instead of using environment variables:

```python
from msgflux.telemetry.config import configure_msgtrace

configure_msgtrace(
    enabled=True,
    exporter="otlp",
    otlp_endpoint="http://otel-collector:4318",
    service_name="my-ai-app",
    sampling_ratio="1.0",
    capture_platform=True,
    max_retries=3,
)
```

!!! note
    Call `configure_msgtrace()` before creating any modules or agents to ensure all spans are captured correctly.

---

## 8. **Sampling**

Use `MSGTRACE_SAMPLING_RATIO` to control the fraction of traces that are recorded:

```bash
# Record 10% of all traces
export MSGTRACE_SAMPLING_RATIO=0.1

# Record everything (default behaviour when unset)
export MSGTRACE_SAMPLING_RATIO=1.0
```

---

## 9. **Reducing Span Payload Size**

For high-throughput systems, disable verbose captures to keep span sizes small:

```bash
# Disable tool response capture
export MSGFLUX_TELEMETRY_CAPTURE_TOOL_CALL_RESPONSES=false

# Disable full agent state capture
export MSGFLUX_TELEMETRY_CAPTURE_AGENT_PREPARE_MODEL_EXECUTION=false

# Disable module state dict capture (already off by default)
export MSGFLUX_TELEMETRY_CAPTURE_STATE_DICT=false
```
