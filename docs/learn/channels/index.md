# ⇄ Channels

Channels expose msgFlux modules through external interfaces. The first channel
is the HTTP server, which exposes registered `Agent` instances through an
OpenAI-compatible Chat Completions endpoint.

This means any OpenAI SDK client can call specialized msgFlux agents as if they
were regular chat models. The request `model` selects the registered agent name.

## ✦₊⁺ Overview

The channel layer is the external boundary for msgFlux modules. In practice,
it lets you:

- expose agents over HTTP
- reuse OpenAI-compatible clients
- control execution with `run_config`
- adapt incoming requests with pre-processors
- adapt outgoing responses with post-processors

The default channel is the HTTP server, but the design is meant to support
other external interfaces as they are added.

## 1. **Quick Start**

The HTTP server depends on FastAPI and Uvicorn:

```bash
uv pip install "msgflux[server,openai]"
```

If you want to run the example without cloning the repository, download it
first:

```bash
curl -L -o server_streaming_agent.py \
  https://raw.githubusercontent.com/msgflux/msgflux/main/examples/server_streaming_agent.py
```

Start the server:

```bash
uv run --with 'msgflux[server,openai]' msgflux server server_streaming_agent.py --host 127.0.0.1
```

Use `--port` to override the default `8010`:

```bash
uv run --with 'msgflux[server,openai]' msgflux server server_streaming_agent.py --host 127.0.0.1 --port 9000
```

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
    "model": "SupportAgent",
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

@registry.agent()
class SupportAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    system_message = "You are a concise customer support specialist."

@registry.agent(name="billing")
class BillingAgent(nn.Agent):
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

    The same decorator also works with instances:

    ```python
    registry.agent(SupportAgent())
    ```

!!! tip "Custom registry object"

    If your registry object has a different name, pass it explicitly as
    `module.py:object_name`.

After registration, `/agents` returns `SupportAgent` and `billing`:

```bash
curl -s http://127.0.0.1:8010/agents
```

Use `model: "billing"` for the billing agent and `model: "SupportAgent"` for
the support agent. The request examples below show each one in context.

## 3. **HTTP API**

| Name | Method | Description |
|---|---|---|
| `/v1/chat/completions` | `POST` | OpenAI-compatible endpoint that runs a registered agent. |
| `/health` | `GET` | Basic status check for the server. |
| `/agents` | `GET` | Lists the registered agent names. |

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
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
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
    run_config={"model_preference": "fast"},
)
```

You can still override it per request through `extra_body`:

```python
response = await model.acall(
    [{"role": "user", "content": "Analyze this carefully."}],
    extra_body={
        "run_config": {
            "model_preference": "quality",
        }
    },
)
```

## 6. **msgFlux Client**

The `msgflux` model provider is a thin OpenAI-compatible client for a msgFlux
server. It uses `MSGFLUX_BASE_URL` when set, otherwise it defaults to
`http://127.0.0.1:8010/v1`.

```python
import msgflux as mf

mf.load_dotenv()

model = mf.Model.chat_completion(
    "msgflux/support",
)
response = await model.acall(
    [{"role": "user", "content": "My order A1002 still has not arrived."}],
    stream=True,
    run_config={
        "vars": {
            "tenant": "acme",
            "tier": "priority",
        },
    },
)

async for chunk in response.consume():
    print(chunk, end="", flush=True)
```

## 7. **Pre-processing**

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
@registry.pre("billing")
def force_billing_vars(_request, _context, run):
    return {
        "messages": run.messages,
        "vars": {
            **run.vars,
            "department": "billing",
        },
    }
```

Returning a mapping replaces the provided fields. Mutate `run` in place when
you want to merge with the current execution state.

### 7.1 Convert Messages to Task

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

### 7.2 Convert OpenAI Image Content to `task_multimodal`

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

## 8. **Post-processing**

Use `registry.post()` to transform the Agent output before the HTTP response is
encoded. Post-processors receive `(output, context, run)`.

```python
@registry.post("support")
def add_support_footer(output, _context, _run):
    if not isinstance(output, str):
        return output
    return output + "\n\nIf this does not solve it, contact support."
```

Return `None` to keep the original output. Return a new value to replace it.
For non-streaming responses, a mapping can control the final assistant message:

```python
@registry.post("billing")
def normalize_billing_output(output, context, _run):
    return {
        "response": str(output),
        "reasoning_content": f"Handled by agent `{context.agent_name}`.",
    }
```

The adapter maps `response` or `answer` to `message.content`, and maps
`reasoning` or `reasoning_content` to `message.reasoning_content`.

For streaming responses, the Agent usually returns a `ModelStreamResponse`.
Avoid converting it into a string in post-processing unless you intentionally
want to buffer the stream and lose incremental token delivery.

## 9. **See Also**

- [Agent](../nn/agent/index.md) - Core module that channels expose over HTTP
- [Message](../nn/message.md) - Structured message passing
- [Model Gateway](../nn/agent/model-gateway.md) - Multi-model routing for `model_preference`
