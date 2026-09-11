# Chat Completion

The `chat_completion` model is the most versatile model type for natural language interactions. It processes messages in conversational format and supports advanced features like multimodal input/output, structured generation, and tool calling.

!!! info "Dependencies"
    Chat completion providers use the HTTPX2 transport included with msgFlux:

    === "uv"
        ```bash
        uv add msgflux
        ```

    === "pip"
        ```bash
        pip install msgflux
        ```

    See [Dependency Management](../../dependency-management.md) for the complete provider matrix.

    OpenAI chat, audio, image, embedding, and moderation models all use the
    included HTTPX2 transport. The OpenAI Python SDK is not required.

## ✦₊⁺ Overview

--8<-- "docs/_includes/init_chat_completion_model.md"

Chat completion models are stateless - they don't maintain conversation history between calls. You must provide all context (previous messages, system prompt, etc.) in each request.

### Quick Start

???+ example

    ```python
    import msgflux as mf

    # mf.set_envs(OPENAI_API_KEY="...")

    # Create model
    model = mf.Model.chat_completion("openai/gpt-5.6-luna")

    response = model("Hello!")
    print(response.consume())
    ```

## 1. **Model Initialization**

### 1.1 **Basic Parameters**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion(
        "openai/gpt-5.6-luna",
        # --- Generation ---
        temperature=0.7,               # Randomness (0-2)
        max_tokens=1000,               # Max output tokens (includes reasoning tokens)
        top_p=0.9,                     # Nucleus sampling (alternative to temperature)
        stop=["\n\n"],                 # Stop sequences (up to 4)
        # --- Reasoning ---
        reasoning_effort="medium",     # Values depend on the selected model
        speed="fast",                  # "fast", "ultrafast", or "nitro"
        enable_thinking=True,          # Enable extended model reasoning
        return_reasoning=True,         # Include reasoning content in response
        reasoning_max_tokens=4096,     # OpenRouter only: max tokens reserved for reasoning/thinking
        reasoning_in_tool_call=True,   # Preserve reasoning context across tool calls
        # --- Output ---
        modalities=["text"],           # ["text"], ["audio"] or ["text", "audio"]
        audio={"voice": "alloy", "format": "mp3"},  # Audio output config
        verbosity="medium",            # Response verbosity: "low", "medium", "high"
        logprobs=True,                 # Include token logprobs in metadata
        top_logprobs=2,                # Return 2 alternatives per token
        parallel_tool_calls=True,      # Allow model to call multiple tools in parallel
                verbose=False,                 # Print raw output before transformation
        # --- Search ---
        web_search_options={},         # Web search config (OpenAI / OpenRouter only)
        extra_body={},                 # Provider-specific OpenAI-compatible extensions
        prompt_cache_retention="24h",  # OpenAI only: "in_memory" or "24h"
        store=None,                    # Provider storage preference; None keeps its default
        # --- Infrastructure ---
        api_mode="chat_completions",   # Wire protocol used by the provider
        base_url="https://api.openai.com/v1",  # Override provider API endpoint
        context_length=128000,         # Override maximum context window
        enable_cache=True,             # Cache identical API responses in-process
        cache_size=128,                # Max number of cached entries
        retry=None,                    # Custom tenacity retry configuration
    )
    ```

Use `extra_body` for provider-specific request body fields supported by
OpenAI-compatible APIs but not modeled directly by msgFlux. The HTTP transport
expands this container into the JSON request body.

You can also pass provider-specific fields directly as keyword arguments
in the model constructor, or per request with `model(...)` / `model.acall(...)`.
msgFlux merges these kwargs into `extra_body`.

```python
import msgflux as mf

model = mf.Model.chat_completion(
    "brave/brave",
    extra_body={"enable_citations": True},
    enable_entities=True,    # merged into extra_body
    enable_research=False,   # merged into extra_body
)

# Per-request extensions are merged with constructor defaults.
response = model(
    "Search the latest release notes.",
    extra_body={"country": "US"},
    enable_citations=True,
)
```

### 1.2 **API Mode**

`api_mode` names the wire protocol, independently of the provider. Most compatible
providers default to `"chat_completions"`. The concrete OpenAI provider defaults
to `"responses"`, following OpenAI's current recommendation for reasoning,
tool-calling, and multi-turn workflows. Groq and vLLM also support Responses
behind the same `Model.chat_completion` frontend:

```python
import msgflux as mf

chat_model = mf.Model.chat_completion(
    "openrouter/openai/gpt-oss-120b", api_mode="chat_completions"
)

responses_model = mf.Model.chat_completion(
    "groq/openai/gpt-oss-20b",
    api_mode="responses",
    reasoning_effort="low",
)

# The call interface does not change.
response = responses_model(
    "Is SKU-1842 available?",
    system_prompt="Answer in one sentence.",
)
print(response.consume())
```

To use an OpenAI feature that remains specific to Chat Completions, select it
explicitly:

```python
audio_model = mf.Model.chat_completion(
    "openai/gpt-audio",
    api_mode="chat_completions",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "mp3"},
)
```

Keeping this identity explicit prevents provider-only state from being replayed
through an incompatible protocol. Anthropic Messages or Google Interactions can
later add their own modes without changing the canonical `ChatMessages` history.

For `api_mode="responses"`, msgFlux converts parameters at the Model boundary:

| Chat-completion frontend | Responses request |
|---|---|
| `messages` / `ChatMessages` | `input` items |
| `system_prompt` | leading system input |
| `max_tokens` | `max_output_tokens` |
| `reasoning_effort` | `reasoning.effort` |
| `verbosity` | `text.verbosity` |
| `generation_schema` | `text.format` |
| function tools and named `tool_choice` | flattened Responses function tools |
| `web_search_options` | a hosted `web_search` tool |

Reasoning text or summaries stay canonical in history. Native item identity,
status, encrypted content, or signatures are stored sparsely under
`provider_state` and replayed only when `provider`, `api_mode`, and codec match.
This lets OpenAI replay its opaque reasoning item while Groq and vLLM rebuild a
clear-text `reasoning_text` item without storing a second copy of the text.
Parameters without a Responses equivalent (`stop`, output `modalities`, and
`audio`) fail at initialization instead of being silently discarded.

OpenAI requests an automatic summary and, with `reasoning_in_tool_call=True`,
encrypted reasoning content. Groq and vLLM do not receive these OpenAI-only
fields. Groq returns reasoning in `output[].content[].reasoning_text`; vLLM does
the same when its server is started with a supported reasoning parser.

GPT-5.6 accepts `reasoning_effort` values `"none"`, `"low"`, `"medium"`,
`"high"`, `"xhigh"`, and `"max"`. A reasoning item may contain only opaque
state and no textual summary; msgFlux replays that item with `summary=[]`, as
required by the GPT-5.6 Responses contract. When GPT-5.6 returns separate
`commentary` and `final_answer` message phases, ordinary generations select the
final-answer phase while `ToolFlowControl` selects commentary as the actionable
trajectory. Both native messages remain in `ChatMessages`, including their
`phase`, so subsequent manual-history requests can replay every Responses output
item without synthesizing or duplicating the selected answer.

### 1.3 **Storage and ZDR preference**

`store` is optional and defaults to `None`, so msgFlux does not impose a data
retention policy when the application does not choose one. Its wire mapping is
provider-specific:

| msgFlux | OpenAI | OpenRouter |
|---|---|---|
| `store=None` | omit `store` | omit `provider.zdr` |
| `store=False` | `store=false` | `provider.zdr=true` |
| `store=True` | `store=true` | `provider.zdr=false` |

```python
import msgflux as mf

openai_model = mf.Model.chat_completion(
    "openai/gpt-5.6-luna",
    api_mode="responses",
    store=False,
)

openrouter_model = mf.Model.chat_completion(
    "openrouter/nvidia/nemotron-3.5-lightning:free",
    store=False,
)
```

For OpenAI Responses, omitting `store` uses OpenAI's storage default;
`store=False` disables application-state storage for that Response. It does not
by itself activate organization-level Zero Data Retention or remove the
provider's standard abuse-monitoring retention. OpenAI ZDR must also be enabled
for the organization or project.

OpenRouter expresses the preference as a routing constraint rather than a
Response storage flag. `store=False` therefore restricts the request to ZDR
endpoints through `provider.zdr=true`. `store=True` removes that per-request
restriction but cannot override a stricter account or guardrail policy.

Ollama defaults to its native `api_mode="ollama_chat"`, sent directly to
`/api/chat` with `httpx2`. This mode supports Ollama's `thinking` history,
native tools, images, structured `format`, and streaming. Select
`api_mode="chat_completions"` to retain the OpenAI-compatible `/v1` transport:

```python
import msgflux as mf

native = mf.Model.chat_completion(
    "ollama/gpt-oss:20b",
    enable_thinking="medium",
)
compatible = mf.Model.chat_completion(
    "ollama/qwen3:0.6b",
    api_mode="chat_completions",
)
```

For native Ollama, `enable_thinking` accepts `True`/`False` for models with a
boolean thinking switch. Models such as GPT-OSS accept the explicit levels
`"low"`, `"medium"`, and `"high"`, which msgFlux forwards unchanged as the
native `think` field. See [Ollama Thinking](https://docs.ollama.com/capabilities/thinking).

For vLLM, configure the server-side parser for the served model, then select the
Responses transport in msgFlux:

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --reasoning-parser deepseek_r1
```

```python
import msgflux as mf

model = mf.Model.chat_completion(
    "vllm/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    api_mode="responses",
)

response = model("What is 19 × 23?")
print(response.consume_reasoning())
print(response.consume())
```

See the provider references for the exact server/model capabilities:
[Groq Responses reasoning](https://console.groq.com/docs/responses-api#reasoning),
[Groq Chat reasoning](https://console.groq.com/docs/reasoning#accessing-reasoning-content),
and [vLLM reasoning outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs/).

### 1.4 **Request Speed**

`speed` expresses a provider-neutral preference for a faster request path. The
active `ChatModelExtension` translates it after the API adapter has prepared the
request and before the transport sends it:

| msgFlux | OpenAI | OpenRouter |
|---|---|---|
| `speed="fast"` | `service_tier="fast"` | `speed="fast"` for Claude; otherwise route through `:nitro` |
| `speed="ultrafast"` | `service_tier="ultrafast"` for supported models | warning and ignore |
| `speed="nitro"` | warning and ignore | append the `:nitro` routing variant |
| `speed=None` | no explicit tier | no explicit speed routing |

```python
import msgflux as mf

model = mf.Model.chat_completion(
    "openai/gpt-5.6-sol",
    speed="fast",
)

response = model("Summarize why connection pooling reduces latency.")

model.set_speed("ultrafast")
faster_response = model("Now give a one-sentence summary.")

model.set_speed(None)
```

The setting affects future requests and is serialized with the model. Unsupported
provider/model combinations emit a warning and keep the previous setting. In
OpenRouter, explicit `nitro` cannot be combined with another model suffix such
as `:free`; msgFlux warns and ignores that update.

The requested and effective values are kept separately because providers may
report a different tier after routing:

```python
print(response.metadata.model.get("requested_speed"))
print(response.metadata.model.get("effective_speed"))
```

### 1.5 **Custom Chat Extensions**

`chat_extensions` adds request transformations after the active API adapter has
prepared its wire request and before the transport sends it. Provider extensions
remain first in the pipeline, and extension names must be unique:

```python
import msgspec
import msgflux as mf
from msgflux.models import ChatModelExtension


class RequestLabelExtension(ChatModelExtension):
    name = "request_label"

    def prepare_request(self, owner, request, context):
        config = owner.get_chat_extension_config(self.name, {})
        if not config:
            return request

        params = dict(request.params)
        headers = dict(params.get("extra_headers") or {})
        headers["X-Request-Label"] = config["label"]
        params["extra_headers"] = headers
        return msgspec.structs.replace(request, params=params)


model = mf.Model.chat_completion(
    "openai/gpt-5.6-luna",
    chat_extensions=[RequestLabelExtension()],
)
model.configure_chat_extension("request_label", label="inventory-audit")

response = model("Explain why SKU-1842 reservations overlapped.")
```

The `ChatRequestContext` received by `prepare_request()` identifies the
`operation` (`"generate"`, `"warmup"`, `"compact"`, or `"token_count"`) and
the active `api_mode`. An extension can therefore avoid adding generation-only
fields to token-count requests.

Extensions can also be changed after initialization:

```python
extension = RequestLabelExtension()
runtime_model = mf.Model.chat_completion("openai/gpt-5.6-luna")

runtime_model.register_chat_extension(extension)
runtime_model.configure_chat_extension("request_label", label="retry-audit")
runtime_model.remove_chat_extension("request_label")
```

Registering an existing name raises by default. Pass `replace=True` to replace
that extension while preserving its position in the request pipeline. Removing
an extension also removes its stored configuration.

Extension objects are runtime behavior and are not serialized. Their model-owned
configuration is serialized, so an application can register the same custom
extension after restoring a model. Built-in provider extensions are restored
automatically from the provider class.

## 2. **System Prompt**

The `system_prompt` parameter sets the model's overarching behavior and role before any user messages. It is a convenience shorthand: when provided, msgFlux automatically inserts a `system` message at the beginning of the conversation, so you don't have to do it manually in the messages list.

???+ example

    === "Basic Usage"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        response = model(
            messages="What is recursion?",
            system_prompt="You are a computer science teacher. Explain concepts clearly with short examples."
        )

        print(response.consume())
        ```

    === "Persona and Tone"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        # Customer support assistant
        response = model(
            messages="My order hasn't arrived yet.",
            system_prompt=(
                "You are a friendly customer support agent for an online store. "
                "Always be empathetic, offer concrete next steps, and avoid technical jargon."
            )
        )

        print(response.consume())
        ```

    === "Format Instructions"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        response = model(
            messages="Summarize the water cycle.",
            system_prompt=(
                "Always respond in bullet points. "
                "Use at most 5 bullets per answer. "
                "Be concise."
            )
        )

        print(response.consume())
        ```

!!! note
    If your messages list already contains a `{"role": "system", ...}` entry, passing `system_prompt` will insert a **second** system message at position 0. Avoid mixing both approaches in the same call.

## 3. **Response Caching**

Response caching avoids redundant API calls by caching identical requests:

???+ example

    === "Enabling Cache"

        ```python
        import msgflux as mf

        # Enable cache on initialization
        model = mf.Model.chat_completion(
            "openai/gpt-4.1-mini",
            enable_cache=True,   # Enable caching
            cache_size=128       # Cache up to 128 responses
        )

        # First call - hits API
        response1 = model(messages=[{"role": "user", "content": "Hello"}])
        print(response1.consume())

        # Second identical call - returns cached response (no API call)
        response2 = model(messages=[{"role": "user", "content": "Hello"}])
        print(response2.consume())

        # Different call - hits API again
        response3 = model(messages=[{"role": "user", "content": "Hi"}])
        print(response3.consume())
        ```

    === "Cache Statistics"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion(
            "openai/gpt-4.1-mini",
            enable_cache=True,
            cache_size=128
        )

        # Make some calls
        model(messages=[{"role": "user", "content": "Test 1"}])
        model(messages=[{"role": "user", "content": "Test 1"}])  # Cache hit
        model(messages=[{"role": "user", "content": "Test 2"}])

        # Check cache stats
        if model._response_cache:
            stats = model._response_cache.cache_info()
            print(stats)
            # {
            #     'hits': 1,
            #     'misses': 2,
            #     'maxsize': 128,
            #     'currsize': 2
            # }

            # Clear cache
            model._response_cache.cache_clear()
        ```

### 3.1 **Cache Behavior**

The cache is sensitive to:

- Message content
- System prompt
- Temperature and sampling parameters
- Generation schema
- Tool definitions

Changing any of these creates a new cache entry.

### 3.2 **Prompt Cache Warmup**

Some providers cache long prompt prefixes server-side. For agent-level warmup, use `Agent.warmup_system_prompt()`. It sends the rendered system prompt and tool schemas without task messages, chat history, or checkpoint state.

See [Agent — Prompt Cache Warmup](../nn/agent/prompt-cache.md) for the recommended usage.

OpenAI-compatible chat providers use `warmup_max_tokens=1` by default:

```python
model = mf.Model.chat_completion(
    "openai/gpt-4.1-mini",
    warmup_max_tokens=1,
)
```

This is separate from `enable_cache`: response caching is local/in-process, while prompt cache warmup targets the provider's prompt cache.

## 4. **Message Formats**

???+ example

    === "Simple String"

        ```python
        response = model(
            messages="What is Python?",
            system_prompt="You are a programming expert."
        )
        ```

    === "ChatML Format"

        ```python
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Tell me a joke."}
        ]

        response = model(messages=messages)
        ```

    === "ChatBlock Format"

        ```python
        import msgflux as mf

        # Text only
        messages = [
            mf.ChatBlock.user("What's in this image?")
        ]

        # With images
        messages = [
            mf.ChatBlock.user(
                "Describe this image",
                media=mf.ChatBlock.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Above_Gotham.jpg/1280px-Above_Gotham.jpg")
            )
        ]

        # Multiple media
        messages = [
            mf.ChatBlock.user(
                "Compare these images",
                media=[
                    mf.ChatBlock.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"),
                    mf.ChatBlock.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Golde33443.jpg/1200px-Golde33443.jpg")
                ]
            )
        ]

        response = model(messages=messages)
        ```

    === "ChatMessages"

        ```python
        import msgflux as mf

        messages = mf.ChatMessages(thread_id="support_42")
        messages.add_system("You are a concise support assistant.")
        messages.add_user("My invoice total looks wrong.")
        messages.add_assistant("I can help check it.")
        messages.add_user("The tax line seems duplicated.")

        response = model(messages=messages)
        print(response.consume())
        ```

## 5. **Async Support**

Async version for concurrent operations:

???+ example

    ```python
    import msgflux as mf
    import asyncio

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    response = await model.acall(
        messages=[{"role": "user", "content": prompt}]
    )
    return response.consume()
    ```

## 6. **Streaming**

Stream tokens as they're generated:

???+ example

    === "Pull One Chunk"

        Use `next_chunk()` when your application needs to decide exactly when the next token is delivered, such as a TUI that can pause rendering or synchronize text with another stream:

        ```python
        import asyncio
        import msgflux as mf

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        async def main():
            response = await model.acall(
                messages=[{"role": "user", "content": "Count to 10"}],
                stream=True
            )

            while True:
                chunk = await response.next_chunk()
                if chunk is None:
                    break

                print(chunk, end="", flush=True)
                await asyncio.sleep(0.05)  # application-controlled pacing

        asyncio.run(main())
        ```

    === "Basic Streaming"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        response = model(
            messages=[{"role": "user", "content": "Count to 10"}],
            stream=True
        )

        async for chunk in response.consume():
            print(chunk, end="", flush=True)
        ```

    === "Async Streaming"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        response = await model.acall(
            messages=[{"role": "user", "content": "Write a short poem"}],
            stream=True
        )

        async for chunk in response.consume():
            print(chunk, end="", flush=True)
        ```

    === "FastAPI"

        ```python
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        import msgflux as mf

        app = FastAPI()
        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        @app.get("/chat")
        async def chat(query: str):
            response = model(
                messages=[{"role": "user", "content": query}],
                stream=True
            )

            return StreamingResponse(
                response.consume(),
                media_type="text/plain"
            )
        ```

## 7. **Multimodal Inputs**

Modern models support multiple input modalities:

???+ example

    === "Image Understanding"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
                    }
                }
            ]
        }]

        response = model(messages=messages)
        print(response.consume())
        ```

    === "ChatBlock Helper"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        messages = [
            mf.ChatBlock.user(
                "Describe this image",
                media=mf.ChatBlock.image("https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg")
            )
        ]

        response = model(messages=messages)
        print(response.consume())
        ```

    === "ChatMessages"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        messages = mf.ChatMessages(thread_id="image_review_42")
        messages.add_user_multimodal(
            text="Describe this image",
            media={
                "image": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
            },
        )

        response = model(messages=messages)
        print(response.consume())
        ```

    === "Base64"

        ```python
        import msgflux as mf
        import base64

        # Read and encode image
        with open("image.jpg", "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        }]

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")
        response = model(messages=messages)
        ```

## 8. **Structured Generation**

Generate structured data conforming to a schema:

???+ example

    === "Basic Schema"

        ```python
        import msgflux as mf
        from msgspec import Struct

        class CalendarEvent(Struct):
            name: str
            date: str
            participants: list[str]

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        response = model(
            messages="Alice and Bob are going to a science fair on Friday.",
            system_prompt="Extract the event information.",
            generation_schema=CalendarEvent
        )

        event = response.consume()
        print(event)
        # {'name': 'science fair', 'date': 'Friday', 'participants': ['Alice', 'Bob']}
        ```

    === "Nested Schemas"

        ```python
        import msgflux as mf
        from msgspec import Struct

        class Address(Struct):
            street: str
            city: str
            country: str

        class Person(Struct):
            name: str
            age: int
            address: Address

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        response = model(
            messages="John Doe, 30 years old, lives at 123 Main St, New York, USA.",
            system_prompt="Extract person information.",
            generation_schema=Person
        )

        person = response.consume()
        print(person)
        # {
        #     'name': 'John Doe',
        #     'age': 30,
        #     'address': {
        #         'street': '123 Main St',
        #         'city': 'New York',
        #         'country': 'USA'
        #     }
        # }
        ```

    === "With System Prompt"

        `system_prompt` and `generation_schema` compose naturally: the system prompt shapes the model's role while the schema enforces the output structure.

        ```python
        import msgflux as mf
        from msgspec import Struct

        class Sentiment(Struct):
            label: str   # "positive", "neutral", or "negative"
            score: float # confidence from 0.0 to 1.0

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        response = model(
            messages="I absolutely loved the new product update!",
            system_prompt="You are a sentiment analysis engine. Classify the user's message.",
            generation_schema=Sentiment
        )

        result = response.consume()
        print(result)
        # {'label': 'positive', 'score': 0.98}
        ```

    === "Planning Schemas"

        ```python
        import msgflux as mf

        # Access built-in planning schemas
        ChainOfThoughts = mf.generation.plan.ChainOfThoughts
        ReAct = mf.generation.plan.ReAct
        SelfConsistency = mf.generation.plan.SelfConsistency

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        # Use Chain of Thoughts
        response = model(
            messages="What is 25 * 4 + 17?",
            generation_schema=ChainOfThoughts
        )

        result = response.consume()
        print(result)
        ```

## 9. **Tool Calling**

Models can suggest calling functions (tools) to gather information:

`ToolCatalogView` is the provider-neutral contract shared by `ToolLibrary`,
`Agent`, and `Model`. It contains logical tool definitions and thread-local
loading state rather than provider wire schemas. Build the view through a
`ToolLibrary`; the concrete Model provider compiles it to Chat Completions,
Responses, or another supported protocol.

`ToolCatalog` is still accepted temporarily when integrating existing code that
already builds OpenAI-style function schemas, but new code should use the
canonical view.

???+ example

    === "Defining Tools"

        ```python
        import msgflux as mf

        def get_weather(location: str, unit: str = "celsius") -> str:
            """Get the current weather for a city."""
            return f"The weather in {location} is 22 degrees {unit}."


        library = mf.nn.ToolLibrary("weather", [get_weather])
        messages = mf.ChatMessages(
            [{"role": "user", "content": "What's the weather in Paris?"}],
            thread_id="weather-demo",
        )
        tool_catalog = library.get_tool_catalog_view(messages)

        model = mf.Model.chat_completion(
            "openai/gpt-5.6-luna",
            api_mode="responses",
            store=False,
        )

        response = model(
            messages=messages,
            tool_catalog=tool_catalog,
        )

        for intent in response.get_tool_intents():
            print(intent.name, intent.arguments)
        # get_weather {'location': 'Paris', 'unit': 'celsius'}
        ```

        `ToolLibrary` derives the schema from the callable and returns a catalog
        view scoped to `weather-demo`. The response exposes provider-neutral
        `ToolIntent` values regardless of the selected API mode.

    === "Tool Choice"

        ```python
        import msgflux as mf

        def get_weather(location: str) -> str:
            """Get the current weather for a city."""
            return f"The weather in {location} is 22 degrees celsius."

        library = mf.nn.ToolLibrary("weather", [get_weather])
        messages = mf.ChatMessages(
            thread_id="weather-choice-demo",
        )
        catalog = library.get_tool_catalog_view(messages)

        auto = catalog.with_choice("auto")
        required = catalog.with_choice("required")
        disabled = catalog.with_choice("none")
        specific = catalog.with_choice("get_weather")

        print(required.choice.mode)
        print(specific.choice.mode, specific.choice.name)
        # required
        # tool get_weather
        ```

        `with_choice(...)` returns a new immutable view. It does not mutate the
        library or the catalog used by another thread.

    === "Full Flow"

        ```python
        import msgflux as mf

        def get_weather(location: str, unit: str = "celsius") -> str:
            """Get the current weather for a city."""
            return f"The weather in {location} is 22 degrees {unit}."


        library = mf.nn.ToolLibrary("weather", [get_weather])

        model = mf.Model.chat_completion(
            "openai/gpt-5.6-luna",
            api_mode="responses",
            reasoning_effort="low",
            store=False,
        )

        messages = mf.ChatMessages(
            [{"role": "user", "content": "What's the weather in Paris?"}],
            thread_id="weather-loop-demo",
        )
        tool_catalog = library.get_tool_catalog_view(messages)

        response = model(messages=messages, tool_catalog=tool_catalog)
        outcomes = library.execute_intents(response.get_tool_intents())

        messages.extend(response.history_items)
        messages.extend(response.render_tool_outcomes(outcomes))

        final_response = model(messages=messages, tool_catalog=tool_catalog)
        print(final_response.consume())
        # The weather in Paris is 22 degrees celsius.
        ```

        The Library executes canonical intents and returns `ToolOutcome`
        values. The response renders those outcomes for its selected API mode:
        Chat Completions uses assistant/tool messages, while Responses uses
        `function_call_output` items. `Agent` manages this loop automatically.

    === "Streaming"

        ```python
        import msgflux as mf

        def get_weather(location: str) -> str:
            """Get the current weather for a city."""
            return f"The weather in {location} is 22 degrees celsius."


        library = mf.nn.ToolLibrary("weather", [get_weather])
        messages = mf.ChatMessages(
            [{"role": "user", "content": "What's the weather in Tokyo?"}],
            thread_id="weather-stream-demo",
        )
        catalog = library.get_tool_catalog_view(messages)
        model = mf.Model.chat_completion(
            "openai/gpt-5.6-luna",
            api_mode="responses",
            store=False,
        )

        response = model(
            messages=messages,
            tool_catalog=catalog,
            stream=True,
        )

        response.consume()
        for intent in response.get_tool_intents():
            print(intent.name, intent.arguments)
        ```

        Tool-call deltas are accumulated into complete intents while the
        response stream is consumed.

## 10. **Prefilling**

Force the model to start its response with specific text. msgFlux appends the value as an assistant message before sending the request to the provider — see [Prefilling](../nn/agent/prefilling.md) for a detailed explanation of the technique.

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    response = model(
        messages=[{"role": "user", "content": "What is 30 * 3 + 33?"}],
        prefilling="Let's think step by step:"
    )

    print(response.consume())
    # Let's think step by step:
    # First, calculate 30 × 3 = 90.
    # Then, add 33 to that: 90 + 33 = 123.
    # So, the answer is 123.
    ```

## 11. **Web Search**

The `web_search_options` parameter enables real-time web search, letting the model ground its answers in up-to-date information retrieved from the internet. It is currently supported by OpenAI search models (`gpt-4o-search-preview`, `gpt-4o-mini-search-preview`) and OpenRouter.

OpenAI-compatible search providers can also expose chat completion models that search the web before answering. Brave uses `BRAVE_SEARCH_API_KEY` with the `brave/brave` model id, and Exa uses `EXA_API_KEY` with the `exa/exa` model id.

!!! info "Dependencies"
    Web-search chat models use the base HTTPX2 transport:

    === "uv"
        ```bash
        uv add msgflux
        ```

    === "pip"
        ```bash
        pip install msgflux
        ```

???+ example

    === "Basic Search"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion(
            "openai/gpt-4o-search-preview",
            web_search_options={"search_context_size": "low"},
        )

        response = model("What is the latest Python version released?")
        print(response.consume())
        # As of March 2026, the latest stable release of Python is version 3.14,
        # released on October 7, 2025. (liquidweb.com) ...
        ```

    === "With User Location"

        Restrict search results to a specific geographic area by providing an approximate user location:

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion(
            "openai/gpt-4o-search-preview",
            web_search_options={
                "search_context_size": "high",
                "user_location": {
                    "type": "approximate",
                    "approximate": {
                        "country": "BR",           # ISO 3166-1 alpha-2
                        "city": "São Paulo",
                        "region": "São Paulo",
                        "timezone": "America/Sao_Paulo",  # IANA timezone
                    },
                },
            },
        )

        response = model("What are the top tech events happening this month?")
        print(response.consume())
        ```

    === "Brave Search"

        Brave Answers uses an OpenAI-compatible endpoint. Set
        `BRAVE_SEARCH_API_KEY` and use the `brave/brave` model id:

        ```python
        import msgflux as mf

        mf.set_envs(BRAVE_SEARCH_API_KEY="...")

        model = mf.Model.chat_completion("brave/brave")
        response = model("What are the latest Python packaging updates?")

        print(response.consume())
        ```

        Brave-specific request fields are passed through `extra_body`:

        ```python
        import msgflux as mf

        mf.set_envs(BRAVE_SEARCH_API_KEY="...")

        model = mf.Model.chat_completion(
            "brave/brave",
            extra_body={
                "country": "BR",   # target country for search results
                "language": "pt",  # answer language
            },
        )

        response = model("Quais foram as principais novidades do Python este mês?")
        print(response.consume())
        ```

        Brave's richer answer fields require streaming mode:

        ```python
        import msgflux as mf

        mf.set_envs(BRAVE_SEARCH_API_KEY="...")

        model = mf.Model.chat_completion(
            "brave/brave",
            extra_body={
                "country": "US",
                "language": "en",
                "enable_citations": True,
                "enable_entities": True,
                "enable_research": False,
            },
        )

        response = model("Summarize the latest Python packaging updates.", stream=True)

        async for chunk in response.consume():
            print(chunk, end="", flush=True)
        ```

        Supported Brave Answers parameters include:

        | Parameter | Description | Default |
        |---|---|---|
        | `country` | Target country for search results | `"us"` |
        | `language` | Response language | `"en"` |
        | `enable_citations` | Include inline citation tags in streamed responses | `False` |
        | `enable_entities` | Include entity tags in streamed responses | `False` |
        | `enable_research` | Enable multi-search research mode for more thorough answers | `False` |

        `enable_citations`, `enable_entities`, and `enable_research` require
        `stream=True`. Research mode can be slower and more expensive because
        Brave may run multiple searches before answering.

    === "Exa Answer"

        ```python
        import msgflux as mf

        mf.set_envs(EXA_API_KEY="...")

        model = mf.Model.chat_completion("exa/exa")
        response = model("What are the latest changes in Python packaging?")

        print(response.consume())
        ```

### 11.1 **search_context_size**

Controls how much web content is retrieved and included in the model's context window:

| Value | Behaviour |
|---|---|
| `"low"` | Minimal context — fastest response, lower cost, may reduce answer depth |
| `"medium"` | Balanced context (default) |
| `"high"` | Maximum context — most comprehensive answers, higher cost |

### 11.2 **Annotations**

Search responses include inline citations. The raw URLs are also available in `response.metadata.annotations`:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion(
        "openai/gpt-4o-search-preview",
        web_search_options={"search_context_size": "low"},
    )

    response = model("What is the latest Python version?")
    response.consume()

    for annotation in response.metadata.get("annotations", []):
        print(annotation["url_citation"]["url"])
    # https://www.liquidweb.com/blog/latest-python-version/
    # ...
    ```

## 12. **Reasoning Models**

Reasoning models "think before answering" — they generate an internal chain of thought before producing a final response. This improves accuracy on complex tasks such as multi-step math, code generation, and logical deduction, at the cost of additional latency and tokens.

In msgFlux, **reasoning is a first-class field on the response object**. The model's chain of thought lives in `response.reasoning`, completely separated from the content in `response.data`. This means `consume()` always returns the final answer in its natural type (`str` for text generation, `dict` for structured output) regardless of whether the model reasoned or not — there is no silent type change.

### 12.1 **Configuration Parameters**

msgFlux exposes five parameters that control reasoning behaviour at model initialization:

| Parameter | Description | Default |
|---|---|---|
| `reasoning_effort` | How much reasoning to do. Values are model-dependent; GPT-5.6 accepts `"none"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, and `"max"`. | — |
| `reasoning_max_tokens` | OpenRouter-only hard cap (in tokens) for the internal thinking budget. Maps to `extra_body.reasoning.max_tokens` and cannot be combined with `reasoning_effort`. | — |
| `return_reasoning` | Expose the reasoning trace in `response.reasoning`. When `False`, it may still be retained in provider-neutral history when `reasoning_in_tool_call=True`. | `True` |
| `enable_thinking` | Activate extended model reasoning. Native Ollama accepts a boolean or the levels `"low"`, `"medium"`, and `"high"`. | `False` |
| `reasoning_in_tool_call` | Preserve reasoning context across tool calls. The selected provider codec reconstructs the API-native reasoning fields when history is sent back; msgFlux does not inject `<think>` tags implicitly. | `True` |

???+ example "Initialization"

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion(
        "groq/openai/gpt-oss-120b",
        reasoning_effort="low",
        return_reasoning=True,
    )
    ```

### 12.2 **Response Anatomy**

When a reasoning model responds, the response object keeps the answer, explicit
reasoning, and a provider-generated summary separate:

```
ModelResponse
├── .data          ← final answer (str, dict, ToolCallAggregator)
├── .reasoning     ← chain of thought (str or None)
├── .reasoning_summary ← summary of reasoning (str or None)
├── .has_reasoning ← True if reasoning is present (bool)
├── .response_type ← "text_generation", "structured", "tool_call"
└── .metadata      ← model audit identity, usage stats, annotations, etc.
```

The key methods on a non-streaming response:

| Method / Property | Returns | Description |
|---|---|---|
| `response.consume()` | `str`, `dict`, or `ToolCallAggregator` | The final answer, always in its natural type. |
| `response.consume_reasoning()` | `str` or `None` | The full reasoning trace, or `None` if the model didn't reason. |
| `response.reasoning` | `str` or `None` | Same as `consume_reasoning()` — direct attribute access. |
| `response.consume_reasoning_summary()` | `str` or `None` | A provider-generated reasoning summary. It is not presented as chain-of-thought. |
| `response.reasoning_summary` | `str` or `None` | Same as `consume_reasoning_summary()` — direct attribute access. |
| `response.has_reasoning` | `bool` | `True` when `reasoning is not None`. Useful for conditional logic without inspecting the string. |

### 12.3 **Provider Behaviour**

Not all reasoning providers behave the same way:

| Provider / protocol | Exposes trace via `return_reasoning` | Multi-turn reasoning replay | Notes |
|---|---|---|---|
| **Groq Chat Completions** | Yes — `response.reasoning` | No | Groq documents extraction, but not a Chat field for returning reasoning on the next turn |
| **Groq Responses** | Yes — `response.reasoning` | Yes | Clear-text `reasoning_text` item is reconstructed for the same provider and protocol |
| **vLLM Chat Completions** | Yes, with a reasoning parser | No | The documented multi-turn Chat example replays assistant content only |
| **vLLM Responses** | Yes, with a reasoning parser | Yes | Clear-text Responses item is reconstructed for the same server protocol |
| **Ollama OpenAI-compatible Chat** | Model/version-dependent | No | Select `api_mode="chat_completions"`; that compatibility contract does not document historical `thinking` messages |
| **Ollama native `/api/chat`** | Yes — `response.reasoning` from `message.thinking` | Yes | This is msgFlux's default Ollama mode; thinking, content, and tool calls are accumulated and replayed together |
| **Ollama Responses** (upstream; not yet exposed by msgFlux) | Reasoning summaries | Not established | Available since Ollama 0.13.3, but documented as non-stateful without `previous_response_id` or `conversation` |
| **OpenAI Chat Completions** | Usually no text trace | No | Availability depends on the selected model |
| **OpenAI Responses** | Yes — `response.reasoning_summary` | Yes | The summary remains distinct from CoT; opaque reasoning items are preserved separately for replay |
| **OpenRouter Chat Completions** | Model-dependent | Yes | Ordered `reasoning_details` are round-tripped by its provider codec |
| **Anthropic** (via `enable_thinking`) | Yes — `response.reasoning` | Provider-dependent | Uses `enable_thinking=True` instead of `reasoning_effort` |

OpenAI-compatible providers inherit their transport implementation from
`OpenAICompatibleChatCompletion`, but each provider can declare its own
`ReasoningCodec`. The codec extracts the provider response and reconstructs
provider-native history fields. For example, OpenRouter declares a codec that
round-trips ordered `reasoning_details`, while OpenAI declares its own default
independently. The Agent only stores the canonical interaction items.

### 12.4 **Reasoning Effort**

`reasoning_effort` is the primary knob. Higher effort means the model spends more tokens on internal reasoning, which typically improves answer quality on hard problems.

The value supplied at initialization is the request-level default. Update that
default between calls with `set_reasoning_effort()`:

```python
import msgflux as mf

model = mf.Model.chat_completion(
    "openai/gpt-5.6-luna",
    reasoning_effort="low",
)

quick = model("Name the capital of France.")

model.set_reasoning_effort("high")
careful = model("Prove that there are infinitely many prime numbers.")

model.set_reasoning_effort(None)  # Remove the explicit request-level setting.
```

The setter affects future requests and returns the model, so it can be chained.
Effort names remain model-dependent. If the active provider/API mode does not
support request-level effort, msgFlux emits a warning and leaves the model
unchanged. This setter changes the parameter at the top of each request; it does
not insert a configuration update into an existing conversation.

???+ example

    === "Low Effort — Quick Tasks"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="low",
        )

        response = model("What is the capital of France?")
        print(response.consume())       # "Paris"
        print(response.has_reasoning)    # True (model still reasons, just briefly)
        ```

    === "High Effort — Hard Problems"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="high",
        )

        response = model(
            "Prove that there are infinitely many prime numbers."
        )
        print(response.consume())  # The proof
        print(response.consume_reasoning())  # The full chain of thought
        ```

### 12.5 **Inspecting the Reasoning Trace**

When `return_reasoning=True` (the default) and the provider exposes the reasoning trace, it is available as a separate field on the response object:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion(
        "groq/openai/gpt-oss-120b",
        reasoning_effort="high",
    )

    response = model("Prove that sqrt(2) is irrational.")

    # The answer — always a plain str for text generation
    answer = response.consume()
    print(answer)
    # **Proof that √2 is irrational**
    # We prove the statement by contradiction...

    # The reasoning trace — separate field
    reasoning = response.consume_reasoning()
    print(reasoning)
    # The user asks to prove sqrt(2) is irrational. This is a classic proof.
    # I'll use proof by contradiction: Suppose sqrt(2)=a/b in lowest terms...
    ```

!!! tip
    Comparing `response.reasoning` with `response.consume()` is a great debugging tool: if the final answer is wrong, the trace usually reveals where the reasoning went astray.

When `return_reasoning=False`, the reasoning is discarded even if the provider sends it:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion(
        "groq/openai/gpt-oss-120b",
        reasoning_effort="low",
        return_reasoning=False,  # Discard reasoning
    )

    response = model("What is 2+2?")
    print(response.consume())             # "4"
    print(response.has_reasoning)          # False
    print(response.consume_reasoning())    # None
    ```

Providers that keep reasoning internal (like OpenAI) still report how many tokens were spent via `response.metadata`:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion(
        "openai/gpt-5-mini",
        reasoning_effort="high",
    )

    response = model("A train travels 120 km in 1.5 hours. What is its average speed?")
    print(response.consume())
    # Average speed = distance / time = 120 km ÷ 1.5 h = 80 km/h.

    # No reasoning trace (OpenAI keeps it internal)
    print(response.has_reasoning)  # False

    # But token counts are available
    usage = response.metadata.usage
    print(f"Reasoning tokens used: {usage.output_tokens_details.reasoning_tokens}")
    # Reasoning tokens used: 64
    ```

### 12.6 **Controlling the Reasoning Budget**

`reasoning_max_tokens` is an OpenRouter-only parameter in msgFlux. It maps to `extra_body={"reasoning": {"max_tokens": ...}}` on the OpenRouter API and cannot be combined with `reasoning_effort`.

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion(
        "openrouter/anthropic/claude-sonnet-5",
        reasoning_max_tokens=2000,
    )

    response = model("What's the weather like in Boston? Then recommend what to wear.")
    print(response.consume_reasoning())
    print(response.consume())
    ```

OpenRouter's own API examples pass the reasoning budget inside `extra_body={"reasoning": {"max_tokens": 2000}}`; msgFlux translates the model-level `reasoning_max_tokens` parameter to that shape for you.

### 12.7 **Streaming with Reasoning**

Streaming keeps the channel-specific content and reasoning consumers while also
providing one ordered LM event stream. `consume_events()` preserves the relative
provider order across `reasoning.delta`, `reasoning_summary.delta`, and
`output.delta`.

??? info "How it works internally"

    When `stream=True`, the model returns a `ModelStreamResponse` instead of a
    `ModelResponse`. Each published chunk enters its compatibility channel and
    the canonical ordered event queue:

    ```text
    Provider stream thread
    │
    ├── reasoning chunk → stream_response.add_reasoning(chunk)
    │                     ├── reasoning queue
    │                     └── reasoning.delta
    ├── summary chunk   → stream_response.add_reasoning_summary(chunk)
    │                     ├── reasoning-summary queue
    │                     └── reasoning_summary.delta
    ├── stream_response.finish_reasoning()                     → closes reasoning queue
    ├── content chunk   → stream_response.add(chunk)
    │                     ├── content queue
    │                     └── output.delta
    ├── ...
    └── stream_response.finish(status="completed")
        ├── closes any still-open queues
        ├── records final status
        └── runs finalizers
    ```

    Reasoning has its own channel lifecycle. When a provider knows the reasoning
    phase has ended, it can call `finish_reasoning()` before normal content
    streaming completes. At the end of the full stream, the provider calls
    `finish()` to close any still-open queues, set the final status, and run any
    finalizers attached by higher-level runtime components. The queue sentinels
    are internal details; providers should publish real chunks with `add()` /
    `add_reasoning()`, close reasoning with `finish_reasoning()` when that
    channel is done, and close the full stream with `finish()`.

    The provider also sets `stream_response.reasoning` with the full accumulated
    reasoning text, so it is available as a single string after the stream
    completes.

    The Agent consumes the ordered queue rather than concurrently draining the
    channel-specific queues. Consequently, a reasoning summary emitted before a
    tool call or answer remains before that operation in the execution event
    stream.

#### The two-event system

Streaming responses use two events to signal different stages of the stream:

| Event | Fires when | Purpose |
|---|---|---|
| `first_chunk_event` | First token arrives (reasoning **or** content) | Lets callers know the stream is alive. Fires early — often on the first reasoning token, before any content appears. |
| `_response_type_event` | `response_type` is determined (`"text_generation"` or `"tool_call"`) | Lets callers that need to branch on response type (like `Agent`) wait for this signal before proceeding. |

This separation exists because reasoning models often emit reasoning tokens before any content. Without it, a caller waiting for the response type would have to block until content arrives, defeating the purpose of streaming. With the two-event system, `first_chunk_event` fires immediately on the first reasoning token, while `_response_type_event` fires later when the actual content type becomes clear.

```
Timeline:
  ┌─ reasoning tokens ──────────────────┐┌── content tokens ───────────┐
  │  think think think think think ...   ││  Hello, the answer is ...   │
  ▲                                      ▲                              ▲
  │                                      │                              │
  first_chunk_event                      _response_type_event           metadata set
  (fires here)                           (fires here)                   (stream done)
```

#### Consuming streams

The `consume()`, `consume_reasoning()`, `consume_reasoning_summary()`, and
`consume_events()` methods become async generators in streaming mode:

For content streams, `next_chunk()` is also available when you want pull-based delivery. Each call returns one content chunk (`str` for chat completion) or `None` when the stream is complete. This controls when your application receives the next chunk; it does not pause the remote provider, which may continue producing chunks in the background.

???+ example

    === "Pull-Based Content"

        ```python
        import asyncio
        import msgflux as mf

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="low",
            return_reasoning=True,
        )

        response = await model.acall(
            "What is 2+2? Explain your reasoning.", stream=True
        )

        while True:
            chunk = await response.next_chunk()
            if chunk is None:
                break

            print(chunk, end="", flush=True)
            await asyncio.sleep(0.05)
        ```

    === "Async Streaming — Content + Reasoning"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="low",
            return_reasoning=True,
        )

        response = await model.acall(
            "What is 2+2? Explain your reasoning.", stream=True
        )

        # Consume content chunks
        async for chunk in response.consume():
            print(chunk, end="", flush=True)

        print()  # newline

        # Consume reasoning chunks
        async for chunk in response.consume_reasoning():
            print(chunk, end="", flush=True)

        # After stream completes, accumulated reasoning is also available
        print(response.reasoning)
        print(response.has_reasoning)  # True
        ```

    === "Reasoning First"

        The queues are independent — you can consume reasoning before content. This is useful when you want to display the chain of thought first:

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="low",
            return_reasoning=True,
        )

        response = await model.acall("Solve: 15 × 7 + 3", stream=True)

        # Read reasoning first
        print("Thinking:")
        async for chunk in response.consume_reasoning():
            print(chunk, end="", flush=True)

        # Then read the answer
        print("\n\nAnswer:")
        async for chunk in response.consume():
            print(chunk, end="", flush=True)
        ```

    === "Sync Streaming (polling)"

        In sync contexts, the stream runs in a background thread. Content and reasoning accumulate in pending buffers until an async consumer binds. For sync-only code, you can poll the response after the stream completes:

        ```python
        import time
        import msgflux as mf

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="low",
            return_reasoning=True,
        )

        response = model("What is 2+2?", stream=True)

        # first_chunk_event fires on the first token (often reasoning)
        response.first_chunk_event.wait(timeout=10)

        # Wait for stream to complete
        for _ in range(50):
            if response.metadata is not None:
                break
            time.sleep(0.1)

        # After completion, the accumulated fields are available
        print(response.reasoning)       # Full reasoning text
        print(response.has_reasoning)   # True
        ```

    === "FastAPI — Dual Stream"

        ```python
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        import msgflux as mf

        app = FastAPI()
        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="low",
            return_reasoning=True,
        )

        @app.get("/chat")
        async def chat(query: str):
            response = await model.acall(
                messages=[{"role": "user", "content": query}],
                stream=True,
            )
            return StreamingResponse(
                response.consume(),
                media_type="text/plain",
            )

        @app.get("/chat/reasoning")
        async def chat_reasoning(query: str):
            response = await model.acall(
                messages=[{"role": "user", "content": query}],
                stream=True,
            )
            return StreamingResponse(
                response.consume_reasoning(),
                media_type="text/plain",
            )
        ```

??? tip "Thread safety"

    Both queues use `threading.Lock` to protect the bind/pending-flush
    operations. The producer (provider stream thread) calls `add()` /
    `add_reasoning()` safely from any thread via `loop.call_soon_threadsafe()`.
    Pending chunks are buffered in a `deque` until a consumer binds the queue to
    an event loop; at that point all pending chunks are flushed into the
    `asyncio.Queue` atomically under the lock.

### 12.8 **Reasoning Across Tool Calls**

Reasoning replay across tool calls is protocol-dependent.
`reasoning_in_tool_call=True` asks the selected provider codec to preserve the
native state when that API defines a replay contract. msgFlux does not invent a
wire representation or inject `<think>` tags for providers that do not define
one.

For example, a Responses trajectory keeps reasoning, function calls, and tool
outputs as separate canonical items:

```
# Message history with reasoning_in_tool_call=True:
[
    {"role": "user", "content": "What is (14 + 28) × 3 − 7?"},
    {"type": "reasoning", "role": "assistant", "text": "I need to break this into steps..."},
    {"type": "function_call", "call_id": "call_1", "name": "calculate",
     "arguments": "{\"expression\":\"14 + 28\"}"},
    {"type": "function_call_output", "call_id": "call_1", "output": "42"},
]
```

OpenAI Responses reattaches opaque state; Groq and vLLM Responses reconstruct
`reasoning_text`; OpenRouter reconstructs `reasoning_details`. Groq and vLLM
Chat Completions remain extract-only until those APIs document how reasoning
must be returned on a later request. `response.reasoning` still exposes the
trace from the current model call independently of replay.

Ollama's default native mode follows its `/api/chat` trajectory: msgFlux appends
the returned assistant thinking and tool calls before the tool results and sends
them back together. The optional OpenAI-compatible mode remains extract-only
because `/v1/chat/completions` does not document `thinking` as historical input.

Ollama also documents `/v1/responses` starting in version 0.13.3. It supports
streaming, tools, and reasoning summaries, but the same page calls the endpoint
non-stateful and excludes `previous_response_id` and `conversation`. Until its
manual-input reasoning-item contract is established and tested, msgFlux does
not claim multi-turn reasoning replay through Ollama Responses.

See Ollama's [native tool-calling trajectory](https://docs.ollama.com/capabilities/tool-calling),
[thinking fields](https://docs.ollama.com/capabilities/thinking), and
[OpenAI-compatible endpoint matrix](https://docs.ollama.com/api/openai-compatibility)
for the three distinct contracts.

???+ example

    ```python
    import msgflux as mf
    from msgflux.tools import ToolCatalog

    tools = [{
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"],
                "additionalProperties": False,
            }
        }
    }]

    model = mf.Model.chat_completion(
        "groq/openai/gpt-oss-120b",
        reasoning_effort="high",
        reasoning_in_tool_call=True,
    )

    response = model(
        messages=[{"role": "user", "content": "What is (14 + 28) × 3 − 7?"}],
        tool_catalog=ToolCatalog.from_function_schemas(schemas=tools),
    )

    tool_call_agg = response.consume()
    calls = tool_call_agg.get_calls()
    print(calls)
    ```

### 12.9 **Structured Output with Reasoning**

Reasoning models pair well with `generation_schema` — the model uses its thinking budget to produce more accurate structured output. The reasoning stays in `response.reasoning` while the structured data lives in `response.consume()`:

???+ example

    ```python
    import msgflux as mf
    from msgspec import Struct

    class MathSolution(Struct):
        answer: float
        confidence: str   # "high", "medium", "low"
        explanation: str

    model = mf.Model.chat_completion(
        "groq/openai/gpt-oss-120b",
        reasoning_effort="high",
    )

    response = model(
        messages="A train travels 120 km in 1.5 hours. What is its average speed?",
        system_prompt="You are a precise problem solver.",
        generation_schema=MathSolution,
    )

    # Structured output — always a dict, never wrapped with reasoning
    result = response.consume()
    print(result)
    # {'answer': 80.0, 'confidence': 'high', 'explanation': '120 km / 1.5 h = 80 km/h'}

    # Reasoning trace — separate field
    print(response.consume_reasoning())
    # The user asks about average speed. Formula: speed = distance / time.
    # distance = 120 km, time = 1.5 h, so speed = 120 / 1.5 = 80 km/h.
    ```

### 12.10 **Choosing the Right Effort Level**

| Task | Recommended effort |
|---|---|
| Simple factual lookup | `"low"` |
| Summarisation, translation | `"low"` – `"medium"` |
| Code generation, debugging | `"medium"` – `"high"` |
| Complex math / formal proofs | `"high"` |
| Multi-step planning with tools | `"high"` + `reasoning_in_tool_call=True` |

### 12.11 **Internal Architecture**

This section explains how reasoning flows through the system for readers who want to understand or extend the internals.

#### Response classes

Reasoning lives on two response base classes in `msgflux._private.response`:

| Class | Used when | Reasoning storage |
|---|---|---|
| `BaseResponse` | Non-streaming (`stream=False`) | `reasoning` stores explicit reasoning text; `reasoning_summary` stores a provider-generated summary. |
| `BaseStreamResponse` | Streaming (`stream=True`) | The same values are accumulated through channel-specific consumers and an ordered LM event stream. |

Both classes inherit from `CoreResponse`, which provides `set_metadata()` and `set_response_type()`.

#### Provider flow (non-streaming)

```
model("prompt")
  │
  ├── OpenAICompatibleChatCompletion._generate()
  │     └── client.chat.completions.create(**params)
  │           └── API response
  │
  └── _process_completion_model_output(model_output)
        ├── reasoning = _extract_reasoning(message)
        ├── response.reasoning = reasoning       # ← set directly on response
        ├── response.add(content)                # ← data is pure content
        └── response.set_response_type("text_generation")
```

The Model delegates extraction to its `reasoning_codec`. `return_reasoning`
controls whether the text is exposed through `response.reasoning`; extraction
can still populate canonical history when reasoning must survive a tool call.
For OpenAI Responses, summaries are exposed through
`response.reasoning_summary` instead, while `response.reasoning` remains `None`.

#### Provider flow (streaming)

```
model("prompt", stream=True)
  │
  ├── OpenAICompatibleChatCompletion._stream_generate()  # background thread
  │     └── for chunk in client.chat.completions.create(stream=True):
  │           ├── reasoning_chunk? → stream_response.add_reasoning(chunk)
  │           │                      ├── has_reasoning = True (first time)
  │           │                      └── first_chunk_event.set() (first time)
  │           │
  │           └── content_chunk?   → stream_response.finish_reasoning()
  │                                  stream_response.add(chunk)
  │                                  ├── set_response_type("text_generation")
  │                                  │   └── _response_type_event.set()
  │                                  └── first_chunk_event.set() (if not already)
  │
  │     finally:
  │           ├── stream_response.reasoning = accumulated_reasoning
  │           ├── stream_response.set_metadata(usage)
  │           ├── _response_type_event.set()            # safety net
  │           └── stream_response.finish(status=final_status)
  │
  └── returns stream_response immediately (stream runs in background)
```

`finish_reasoning()` lets a consumer observe the end of the reasoning channel
before the content channel is done. `finish()` signals end-of-stream to any
remaining `next_chunk()`, `consume()`, `consume_reasoning()`,
`consume_reasoning_summary()`, and `consume_events()` consumers by closing
still-open queues. `consume()` is implemented as a convenience async generator
over repeated `next_chunk()` calls. `finish()` also records the final
stream status (`completed`, `failed`, or `interrupted`) and runs registered
finalizers, which durable runtimes use to checkpoint streamed output after the
consumer finishes reading it. The same finalizer updates the caller's
`ChatMessages` object, so its turn and assistant output agree with the durable
snapshot after completion. SQLite checkpoint operations are serialized around a
cross-thread connection because synchronous model streams finish in a background
worker. The safety net `_response_type_event.set()` in
the `finally` block ensures the event is always fired, even if the stream errors
out or the model returns no content chunks (e.g., a pure tool call response).

`reasoning_summary_event` supports summary-specific discovery. It is set on the
first summary chunk, or when the summary channel closes without producing one.
After waiting for it, inspect `has_reasoning_summary`: `True` means
`consume_reasoning_summary()` can yield summary chunks; `False` means the stream
completed that channel without a summary.

At the same time, `ChatStreamAccumulator` builds a provider-neutral ordered
snapshot from reasoning, content, and tool-call deltas. Stream finalizers expose
that snapshot as `StreamFinalState.items`. Durable agents append those items
once at the terminal stream boundary, avoiding a second copy assembled from
the aggregate output and reasoning strings.

Normalized reasoning uses `text` and `summary`. Exact provider-only data is
kept under `provider_state`. New state records identify the `provider`,
`api_mode`, and reasoning `codec`; opaque data is replayed only when that
identity matches. For example, OpenRouter `reasoning_details` remains ordered
and unchanged instead of being flattened into text or replayed to another API.
This distinction is also public: `text` feeds `response.reasoning`, whereas
`summary` feeds `response.reasoning_summary`; a summary is never injected as a
`<think>` trace.

#### Agent integration

The `Agent` module waits on `_response_type_event` before deciding how to process the response:

```python
# Inside Agent._process_model_response():
if isinstance(model_response, ModelStreamResponse):
    wait_for_event(model_response._response_type_event)

# Now response_type is guaranteed to be set
if "tool_call" in model_response.response_type:
    # process tool calls...
```

The Agent reads `model_response.reasoning` to pass it downstream. If the Agent's `config["reasoning_in_response"]` is `True`, the final output is wrapped as `dotdict(answer=raw_response, reasoning=reasoning)` — this is an explicit opt-in at the Agent level, not a silent model-level behaviour.

## 13. **Token Logprobs**

`logprobs` and `top_logprobs` are forwarded for `openai/...` chat completion models. OpenAI declares this capability on its concrete provider class; other subclasses of `OpenAICompatibleChatCompletion` do not receive the initialization fields unless they declare equivalent support.

Use `logprobs=True` to request token log probabilities. Set `top_logprobs` to the number of alternative tokens you want returned for each generated token.

| Parameter | Description |
|---|---|
| `logprobs` | Enables token-level logprob data in the response metadata. |
| `top_logprobs` | Number of alternative tokens returned per generated token. Use with `logprobs=True`. |

The returned payload is exposed in `response.metadata.logprobs` and follows OpenAI's native shape. It includes the `content` list with token entries and nested `top_logprobs` alternatives.

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion(
        "openai/gpt-4.1-mini",
        logprobs=True,
        top_logprobs=2,
    )

    response = model("Hello!")
    print(response.metadata.logprobs["content"][0]["token"])
    print(response.metadata.logprobs["content"][0]["top_logprobs"][0]["token"])
    ```

## 14. **OpenAI Prompt Caching**

`prompt_cache_retention` is an OpenAI-only initialization parameter. msgFlux forwards it only for `openai/...` chat completion models. Other providers that inherit from `OpenAICompatibleChatCompletion` do not use it.

Use it when you want OpenAI to keep cached prefixes in memory or retain them for longer:

| Value | Behaviour |
|---|---|
| `"in_memory"` | Default. Keeps the cache in volatile memory for short-lived reuse. |
| `"24h"` | Extended retention. Keeps cached prefixes available for longer. |

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion(
        "openai/gpt-5.1",
        prompt_cache_retention="24h",
    )

    response = model("Summarize the attached policy.")
    print(response.consume())
    ```

## 15. **Response Metadata**

Every chat response, including a completed stream, identifies the LM request in
`response.metadata.model`:

```python
model = mf.Model.chat_completion(
    "openai/gpt-5.6-luna",
    api_mode="responses",
    reasoning_effort="medium",
    speed="fast",
)
response = model("Summarize the incident.")

print(response.metadata.model)
# {
#     "provider": "openai",
#     "model_id": "gpt-5.6-luna",
#     "api_mode": "responses",
#     "reasoning_effort": "medium",
#     "requested_speed": "fast",
#     "effective_speed": "priority",
# }
```

`reasoning_effort` is present only when that LM transport uses the setting.
Speed fields are present only when requested or reported by the provider. The
other three fields are always produced by OpenAI-compatible chat LMs. An
Agent can persist this small audit record with the generated timeline item
without reconstructing provider details itself.

The same response metadata includes operational request timing:

```python
timing = response.metadata.timing

print(timing.source)      # "provider" or "cache"
print(timing.latency_ms)  # request start through the settled response
```

For a streamed response, `ttft_ms` measures the time to the first non-empty
piece of content exposed by msgFlux. Content may be answer text, returned
reasoning, or a tool call. Empty protocol and role-only events do not count:

```python
stream = await model.acall("Summarize the incident.", stream=True)

async for chunk in stream.consume():
    print(chunk, end="")

print(stream.metadata.timing.latency_ms)
print(stream.metadata.timing.get("ttft_ms"))
```

`ttft_ms` is omitted for non-streaming requests and for streams that settle
without observable output. A msgFlux response-cache hit reports the latency of
the cache lookup with `source="cache"`; it does not reuse the provider request's
original timing. These operational measurements stay on the response and are
not written to Agent checkpoints.

Chat completion providers also expose the same canonical usage shape regardless
of whether their native API calls tokens `prompt`/`completion`, `input`/`output`,
or uses another convention:

| Field | Description |
|---|---|
| `input_tokens` | Tokens consumed by the request input. |
| `output_tokens` | Tokens generated by the model. |
| `total_tokens` | Provider total, or input plus output when omitted. |
| `input_tokens_details.cached_tokens` | Input tokens read from a provider cache. A value greater than zero confirms a cache hit. |
| `cache_hit_percentage` | Percentage of input tokens served from the provider cache (`cached_tokens / input_tokens * 100`). Returns `None` when the provider does not report a valid input-token denominator. |
| `input_tokens_details.cache_write_tokens` | Input tokens written to a provider cache. |
| `output_tokens_details.reasoning_tokens` | Output tokens used for reasoning. |
| `cost` | Provider-reported request cost when available. |
| `raw` | Deep copy of the complete provider-native usage payload. |

The detail objects also normalize audio, video, and prediction-token counters.
Missing optional counters are `0`; unavailable provider cost is `None`.

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    response = model(messages=[{"role": "user", "content": "Hello"}])

    usage = response.metadata.usage
    print(usage.input_tokens)
    print(usage.output_tokens)
    print(usage.total_tokens)

    # Provider-specific fields remain available for diagnostics.
    print(usage.raw)

    # Calculate cost using profile
    from msgflux.models.profiles import get_model_profile

    profile = get_model_profile("gpt-4.1-mini", provider_id="openai")
    if profile:
        cost = profile.cost.calculate(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_tokens=usage.input_tokens_details.cached_tokens,
        )
        print(f"Request cost: ${cost:.4f}")
    ```

The same structure is populated after streaming finishes:

```python
import asyncio


async def main():
    stream = await model.acall("Summarize the report.", stream=True)

    # Consume the stream before reading terminal usage metadata.
    async for chunk in stream.consume():
        print(chunk, end="")

    print(stream.metadata.usage.input_tokens_details.cached_tokens)


asyncio.run(main())
```

## 16. **Error Handling**

Handle common errors gracefully:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    try:
        response = model(messages=[{"role": "user", "content": "Hello"}])
        result = response.consume()
    except ImportError:
        print("Provider not installed")
    except ValueError as e:
        print(f"Invalid parameters: {e}")
    except Exception as e:
        print(f"API error: {e}")
    ```

## 17. **Model Profiles**

Model profiles provide metadata about capabilities, pricing, and limits from [models.dev](https://models.dev).

Every initialized model exposes a `.profile` property that returns this metadata without any extra setup:

???+ example

    === "Instance Profile"

        ```python
        import msgflux as mf

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        # Access profile directly from the instance
        profile = model.profile  # ModelProfile | None
        if profile:
            print(f"Context window: {profile.limits.context}")
            print(f"Tool calling: {profile.capabilities.tool_call}")
        ```

    === "Profile Information"

        ```python
        import msgflux as mf
        from msgflux.models.profiles import get_model_profile

        # Get profile for a model
        profile = get_model_profile("gpt-4.1-mini", provider_id="openai")

        if profile:
            # Check capabilities
            print(f"Tool calling: {profile.capabilities.tool_call}")
            print(f"Structured output: {profile.capabilities.structured_output}")
            print(f"Reasoning: {profile.capabilities.reasoning}")

            # Check modalities
            print(f"Input: {profile.modalities.input}")   # ['text', 'image']
            print(f"Output: {profile.modalities.output}") # ['text']

            # Check limits
            print(f"Context window: {profile.limits.context}")  # 128000
            print(f"Max output: {profile.limits.output}")       # 16384

            # Check pricing
            print(f"Input: ${profile.cost.input_per_million}/M tokens")
            print(f"Output: ${profile.cost.output_per_million}/M tokens")
        ```

    === "Cost Calculation"

        ```python
        from msgflux.models.profiles import get_model_profile

        profile = get_model_profile("gpt-4.1-mini", provider_id="openai")

        if profile:
            # Calculate cost for a request
            cost = profile.cost.calculate(
                input_tokens=1000,
                output_tokens=500
            )
            print(f"Estimated cost: ${cost:.4f}")
        ```

## 18. **Adding a Custom Provider**

If the service you want to use exposes an **OpenAI-compatible API**, add it by
subclassing `OpenAICompatibleChatCompletion`. `OpenAIChatCompletion` is the
concrete OpenAI provider and should not be used as the base for another
provider. The process has three stages depending on how compatible the endpoint
is.

### 18.1 **Stage 1 — URL and API key only**

When the target API is fully OpenAI-compatible and only requires a different base URL and authentication key, the entire subclass is a small configuration mixin plus the `@register_model` decorator.

???+ example "Custom provider — minimal setup"

    ```python
    from os import getenv
    from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
    from msgflux.models.registry import register_model


    class _BaseMyProvider:
        """Configuration mixin for MyProvider."""

        provider: str = "myprovider"  # used in "myprovider/model-name"

        def _get_base_url(self):
            return getenv("MYPROVIDER_BASE_URL", "https://api.myprovider.com/v1")

        def _get_api_key(self):
            key = getenv("MYPROVIDER_API_KEY")
            if not key:
                raise ValueError("Please set `MYPROVIDER_API_KEY`")
            return key


    @register_model
    class MyProviderChatCompletion(
        _BaseMyProvider,
        OpenAICompatibleChatCompletion,
    ):
        """MyProvider Chat Completion."""
    ```

After registering, the model is available through the standard factory. The string before the `/` must match the `provider` class attribute:

???+ example "Using the custom provider"

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion("myprovider/my-model-name")
    response = model("Hello!")
    print(response.consume())
    ```

### 18.2 **Stage 2 — Adapting parameters**

Some providers are mostly OpenAI-compatible but have small differences: renamed fields, required extra headers, or unsupported parameters. Override `_adapt_params` to transform the parameter dict before it reaches the API.

`_adapt_params` receives the fully-populated `params: dict` (call kwargs merged with model-level sampling params) and must return the modified dict.

The built-in OpenRouter provider is a real-world example:

???+ example "Custom provider — with parameter adaptation"

    ```python
    from os import getenv
    from typing import Any, Dict

    from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
    from msgflux.models.registry import register_model


    class _BaseMyProvider:
        provider: str = "myprovider"

        def _get_base_url(self):
            return getenv("MYPROVIDER_BASE_URL", "https://api.myprovider.com/v1")

        def _get_api_key(self):
            key = getenv("MYPROVIDER_API_KEY")
            if not key:
                raise ValueError("Please set `MYPROVIDER_API_KEY`")
            return key


    @register_model
    class MyProviderChatCompletion(
        _BaseMyProvider,
        OpenAICompatibleChatCompletion,
    ):
        """MyProvider Chat Completion."""

        def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
            # 1. Rename max_tokens to the provider-specific field
            params["max_completion_tokens"] = params.pop("max_tokens")

            # 2. Provider requires tool_choice to be set explicitly
            if params["tool_choice"] is None:
                params["tool_choice"] = "auto" if params["tools"] else "none"

            # 3. Map reasoning parameters to the provider format
            reasoning_effort = params.pop("reasoning_effort", None)
            reasoning_max_tokens = params.pop("reasoning_max_tokens", None)
            if reasoning_effort is not None and reasoning_max_tokens is not None:
                raise ValueError(
                    "`reasoning_max_tokens` cannot be used together with "
                    "`reasoning_effort` for OpenRouter."
                )
            if reasoning_effort is not None:
                extra_body = params.get("extra_body", {})
                extra_body["reasoning"] = {"effort": reasoning_effort}
                params["extra_body"] = extra_body
            if reasoning_max_tokens is not None:
                extra_body = params.get("extra_body", {})
                extra_body["reasoning"] = {"max_tokens": reasoning_max_tokens}
                params["extra_body"] = extra_body

            # 4. Add required headers
            params["extra_headers"] = {
                "X-App-Name": "myapp",
            }

            return params
    ```

Common adaptations inside `_adapt_params`:

| Situation | What to do |
|---|---|
| Provider uses `max_completion_tokens` instead of `max_tokens` | `params["max_completion_tokens"] = params.pop("max_tokens")` |
| Provider rejects `tool_choice=None` | Set it explicitly to `"auto"` or `"none"` |
| Provider uses a different field for reasoning | `params.pop("reasoning_effort")` or `params.pop("reasoning_max_tokens")` and remap into `extra_body` |
| Provider requires extra headers | Add keys to `params["extra_headers"]` |
| Provider accepts non-standard extensions | Add keys to `params["extra_body"]` |

### 18.3 **Declaring a reasoning codec**

The compatible base uses `OpenAICompatibleReasoningCodec`, which reads common
parsed-text fields but is extract-only by default. There is no universal Chat
Completions field for replaying reasoning. A provider with a documented
convention declares that behavior in its codec:

```python
from msgflux.models import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
    OpenAIResponsesContextAdapter,
)
from msgflux.models.openai_compatible import (
    OpenAIChatCompletionsAPI,
    OpenAICompatibleChatCompletion,
)
from msgflux.models.reasoning import OpenAICompatibleReasoningCodec
from msgflux.models.registry import register_model


class MyProviderReasoningCodec(OpenAICompatibleReasoningCodec):
    name = "myprovider_reasoning_v1"
    text_fields = ("reasoning_text",)
    history_text_field = "reasoning_text"

    def encode_chat_message(self, items, *, provider, api_mode):
        # Only implement this when the provider documents Chat replay.
        del provider, api_mode
        text = "".join(filter(None, (self._item_text(item) for item in items)))
        return {"reasoning_text": text} if text else {}


@register_model
class MyProviderChatCompletion(
    _BaseMyProvider,
    OpenAICompatibleChatCompletion,
):
    capabilities = ChatProviderCapabilities(
        default_api_mode="chat_completions",
        api_modes=(
            ChatAPIModeCapabilities(
                name="chat_completions",
                adapter=OpenAIChatCompletionsAPI(),
            ),
        ),
        default_reasoning_codec=MyProviderReasoningCodec(),
    )
```

The codec runs inside the Model for normal responses, streaming deltas, and
history conversion. Override `encode_chat_message()` only when the provider
documents Chat replay. For Responses, override `encode_responses_item()`. If an
API returns opaque blocks, IDs, or signatures, `extract_state()` keeps those
values under `provider_state`; reconstruction only occurs for a matching
`provider`, `api_mode`, and codec name. OpenRouter's
`OpenRouterReasoningCodec` and the Groq/vLLM clear-text Responses codec are
built-in examples of both patterns.

You can also pass a `ReasoningCodec` instance through `reasoning_codec=` when
constructing a model. Declare `default_reasoning_codec` in the provider's
`ChatProviderCapabilities` when the wire convention is stable across its API
modes. Put a codec on one `ChatAPIModeCapabilities` when only that protocol
supports the convention.

### 18.4 **Using a different transport**

OpenAI-compatible chat providers use `HTTPChatTransport` by default. Do not
override `_initialize` to replace HTTP clients: that method owns shared cache
and retry setup. When a service requires different request mechanics, implement
`ChatTransport.create()` and `ChatTransport.acreate()` and pass the transport
class or an instance through `chat_transport=`.

The transport receives a `PreparedChatRequest`. Protocol-specific envelope and
response conversion remain in `ChatAPIAdapter`, while authentication remains in
`ModelCredentialResolver`. This keeps alternative networking code independent
from conversation history, tools, reasoning, caching, and model output.

### 18.5 **Adding another API protocol**

Client transport and wire protocol are separate extension points. Register a
`ChatAPIAdapter` when the request envelope, endpoint, response items, or stream
events form a different protocol.

An adapter owns these operations:

| Method | Responsibility |
|---|---|
| `prepare_request` | Convert shared generation parameters into the wire request |
| `build_generation_params` | Build the protocol input envelope |
| `process_output` | Decode a complete response into `ModelResponse` |
| `stream` / `astream` | Decode the protocol's stream events |

`prepare_request` returns a `PreparedChatRequest`. It retains `extra_body` and
`extra_headers` while exposing their expanded HTTP wire representation.
`ChatTransport.create` and `ChatTransport.acreate` send that request without
changing conversation semantics.

The model class declares the available adapters by mode. Adapter instances are
stateless and can be shared. This example registers a provider whose endpoint
implements the Responses protocol:

```python
from os import getenv

from msgflux.models import ChatAPIModeCapabilities, ChatProviderCapabilities
from msgflux.models.reasoning import OpenAICompatibleReasoningCodec
from msgflux.models.openai_compatible import (
    OpenAICompatibleChatCompletion,
    OpenAIResponsesAPI,
)
from msgflux.models.registry import register_model


class _BaseMyProvider:
    provider = "myprovider"

    def _get_base_url(self):
        return getenv("MYPROVIDER_BASE_URL", "https://api.example.com/v1")

    def _get_api_key(self):
        key = getenv("MYPROVIDER_API_KEY")
        if not key:
            raise ValueError("Please set `MYPROVIDER_API_KEY`")
        return key


@register_model
class MyProviderChatCompletion(
    _BaseMyProvider,
    OpenAICompatibleChatCompletion,
):
    capabilities = ChatProviderCapabilities(
        default_api_mode="responses",
        api_modes=(
            ChatAPIModeCapabilities(
                name="responses",
                adapter=OpenAIResponsesAPI(),
                context_adapter=OpenAIResponsesContextAdapter(),
            ),
        ),
        default_reasoning_codec=OpenAICompatibleReasoningCodec(),
    )
```

The capability declaration is validated when the class is imported: mode names
must be unique, the default must exist, and every mode must contain a
`ChatAPIAdapter`. Protocol-specific reasoning, native compaction, and hosted
tool search support also belong to the corresponding mode. Provider-wide
parameter behavior, such as logprobs or prompt-cache retention, belongs to
`ChatProviderCapabilities`.

`context_adapter` is an explicit opt-in. In this example the remote provider
implements OpenAI-compatible `POST /responses/input_tokens` and
`POST /responses/compact` endpoints, so it can reuse
`OpenAIResponsesContextAdapter`. A provider that only implements response
generation omits it and keeps the provider-neutral token estimate and portable
summary fallback.

`Model.chat_completion("myprovider/model-name")` remains the frontend after the
provider class is registered. A genuinely new protocol, such as Google
Interactions, implements the same `ChatAPIAdapter` methods instead of adding
another conditional branch to the shared model lifecycle.

### 18.6 **Selecting a transport and credentials**

`OpenAICompatibleChatCompletion` uses direct HTTP by default for every declared
chat protocol. Providers only need to override the transport when the remote
service requires behavior that cannot be represented by the shared JSON/SSE
transport:

```python
from msgflux.models.chat_transport import HTTPChatTransport


class MyProviderChatCompletion(
    _BaseMyProvider,
    OpenAICompatibleChatCompletion,
):
    chat_transport = HTTPChatTransport
```

The transport uses HTTPX2, which is installed with the base msgFlux package.
Injected clients must therefore implement the HTTPX2 client interface; legacy
HTTPX v1 clients are not supported.

Passing the class creates an independent transport for every model. For tests,
proxies, or custom networking, pass an instance with injected clients through
the model constructor:

```python
import httpx2

from msgflux.models.chat_transport import HTTPChatTransport

client = httpx2.Client(proxy="http://127.0.0.1:8080")
model = MyProviderChatCompletion(
    "model-name",
    chat_transport=HTTPChatTransport(client=client),
)
```

Injected clients remain owned by the caller. Call `model.close()` and
`await model.aclose()` to release clients created by the model.

Authentication is resolved immediately before each request through
`ModelCredentialResolver`. The default resolver calls the provider's
`_get_api_key()` and produces a Bearer header. Providers with refreshable or
file-backed credentials can supply another resolver without changing their API
adapter or transport.

The resolver may return both headers and a request-specific base URL. They are
resolved together for every attempt, so a refreshed identity cannot be paired
with the endpoint from a previous attempt:

```python
from os import getenv

from msgflux.models import (
    ModelCredentialResolver,
    ResolvedModelCredentials,
)


class SubscriptionCredentials(ModelCredentialResolver):
    def resolve(self, owner):
        token = getenv("MY_SUBSCRIPTION_TOKEN")
        if not token:
            raise ValueError("Please set `MY_SUBSCRIPTION_TOKEN`")
        return ResolvedModelCredentials(
            base_url="https://subscription.example.com/backend-api/codex",
            headers={"Authorization": f"Bearer {token}"},
        )


model = MyProviderChatCompletion(
    "model-name",
    credential_resolver=SubscriptionCredentials(),
)
```

Override `aresolve()` when token refresh requires asynchronous I/O. The async
transport calls it directly; the default implementation delegates to
`resolve()`. Resolvers and resolved request material are excluded from model
serialization, and `ResolvedModelCredentials` does not expose its URL or headers
in `repr()`.

### 18.7 **Inspecting provider HTTP errors**

Direct transports raise `ModelProviderHTTPError` after retries are exhausted or
for a non-retryable response. The exception always contains the HTTP status and
a provider description. Structured provider fields remain available for quota,
context-window, authentication, routing, and data-policy handling:

```python
from msgflux.exceptions import ModelProviderHTTPError

try:
    response = model("Summarize the attached report.")
except ModelProviderHTTPError as error:
    print(error.status_code)
    print(error.description)
    print(error.code, error.error_type, error.param)
    print(error.request_id)
```

OpenAI-style `{ "error": { ... } }` payloads and generic top-level `message`
or `detail` payloads are normalized. Arbitrary response bodies and request
headers are not copied into the exception message, so authentication material
does not appear in normal logs.

## 19. OpenAI SSL Verification

OpenAI-compatible chat completion providers verify SSL certificates by default.
Set `OPENAI_SSL_VERIFY=false` only when you intentionally need to disable
certificate verification, such as when testing behind a local proxy or a
controlled internal network with a custom certificate setup.

```bash
export OPENAI_SSL_VERIFY=false
```

The values `0`, `false`, and `no` disable SSL verification. When the variable
is unset, or set to any other value, SSL verification remains enabled.
