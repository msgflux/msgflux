# Chat Completion

The `chat_completion` model is the most versatile model type for natural language interactions. It processes messages in conversational format and supports advanced features like multimodal input/output, structured generation, and tool calling.

!!! info "Dependencies"
    Most providers use the OpenAI Python client under the hood, so a single extra covers all of them:

    === "uv"
        ```bash
        uv add msgflux[openai]
        ```

    === "pip"
        ```bash
        pip install msgflux[openai]
        ```

    See [Dependency Management](../../dependency-management.md) for the complete provider matrix.

## ✦₊⁺ Overview

--8<-- "docs/_includes/init_chat_completion_model.md"

Chat completion models are stateless - they don't maintain conversation history between calls. You must provide all context (previous messages, system prompt, etc.) in each request.

### Quick Start

???+ example

    ```python
    import msgflux as mf

    # mf.set_envs(OPENAI_API_KEY="...")

    # Create model
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    response = model("Hello!")
    print(response.consume())
    ```

## 1. **Model Initialization**

### 1.1 **Basic Parameters**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion(
        "openai/gpt-4.1-mini",
        # --- Generation ---
        temperature=0.7,               # Randomness (0-2)
        max_tokens=1000,               # Max output tokens (includes reasoning tokens)
        top_p=0.9,                     # Nucleus sampling (alternative to temperature)
        stop=["\n\n"],                 # Stop sequences (up to 4)
        # --- Reasoning ---
        reasoning_effort="medium",     # "minimal", "low", "medium", "high"
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
        validate_typed_parser_output=False,  # Validate typed parser output with schema
        verbose=False,                 # Print raw output before transformation
        # --- Search ---
        web_search_options={},         # Web search config (OpenAI / OpenRouter only)
        prompt_cache_retention="24h",  # OpenAI only: "in_memory" or "24h"
        # --- Infrastructure ---
        base_url="https://api.openai.com/v1",  # Override provider API endpoint
        context_length=128000,         # Override maximum context window
        enable_cache=True,             # Cache identical API responses in-process
        cache_size=128,                # Max number of cached entries
        retry=None,                    # Custom tenacity retry configuration
    )
    ```

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

???+ example

    === "Defining Tools"

        ```python
        import msgflux as mf
        from msgflux.tools import ToolDefinitions

        # Define tool schema
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country, e.g. Paris, France"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False
                }
            }
        }]

        tool_definitions = ToolDefinitions(schemas=tools)

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        response = model(
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tool_definitions=tool_definitions,
        )

        # Get tool calls
        tool_call_agg = response.consume()
        calls = tool_call_agg.get_calls()

        for call in calls:
            print(f"Tool: {call['function']['name']}")
            print(f"Arguments: {call['function']['arguments']}")
        # Tool: get_weather
        # Arguments: {'location': 'Paris, France', 'unit': 'celsius'}
        ```

    === "Tool Choice"

        ```python
        import msgflux as mf
        from msgflux.tools import ToolDefinitions

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        # Auto - model decides
        response = model(
            messages=[{"role": "user", "content": "What's the weather?"}],
            tool_definitions=ToolDefinitions(schemas=tools, choice="auto"),
        )

        # Required - must call at least one tool
        response = model(
            messages=[{"role": "user", "content": "What's the weather?"}],
            tool_definitions=ToolDefinitions(schemas=tools, choice="required"),
        )

        # Specific function - must call this exact function
        response = model(
            messages=[{"role": "user", "content": "Paris weather"}],
            tool_definitions=ToolDefinitions(schemas=tools, choice="get_weather"),
        )
        ```

    === "Full Flow"

        ```python
        import msgflux as mf
        from msgflux.tools import ToolDefinitions

        def get_weather(location, unit="celsius"):
            """Simulate weather API call."""
            return f"The weather in {location} is 22°{unit[0].upper()}"

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }]

        tool_definitions = ToolDefinitions(schemas=tools)

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        # Initial request
        messages = [{"role": "user", "content": "What's the weather in Paris?"}]

        response = model(messages=messages, tool_definitions=tool_definitions)
        tool_call_agg = response.consume()

        # Execute tool calls
        tool_functions = {"get_weather": get_weather}
        calls = tool_call_agg.get_calls()

        for call in calls:
            func_name = call['function']['name']
            func_args = call['function']['arguments']

            # Execute function
            result = tool_functions[func_name](**func_args)

            # Add result to aggregator
            tool_call_agg.insert_results(call['id'], result)

        # Get messages with tool results
        tool_messages = tool_call_agg.get_messages()
        messages.extend(tool_messages)

        # Final response with tool results
        final_response = model(messages=messages)
        print(final_response.consume())
        # "The weather in Paris is currently 22°C."
        ```

    === "Streaming"

        ```python
        import msgflux as mf
        from msgflux.tools import ToolDefinitions

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")

        response = model(
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tool_definitions=ToolDefinitions(schemas=tools),
            stream=True
        )

        # Tool calls are aggregated during streaming
        tool_call_agg = response.consume()

        # After stream completes, get calls
        calls = tool_call_agg.get_calls()
        print(calls)
        ```

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

!!! info "Dependencies"
    Install the OpenAI extra if you haven't already:

    === "uv"
        ```bash
        uv add msgflux[openai]
        ```

    === "pip"
        ```bash
        pip install msgflux[openai]
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
| `reasoning_effort` | How much reasoning to do. One of `"minimal"`, `"low"`, `"medium"`, `"high"`. | — |
| `reasoning_max_tokens` | OpenRouter-only hard cap (in tokens) for the internal thinking budget. Maps to `extra_body.reasoning.max_tokens` and cannot be combined with `reasoning_effort`. | — |
| `return_reasoning` | Store the reasoning trace in `response.reasoning`. When `False`, reasoning is discarded even if the provider returns it. | `True` |
| `enable_thinking` | Activate extended model reasoning. | `False` |
| `reasoning_in_tool_call` | Preserve reasoning context across tool calls so the model keeps its chain of thought intact. When enabled, the `ToolCallAggregator` embeds the reasoning in `<think>` tags inside the assistant message history, allowing the model to see its previous reasoning when processing tool results. | `False` |

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

When a reasoning model responds, the response object has two independent data paths:

```
ModelResponse
├── .data          ← final answer (str, dict, ToolCallAggregator)
├── .reasoning     ← chain of thought (str or None)
├── .has_reasoning ← True if reasoning is present (bool)
├── .response_type ← "text_generation", "structured", "tool_call"
└── .metadata      ← usage stats, annotations, etc.
```

The key methods on a non-streaming response:

| Method / Property | Returns | Description |
|---|---|---|
| `response.consume()` | `str`, `dict`, or `ToolCallAggregator` | The final answer, always in its natural type. |
| `response.consume_reasoning()` | `str` or `None` | The full reasoning trace, or `None` if the model didn't reason. |
| `response.reasoning` | `str` or `None` | Same as `consume_reasoning()` — direct attribute access. |
| `response.has_reasoning` | `bool` | `True` when `reasoning is not None`. Useful for conditional logic without inspecting the string. |

!!! info "Why `consume()` never changes type"
    In earlier versions, when a model reasoned, `consume()` returned a `dotdict(answer=..., reasoning=...)` instead of a plain `str`. This caused silent type changes that broke downstream code. Now `consume()` always returns the answer and `consume_reasoning()` returns the reasoning — two separate channels, predictable types.

### 12.3 **Provider Behaviour**

Not all reasoning providers behave the same way:

| Provider | Exposes trace via `return_reasoning` | Reasoning tokens in metadata | Notes |
|---|---|---|---|
| **Groq** (`groq/openai/gpt-oss-*`) | Yes — `response.reasoning` | Yes | Reasoning returned as raw text in API response |
| **OpenAI** (`openai/o*`, `openai/gpt-5-*`) | No — reasoning is fully internal | Yes | Only token counts available via `response.metadata` |
| **Anthropic** (via `enable_thinking`) | Yes — `response.reasoning` | Yes | Uses `enable_thinking=True` instead of `reasoning_effort` |

All providers that inherit from `OpenAIChatCompletion` (Groq, vLLM, Ollama, OpenRouter, Together, SambaNova, Cerebras) share the same reasoning extraction logic. When the provider returns a reasoning field, it is automatically separated from the content and placed in `response.reasoning`.

### 12.4 **Reasoning Effort**

`reasoning_effort` is the primary knob. Higher effort means the model spends more tokens on internal reasoning, which typically improves answer quality on hard problems.

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
    print(f"Reasoning tokens used: {usage['completion_tokens_details']['reasoning_tokens']}")
    # Reasoning tokens used: 64
    ```

### 12.6 **Controlling the Reasoning Budget**

`reasoning_max_tokens` is an OpenRouter-only parameter in msgFlux. It maps to `extra_body={"reasoning": {"max_tokens": ...}}` on the OpenRouter API and cannot be combined with `reasoning_effort`.

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion(
        "openrouter/anthropic/claude-sonnet-4.5",
        reasoning_max_tokens=2000,
    )

    response = model("What's the weather like in Boston? Then recommend what to wear.")
    print(response.consume_reasoning())
    print(response.consume())
    ```

OpenRouter's own API examples pass the reasoning budget inside `extra_body={"reasoning": {"max_tokens": 2000}}`; msgFlux translates the model-level `reasoning_max_tokens` parameter to that shape for you.

### 12.7 **Streaming with Reasoning**

Streaming introduces a dual-queue architecture. Content and reasoning flow through independent queues, allowing consumers to process them in parallel or sequentially.

#### How it works internally

When `stream=True`, the model returns a `ModelStreamResponse` instead of a `ModelResponse`. Internally, two separate `asyncio.Queue` instances handle the data flow:

```
Provider stream thread
│
├── reasoning chunk → stream_response.add_reasoning(chunk) → reasoning queue
├── reasoning chunk → stream_response.add_reasoning(chunk) → reasoning queue
├── content chunk   → stream_response.add(chunk)           → content queue
├── content chunk   → stream_response.add(chunk)           → content queue
├── ...
├── stream_response.add_reasoning(None)  ← reasoning sentinel (end of reasoning)
└── stream_response.add(None)            ← content sentinel (end of content)
```

At the end of the stream, the provider also sets `stream_response.reasoning` with the full accumulated reasoning text, so it is available as a single string after the stream completes.

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

The `consume()` and `consume_reasoning()` methods become async generators in streaming mode:

???+ example

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

#### Thread safety

Both queues use `threading.Lock` to protect the bind/pending-flush operations. The producer (provider stream thread) calls `add()` / `add_reasoning()` safely from any thread via `loop.call_soon_threadsafe()`. Pending chunks are buffered in a `deque` until a consumer binds the queue to an event loop — at that point all pending chunks are flushed into the `asyncio.Queue` atomically under the lock.

### 12.8 **Reasoning Across Tool Calls**

When a reasoning model uses tools it normally loses its chain of thought between calls. `reasoning_in_tool_call=True` preserves the reasoning context so the model can continue thinking coherently after each tool result.

Internally, when this flag is enabled, the `ToolCallAggregator` embeds the reasoning in `<think>` tags inside the assistant message that gets appended to the conversation history:

```
# Message history with reasoning_in_tool_call=True:
[
    {"role": "user", "content": "What is (14 + 28) × 3 − 7?"},
    {"role": "assistant", "content": "<think>I need to break this into steps...</think>",
     "tool_calls": [{"function": {"name": "calculate", "arguments": {"expression": "14 + 28"}}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "42"},
    # Model sees its previous reasoning and can continue the chain
]
```

This is separate from the response-level `reasoning` field. The `ToolCallAggregator` keeps its own copy of the reasoning for message formatting, while `response.reasoning` on the final `ModelResponse` reflects the reasoning from the last model call in the loop.

???+ example

    ```python
    import msgflux as mf
    from msgflux.tools import ToolDefinitions

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
        tool_definitions=ToolDefinitions(schemas=tools),
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
| `BaseResponse` | Non-streaming (`stream=False`) | `self.reasoning: str \| None` — set once by the provider after the full API response arrives. `has_reasoning` is a `@property` that checks `self.reasoning is not None`. |
| `BaseStreamResponse` | Streaming (`stream=True`) | `self.reasoning: str \| None` — accumulated by the provider as chunks arrive. `has_reasoning` is a mutable `bool` flag, flipped to `True` on the first non-None reasoning chunk via `add_reasoning()`. |

Both classes inherit from `CoreResponse`, which provides `set_metadata()` and `set_response_type()`.

#### Provider flow (non-streaming)

```
model("prompt")
  │
  ├── OpenAIChatCompletion._generate()
  │     └── client.chat.completions.create(**params)
  │           └── API response
  │
  └── _process_completion_model_output(model_output)
        ├── reasoning = _extract_reasoning(message)
        ├── response.reasoning = reasoning       # ← set directly on response
        ├── response.add(content)                # ← data is pure content
        └── response.set_response_type("text_generation")
```

The `_extract_reasoning()` method checks for provider-specific reasoning fields in the API response (e.g., `message.reasoning_content` for Groq/OpenAI-compatible providers). If `return_reasoning=False`, it skips extraction entirely.

#### Provider flow (streaming)

```
model("prompt", stream=True)
  │
  ├── OpenAIChatCompletion._stream_generate()  # runs in background thread
  │     └── for chunk in client.chat.completions.create(stream=True):
  │           ├── reasoning_chunk? → stream_response.add_reasoning(chunk)
  │           │                      ├── has_reasoning = True (first time)
  │           │                      └── first_chunk_event.set() (first time)
  │           │
  │           └── content_chunk?   → stream_response.add(chunk)
  │                                  ├── set_response_type("text_generation")
  │                                  │   └── _response_type_event.set()
  │                                  └── first_chunk_event.set() (if not already)
  │
  │     finally:
  │           ├── stream_response.reasoning = accumulated_reasoning
  │           ├── stream_response.add_reasoning(None)  # sentinel
  │           ├── stream_response.add(None)             # sentinel
  │           ├── _response_type_event.set()            # safety net
  │           └── stream_response.set_metadata(usage)
  │
  └── returns stream_response immediately (stream runs in background)
```

The `None` sentinels signal end-of-stream to the `consume()` / `consume_reasoning()` async generators. The safety net `_response_type_event.set()` in the `finally` block ensures the event is always fired, even if the stream errors out or the model returns no content chunks (e.g., a pure tool call response).

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

`logprobs` and `top_logprobs` are forwarded for `openai/...` chat completion models. Providers that inherit from `OpenAIChatCompletion` but are not OpenAI do not receive these fields.

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

`prompt_cache_retention` is an OpenAI-only initialization parameter. msgFlux forwards it only for `openai/...` chat completion models. OpenAI-compatible providers that inherit from `OpenAIChatCompletion` do not use it.

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

All responses include metadata with usage information:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

    response = model(messages=[{"role": "user", "content": "Hello"}])

    # Access metadata
    print(response.metadata)
    # {
    #     'usage': {
    #         'completion_tokens': 9,
    #         'prompt_tokens': 19,
    #         'total_tokens': 28
    #     }
    # }

    # Calculate cost using profile
    from msgflux.models.profiles import get_model_profile

    profile = get_model_profile("gpt-4.1-mini", provider_id="openai")
    if profile:
        usage = response.metadata.usage
        cost = profile.cost.calculate(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens
        )
        print(f"Request cost: ${cost:.4f}")
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

If the service you want to use exposes an **OpenAI-compatible API**, you can add it as a provider by subclassing `OpenAIChatCompletion`. The process has two stages depending on how compatible the endpoint is.

### 18.1 **Stage 1 — URL and API key only**

When the target API is fully OpenAI-compatible and only requires a different base URL and authentication key, the entire subclass is a small configuration mixin plus the `@register_model` decorator.

???+ example "Custom provider — minimal setup"

    ```python
    from os import getenv
    from msgflux.models.providers.openai import OpenAIChatCompletion
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
    class MyProviderChatCompletion(_BaseMyProvider, OpenAIChatCompletion):
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

    from msgflux.models.providers.openai import OpenAIChatCompletion
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
    class MyProviderChatCompletion(_BaseMyProvider, OpenAIChatCompletion):
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

### 18.3 **Stage 3 — Using a different client**

The two previous stages assume the service is reached through the `openai` Python package. If you want to use a completely different HTTP client or SDK — one that is **not** the `openai` package but still exposes a compatible interface — override `_initialize` instead.

`_initialize` is called once at construction time. Its job is to populate three things on `self`:

| Attribute | Type | Purpose |
|---|---|---|
| `self.client` | any object | Sync client; must expose `.chat.completions.create(**params)` |
| `self.aclient` | any object | Async client; must expose `await .chat.completions.create(**params)` |
| `self._response_cache` | `ResponseCache \| None` | In-memory response cache (set to `None` to disable) |

It must also wrap `self.__call__` and `self.acall` with the retry decorator so that the model's retry logic still works.

The response object returned by `.chat.completions.create()` must be OpenAI-compatible: it needs `.choices[0].message` and `.usage` attributes. Any SDK that advertises OpenAI compatibility will satisfy this contract.

???+ example "Custom provider — with a different client"

    ```python
    from os import getenv

    from msgflux.models.cache import ResponseCache
    from msgflux.models.providers.openai import OpenAIChatCompletion
    from msgflux.models.registry import register_model
    from msgflux.utils.tenacity import apply_retry, default_model_retry

    # Replace with the SDK you actually want to use.
    # It must expose client.chat.completions.create() / aclient.chat.completions.create().
    import my_sdk


    class _BaseMyProvider:
        provider: str = "myprovider"

        def _get_base_url(self):
            return getenv("MYPROVIDER_BASE_URL", "https://api.myprovider.com/v1")

        def _get_api_key(self):
            key = getenv("MYPROVIDER_API_KEY")
            if not key:
                raise ValueError("Please set `MYPROVIDER_API_KEY`")
            return key

        def _initialize(self):
            base_url = self._get_base_url()
            api_key = self._get_api_key()

            # Sync and async clients from your chosen SDK.
            self.client = my_sdk.Client(base_url=base_url, api_key=api_key)
            self.aclient = my_sdk.AsyncClient(base_url=base_url, api_key=api_key)

            # Preserve response caching (reads enable_cache / cache_size set by __init__).
            cache_size = getattr(self, "cache_size", 128)
            enable_cache = getattr(self, "enable_cache", None)
            self._response_cache = (
                ResponseCache(maxsize=cache_size) if enable_cache else None
            )

            # Preserve retry logic.
            retry_config = getattr(self, "retry", None)
            self.__call__ = apply_retry(
                self.__call__, retry_config, default=default_model_retry
            )
            self.acall = apply_retry(
                self.acall, retry_config, default=default_model_retry
            )


    @register_model
    class MyProviderChatCompletion(_BaseMyProvider, OpenAIChatCompletion):
        """MyProvider Chat Completion using a custom SDK."""
    ```

The pattern above keeps caching and retry behaviour identical to every other built-in provider. The only thing that changes is the objects assigned to `self.client` and `self.aclient`.

!!! note
    The response returned by `.chat.completions.create()` is consumed by `_process_model_output`. That method reads `model_output.choices[0].message` and `model_output.usage.to_dict()`. If your SDK returns a different structure, also override `_process_model_output` to adapt it.
