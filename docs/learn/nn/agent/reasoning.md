# Reasoning

Modern language models can "think before answering" — generating an internal chain of thought before producing a final response. In msgFlux there are **two distinct mechanisms** for reasoning, and understanding the difference is key to using them effectively.

## Two Kinds of Reasoning

| | Model-level reasoning | Schema-level reasoning |
|---|---|---|
| **What it is** | The model's native thinking capability (e.g. `reasoning_effort="high"`) | A `generation_schema` that forces structured thinking (CoT, ReAct, SelfConsistency) |
| **Where reasoning lives** | `response.reasoning` — a first-class field on the response object | Inside `response.consume()` — as a field of the structured output (e.g. `result.reasoning`) |
| **Configured via** | Model params: `reasoning_effort`, `return_reasoning`, `reasoning_max_tokens` (OpenRouter only) | Agent param: `generation_schema=ChainOfThought` |
| **Works with any model** | No — requires a reasoning-capable model (Groq gpt-oss, OpenAI o-series, etc.) | Yes — any model that supports structured output |
| **Controllable budget** | Yes — `reasoning_effort`; OpenRouter also supports `reasoning_max_tokens` | No — the model decides how much to write in the schema field |
| **Can combine with tools** | Yes — `reasoning_in_tool_call=True` preserves the chain across calls | Yes — ReAct is specifically designed for tool use |

!!! tip
    You can use both simultaneously. A reasoning model with `reasoning_effort="high"` and `generation_schema=ChainOfThought` will think internally **and** produce a structured reasoning field. The internal trace goes to `response.reasoning`, the schema field goes to `response.consume().reasoning`.

For schema-level reasoning (CoT, ReAct, SelfConsistency), see [Generation Schemas](generation-schemas.md#reasoning-schemas). This page focuses on **model-level reasoning** — the first-class `response.reasoning` field.

---

## 1. **Configuration**

Model-level reasoning is configured at model initialization through parameters forwarded to the provider:

```python
import msgflux as mf

model = mf.Model.chat_completion(
    "openrouter/anthropic/claude-sonnet-4.5",
    return_reasoning=True,        # Store the trace in response.reasoning (default: True)
    reasoning_max_tokens=1024,    # OpenRouter only: cap the thinking budget in tokens
    reasoning_in_tool_call=True,  # Preserve reasoning across tool call rounds
)
```

At the Agent level, one additional config key controls how reasoning is surfaced in the Agent's output:

| Config key | Type | Default | Effect |
|---|---|---|---|
| `reasoning_in_response` | `bool` | `False` | When `True`, the Agent wraps its output as `dotdict(answer=raw_response, reasoning=reasoning)` instead of returning the raw response. |

```python
import msgflux.nn as nn

class Solver(nn.Agent):
    model = model
    instructions = "Solve the problem step by step."
    config = {"reasoning_in_response": True}
```

!!! info "Why `reasoning_in_response` exists"
    By default, the Agent returns the raw response (`str` for text, `dict` for structured output). The reasoning is still available on the underlying `ModelResponse`, but the Agent's `forward()` returns only the content. If your downstream code needs **both** the answer and the reasoning in a single object, enable `reasoning_in_response`. The contract is uniform: always `dotdict(answer=..., reasoning=...)` regardless of whether the raw response is a `str` or `dict`.

---

## 2. **Non-Streaming**

In non-streaming mode, the model completes its full response before returning. Reasoning is available immediately as a string field.

### Basic usage

???+ example

    === "Default (raw response)"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="low",
            return_reasoning=True,
        )

        class Solver(nn.Agent):
            model = model
            instructions = "Solve the problem. Answer with just the result."

        agent = Solver()
        response = agent("What is 15 * 7 + 3?")

        print(type(response))  # <class 'str'>
        print(response)        # "108"
        ```

        The reasoning trace is **not** in the Agent's return value. It lives on the `ModelResponse` inside the pipeline. To access it, use `reasoning_in_response`.

    === "With reasoning_in_response"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="low",
            return_reasoning=True,
        )

        class Solver(nn.Agent):
            model = model
            instructions = "Solve the problem. Answer with just the result."
            config = {"reasoning_in_response": True}

        agent = Solver()
        response = agent("What is 15 * 7 + 3?")

        print(type(response))       # <class 'dotdict'>
        print(response.answer)      # "108"
        print(response.reasoning)   # "15 * 7 = 105, then 105 + 3 = 108"
        ```

    === "Without reasoning"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="low",
            return_reasoning=False,  # Discard reasoning
        )

        class Solver(nn.Agent):
            model = model
            instructions = "Solve the problem."
            config = {"reasoning_in_response": True}

        agent = Solver()
        response = agent("What is 2 + 2?")

        # No reasoning available → no wrapping, plain str
        print(type(response))  # <class 'str'>
        print(response)        # "4"
        ```

        When `return_reasoning=False` (or the model simply doesn't reason), `reasoning` is `None` and `reasoning_in_response` has no effect — the Agent returns the raw response as-is.

    === "Async"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="low",
            return_reasoning=True,
        )

        class Solver(nn.Agent):
            model = model
            instructions = "Solve the problem. Answer with just the result."
            config = {"reasoning_in_response": True}

        agent = Solver()
        response = await agent.acall("What is 15 * 7 + 3?")

        print(response.answer)     # "108"
        print(response.reasoning)  # "15 * 7 = 105, then 105 + 3 = 108"
        ```

### How it works internally

When the Agent calls the model in non-streaming mode, the pipeline is:

```
Agent.forward("What is 15 * 7?")
  │
  ├── _execute_model() → ModelResponse
  │     ├── .data = "108"                     ← the answer
  │     ├── .reasoning = "15*7=105..."        ← the trace
  │     ├── .has_reasoning = True
  │     └── .response_type = "text_generation"
  │
  └── _process_model_response()
        ├── raw_response = model_response.consume()        → "108"
        ├── reasoning = model_response.reasoning           → "15*7=105..."
        │
        └── _prepare_response()
              └── _apply_reasoning_in_response(raw_response, reasoning)
                    ├── reasoning_in_response=True  → dotdict(answer="108", reasoning="15*7=105...")
                    └── reasoning_in_response=False → "108"
```

The key insight is that `model_response.reasoning` is **always** populated (when the provider returns it), regardless of the `reasoning_in_response` config. The config only controls whether the Agent wraps the output.

---

## 3. **Streaming**

Streaming with reasoning introduces a **dual-queue architecture**. Content and reasoning flow through independent queues, allowing consumers to process them in parallel or sequentially.

### Consuming streams

When `stream=True`, the Agent returns a `ModelStreamResponse`. Both `consume()` and `consume_reasoning()` are async generators:

???+ example

    === "Content + Reasoning"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="low",
            return_reasoning=True,
        )

        class Assistant(nn.Agent):
            model = model
            instructions = "Answer concisely."
            config = {"stream": True}

        agent = Assistant()
        response = await agent.acall("What is 2+2?")

        # Consume content chunks
        async for chunk in response.consume():
            print(chunk, end="", flush=True)

        print()

        # Consume reasoning chunks
        async for chunk in response.consume_reasoning():
            print(chunk, end="", flush=True)
        ```

    === "Reasoning First"

        The queues are independent — you can consume reasoning before content. This is useful when you want to display the chain of thought in a UI before showing the answer:

        ```python
        response = await agent.acall("Solve: 15 * 7 + 3")

        # Read reasoning first
        print("Thinking:")
        async for chunk in response.consume_reasoning():
            print(chunk, end="", flush=True)

        # Then read the answer
        print("\n\nAnswer:")
        async for chunk in response.consume():
            print(chunk, end="", flush=True)
        ```

    === "Sync Polling"

        In sync contexts, the stream runs in a background thread. After the stream completes, the accumulated fields are available:

        ```python
        import time

        agent = Assistant()
        response = agent("What is 2+2?")

        # first_chunk_event fires on the first token (often reasoning)
        response.first_chunk_event.wait(timeout=10)

        # Wait for stream to complete
        for _ in range(50):
            if response.metadata is not None:
                break
            time.sleep(0.1)

        # After completion
        print(response.reasoning)       # Full accumulated reasoning
        print(response.has_reasoning)   # True
        ```

    === "FastAPI — Dual Endpoints"

        ```python
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        import msgflux as mf
        import msgflux.nn as nn

        app = FastAPI()

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="low",
            return_reasoning=True,
        )

        class Assistant(nn.Agent):
            model = model
            instructions = "Answer concisely."
            config = {"stream": True}

        agent = Assistant()

        @app.get("/chat")
        async def chat(query: str):
            response = await agent.acall(query)
            return StreamingResponse(
                response.consume(),
                media_type="text/plain",
            )

        @app.get("/chat/reasoning")
        async def chat_reasoning(query: str):
            response = await agent.acall(query)
            return StreamingResponse(
                response.consume_reasoning(),
                media_type="text/plain",
            )
        ```

### The two-event system

Streaming responses use two events to signal different stages:

| Event | Fires when | Purpose |
|---|---|---|
| `first_chunk_event` | First token arrives (reasoning **or** content) | Signals the stream is alive. Fires early — often on the first reasoning token. |
| `_response_type_event` | `response_type` is determined (`"text_generation"` or `"tool_call"`) | The Agent waits on this event before deciding how to process the response. |

This separation matters because reasoning models emit reasoning tokens **before** any content. Without it, the Agent would block on `response_type` until the first content token arrives — potentially seconds of wasted time.

```
Timeline:
  ┌─ reasoning tokens ──────────────────┐┌── content tokens ───────────┐
  │  think think think think think ...   ││  Hello, the answer is ...   │
  ▲                                      ▲                              ▲
  │                                      │                              │
  first_chunk_event                      _response_type_event           metadata set
  (fires here)                           (fires here)                   (stream done)
```

Inside the Agent, the flow is:

```python
# Agent._process_model_response():
if isinstance(model_response, ModelStreamResponse):
    wait_for_event(model_response._response_type_event)  # blocks here, not on first_chunk

# Now response_type is guaranteed to be set
if "tool_call" in model_response.response_type:
    # enter tool call loop...
else:
    # return stream response to caller
```

### How the dual-queue works internally

```
Provider stream (background thread)
│
├── reasoning chunk → add_reasoning(chunk)
│     ├── has_reasoning = True         (first time only)
│     ├── first_chunk_event.set()      (first time only)
│     └── chunk → reasoning queue
│
├── content chunk  → add(chunk)
│     ├── set_response_type("text_generation")
│     │   └── _response_type_event.set()   (first time only)
│     ├── first_chunk_event.set()          (if not already)
│     └── chunk → content queue
│
└── finally:
      ├── stream_response.reasoning = accumulated   ← full text available
      ├── add_reasoning(None)                       ← sentinel (end of reasoning)
      ├── add(None)                                 ← sentinel (end of content)
      ├── _response_type_event.set()                ← safety net
      └── set_metadata(usage)
```

Each queue uses a `deque` as a pending buffer that is flushed into an `asyncio.Queue` when a consumer first calls `consume()` / `consume_reasoning()`. The `None` sentinel signals end-of-stream to the async generator.

!!! info "reasoning_in_response has no effect in streaming"
    When `stream=True`, the Agent returns the `ModelStreamResponse` directly — it does not wrap it. The consumer accesses reasoning through `consume_reasoning()` and content through `consume()`. The `reasoning_in_response` config only applies to non-streaming responses.

---

## 4. **Reasoning Across Tool Calls**

When a reasoning model calls tools, it normally loses its chain of thought between rounds. Enable `reasoning_in_tool_call=True` on the model to preserve the reasoning context.

### How it works

After each tool call round, the `ToolCallAggregator` formats the assistant message that goes back into the conversation history. When `reasoning_in_tool_call=True`, the reasoning is embedded in `<think>` tags inside that message:

```
Message history:
[
  {"role": "user", "content": "What is (14+28)*3-7?"},

  {"role": "assistant",
   "content": "<think>I need to compute (14+28) first, then multiply by 3, then subtract 7. Let me use the calculator.</think>",
   "tool_calls": [{"function": {"name": "calc", "arguments": {"expr": "14+28"}}}]},

  {"role": "tool", "tool_call_id": "call_1", "content": "42"},

  {"role": "assistant",
   "content": "<think>14+28=42. Now I need 42*3. Let me call calc again.</think>",
   "tool_calls": [{"function": {"name": "calc", "arguments": {"expr": "42*3"}}}]},

  {"role": "tool", "tool_call_id": "call_2", "content": "126"},

  {"role": "assistant", "content": "The answer is 119."}
]
```

The model sees its own previous reasoning at each step, enabling coherent multi-step problem solving.

!!! info "Two separate reasoning stores"
    The `ToolCallAggregator` keeps its own copy of the reasoning for message formatting (`<think>` tags in the conversation). The `ModelResponse.reasoning` field on the **final** model call reflects only the reasoning from that last call. These are intentionally separate — the conversation history needs the full chain, while the response field exposes the latest trace.

???+ example

    === "Basic Tool Call with Reasoning"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        def multiply(a: int, b: int) -> int:
            """Multiply two numbers together."""
            return a * b

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="high",
            return_reasoning=True,
            reasoning_in_tool_call=True,
        )

        class Calculator(nn.Agent):
            model = model
            instructions = "Use the tools to compute the result. Answer with just the number."
            tools = [add, multiply]

        agent = Calculator()
        response = agent("What is (3 + 5) * 4?")
        print(response)  # "32"
        ```

    === "With reasoning_in_response"

        ```python
        class Calculator(nn.Agent):
            model = model
            instructions = "Use the tools to compute the result. Answer with just the number."
            tools = [add, multiply]
            config = {"reasoning_in_response": True}

        agent = Calculator()
        response = agent("What is (3 + 5) * 4?")

        # If the final model call includes reasoning:
        if isinstance(response, dict):
            print(response.answer)     # "32"
            print(response.reasoning)  # "3+5=8, 8*4=32"
        else:
            # Final call had no reasoning (model may skip it on simple answers)
            print(response)  # "32"
        ```

    === "Async with Tools"

        ```python
        agent = Calculator()
        response = await agent.acall("What is (3 + 5) * 4?")
        print(response)  # "32"
        ```

!!! warning "Reasoning on the final call is not guaranteed"
    After the tool call loop completes, the model makes a final call to produce the answer. This final call may or may not include reasoning — it depends on the model and the complexity of the remaining task. When `reasoning_in_response=True` and the final call has no reasoning, the Agent returns the raw response (no `dotdict` wrapping).

---

## 5. **Combining with Generation Schemas**

Model-level reasoning and schema-level reasoning serve different purposes and can be combined:

| Approach | Reasoning lives in | Use case |
|---|---|---|
| Model-level only | `response.reasoning` | When you want the model's native thinking without constraining the output format |
| Schema-level only | `response.consume().reasoning` | When you want explicit, structured reasoning visible in the output |
| Both combined | Both fields populated | Maximum reasoning quality — the model thinks internally **and** produces structured reasoning |

???+ example

    === "Model-level only"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="high",
            return_reasoning=True,
        )

        class Solver(nn.Agent):
            model = model
            instructions = "Solve the problem."
            config = {"reasoning_in_response": True}

        agent = Solver()
        response = agent("What is 25 * 4 + 17?")

        # response.answer = "117"
        # response.reasoning = "25 * 4 = 100, 100 + 17 = 117"  (model's internal trace)
        ```

    === "Schema-level only"

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.generation.reasoning import ChainOfThought

        model = mf.Model.chat_completion("openai/gpt-4.1-mini")  # no reasoning_effort

        class Solver(nn.Agent):
            model = model
            generation_schema = ChainOfThought

        agent = Solver()
        result = agent("What is 25 * 4 + 17?")

        # result.reasoning = "Step 1: 25 * 4 = 100..."  (schema field)
        # result.final_answer = "117"
        ```

    === "Both combined"

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.generation.reasoning import ChainOfThought

        model = mf.Model.chat_completion(
            "groq/openai/gpt-oss-120b",
            reasoning_effort="high",
            return_reasoning=True,
        )

        class Solver(nn.Agent):
            model = model
            generation_schema = ChainOfThought
            config = {"reasoning_in_response": True}

        agent = Solver()
        response = agent("What is 25 * 4 + 17?")

        # response.answer = {"reasoning": "Step 1: ...", "final_answer": "117"}  (schema)
        # response.reasoning = "The user asks 25*4+17. Let me compute..."        (model trace)
        ```

        The schema reasoning is a structured, user-facing explanation. The model reasoning is the raw internal trace — often more detailed and less polished.

---

## 6. **Verbose Mode**

When `verbose=True`, the Agent prints both the reasoning trace and the response to the console:

```python
class Solver(nn.Agent):
    model = model
    instructions = "Solve the problem."
    config = {"verbose": True}

agent = Solver()
agent("What is 15 * 7?")
```

Console output:

```
[solver][reasoning] 15 * 7 = 105
[solver][response] 105
```

This is useful for debugging the relationship between the model's thinking and its final answer.

---

## 7. **Quick Reference**

### Model parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reasoning_effort` | `str` | — | `"minimal"`, `"low"`, `"medium"`, `"high"` |
| `return_reasoning` | `bool` | `True` | Store reasoning in `response.reasoning` |
| `reasoning_max_tokens` | `int` | — | OpenRouter-only cap on reasoning token budget |
| `reasoning_in_tool_call` | `bool` | `False` | Embed reasoning in `<think>` tags across tool call rounds |
| `enable_thinking` | `bool` | `False` | Provider-level switch (e.g. Anthropic) |

### Agent config

| Config key | Type | Default | Description |
|---|---|---|---|
| `reasoning_in_response` | `bool` | `False` | Wrap output as `dotdict(answer=..., reasoning=...)` |

### Response API

| Non-streaming | Streaming | Description |
|---|---|---|
| `response.consume()` → `str` or `dict` | `response.consume()` → `AsyncGenerator[str, None]` | Final answer |
| `response.consume_reasoning()` → `str` or `None` | `response.consume_reasoning()` → `AsyncGenerator[str, None]` | Reasoning trace |
| `response.reasoning` → `str` or `None` | `response.reasoning` → `str` or `None` (after stream ends) | Direct attribute |
| `response.has_reasoning` → `bool` (property) | `response.has_reasoning` → `bool` (mutable flag) | Discoverability |

### See also

- [Chat Completion — Reasoning Models](../../models/chat_completion.md#12-reasoning-models) — Model-level reasoning reference with internal architecture details
- [Generation Schemas — Reasoning Schemas](generation-schemas.md#reasoning-schemas) — CoT, ReAct, SelfConsistency
- [Tools](tools.md) — Tool calling overview
- [Streaming](streaming.md) — General streaming overview
