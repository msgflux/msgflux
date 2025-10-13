# `Chat Completion`

The [`chat_completion`](../api-reference/models/types/chat_completion.md) model is the most common and versatile model for natural language interactions. It processes messages in a conversational format and supports advanced features such as multimodal input and output, structured data generation, and tool (function) calls.

We will explain it's features to understand how it works and its limitations. And why we need a higher-level abstraction like `nn.Agent` to make it easier to create applications.

## ✦₊⁺ Overview

All models have the same calling interface, differing only in the initialization of their classes.

=== "__init__"
    ::: msgflux.models.providers.openai.OpenAIChatCompletion.__init__ 
        options:
            show_signature: false
            show_source: false
            show_root_heading: false

=== "__call__"
    ::: msgflux.models.providers.openai.OpenAIChatCompletion.__call__
        options:
            show_signature: false
            show_source: false
            show_root_heading: false

{! ../_includes/init_chat_completion_model.md !}

### 1. **Stateless**

Chat completion models do **not maintain** state between calls. All context information (previous messages, system prompt, etc.) must be provided on each new call.

```python
prompt = "What is Deep Learning?"
response = model(prompt)

assistant = response.consume()
print(assistant)

prompt_2 = "What are the most impactful architectures?"

messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": assistant}
    {"role": "user", "content": prompt_2},     
]

response = model(messages)
print(response.consume())
```

### 2. **Multimodal**

Modern models are natively multimodal. This means they can:

- Understand and generate text

- Interpret and generate images

- Listen to and generate speech

```python
messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "What do you see in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]}
]
```

### 3. **Structured Generation**

The model can be guided to produce structured responses according to a user-defined schema.

In msgflux this is called `generation_schema`. The name shows not only that the model produces a `structured output`, but also that it follows a schema.

For this, we use `msgspec.Struct` as the structure format:

```python
import msgspec

class Weather(msgspec.Struct):
    temperature: float
    condition: str

response = model(
    messages=[...],
    generation_schema=Weather
)
```

The response is decoded directly as an instance of the Weather class, which simplifies consumption of the response and avoids manual text parsing.

`generation_schema` maybe the most important feature in `chat_completion` models. This feature enables things like `ReAct`, `CoT`, new content generation, guided data extraction, etc. In this framework following tutorials we make extensive use of it.

We offer a set of schemes that assist in model planning

```python
from msgflux.generation.plan import (
    ChainOfThoughts,
    ReAct,
    SelfConsistency,
    TreeOfThoughts
)
```

In the case of `ReAct`, where tool calls occur, it's necessary to implement control flow to feed back into the model. This is already present in `nn.Agent`.

---

### 4. **Tools**

When we provide a set of tools (`tool_schemas`), the model may **suggest** that one of them be called—but it does not automatically execute them.

Instead, it returns a call intent, which is captured and processed by an internal component called the `ToolCallAggregator`.

This class collects and organizes tool calls suggested by the model, especially in streaming mode, where arguments arrive in fragmented form.

Main responsibilities:

- Reassemble parts of calls during the stream (`process`)

- Convert raw data into complete functional calls (`get_calls`)

- Insert tool results (`insert_results`)

- Generate messages in the correct format to follow the flow with the model (`get_messages`)

---

### 5. **Prefilling**

Prefilling is a technique used to start the model message. A classic usage is: 
    
`let's think step by step`

The model detects that it has started the sequence and then the message it will send next is the **continuation** of it. This technique is particularly very useful for generating structured outputs.
