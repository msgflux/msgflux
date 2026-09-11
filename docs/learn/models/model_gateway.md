
# `ModelGateway` — Resilient Model Manager

The [`ModelGateway`](../../api-reference/models/model_gateway.md) class is an **orchestration layer** over multiple models of the same type (e.g., multiple `chat_completion` models), allowing:

- **Automatic fallback** between models.
- **Time-based** model availability constraints.
- **Model preference** selection via aliases.
- **Optional strict routing** with fallback disabled.
- **Model capability descriptions** for model-facing selectors.
- **Control of execution attempts** with exception handling.
- **Consistent model typing validation**.

It's ideal for production-grade model orchestration where reliability and control over model usage are required.

## ✦₊⁺ Overview

## 1. **Usage**

```bash
pip install msgflux
```

All you need is:

- Models may be `BaseModel` instances or chat-completion shorthand strings in
  `"provider/model-id"` form.
- All models **must be of the same `model_type`**.
- Each deployment **must have a unique `model_name`**.
- At least **2 deployments** are recommended for effective fallback.
---

### 1.1 **Query**

```python
import msgflux as mf

# mf.set_envs(OPENAI_API_KEY="...")

gateway = mf.ModelGateway([
    {
        "model_name": "fast",
        "model": "openai/gpt-5.6-luna",
        "description": "Fast for clear, repeatable tasks.",
    },
    {
        "model_name": "deep",
        "model": "openai/gpt-5.6-sol",
        "description": "Deep reasoning for ambiguous, multi-step tasks.",
    },
])

response = gateway(
    messages="Explain why shared mutable state is risky under concurrency.",
    model_preference="deep",
)
print(response.consume())
```

The selected alias is attempted first. With the default `fallback=True`, a
failure or time restriction causes the Gateway to try another available
deployment. Use `fallback=False` when it must attempt only the selected model:

```python
strict_gateway = mf.ModelGateway(
    [
        {
            "model_name": "fast",
            "model": "openai/gpt-5.6-luna",
            "description": "Fast for clear, repeatable tasks.",
        },
        {
            "model_name": "deep",
            "model": "openai/gpt-5.6-sol",
            "description": "Deep reasoning for ambiguous, multi-step tasks.",
        },
    ],
    fallback=False,
)
```

`description` is optional for direct routing. It is required when an Agent
using the Gateway is exposed through
[`AgentTool`](../nn/agent/tools/agent-tool.md#selecting-a-subagent-model).

### 1.2 **Simulated Failure**

```python
import msgflux as mf
from msgflux.models.base import BaseModel
from msgflux.models.types import ChatCompletionModel

# Simulate a model that fails
class BrokenModel(BaseModel, ChatCompletionModel):
    provider = "mock"
    model_id = "broken-model"

    def _initialize(self):
        pass

    def __call__(self, **kwargs):
        raise RuntimeError("Simulate failure")

broken = BrokenModel()
fallback = mf.Model.chat_completion("openai/gpt-5.6-luna")

gateway_broken = mf.ModelGateway([
    {"model_name": "broken", "model": broken},
    {"model_name": "fallback", "model": fallback},
])

response = gateway_broken(messages="Summarize why the primary model failed.")
print(response.consume())
```

### 1.3 **Time constraints**

```python
import random
from typing import Any

from msgflux.exceptions import ModelRouterError
from msgflux.models.base import BaseModel
from msgflux.models.gateway import ModelGateway
from msgflux.models.response import ModelResponse
from msgflux.models.types import ChatCompletionModel

class MockChatCompletion(BaseModel, ChatCompletionModel):

    provider = "mock"

    def __init__(
        self,
        model_id: str,
        fail_sometimes: bool = False,
        success_rate: float = 0.7
    ):
        self.model_id = model_id
        self._fail_sometimes = fail_sometimes
        self._success_rate = success_rate
        self._call_count = 0

    def _initialize(self):
        pass

    def __call__(self, **kwargs: Any):
        response = ModelResponse()
        response.set_response_type("text_generation")
        self._call_count += 1
        if self._fail_sometimes:
            if random.random() > self._success_rate:
                raise ValueError(f"Simulated failure for {self.model_id}")
        messages = kwargs.get("messages", "Default prompt")
        response_text = f"Response from {self.model_id} to messages: '{messages}' (Call #{self._call_count})"
        response.add(response_text)
        return response

model1 = MockChatCompletion(model_id="model-A", fail_sometimes=True, success_rate=0.3)
model2 = MockChatCompletion(model_id="model-B", fail_sometimes=True, success_rate=0.5)
model3 = MockChatCompletion(model_id="model-C") # Always works
model4 = MockChatCompletion(model_id="model-D") # Always works

gateway_mock = ModelGateway([
    {
        "model_name": "unstable-A",
        "model": model1,
    },
    {
        "model_name": "unstable-B",
        "model": model2,
        "time_constraints": [("23:00", "07:00")],
    },
    {
        "model_name": "reliable-C",
        "model": model3,
        "time_constraints": [("10:00", "11:00")],
    },
    {
        "model_name": "reliable-D",
        "model": model4,
    },
])

try:
    response = gateway_mock(messages="Hi")
    print("Result:", response.consume())
except ModelRouterError as e:
    print("Error:", e)
```

## 2. **Model Metadata**

The Gateway exposes stable aliases, normalized models, and descriptions:

```python
print(gateway.model_names)
# ["fast", "deep"]

print(gateway.model_descriptions)
# {
#     "fast": "Fast for clear, repeatable tasks.",
#     "deep": "Deep reasoning for ambiguous, multi-step tasks.",
# }

print(gateway.model_type)
# "chat_completion"
```

Use `gateway.validate_model_name(alias)` before storing or forwarding a
user-selected alias. Unknown aliases raise instead of silently falling back.

### 2.1 **Updating Reasoning Effort**

`set_reasoning_effort()` proxies request-level configuration to the models
behind the Gateway. By default it updates every deployment so a fallback keeps
the requested behavior:

```python
gateway.set_reasoning_effort("high")
response = gateway(messages="Compare two recovery strategies.")
```

Target one deployment by its public alias when the configuration should not be
shared:

```python
gateway.set_reasoning_effort("low", model_name="fast")
```

Pass `None` to remove the explicit setting. A deployment whose provider or API
mode does not support request-level reasoning effort emits a warning and is left
unchanged.

### 2.2 **Updating Request Speed**

`set_speed()` uses the same proxy behavior. Without an alias, every fallback
deployment receives the canonical preference and translates it independently:

```python
gateway.set_speed("fast")
```

Target one deployment when its provider offers a distinct tier:

```python
gateway.set_speed("ultrafast", model_name="deep")
```

Supported values are `"fast"`, `"ultrafast"`, and `"nitro"`. Pass `None` to
remove the preference. Unsupported combinations emit a warning without failing
the Gateway.

## 3. **Serialization**

`serialize()` persists every normalized model together with `model_name`,
`description`, `time_constraints`, and the Gateway's `fallback` policy.
`ModelGateway.from_serialized(...)` also accepts older state without the policy
field and restores it with `fallback=True`.
