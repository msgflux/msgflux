
# `ModelGateway` — Resilient Model Manager

The [`ModelGateway`](../../api-reference/models/gateway.md) class is an **orchestration layer** over multiple models of the same type (e.g., multiple `chat_completion` models), allowing:

- **Automatic fallback** between models.
- **Time-based** model availability constraints.
- **Model preference** selection via aliases.
- **Control of execution attempts** with exception handling.
- **Consistent model typing validation**.

It's ideal for production-grade model orchestration where reliability and control over model usage are required.

## ✦₊⁺ Overview

## 1. **Usage**

```bash
pip install msgflux[openai]
```

All you need is:

- All models **must inherit from `BaseModel`**.
- All models **must be of the same `model_type`**.
- Each deployment **must have a unique `model_name`**.
- At least **2 deployments** are recommended for effective fallback.
---

### 1.1 **Query**

```python
import msgflux as mf

mf.set_envs(OPENAI_API_KEY="sk-...", TOGETHER_API_KEY="<>")

gateway = mf.ModelGateway([
    {
        "model_name": "primary",
        "model": mf.Model.chat_completion("openai/gpt-4.1-nano"),
    },
    {
        "model_name": "fallback",
        "model": mf.Model.chat_completion("together/gpt-oss-120B"),
    },
])

response = gateway(messages="Who was Frank Rosenblatt?")
print(response.consume())
```

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
fallback = mf.Model.chat_completion("openai/gpt-4.1-nano")

gateway_broken = mf.ModelGateway([
    {"model_name": "broken", "model": broken},
    {"model_name": "fallback", "model": fallback},
])

response = gateway_broken(messages="Who were Warren McCulloch and Walter Pitts?")
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

## 2. **Model Info**

Returns information for all managed models:

```python
print(gateway.get_model_info())
```

```python
[
    {'model_id': 'gpt-4.1-nano', 'provider': 'openai'},
    {'model_id': 'mistral-7b', 'provider': 'together'}
]
```


Returns the type of the models:

```python
print(gateway.model_type)
```

```python
'chat_completion'
```


---

## 3. **Serialization**

Serializes the state of the gateway and models.

```python
print(gateway.serialize())
```

```python
{
    'msgflux_type': 'model_gateway',
    'state': {
        'models': [
            {
                'model_name': 'primary',
                'model': {
                    'msgflux_type': 'model',
                    'provider': 'openai',
                    'model_type': 'chat_completion',
                    'state': {
                        'model_id': 'gpt-4.1-nano',
                        'sampling_params': {'organization': None, 'project': None},
                        'sampling_run_params': {
                            'max_tokens': 512,
                            'temperature': None,
                            'top_p': None,
                            'modalities': ['text'],
                            'audio': None
                        }
                    }
                }
            },
            {
                'model_name': 'fallback',
                'model': {
                    'msgflux_type': 'model',
                    'provider': 'together',
                    'model_type': 'chat_completion',
                    'state': {
                        'model_id': 'mistral-7b',
                        'sampling_params': {'organization': None, 'project': None},
                        'sampling_run_params': {
                                'max_tokens': 512,
                                'temperature': None,
                                'top_p': None,
                                'modalities': ['text'],
                                'audio': None
                        }
                    }
                }
            },
        ]
    }
}
```

---
