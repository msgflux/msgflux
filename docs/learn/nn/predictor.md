# nn.Predictor

`nn.Predictor` is the most generic Module type — it feeds data to a model and returns predictions. It works with any msgflux model (classifiers, regressors, detectors, moderators) or custom models that inherit from `BaseModel`.

## Quick Start

```python
import msgflux as mf
import msgflux.nn as nn

class ContentModerator(nn.Predictor):
    model = mf.Model.moderation("openai/omni-moderation-latest")

moderator = ContentModerator()
result = moderator("This is a great day!")
print(result.safe)  # True
```

---

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `BaseModel \| ModelGateway` | Any msgflux model or custom model |
| `message_fields` | `dict \| None` | Map `Message` field names to inputs. Valid keys: `task_inputs`, `model_preference` |
| `response_mode` | `str \| None` | Field path on the `Message` where the result is written. `None` returns the result directly |
| `response_template` | `str \| None` | Jinja template to format the response |
| `config` | `dict \| None` | Extra parameters passed directly to the model |
| `hooks` | `list[Hook] \| None` | Hook instances registered on the module |
| `name` | `str \| None` | Module name in snake_case |

---

## Compatible Models

Any model that accepts `data` as input works with Predictor:

| Type | Factory | Description |
|------|---------|-------------|
| `ModerationModel` | `Model.moderation()` | Content safety classification |
| `TextClassifierModel` | `Model.text_classifier()` | Text classification |
| `ImageClassifierModel` | `Model.image_classifier()` | Image classification |
| Custom | Inherit from `BaseModel` | Any custom model (sklearn, etc.) |

---

## Content Moderation

```python
class ContentModerator(nn.Predictor):
    model          = mf.Model.moderation("openai/omni-moderation-latest")
    message_fields = {"task_inputs": "user_message"}
    response_mode  = "moderation"

moderator = ContentModerator()

msg = mf.Message()
msg.user_message = "I love programming in Python!"

moderator(msg)
print(msg.moderation)
```

---

## Text Classification

Using vLLM with a self-hosted classifier:

```python
class SentimentClassifier(nn.Predictor):
    model          = mf.Model.text_classifier("vllm/my-sentiment-model")
    message_fields = {"task_inputs": "text"}
    response_mode  = "sentiment"

classifier = SentimentClassifier()

msg = mf.Message()
msg.text = "This movie was absolutely wonderful"

classifier(msg)
print(msg.sentiment)  # ["positive"]
```

---

## Custom Models

Create custom models by inheriting from `BaseModel`. This allows integrating any ML framework (sklearn, XGBoost, PyTorch, etc.) into the msgflux module system.

### sklearn

```python
import joblib

from msgflux.core.dotdict import dotdict
from msgflux.models.base import BaseModel
from msgflux.models.response import ModelResponse


class SklearnClassifier(BaseModel):
    """Wraps a scikit-learn classifier as a msgflux model."""

    model_type = "tabular_classifier"
    provider = "sklearn"

    def __init__(self, path: str):
        self.model_id = path
        self._path = path
        self._initialize()

    def _initialize(self):
        self.clf = joblib.load(self._path)

    def __call__(self, *, data, **kwargs):
        response = ModelResponse()
        response.set_response_type("text_classification")
        predictions = self.clf.predict(data)
        labels = [self.clf.classes_[p] for p in predictions]
        response.add(labels)
        return response

    async def acall(self, *, data, **kwargs):
        return self(data=data, **kwargs)
```

Then use it like any other model:

```python
class ChurnPredictor(nn.Predictor):
    model = SklearnClassifier("models/churn_v2.pkl")

predictor = ChurnPredictor()
result = predictor([[0.5, 1.2, 3.0, 0.8]])
print(result)  # ["churn"]
```

---

## With Message

```python
class SafetyFilter(nn.Predictor):
    model          = mf.Model.moderation("openai/omni-moderation-latest")
    message_fields = {"task_inputs": "content"}
    response_mode  = "safety"

filter = SafetyFilter()

msg = mf.Message()
msg.content = "Hello, how are you?"

filter(msg)
print(msg.safety.safe)  # True
```

---

## Integration with Agents

Predictors work as preprocessing or guardrail steps in agent pipelines.

```python
class Moderator(nn.Predictor):
    model          = mf.Model.moderation("openai/omni-moderation-latest")
    message_fields = {"task_inputs": "user_input"}
    response_mode  = "moderation"

class Assistant(nn.Agent):
    model          = mf.Model.chat_completion("openai/gpt-4.1-mini")
    message_fields = {"task_inputs": "user_input"}
    response_mode  = "response"

class SafePipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.moderator = Moderator()
        self.assistant = Assistant()

    def forward(self, msg):
        self.moderator(msg)
        if msg.moderation.safe:
            self.assistant(msg)
        else:
            msg.response = "I can't process this request."
        return msg

pipeline = SafePipeline()

msg = mf.Message()
msg.user_input = "Tell me about machine learning"

pipeline(msg)
print(msg.response)
```

---

## Predictor Hierarchies

Share configuration across related predictors.

```python
class BaseClassifier(nn.Predictor):
    """Base class for all text classifiers."""
    model = mf.Model.text_classifier("vllm/my-model")

class SpamDetector(BaseClassifier):
    message_fields = {"task_inputs": "email_body"}
    response_mode  = "spam_result"

class TopicClassifier(BaseClassifier):
    message_fields = {"task_inputs": "article_text"}
    response_mode  = "topic"
```

---

## Async

```python
result = await predictor.acall("some input data")
```

---

## Debugging

```python
# Inspect parameters before execution
params = predictor.inspect_model_execution_params("test input")
print(params)
```
