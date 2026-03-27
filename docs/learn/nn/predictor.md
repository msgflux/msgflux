# nn.Predictor

## ✦₊⁺ Overview

`nn.Predictor` is the most generic Module type — it feeds data to a model and returns predictions. It works with any msgflux model (classifiers, regressors, detectors, moderators) or custom models that inherit from `BaseModel`.

---

## 1. **Quick Start**

!!! info "Initialization styles"

    === "Declarative (recommended)"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class ContentModerator(nn.Predictor):
            model = mf.Model.moderation("openai/omni-moderation-latest")

        moderator = ContentModerator()
        result = moderator("This is a great day!")
        print(result.safe)  # True
        ```

    === "Direct"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        predictor = nn.Predictor(
            model=mf.Model.moderation("openai/omni-moderation-latest")
        )
        result = predictor("This is a great day!")
        ```

---

## 2. **Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `BaseModel \| ModelGateway` | Any msgflux model or custom model |
| `message_fields` | `dict \| None` | Map `Message` field names to inputs. Valid keys: `task_inputs`, `model_preference` |
| `response_mode` | `str \| None` | Field path on the `Message` where the result is written. `None` returns the result directly |
| `templates` | `dict[str, str] \| None` | Jinja templates dict. Valid keys: `response` |
| `config` | `dict \| None` | Extra parameters passed directly to the model |
| `hooks` | `list[Hook] \| None` | Hook instances registered on the module |
| `name` | `str \| None` | Module name in snake_case |

---

## 3. **Compatible Models**

Any model that accepts `data` as input works with Predictor:

| Type | Factory | Description |
|------|---------|-------------|
| `ModerationModel` | `Model.moderation()` | Content safety classification |
| `TextClassifierModel` | `Model.text_classifier()` | Text classification |
| `ImageClassifierModel` | `Model.image_classifier()` | Image classification |
| Custom | Inherit from `BaseModel` | Any custom model (sklearn, etc.) |

---

## 4. **Usage Examples**

!!! info "Examples by use case"

    === "Content Moderation"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class ContentModerator(nn.Predictor):
            model          = mf.Model.moderation("openai/omni-moderation-latest")
            message_fields = {"task_inputs": "user_message"}
            response_mode  = "moderation"

        moderator = ContentModerator()

        msg = mf.dotdict(user_message="I love programming in Python!")
        moderator(msg)
        print(msg.moderation.safe)  # True
        ```

    === "Text Classification"

        Using vLLM with a self-hosted classifier:

        ```python
        class SentimentClassifier(nn.Predictor):
            model          = mf.Model.text_classifier("vllm/my-sentiment-model")
            message_fields = {"task_inputs": "text"}
            response_mode  = "sentiment"

        classifier = SentimentClassifier()

        msg = mf.dotdict(text="This movie was absolutely wonderful")
        classifier(msg)
        print(msg.sentiment)  # ["positive"]
        ```

    === "Templates"

        Format the raw prediction output with Jinja templates:

        ```python
        class ContentModerator(nn.Predictor):
            model     = mf.Model.moderation("openai/omni-moderation-latest")
            templates = {"response": "safe={{ safe }}, flagged={{ results.flagged }}"}

        moderator = ContentModerator()
        result = moderator("Hello!")
        print(result)  # "safe=True, flagged=False"
        ```

    === "Hierarchies"

        Share configuration across related predictors via inheritance:

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

## 5. **Custom Models**

Integrate any ML framework (sklearn, XGBoost, PyTorch, etc.) by inheriting from `BaseModel`:

!!! info "Custom model examples"

    === "sklearn"

        ```python
        import joblib
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

        class ChurnPredictor(nn.Predictor):
            model = SklearnClassifier("models/churn_v2.pkl")

        predictor = ChurnPredictor()
        result = predictor([[0.5, 1.2, 3.0, 0.8]])
        print(result)  # ["churn"]
        ```

---

## 6. **Integration with Agents**

Predictors work as preprocessing or guardrail steps in agent pipelines.

!!! info "Predictor + Agent pipeline"

    ```python
    import msgflux as mf
    import msgflux.nn as nn

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

    msg = mf.dotdict(user_input="Tell me about machine learning")
    pipeline(msg)
    print(msg.response)
    ```

---

## 7. **Async**

```python
result = await predictor.acall("some input data")
```

---

## 8. **Debugging**

```python
params = predictor.inspect_model_execution_params("test input")
print(params)
```
