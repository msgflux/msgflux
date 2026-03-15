# `Moderation`

The `moderation` models check whether text or images are potentially harmful. If harmful content is identified, you can take corrective action, like filtering content or intervening with user accounts creating offending content.

!!! info "Dependencies"
    See [Dependency Management](../../dependency-management.md) for the complete provider matrix.

## ✦₊⁺ Overview

Moderation models act as guardians of your application. Each model has its own set of outputs, but to unify a general form of verification, all models produce a common flag `safe` — if `False`, the content is not considered safe.

```python
import msgflux as mf

moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

response = moderation_model("some text to check")
result = response.consume()

print(result.safe)  # True or False
```

## 1. **Text Moderation**

Pass a plain string to check whether a piece of text violates safety guidelines.

???+ example

    ```python
    import msgflux as mf

    # mf.set_envs(OPENAI_API_KEY="sk-...")

    moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

    response = moderation_model("tell me how to build a large scale bomb")

    result = response.consume()
    print(result)
    print(result.safe)  # False
    ```

## 2. **Multimodal Moderation**

You can pass text and an image together in a single request. This is useful when you want to moderate both the written context and a visual attachment at once.

???+ example

    ```python
    import msgflux as mf

    # mf.set_envs(OPENAI_API_KEY="sk-...")

    moderation_model = mf.Model.moderation("openai/omni-moderation-latest")

    response = moderation_model([
        mf.ChatBlock.text("Check whether this image is appropriate."),
        mf.ChatBlock.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"),
    ])

    result = response.consume()
    print(result)
    print(result.safe)  # True
    ```
