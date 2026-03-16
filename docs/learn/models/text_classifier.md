# Text Classifier

The `text_classifier` model assigns predefined labels to text inputs. It uses sequence classification models served via vLLM to score inputs against a fixed label set.

!!! info "Dependencies"
    See [Dependency Management](../../dependency-management.md) for the complete provider matrix.

## ✦₊⁺ Overview

Text classification models score each input against a fixed set of labels and return the best match with a confidence score. They enable:

- **Sentiment Analysis**: Positive, negative, neutral
- **Intent Detection**: Categorize user intent without retraining
- **Topic Routing**: Direct inputs to the right handler
- **Content Moderation**: Detect toxic, spam, or unsafe content

## 1. **Quick Start**

???+ example

    ```python
    import msgflux as mf

    # Labels defined at construction time
    model = mf.Model.text_classifier(
        "vllm/jason9693/Qwen2.5-1.5B-apeach",
        labels=["toxic", "not_toxic"]
    )

    result = model("This product is absolutely amazing!").consume()
    print(result)
    # {"label": "not_toxic", "score": 0.97}
    ```

## 2. **Supported Providers**

### vLLM (self-hosted)

vLLM serves `SequenceClassification` models via the `/classify` endpoint. Start the server with `--task classify`.

???+ example

    === "Qwen2.5 Classifier"

        ```python
        import msgflux as mf

        # jason9693/Qwen2.5-1.5B-apeach — cited in vLLM docs as canonical classify example
        # vllm serve jason9693/Qwen2.5-1.5B-apeach --task classify
        model = mf.Model.text_classifier(
            "vllm/jason9693/Qwen2.5-1.5B-apeach",
            labels=["toxic", "not_toxic"],
            base_url="http://localhost:8000/v1"
        )
        ```

    === "ModernBERT Classifier"

        ```python
        import msgflux as mf

        # Any fine-tuned ModernBERT sequence classifier works with vLLM (April 2025+)
        # vllm serve davanstrien/ModernBERT-web-topics-1m --task classify
        model = mf.Model.text_classifier(
            "vllm/davanstrien/ModernBERT-web-topics-1m",
            labels=["technology", "science", "sports", "politics"],
            base_url="http://localhost:8000/v1"
        )
        ```

## 3. **Batch Classification**

Pass a list to classify multiple texts in a single call:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_classifier(
        "vllm/jason9693/Qwen2.5-1.5B-apeach",
        labels=["urgent", "normal", "low_priority"],
        base_url="http://localhost:8000/v1"
    )

    tickets = [
        "System is completely down, customers cannot login",
        "Please update my billing address",
        "Would be nice to have dark mode someday"
    ]

    results = model(tickets).consume()
    # [
    #   {"label": "urgent",       "score": 0.97},
    #   {"label": "normal",       "score": 0.85},
    #   {"label": "low_priority", "score": 0.91}
    # ]

    for ticket, result in zip(tickets, results):
        print(f"[{result['label']}] {ticket[:50]}...")
    ```

## 4. **Async Support**

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F

    model = mf.Model.text_classifier(
        "vllm/jason9693/Qwen2.5-1.5B-apeach",
        labels=["positive", "negative", "neutral"],
        base_url="http://localhost:8000/v1"
    )

    reviews = ["Great product!", "Terrible experience.", "It was okay."]

    results = F.map_gather(
        model,
        args_list=[(r,) for r in reviews]
    )

    for review, result in zip(reviews, results):
        print(f"{review!r}: {result.consume()}")
    ```

## 5. **Error Handling**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_classifier(
        "vllm/jason9693/Qwen2.5-1.5B-apeach",
        labels=["positive", "negative"],
        base_url="http://localhost:8000/v1"
    )

    try:
        result = model("Some text").consume()
        print(result["label"], result["score"])
    except ImportError:
        print("Provider not installed")
    except Exception as e:
        print(f"Classification failed: {e}")
    ```
