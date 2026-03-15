# Model

The `Model` class provides a unified interface for working with AI models across different providers and modalities. It acts as a factory that creates provider-specific model instances with a consistent API.

!!! info "Dependency management"
    For the full list of providers and their required extras, see [Dependency Management](../../dependency-management.md).

```python
import msgflux as mf
```

## ✦₊⁺ Overview

Instead of learning different APIs for each provider, you use a single factory method:

```python
model = mf.Model.chat_completion("openai/gpt-4.1-mini", temperature=0.7)
```

All models follow the pattern: `provider/model-id`

## 1. **Quick Start**

### 1.1 **Installation**

```bash
pip install msgflux[openai]
```

### 1.2 **Basic Usage**

```python
import msgflux as mf

# Set API key
mf.set_envs(OPENAI_API_KEY="sk-...")

# Create model
model = mf.Model.chat_completion("openai/gpt-4.1-mini")

# Use model (see chat_completion.md for details)
response = model(messages=[{"role": "user", "content": "Hello!"}])
print(response.consume())  # "Hello! How can I help you today?"
```

## 2. **Model Types**

The `Model` class supports multiple model types:

### 2.1 **Factory Methods**

Each model type has a dedicated factory method:

| Model Type | Factory Method | Use Case |
|------------|---------------|----------|
| **chat_completion** | `Model.chat_completion()` | Chat and text generation |
| **text_embedder** | `Model.text_embedder()` | Convert text to vectors |
| **text_to_image** | `Model.text_to_image()` | Generate images from text |
| **image_text_to_image** | `Model.image_text_to_image()` | Edit images with text |
| **text_to_speech** | `Model.text_to_speech()` | Convert text to audio |
| **speech_to_text** | `Model.speech_to_text()` | Transcribe audio to text |
| **moderation** | `Model.moderation()` | Content moderation |
| **text_classifier** | `Model.text_classifier()` | Classify text |
| **image_classifier** | `Model.image_classifier()` | Classify images |
| **image_embedder** | `Model.image_embedder()` | Convert images to vectors |
| **text_reranker** | `Model.text_reranker()` | Rerank text results |

## 3. **Usage Examples**

### 3.1 **Chat Completion**

```python
import msgflux as mf

# Create model
model = mf.Model.chat_completion(
    "openai/gpt-4.1-mini",
    temperature=0.7,
    max_tokens=1000
)

# Single completion
response = model(messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
])

print(response.consume())  # "The capital of France is Paris."
```

### 3.2 **Text Embeddings**

```python
import msgflux as mf

# Create embedder
embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

# Generate embedding
response = embedder("Hello, world!")

embedding = response.consume()
print(len(embedding))  # 1536
print(embedding[:5])  # [0.123, -0.456, 0.789, ...]
```

### 3.3 **Speech to Text**

```python
import msgflux as mf

# Create transcription model
model = mf.Model.speech_to_text("openai/whisper-1")

# Transcribe audio file
response = model("path/to/audio.mp3")

transcription = response.consume()
print(transcription)  # "Hello, this is a test."
```

## 4. **Model Information**

### 4.1 **Getting Model Metadata**

```python
import msgflux as mf

model = mf.Model.chat_completion("openai/gpt-4.1-mini")

# Get model type
print(model.model_type)  # "chat_completion"

# Get instance type information
print(model.instance_type())
# {'model_type': 'chat_completion'}

# Get model info
print(model.get_model_info())
# {'model_id': 'gpt-4.1-mini', 'provider': 'openai'}
```

## 5. **Serialization**

Models can be serialized and deserialized for storage or transfer:

### 5.1 **Serializing a Model**

```python
import msgflux as mf

# Create and configure model
model = mf.Model.chat_completion(
    "openai/gpt-4.1-mini",
    temperature=0.7,
    max_tokens=500
)

# Serialize
state = model.serialize()
print(state)
# {
#     'msgflux_type': 'model',
#     'provider': 'openai',
#     'model_type': 'chat_completion',
#     'state': {
#         'model_id': 'gpt-4.1-mini',
#         'sampling_params': {...},
#         'sampling_run_params': {
#             'temperature': 0.7,
#             'max_tokens': 500,
#             ...
#         }
#     }
# }

# Save to file
mf.save(state, "model_config.json")
```

### 5.2 **Deserializing a Model**

```python
import msgflux as mf

# Load from file
state = mf.load("model_config.json")

# Recreate model
model = mf.Model.from_serialized(
    provider=state['provider'],
    model_type=state['model_type'],
    state=state['state']
)

# Model is ready to use
response = model(messages=[{"role": "user", "content": "Hello"}])
```

## 6. **Response Types**

All models return one of two response types:

### 6.1 **ModelResponse**

For non-streaming responses (embeddings, transcription, etc.):

```python
import msgflux as mf

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")
response = embedder("Hello")

# Response is ModelResponse
print(type(response))  # <class 'msgflux.models.response.ModelResponse'>

# Get response type
print(response.response_type)  # "text_embedding"

# Consume the response
result = response.consume()
print(result)  # [0.123, -0.456, ...]
```

### 6.2 **ModelStreamResponse**

For streaming responses (chat, text-to-speech, etc.):

```python
import msgflux as mf

model = mf.Model.chat_completion("openai/gpt-4.1-mini")

# Stream enabled
response = model(
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True
)

# Response is ModelStreamResponse
print(type(response))  # <class 'msgflux.models.response.ModelStreamResponse'>

# Consume stream
async for chunk in response.consume():
    print(chunk, end="", flush=True)
# Output: "1, 2, 3, 4, 5"
```

## 7. **Batch Processing**

Some providers expose native batch support for embeddings — they accept a `List[str]` in a single API call, which is more efficient than sending one request per text. You can check if a model supports this via the `batch_support` attribute.

```python
import msgflux as mf

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")
print(embedder.batch_support)  # True
```

When `batch_support` is `True`, pass the full list directly:

```python
import msgflux as mf

embedder = mf.Model.text_embedder("openai/text-embedding-3-small")

texts = ["Hello", "World", "AI", "Embedding"]

# Single API call — returns a list of embeddings
response = embedder(texts)
embeddings = response.consume()  # List[List[float]]
print(f"Generated {len(embeddings)} embeddings")
```

When `batch_support` is `False`, use `F.map_gather` to process texts in parallel across multiple requests:

```python
import msgflux as mf
import msgflux.nn.functional as F

embedder = mf.Model.text_embedder("some-provider/some-model")
print(embedder.batch_support)  # False

texts = ["Hello", "World", "AI", "Embedding"]

results = F.map_gather(
    embedder,
    args_list=[(text,) for text in texts]
)

embeddings = [r.consume() for r in results]
print(f"Generated {len(embeddings)} embeddings")
```
