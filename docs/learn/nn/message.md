# Message

## ✦₊⁺ Overview

`Message` is a structured container for **data flow between modules**. It extends [`dotdict`](../dotdict.md) with default fields tailored for AI workflows.

For general `dotdict` features — dot access, nested paths, `get()`/`set()`, serialization, immutability, and hidden keys — see the [`dotdict` docs](../dotdict.md).

## 1. **Quick Start**

```python
from msgflux import Message

msg = Message(
    content="Analyze this text",
    user_id="user123",
    chat_id="chat456",
)

msg.set("context.data", {"key": "value"})
print(msg.content)       # "Analyze this text"
print(msg.context.data)  # {"key": "value"}
```

---

## 2. **Default Fields**

`Message` pre-defines a set of fields that modules and agents expect:

| Field | Description |
|-------|-------------|
| `content` | Main content (text, dict) |
| `texts` | Text data container |
| `context` | Context information |
| `audios` | Audio data |
| `images` | Image data |
| `videos` | Video data |
| `extra` | Extra data |
| `outputs` | Module outputs |
| `response` | Final response |

### Metadata

`Message` also initializes metadata fields that help track workflow executions:

| Field | Description |
|-------|-------------|
| `execution_id` | Auto-generated unique ID |
| `user_id` | User identifier |
| `user_name` | User name |
| `chat_id` | Chat/session identifier |

```python
msg = Message(
    user_id="123",
    user_name="Bruce Wayne",
    chat_id="456",
)

print(msg.metadata["execution_id"])  # Auto-generated UUID
```

---

## 3. **With nn.Agent**

### message_fields

`message_fields` controls which parts of the `Message` an agent reads as inputs:

```python
import msgflux.nn as nn

class Analyzer(nn.Agent):
    model = model
    message_fields = {
        "task": "content",                    # Read task
        "task_multimodal": {"image": "images.user"},  # Read image
        "task_context": "context.data",       # Read task context
        "vars": "extra.vars"                         # Read vars
    }
    response_mode = "outputs.analysis"               # Write response

analyzer = Analyzer()

msg = Message(
    content="Analyze this image",
    context={"data": {"type": "product"}}
)
msg.set("images.user", "https://example.com/image.jpg")

analyzer(msg)

print(msg.outputs.analysis)  # Agent's response
```

### response_mode Options

`response_mode` controls how and where the module delivers its output:

| Mode | Behavior |
|------|----------|
| `None` (default) | Return response directly |
| `"<path>"` | Write to `msg.<path>` in place and return `None` |
| `"<path>:"` | Return a new `dotdict` with the response under `<path>` |

**Writing to a Message** — pass a string path *without* a trailing colon. The
module writes to that field of the `Message` object and returns `None`:

```python
# Write response to msg.outputs.result, return None
agent = nn.Agent(model, response_mode="outputs.result")
msg = Message(content="Hi")
result = agent(msg)
print(result)  # None
print(msg.outputs.result)
```

**Returning a dotdict** — add a trailing colon (`:`). The module creates a new
`dotdict` whose structure mirrors the path, without needing a `Message` at all:

```python
# Returns dotdict({"outputs": {"result": <response>}})
agent = nn.Agent(model, response_mode="outputs.result:")
result = agent("What is Python?")
print(result.get("outputs.result"))

# Simple key
agent = nn.Agent(model, response_mode="answer:")
result = agent("What is Python?")
print(result.answer)
```

**Writing structured extraction** (for signatures):

```python
agent = nn.Agent(model, signature="text -> label: str", response_mode="extraction")
agent(msg)
print(msg.extraction)  # {"label": "..."}
```

---

## 4. **In Workflows**

### Inline DSL

```python
import msgflux.nn.functional as F


def preprocess(msg):
    msg.preprocessed_content = msg.content.upper()
    return msg

def analyze(msg):
    msg.outputs.analysis = f"Analyzed: {msg.preprocessed_content}"
    return msg

modules = {"preprocess": preprocess, "analyze": analyze}

msg = Message(content="hello world")
F.inline("preprocess -> analyze", modules, msg)

print(msg.outputs.analysis)  # "Analyzed: HELLO WORLD"
```

### Passing Between Modules

```python
import msgflux.nn as nn


class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = nn.Transcriber(...)
        self.analyzer = nn.Agent(...)

    def forward(self, msg):
        # Transcriber writes to msg.content
        self.transcriber(msg)

        # Analyzer reads from msg.content
        self.analyzer(msg)

        return msg
```

---

## 5. **Multimodal Data**

Store multimodal inputs directly on the message:

```python
msg = Message()

# Audio
msg.set("user_audio", "/path/to/audio.mp3")

# Images
msg.set("user_image", "https://example.com/image.jpg")
msg.set("images.product", ["img1.jpg", "img2.jpg"])

# Files
msg.set("user_file", "/path/to/document.pdf")
```

---

## 6. **Complete Example**

```python
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Message


class Speech2Text(nn.Transcriber):
    model = mf.Model.speech_to_text("openai/whisper-1")
    message_fields = {"task_multimodal": {"audio": "user_audio"}}
    response_mode = "content"


class Analyzer(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    message_fields = {"task": "content"}
    response_mode = "outputs.analysis"


class Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = Speech2Text()
        self.analyzer = Analyzer()
        self.components = nn.ModuleDict(
            {
                "transcriber": self.transcriber,
                "analyzer": self.analyzer,
            }
        )
        self.register_buffer("flux", "{user_audio is not None? transcriber} -> analyzer")

    def forward(self, msg):
        return F.inline(self.flux, self.components, msg)


pipeline = Pipeline()

# Text input
msg = Message(content="Analyze this text for sentiment.")
pipeline(msg)
print(msg.outputs.analysis)

# Audio input
msg = Message()
msg.user_audio = "/path/to/audio.mp3"
pipeline(msg)
print(msg.content)           # Transcription
print(msg.outputs.analysis)  # Analysis result
```
