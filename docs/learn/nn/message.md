# Message

## ✦₊⁺ Overview

`Message` is a structured container for **data flow between modules**. It extends [`dotdict`](../dotdict.md) with default fields tailored for AI workflows.

For general `dotdict` features — dot access, nested paths, `get()`/`set()`, serialization, immutability, and hidden keys — see the [`dotdict` docs](../dotdict.md).

## 1. **Quick Start**

```python
from msgflux import Message

msg = Message(
    content="Analyze this text",
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
| `"<path>"` | Write to `msg.<path>` and return the `Message` |

**Writing to a Message** — pass a string path *without* a trailing colon. The
module writes to that field of the `Message` object and returns it:

```python
# Write response to msg.outputs.result, return the Message
agent = nn.Agent(model, response_mode="outputs.result")
msg = agent(Message(content="Hi"))
print(msg.outputs.result)
```

---

## 4. **Inline**

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
