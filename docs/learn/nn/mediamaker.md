# nn.MediaMaker

## ✦₊⁺ Overview

`nn.MediaMaker` is a Module for **generating visual and audio content** (images, videos, 3D models, audio) from prompts.

## 1. **Quick Start**

```python
import msgflux as mf
import msgflux.nn as nn

model = mf.Model.text_to_image("openai/gpt-image-1")

maker = nn.MediaMaker(model=model)

# Returns base64 string (gpt-image-1 default)
image_b64 = maker("A futuristic city at sunset, cyberpunk style")
```

---

## 2. **Declarative Style**

```python
import msgflux as mf
import msgflux.nn as nn

class ImageGenerator(nn.MediaMaker):
    model = mf.Model.text_to_image("openai/gpt-image-1")
    message_fields = {"task_inputs": "prompt"}
    response_mode = "generated_image"
    config = {"quality": "high"}  # size, quality, n, background go in config

generator = ImageGenerator()
```

---

## 3. **Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `MEDIA_MODEL_TYPES` | Media generation model |
| `message_fields` | `dict` | Map `Message` field names to inputs. Valid keys: `task_inputs`, `task_multimodal_inputs` |
| `response_mode` | `str \| None` | Field path on the `Message` where the result is written. `None` returns the result directly |
| `response_format` | `"base64" \| "url" \| None` | Output format for the generated media |
| `negative_prompt` | `str \| None` | What to avoid in generation |
| `config` | `dict \| None` | Extra parameters passed to the model. Common keys: `fps`, `duration_seconds`, `aspect_ratio`, `n` |
| `hooks` | `list[Hook] \| None` | Hook instances registered on the module |
| `name` | `str \| None` | Module name in snake_case |

---

## 4. **Text to Image**

```python
# OpenAI — superior detail and prompt adherence
model = mf.Model.text_to_image("openai/gpt-image-1.5")

# Replicate — FLUX 2 Flex open model
model = mf.Model.text_to_image("replicate/black-forest-labs/flux-2-flex")

maker = nn.MediaMaker(model=mf.Model.text_to_image("openai/gpt-image-1"))

# Returns base64 string
image_b64 = maker("A serene Japanese garden with cherry blossoms")
```

### Sizes, Quality & Background

`size`, `quality`, `background`, and `n` are **not** call kwargs on `MediaMaker` — pass them via `config`:

```python

class Maker(nn.MediaMaker):
    model = mf.Model.text_to_image("openai/gpt-image-1")
    config = {
        "size": "1536x1024",   # landscape
        "quality": "high",
        "background": "transparent",  # useful for product shots
    }

maker = Maker()
image_b64 = maker("A panoramic mountain view")
```

### Multiple Images

```python
class Maker(nn.MediaMaker):
    model = mf.Model.text_to_image("openai/gpt-image-1")
    config = {"n": 4, "size": "1024x1024", "quality": "low"}

maker = Maker()

# Returns list of base64 strings when n > 1
images = maker("A cute robot")

---

## 5. **Image Editing**

```python
model = mf.Model.image_text_to_image("openai/gpt-image-1.5")
maker = nn.MediaMaker(model=model)

# Edit with reference image
edited = maker(
    "Make it look like sunset",
    task_multimodal_inputs={"image": "/path/to/photo.jpg"}
)

# Edit with mask
edited = maker(
    "Add a flamingo in the pool",
    task_multimodal_inputs={
        "image": "/path/to/pool.jpg",
        "mask": "/path/to/mask.png"
    }
)
```

---

## 6. **Negative Prompts**

Specify what to avoid:

```python
class Designer(nn.MediaMaker):
    model = model
    negative_prompt = "blurry, low quality, distorted, watermark"

maker = Designer()

image = maker("A professional portrait photo")
```

---

## 7. **With Message**

```python
from msgflux import Message

msg = Message()
msg.prompt = "A cozy cabin in the mountains during winter"

class Maker(nn.MediaMaker):
    model = model
    message_fields = {"task_inputs": "prompt"}
    response_mode = "generated_image"

maker(msg)
# msg.generated_image contains the base64 string
```

---

## 8. **Config Options**

Pass extra generation parameters via `config`:

```python
# Video generation with specific settings
video_model = mf.Model.text_to_video("sora/text-to-video")

class Maker(nn.MediaMaker):
    model = video_model
    config = {
        "duration_seconds": 5,
        "aspect_ratio": "16:9",
        "fps": 24,
    }

maker = Maker()

video = maker("A timelapse of a blooming flower")
```

---

## 9. **Creative Pipeline**

Chain an `Agent` (prompt engineer) and a `MediaMaker` (image generator) using `Inline`:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Inline


class StoryToPrompt(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    instructions = """
    Create a detailed image generation prompt from the story scene.
    Focus on visual elements, lighting, style, and composition.
    """
    message_fields = {"task_inputs": "scene"}
    response_mode = "prompt"


class ImageGenerator(nn.MediaMaker):
    model = mf.Model.text_to_image("openai/gpt-image-1")
    message_fields = {"task_inputs": "prompt"}
    response_mode = "illustration"


prompter = StoryToPrompt()
generator = ImageGenerator()

pipeline = Inline(
    "prompter -> generator",
    {"prompter": prompter, "generator": generator},
)

msg = Message()
msg.scene = """
Chapter 3: The hero stood at the edge of the cliff,
watching the dragon descend from the storm clouds.
Lightning illuminated the ancient castle behind them.
"""

pipeline(msg)
print(msg.prompt)       # Detailed image prompt
# msg.illustration contains the base64 string
```

---

## 10. **Guardrails**

Use `Guard` hooks to validate prompts before generation — useful for blocking unsafe content or enforcing prompt policies.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.nn.hooks import Guard
from msgflux.exceptions import UnsafeUserInputError

BLOCKED = {"violence", "explicit", "nsfw"}

def content_policy(data):
    text = str(data).lower()
    return {"safe": not any(w in text for w in BLOCKED)}

# With message: short-circuits and returns the message directly (model never called)
class SafeImageMaker(nn.MediaMaker):
    model = mf.Model.text_to_image("openai/gpt-image-1")
    hooks = [
        Guard(
            validator=content_policy,
            on="pre",
            message="Prompt violates content policy.",
        )
    ]

maker = SafeImageMaker()
result = maker("explicit content")  # → "Prompt violates content policy."
```

Without `message`, a `UnsafeUserInputError` is raised instead:

```python
class StrictMaker(nn.MediaMaker):
    model = mf.Model.text_to_image("openai/gpt-image-1")
    hooks = [Guard(validator=content_policy, on="pre")]

maker = StrictMaker()

try:
    maker("violence scene")
except UnsafeUserInputError:
    print("Prompt blocked.")
```

---

## 11. **Async**

```python
image = await maker.acall("A colorful abstract painting")
```

---

## 12. **Debugging**

```python
# Inspect parameters before execution
params = maker.inspect_model_execution_params("test prompt")
print(params)
```
