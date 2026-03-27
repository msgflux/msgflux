# nn.Speaker

## ✦₊⁺ Overview

The `nn.Speaker` module converts text into natural-sounding speech using text-to-speech models.

## 1. **Quick Start**

### AutoParams Initialization (Recommended)

Define reusable voice personas.

```python
import msgflux as mf
import msgflux.nn as nn
import shutil

class NaturalVoiceSpeaker(nn.Speaker):
    """Natural-sounding speaker for user-facing applications."""
    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
    response_format = "pcm"
    config = {"voice": "nova", "speed": 1.0}

speaker = NaturalVoiceSpeaker()

# Generate speech — returns a temp file path
audio_path = speaker("Hello, welcome to msgFlux!")

# Copy to a permanent location
shutil.copy(audio_path, "welcome.mp3")
```

### Traditional Initialization

```python
model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")

speaker = nn.Speaker(
    model=model,
    response_format="mp3",
    config={"voice": "nova"}
)

# Returns a temp file path (str)
audio_path = speaker("Hello world")
```

---

## 2. **Audio Formats**

Choose the right format for your use case.

| Format | Description | Use Case |
|--------|-------------|----------|
| `"mp3"` | Universal, compressed | Podcasts, UI sounds |
| `"opus"` | Low latency, high efficiency | Streaming, RTC |
| `"flac"` | Lossless compressed | Archival, high-end audio |
| `"wav"` | Uncompressed | Editing, post-processing |
| `"aac"` | Standard compressed | Mobile apps |
| `"pcm"` | Raw audio bytes | Real-time playback processing |

```python
class StreamingSpeaker(nn.Speaker):
    """Low-latency speaker for chunks."""
    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
    response_format = "opus"

streamer = StreamingSpeaker()
```

---

## 3. **Configuration**

### Voice & Speed

Control characteristics via `config`.

```python
class NarratorSpeaker(nn.Speaker):
    """Clear, neutral voice for audiobooks."""
    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
    response_format = "aac"
    config = {
        "voice": "echo",  # Provider-specific voice ID
        "speed": 0.9      # 1.0 is normal speed
    }

narrator = NarratorSpeaker()
narrator("Hello world")
```

### Guardrails

Validate input text before generation to save costs and ensure safety. Use `Guard` hooks — `on="pre"` runs before the model call.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux.nn.hooks import Guard
from msgflux.exceptions import UnsafeUserInputError

def length_validator(data):
    return {"safe": len(str(data)) <= 4096}

# With message: short-circuits and returns the message directly (model never called)
class SafeSpeaker(nn.Speaker):
    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
    hooks = [
        Guard(
            validator=length_validator,
            on="pre",
            message="Input too long, generation blocked.",
        )
    ]

speaker = SafeSpeaker()
result = speaker(very_long_text)  # → "Input too long, generation blocked."
```

Without `message`, a `UnsafeUserInputError` is raised instead:

```python
class StrictSpeaker(nn.Speaker):
    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
    hooks = [Guard(validator=length_validator, on="pre")]

speaker = StrictSpeaker()

try:
    speaker(very_long_text)
except UnsafeUserInputError:
    print("Input too long, generation blocked.")
```

### Prompt Guidance

Some models accept a system prompt or style guidance. `gpt-4o-mini-tts` has native steerability — instruct not just *what* to say but *how* to say it.

```python
class StorytellerSpeaker(nn.Speaker):
    """Expressive speaker."""
    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
    prompt = "Speak with dramatic pauses and emotional variation."

storyteller = StorytellerSpeaker()

# Override prompt at call time
audio = storyteller(
    "Welcome to the show!",
    prompt="Speak as a radio host, upbeat and friendly"
)
```

---

## 4. **Streaming**

Enable streaming via `config={"stream": True}`. The result is a `ModelStreamResponse` — consume it with `async for` via `.consume()`.

```python
import asyncio
import msgflux as mf
import msgflux.nn as nn

class StreamingSpeaker(nn.Speaker):
    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
    response_format = "opus"
    config = {"stream": True}

speaker = StreamingSpeaker()
```

**Save to file:**

```python
async def save_to_file():
    stream = speaker("This will be streamed to a file.")
    with open("output.opus", "wb") as f:
        async for chunk in stream.consume():
            if chunk is None:
                break
            f.write(chunk)

asyncio.run(save_to_file())
```

**Real-time playback** with `pyaudio` (use `pcm` format):

```python
# pip install pyaudio
import pyaudio

class RealtimeSpeaker(nn.Speaker):
    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
    response_format = "pcm"
    config = {"stream": True}

async def play_realtime():
    speaker = RealtimeSpeaker()
    stream = speaker("Streaming audio in real time.")

    pa = pyaudio.PyAudio()
    audio_out = pa.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    async for chunk in stream.consume():
        if chunk is None:
            break
        audio_out.write(chunk)

    audio_out.close()
    pa.terminate()

asyncio.run(play_realtime())
```

---

## 5. **Integration with Agents**

Speakers typically sit at the end of a voice pipeline (Agent -> Speaker).

```python
class VoiceAssistant(nn.Agent):
    """Voice-enabled assistant."""
    model = mf.Model.chat_completion("openai/gpt-4o-mini")

class ResponseSpeaker(nn.Speaker):
    """Converts agent responses to speech."""
    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
    response_format = "mp3"

# simple pipeline
assistant = VoiceAssistant()
speaker = ResponseSpeaker()

audio_path = speaker(assistant("What's the weather?"))
```

---

## 6. **Message Field Mapping**

Automatically extract text from structured messages.

```python
class NotificationSpeaker(nn.Speaker):
    """Reads notifications."""
    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
    response_mode = "audio"
    message_fields = {"task_inputs": "notification.text"}
    response_format = "mp3"

speaker = NotificationSpeaker()

msg = mf.Message()
msg.set("notification.text", "You have a new meeting.")

speaker(msg)  # mutates msg in place, returns None
audio_path = msg.get("audio")
```

---

## 7. **Creating Speaker Hierarchies**

Share configuration across related speakers.

```python
# Base speaker for announcements
class AnnouncementSpeaker(nn.Speaker):
    response_format = "mp3"
    config = {"voice": "onyx"}

# Urgent announcements
class EmergencySpeaker(AnnouncementSpeaker):
    config = {"voice": "onyx", "speed": 1.1}

# Casual announcements
class CasualSpeaker(AnnouncementSpeaker):
    config = {"voice": "nova", "speed": 1.0}
```
