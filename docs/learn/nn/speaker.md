# nn.Speaker

## ✦₊⁺ Overview

The `nn.Speaker` module converts text into natural-sounding speech using text-to-speech models.

---

## 1. **Quick Start**

!!! info "Initialization styles"

    === "Declarative (recommended)"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class NaturalVoiceSpeaker(nn.Speaker):
            """Natural-sounding speaker for user-facing applications."""
            model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
            response_format = "pcm"
            config = {"voice": "nova", "speed": 1.0}

        speaker = NaturalVoiceSpeaker()
        audio_path = speaker("Hello, welcome to msgFlux!")
        ```

    === "Direct"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        speaker = nn.Speaker(
            model=mf.Model.text_to_speech("openai/gpt-4o-mini-tts"),
            response_format="mp3",
            config={"voice": "nova"},
        )

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

---

## 3. **Configuration**

!!! info "Controlling voice and behavior"

    === "Voice & Speed"

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

    === "Prompt Guidance"

        `gpt-4o-mini-tts` has native steerability — instruct not just *what* to say but *how* to say it:

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

    === "Hierarchies"

        Share configuration across related speakers via inheritance:

        ```python
        class AnnouncementSpeaker(nn.Speaker):
            model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
            response_format = "mp3"
            config = {"voice": "onyx"}

        class EmergencySpeaker(AnnouncementSpeaker):
            config = {"voice": "onyx", "speed": 1.1}

        class CasualSpeaker(AnnouncementSpeaker):
            config = {"voice": "nova", "speed": 1.0}
        ```

---

## 4. **Guardrails**

Use `Guard` hooks to validate input text before generation.

!!! info "Guard patterns"

    === "Short-circuit (with message)"

        When `message` is provided, the guard returns it directly — the model is never called:

        ```python
        import msgflux as mf
        import msgflux.nn as nn
        from msgflux.nn.hooks import Guard

        def length_validator(data):
            return {"safe": len(str(data)) <= 4096}

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

    === "Raises exception"

        Without `message`, a `UnsafeUserInputError` is raised instead:

        ```python
        from msgflux.exceptions import UnsafeUserInputError

        class StrictSpeaker(nn.Speaker):
            model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
            hooks = [Guard(validator=length_validator, on="pre")]

        speaker = StrictSpeaker()

        try:
            speaker(very_long_text)
        except UnsafeUserInputError:
            print("Input too long, generation blocked.")
        ```

---

## 5. **Streaming**

Enable streaming via `config={"stream": True}`. The result is a `ModelStreamResponse` — consume it with `async for` via `.consume()`.

!!! info "Streaming patterns"

    === "Save to file"

        ```python
        import asyncio
        import msgflux as mf
        import msgflux.nn as nn

        class StreamingSpeaker(nn.Speaker):
            model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
            response_format = "opus"
            config = {"stream": True}

        speaker = StreamingSpeaker()

        async def save_to_file():
            stream = speaker("This will be streamed to a file.")
            with open("output.opus", "wb") as f:
                async for chunk in stream.consume():
                    if chunk is None:
                        break
                    f.write(chunk)

        asyncio.run(save_to_file())
        ```

    === "Real-time playback"

        Real-time playback with `pyaudio` (use `pcm` format):

        ```python
        # pip install pyaudio
        import asyncio
        import pyaudio
        import msgflux as mf
        import msgflux.nn as nn

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

## 6. **Integration with Agents**

Speakers typically sit at the end of a voice pipeline (Agent → Speaker).

!!! info "Agent → Speaker pipeline"

    === "Simple pipeline"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class VoiceAssistant(nn.Agent):
            """Voice-enabled assistant."""
            model = mf.Model.chat_completion("openai/gpt-4o-mini")

        class ResponseSpeaker(nn.Speaker):
            """Converts agent responses to speech."""
            model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
            response_format = "mp3"

        assistant = VoiceAssistant()
        speaker = ResponseSpeaker()

        audio_path = speaker(assistant("What's the weather?"))
        ```

    === "With Message"

        Bind to fields on a shared `Message` for pipeline composition:

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class NotificationSpeaker(nn.Speaker):
            """Reads notifications."""
            model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
            response_mode = "audio"
            message_fields = {"task_inputs": "notification.text"}
            response_format = "mp3"

        speaker = NotificationSpeaker()

        msg = mf.dotdict()
        msg.notification = mf.dotdict(text="You have a new meeting.")

        speaker(msg)  # mutates msg in place, returns None
        audio_path = msg.audio
        ```
