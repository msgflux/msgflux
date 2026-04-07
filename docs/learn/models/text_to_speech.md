# Text to Speech

The `text_to_speech` model converts text into natural-sounding spoken audio. These models enable voice generation for accessibility, content creation, virtual assistants, and more.

## ✦₊⁺ Overview

Text-to-speech (TTS) models transform written text into spoken audio. They enable:

- **Voice Synthesis**: Convert any text to natural-sounding speech
- **Voice Selection**: Choose from different voice profiles
- **Speed Control**: Adjust speaking rate
- **Format Options**: Generate audio in various formats
- **Streaming**: Real-time audio generation

### Common Use Cases

- **Accessibility**: Convert text to speech for visually impaired users
- **Content Creation**: Generate voiceovers for videos and podcasts
- **Virtual Assistants**: Add voice to chatbots and AI assistants
- **Audio Books**: Convert written content to audio format
- **Language Learning**: Pronunciation examples
- **Notifications**: Voice alerts and announcements

## 1. **Supported Providers**

!!! info "Dependencies"
    See [Dependency Management](../../dependency-management.md) for the complete provider matrix.

???+ example

    === "OpenAI"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf

        # mf.set_envs(OPENAI_API_KEY="...")

        model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
        ```

    === "Together AI"

        ```python
        # pip install msgflux[openai]
        import msgflux as mf

        # mf.set_envs(TOGETHER_API_KEY="...")

        model = mf.Model.text_to_speech(
            "together/canopylabs/orpheus-3b-0.1-ft",
            voice="tara",
            response_format="mp3"
        )
        ```

## 2. **Quick Start**

???+ example

    === "Basic"

        ```python
        import msgflux as mf

        # Create TTS model
        model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")

        # Generate speech
        response = model("Hello, how are you today?")

        # Get audio file path
        audio_path = response.consume()
        print(audio_path)  # /tmp/tmpXXXXXX.opus
        ```

    === "With Voice"

        ```python
        import msgflux as mf

        model = mf.Model.text_to_speech(
            "openai/gpt-4o-mini-tts",
            voice="nova",  # Female voice
            speed=1.0      # Normal speed
        )

        response = model("Welcome to our service!")
        audio_path = response.consume()
        ```

## 3. **Audio Formats**

### Format Comparison

| Format | Quality | Size | Use Case |
|--------|---------|------|----------|
| **opus** | High | Small | Streaming, real-time |
| **mp3** | Good | Medium | Universal playback |
| **aac** | High | Small | Mobile, web |
| **flac** | Lossless | Large | Archival, editing |
| **wav** | Lossless | Large | Professional audio |
| **pcm** | Raw | Largest | Audio processing |

## 4. **Voice Instructions**

Control voice characteristics with prompts:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")

    # Add emotional tone
    response = model(
        "I'm so excited about this!",
        prompt="Speak with enthusiasm and energy"
    )

    # Control pacing
    response = model(
        "This is an important announcement.",
        prompt="Speak slowly and clearly, emphasizing each word"
    )

    # Set context
    response = model(
        "Welcome to the show!",
        prompt="Speak as a radio host, upbeat and friendly"
    )
    ```

Note: `gpt-4o-mini-tts` has native steerability — you can instruct not just *what* to say but *how* to say it.

## 5. **Streaming Audio**

Generate and play audio in real-time:

???+ example

    === "Basic"

        ```python
        import msgflux as mf
        import asyncio

        model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")

        response = model(
            "This is a long text that will be streamed as audio chunks...",
            stream=True
        )

        # consume() returns an async generator
        async def handle():
            async for chunk in response.consume():
                if chunk is None:  # End of stream
                    break
                # chunk is bytes - play or save incrementally
                process_audio_chunk(chunk)

        asyncio.run(handle())
        ```

    === "To File"

        ```python
        import msgflux as mf
        import asyncio

        model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")

        async def save():
            response = model(
                "This will be streamed to a file.",
                stream=True,
                response_format="mp3"
            )

            with open("output.mp3", "wb") as f:
                async for chunk in response.consume():
                    if chunk is None:
                        break
                    f.write(chunk)

            print("Audio saved to output.mp3")

        asyncio.run(save())
        ```

    === "Together AI"

        ```python
        import msgflux as mf
        import asyncio

        model = mf.Model.text_to_speech(
            "together/canopylabs/orpheus-3b-0.1-ft",
            voice="tara",
            response_format="mp3"
        )

        async def save():
            response = model(
                "Today is a wonderful day to build something people love!",
                stream=True
            )

            with open("output.mp3", "wb") as f:
                async for chunk in response.consume():
                    if chunk is None:
                        break
                    f.write(chunk)

        asyncio.run(save())
        ```

    === "Playback"

        ```python
        import msgflux as mf
        import asyncio
        import pyaudio  # pip install pyaudio

        model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")

        async def play():
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,  # 24kHz for TTS
                output=True
            )

            response = model(
                "This will be played in real-time.",
                stream=True,
                response_format="pcm"
            )

            async for chunk in response.consume():
                if chunk is None:
                    break
                stream.write(chunk)

            stream.stop_stream()
            stream.close()
            p.terminate()

        asyncio.run(play())
        ```

## 6. **Async Support**

Generate audio asynchronously:

???+ example

    === "Basic"

        ```python
        import msgflux as mf
        import asyncio

        model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")

        async def main():
            response = await model.acall("Hello, how are you today?")
            audio_path = response.consume()
            print(audio_path)

        asyncio.run(main())
        ```

    === "Streaming"

        ```python
        import msgflux as mf
        import asyncio

        model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")

        async def stream_speech(text):
            response = await model.acall(text, stream=True)

            async for chunk in response.consume():
                if chunk is None:
                    break
                # Process chunk asynchronously
                await process_chunk(chunk)

        asyncio.run(stream_speech("Hello world"))
        ```

## 7. **Batch Processing**

Generate multiple audio files:

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F

    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts", voice="nova")

    texts = [
        "Welcome to chapter one.",
        "Welcome to chapter two.",
        "Welcome to chapter three."
    ]

    # Generate in parallel
    results = F.map_gather(
        model,
        args_list=[(text,) for text in texts]
    )

    # Save all files
    for i, result in enumerate(results):
        audio_path = result.consume()
        import shutil
        shutil.copy(audio_path, f"chapter_{i+1}.opus")
    ```

## 8. **Working with Audio Files**

???+ example

    === "Save"

        ```python
        import msgflux as mf
        import shutil

        model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")

        response = model("Save this audio", response_format="mp3")

        # Get temporary file path and copy to desired location
        temp_path = response.consume()
        shutil.copy(temp_path, "output.mp3")
        print("Saved to output.mp3")
        ```

    === "Play"

        ```python
        import msgflux as mf
        import subprocess

        model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")

        response = model("Play this message")
        audio_path = response.consume()

        # Play with system player
        subprocess.run(["mpv", audio_path])  # Or use "afplay" on macOS
        ```

    === "Audio Info"

        ```python
        import msgflux as mf
        from pydub import AudioSegment

        model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")

        response = model("Get information about this audio", response_format="mp3")
        audio_path = response.consume()

        # Load with pydub
        audio = AudioSegment.from_mp3(audio_path)

        print(f"Duration: {len(audio) / 1000:.2f} seconds")
        print(f"Channels: {audio.channels}")
        print(f"Frame rate: {audio.frame_rate} Hz")
        print(f"Sample width: {audio.sample_width} bytes")
        ```

## 9. **Error Handling**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")

    try:
        response = model("Hello world")
        audio_path = response.consume()
    except ImportError:
        print("Provider not installed")
    except ValueError as e:
        print(f"Invalid parameters: {e}")
        # Common issues:
        # - Invalid voice name
        # - Speed out of range (0.25-4.0)
        # - Invalid response_format
    except Exception as e:
        print(f"Generation failed: {e}")
        # Common errors:
        # - Rate limits
        # - Network issues
        # - Text too long
    ```
