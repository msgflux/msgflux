# Speech to Text

The `speech_to_text` model transcribes spoken audio into written text. These models enable voice-to-text conversion for accessibility, transcription services, voice commands, and more.

## ✦₊⁺ Overview

Speech-to-text (STT) models convert audio recordings into text transcripts. They enable:

- **Transcription**: Convert spoken audio to written text
- **Timestamping**: Get word and segment-level timestamps
- **Multiple Formats**: Output as text, JSON, SRT, VTT subtitles
- **Language Detection**: Automatic or manual language specification
- **Context Awareness**: Use prompts to improve accuracy

### Common Use Cases

- **Meeting Transcription**: Convert recordings to searchable text
- **Subtitle Generation**: Create subtitles for videos
- **Voice Commands**: Process spoken user commands
- **Accessibility**: Provide captions for audio content
- **Interview Analysis**: Transcribe interviews and podcasts
- **Call Center**: Analyze customer service calls

## 1. **Quick Start**

### Basic Usage

???+ example

    ```python
    import msgflux as mf

    # Recommended — best accuracy, supports streaming
    model = mf.Model.speech_to_text("openai/whisper-1")

    # Transcribe audio file
    response = model("path/to/audio.mp3")

    # Get transcript
    transcript = response.consume()
    print(transcript["text"])
    # "Hello, this is a test recording."
    ```

### From URL

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/whisper-1")

    # Transcribe from URL
    response = model("https://example.com/audio.mp3")
    transcript = response.consume()
    print(transcript["text"])
    ```

## 2. **Audio Format Support**

### Supported Formats

- **MP3** (.mp3)
- **MP4** (.mp4, audio track)
- **MPEG** (.mpeg)
- **MPGA** (.mpga)
- **M4A** (.m4a)
- **WAV** (.wav)
- **WEBM** (.webm)

### File Size Limits

- Maximum file size: **25 MB**
- For larger files, split into chunks or compress

## 3. **Supported Providers**

!!! info "Dependencies"
    See [Dependency Management](../../dependency-management.md) for the complete provider matrix.

### OpenAI

???+ example

    ```python
    import msgflux as mf

    # Recommended — best accuracy, lower WER, streaming support
    model = mf.Model.speech_to_text("openai/gpt-4o-transcribe")

    # Recommended — faster and cheaper, streaming support
    model = mf.Model.speech_to_text("openai/gpt-4o-mini-transcribe")

    # Pinned snapshot (~90% fewer hallucinations vs Whisper v2)
    model = mf.Model.speech_to_text("openai/gpt-4o-mini-transcribe-2025-12-15")

    # Speaker diarization (identifies who is speaking)
    model = mf.Model.speech_to_text("openai/gpt-4o-transcribe-diarize")

    # Legacy — rich format support (verbose_json, srt, vtt, timestamps, temperature)
    model = mf.Model.speech_to_text("openai/whisper-1")
    ```

## 4. **Response Formats**

!!! info "Format support by model"

    | Format | `gpt-4o-transcribe` / `gpt-4o-mini-transcribe` | `whisper-1` |
    |--------|------------------------------------------------|-------------|
    | `text` | Yes | Yes |
    | `json` | Yes | Yes |
    | `verbose_json` | No | Yes |
    | `srt` | No | Yes |
    | `vtt` | No | Yes |

### Text Format (Default)

Simple text output:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/whisper-1")

    response = model(
        "audio.mp3",
        response_format="text"
    )

    transcript = response.consume()
    print(transcript["text"])
    # "This is the transcribed text."
    ```

### JSON Format

Structured output:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/whisper-1")

    response = model(
        "audio.mp3",
        response_format="json"
    )

    transcript = response.consume()
    print(transcript)
    # {"text": "This is the transcribed text."}
    ```

### Verbose JSON

Detailed output with metadata:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/whisper-1")

    response = model(
        "audio.mp3",
        response_format="verbose_json"
    )

    transcript = response.consume()
    print(transcript)
    # {
    #     "text": "This is the transcribed text.",
    #     "language": "en",
    #     "duration": 5.2,
    #     "segments": [...]
    # }
    ```

### SRT (SubRip) Format

Subtitle format for videos:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/whisper-1")

    response = model(
        "audio.mp3",
        response_format="srt"
    )

    transcript = response.consume()
    print(transcript["text"])
    # 1
    # 00:00:00,000 --> 00:00:02,000
    # This is the first subtitle
    #
    # 2
    # 00:00:02,000 --> 00:00:05,000
    # This is the second subtitle
    ```

### VTT (WebVTT) Format

Web-friendly subtitle format:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/whisper-1")

    response = model(
        "audio.mp3",
        response_format="vtt"
    )

    transcript = response.consume()
    print(transcript["text"])
    # WEBVTT
    #
    # 00:00:00.000 --> 00:00:02.000
    # This is the first subtitle
    #
    # 00:00:02.000 --> 00:00:05.000
    # This is the second subtitle
    ```

!!! info "Whisper-only"

    `timestamp_granularities` requires `response_format="verbose_json"` and is only supported by `whisper-1`. The `gpt-4o-transcribe` family does not support granular timestamps.

## 5. **Timestamp Granularities**

### Word-Level Timestamps

Get timestamp for each word:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/whisper-1")

    response = model(
        "audio.mp3",
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )

    transcript = response.consume()
    print(transcript["words"])
    # [
    #     {"word": "Hello", "start": 0.0, "end": 0.5},
    #     {"word": "world", "start": 0.6, "end": 1.1}
    # ]
    ```

### Segment-Level Timestamps

Get timestamps for phrases/segments:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/whisper-1")

    response = model(
        "audio.mp3",
        response_format="verbose_json",
        timestamp_granularities=["segment"]
    )

    transcript = response.consume()
    print(transcript["segments"])
    # [
    #     {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello world."},
    #     {"id": 1, "start": 2.5, "end": 5.0, "text": "How are you?"}
    # ]
    ```

### Both Word and Segment Timestamps

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/whisper-1")

    response = model(
        "audio.mp3",
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"]
    )

    transcript = response.consume()
    print("Words:", transcript["words"])
    print("Segments:", transcript["segments"])
    ```

## 6. **Language Specification**

### Automatic Detection

By default, Whisper auto-detects the language:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/whisper-1")

    response = model("audio.mp3")
    transcript = response.consume()
    # Language automatically detected
    ```

### Manual Language Specification

Improve accuracy and speed by specifying the language:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/whisper-1")

    response = model("audio.mp3", language="en")  # ISO 639-1 code
    ```

### ISO 639-1 Language Codes

Common language codes:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese
- `ar` - Arabic
- `hi` - Hindi

## 7. **Context and Prompts**

Improve transcription accuracy with context:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/whisper-1")

    # Technical content
    response = model(
        "meeting.mp3",
        prompt="This is a technical discussion about machine learning, neural networks, and AI"
    )

    # Names and terminology
    response = model(
        "interview.mp3",
        prompt="Interview with Dr. Smith about quantum computing"
    )

    # Continuing previous segment
    response = model(
        "part2.mp3",
        prompt="Previous text ended with: ...and that's how we solved the problem."
    )
    ```

!!! info "Whisper-only"

    `temperature` is only supported by `whisper-1`. The `gpt-4o-transcribe` family ignores this parameter.

## 8. **Temperature Control**

Control transcription randomness:

???+ example

    ```python
    import msgflux as mf

    # Deterministic (temperature=0)
    model = mf.Model.speech_to_text("openai/whisper-1", temperature=0.0)

    # More creative (higher temperature)
    model = mf.Model.speech_to_text("openai/whisper-1", temperature=0.3)
    ```

Note: Lower temperature = more conservative/repetitive, Higher temperature = more creative but potentially less accurate.

## 9. **Streaming**

!!! info "Requires gpt-4o-transcribe or gpt-4o-mini-transcribe"

    Streaming is **not supported** by `whisper-1` — if passed, the parameter is silently ignored.
    Use `gpt-4o-transcribe` or `gpt-4o-mini-transcribe` for real-time transcription.

Process transcription in real-time:

???+ example

    ```python
    import msgflux as mf
    import asyncio

    # Streaming only works with the gpt-4o-transcribe family
    model = mf.Model.speech_to_text("openai/gpt-4o-mini-transcribe")

    # Stream transcription — consume() returns an async generator
    async def transcribe():
        response = model("long_audio.mp3", stream=True)
        async for chunk in response.consume():
            if chunk is None:
                break
            print(chunk, end="", flush=True)

    asyncio.run(transcribe())
    ```

## 10. **Speaker Diarization**

Identify who is speaking with `gpt-4o-transcribe-diarize`:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/gpt-4o-transcribe-diarize")

    response = model("meeting.mp3")
    transcript = response.consume()

    # Each segment includes speaker identification
    for segment in transcript.get("segments", []):
        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "")
        print(f"[{speaker}] {text}")
    ```

!!! info

    `gpt-4o-transcribe-diarize` is available via the transcriptions endpoint only and is not yet supported in the Realtime API.

## 11. **Async Support**

Transcribe audio asynchronously:

???+ example

    ```python
    import msgflux as mf
    import asyncio

    model = mf.Model.speech_to_text("openai/whisper-1")

    audio_files = ["audio1.mp3", "audio2.mp3", "audio3.mp3"]

    responses = await asyncio.gather(*[model.acall(f) for f in audio_files])

    for file, response in zip(audio_files, responses):
        transcript = response.consume()
        print(f"{file}: {transcript['text']}")
    ```

## 12. **Batch Processing**

Transcribe multiple files:

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F

    model = mf.Model.speech_to_text("openai/whisper-1")

    audio_files = [
        "meeting1.mp3",
        "meeting2.mp3",
        "meeting3.mp3"
    ]

    # Process in parallel
    results = F.map_gather(
        model,
        args_list=[(f,) for f in audio_files]
    )

    # Get all transcripts
    for file, result in zip(audio_files, results):
        transcript = result.consume()
        print(f"{file}:")
        print(transcript["text"])
        print()
    ```

## 13. **Error Handling**

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.speech_to_text("openai/whisper-1")

    try:
        response = model("audio.mp3")
        transcript = response.consume()
    except ImportError:
        print("Provider not installed")
    except ValueError as e:
        print(f"Invalid parameters: {e}")
        # Common issues:
        # - Invalid language code
        # - Invalid response_format
        # - File too large (>25MB)
    except FileNotFoundError:
        print("Audio file not found")
    except Exception as e:
        print(f"Transcription failed: {e}")
        # Common errors:
        # - Unsupported audio format
        # - Corrupted audio file
        # - Network issues
        # - Rate limits
    ```
