# nn.Transcriber

## ✦₊⁺ Overview

The `nn.Transcriber` module wraps speech-to-text models to **transcribe audio** into text or structured data.

---

## 1. **Quick Start**

!!! info "Initialization styles"

    === "Declarative (recommended)"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class Speech2Text(nn.Transcriber):
            """Transcribes user voice notes."""
            model          = mf.Model.speech_to_text("openai/whisper-1")
            response_mode  = "content"
            message_fields = {"task_multimodal": {"audio": "user_audio"}}

        transcriber = Speech2Text()
        result = transcriber("/path/to/audio.mp3")
        ```

    === "Direct"

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        transcriber = nn.Transcriber(
            model=mf.Model.speech_to_text("openai/whisper-1")
        )
        result = transcriber("/path/to/audio.mp3")
        ```

## String shorthand

When you do not need to configure extra model parameters, you can pass a
`"provider/model-id"` string directly as the `model` argument. msgFlux will
call `Model.speech_to_text` internally.

```python
import msgflux.nn as nn

transcriber = nn.Transcriber("openai/whisper-1")
result = transcriber("/path/to/audio.mp3")
```

The shorthand also works when reassigning `transcriber.model` after
construction:

```python
transcriber.model = "openai/whisper-1"
```

---

## 2. **Input Types**

!!! info "Supported audio inputs"

    === "File path"

        ```python
        result = transcriber("/path/to/audio.mp3")
        ```

    === "URL"

        ```python
        result = transcriber("https://example.com/audio.wav")
        ```

    === "Bytes"

        ```python
        with open("audio.mp3", "rb") as f:
            audio_bytes = f.read()

        result = transcriber(audio_bytes)
        ```

    === "Message"

        Extract audio from a structured message via `message_fields`:

        ```python
        import msgflux as mf
        import msgflux.nn as nn

        class Speech2Text(nn.Transcriber):
            model          = mf.Model.speech_to_text("openai/whisper-1")
            message_fields = {"task_multimodal": {"audio": "user_audio"}}
            response_mode  = "transcription"

        transcriber = Speech2Text()

        msg = mf.dotdict(user_audio="/path/to/audio.mp3")
        transcriber(msg)
        print(msg.transcription)
        ```

---

## 3. **Parameters**

| Parameter | Description |
|-----------|-------------|
| `model` | Speech-to-text model client instance |
| `message_fields` | Map inputs (audio) from Message fields |
| `response_mode` | Where to write the output in the Message |
| `response_template` | Jinja template to format the output string |
| `response_format` | `"text"` (default), `"json"`, `"verbose_json"`, `"srt"`, `"vtt"` |
| `prompt` | Optional text prompt to guide style or vocabulary |
| `config` | Runtime params passed to the model: `language`, `stream`, `timestamp_granularities` |

---

## 4. **Configuration**

!!! info "Controlling transcription behavior"

    === "Language"

        Specify the spoken language to improve accuracy and speed via `config`:

        ```python
        class PortugueseTranscriber(nn.Transcriber):
            model  = mf.Model.speech_to_text("openai/whisper-1")
            config = {"language": "pt"}  # ISO 639-1 code

        transcriber = PortugueseTranscriber()
        result = transcriber("audio.mp3")
        ```

    === "Timestamps"

        Request word or segment-level timestamps. Requires `response_format="verbose_json"` and `whisper-1`:

        ```python
        class TimestampTranscriber(nn.Transcriber):
            model           = mf.Model.speech_to_text("openai/whisper-1")
            response_format = "verbose_json"
            config          = {"timestamp_granularities": ["word"]}

        transcriber = TimestampTranscriber()
        result = transcriber("audio.mp3")
        # result["text"]  — full transcript
        # result["words"] — [{"word": "Hello", "start": 0.0, "end": 0.5}, ...]
        ```

    === "Subtitles"

        Export directly as SRT or VTT for video workflows:

        ```python
        class SubtitleGenerator(nn.Transcriber):
            model           = mf.Model.speech_to_text("openai/whisper-1")
            response_format = "srt"

        gen = SubtitleGenerator()
        srt_content = gen("video_audio.mp3")
        ```

---

## 5. **Integration with Agents**

Transcribers are often the first step in a voice processing pipeline.

!!! info "Transcriber → Agent pipeline"

    ```python
    import msgflux as mf
    import msgflux.nn as nn

    class Speech2Text(nn.Transcriber):
        model          = mf.Model.speech_to_text("openai/whisper-1")
        message_fields = {"task_multimodal": {"audio": "user_audio"}}
        response_mode  = "content"

    class Analyzer(nn.Agent):
        """Analyzes the transcribed text."""
        model          = mf.Model.chat_completion("openai/gpt-4.1-mini")
        message_fields = {"task": "content"}
        response_mode  = "analysis"

    transcriber = Speech2Text()
    analyzer = Analyzer()

    pipeline = mf.Inline(
        "{user_audio is not None? transcriber} -> analyzer",
        {"transcriber": transcriber, "analyzer": analyzer},
    )

    msg = mf.dotdict(user_audio="/path/to/voice_note.mp3")
    pipeline(msg)
    print(f"Transcript: {msg.content}")
    print(f"Analysis: {msg.analysis}")
    ```

---

## 6. **Async**

```python
result = await transcriber.acall("/path/to/audio.mp3")
```

---

## 7. **Debugging**

```python
params = transcriber.inspect_model_execution_params("/path/to/audio.mp3")
print(params)
```
