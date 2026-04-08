# Meeting Assistant

<span class="tag tag-purple">Intermediate</span><span class="tag tag-gray">Signature</span><span class="tag tag-gray">Multimodal</span>

Turn a meeting recording into structured notes: transcribe the audio, then extract decisions, action items, open questions, sentiment, and a TL;DR — all with typed outputs from a single `Signature`.

---

## The Problem

Every meeting produces the same result: a recording that no one re-listens to and notes that vary by who took them. Action items get lost. Decisions that were made get re-discussed. Open questions go untracked.

The gap is not the recording — it is the structured summary that never gets produced. Asking a model to summarize without a schema produces prose. Asking with a schema but without explicit typed fields produces inconsistent JSON. And a field like `action_items: List[dict]` — with no fixed shape — is not a valid structured output at all.

---

## The Plan

We will build a pipeline that accepts an audio recording or a raw transcript and returns a fully typed summary: a one-sentence TL;DR, the list of decisions made, action items with owner and deadline, open questions that need follow-up, overall sentiment, and whether a follow-up meeting was agreed.

Action items use three parallel lists — owner, task, deadline — instead of a list of dicts. Each list has a well-defined element type, which makes the output schema valid. The pipeline zips them back into records before returning.

---

## Architecture

```
Meeting input (audio or transcript)
        │
        ├── audio? → MeetingTranscriber (Whisper) → transcript
        │
        ▼
   MeetingAnalyzer (nn.Agent + Signature)
        │
        ▼
   {tldr, decisions, action_items,
    open_questions, sentiment, follow_up_meeting}
```

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1 — Models

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Signature, InputField, OutputField
from typing import List, Literal

mf.load_dotenv()

chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model  = mf.Model.speech_to_text("openai/whisper-1")
```

---

## Step 2 — Signature

Action items are expressed as three correlated parallel lists — `action_owners[i]`, `action_tasks[i]`, and `action_deadlines[i]` describe a single item. This keeps every type expressible in the structured output schema. `List[dict]` has no fixed shape and is rejected at validation time.

```python
class MeetingAnalysis(Signature):
    """Extract structured notes from a meeting transcript."""

    transcript: str = InputField(
        desc="Full verbatim transcript of the meeting"
    )

    tldr: str = OutputField(
        desc="One-sentence summary of what the meeting accomplished"
    )
    decisions: List[str] = OutputField(
        desc="Decisions that were made and agreed upon during the meeting"
    )
    action_owners: List[str] = OutputField(
        desc="Person responsible for each action item (correlated with action_tasks and action_deadlines)"
    )
    action_tasks: List[str] = OutputField(
        desc="What needs to be done for each action item"
    )
    action_deadlines: List[str] = OutputField(
        desc="Due date for each action item, empty string if none stated"
    )
    open_questions: List[str] = OutputField(
        desc="Questions raised during the meeting that were not resolved — need follow-up"
    )
    sentiment: Literal["positive", "neutral", "tense"] = OutputField(
        desc="Overall tone of the meeting"
    )
    follow_up_meeting: bool = OutputField(
        desc="True if the group explicitly agreed to schedule a follow-up meeting"
    )
```

---

## Step 3 — Transcriber and Analyzer

```python
class MeetingTranscriber(nn.Transcriber):
    """Transcribes meeting audio into msg.meeting.transcript."""
    model          = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode  = "meeting.transcript"


class MeetingAnalyzer(nn.Agent):
    """Extracts structured notes from a transcript."""
    model     = chat_model
    signature = MeetingAnalysis
    config    = {"verbose": True}
```

---

## Step 4 — Pipeline

`MeetingAssistant` accepts either audio bytes or a plain transcript string. The `_zip_action_items` method reassembles the three parallel lists into a list of dicts before returning — callers see clean records, not raw arrays.

```python
class MeetingAssistant(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = MeetingTranscriber()
        self.analyzer    = MeetingAnalyzer()

    def _zip_action_items(self, result: dict) -> list:
        return [
            {"owner": o, "task": t, "deadline": d or None}
            for o, t, d in zip(
                result.get("action_owners",    []),
                result.get("action_tasks",     []),
                result.get("action_deadlines", []),
            )
        ]

    def forward(
        self,
        audio:      bytes | None = None,
        transcript: str   | None = None,
    ) -> dict:
        if audio:
            msg = mf.Message()
            msg.audio_content = audio
            self.transcriber(msg)
            transcript = msg.meeting.transcript

        result = self.analyzer(transcript=transcript)
        return {
            "transcript":      transcript,
            "tldr":            result.get("tldr", ""),
            "decisions":       result.get("decisions", []),
            "action_items":    self._zip_action_items(result),
            "open_questions":  result.get("open_questions", []),
            "sentiment":       result.get("sentiment", ""),
            "follow_up_meeting": result.get("follow_up_meeting", False),
        }

    async def aforward(
        self,
        audio:      bytes | None = None,
        transcript: str   | None = None,
    ) -> dict:
        if audio:
            msg = mf.Message()
            msg.audio_content = audio
            await self.transcriber.acall(msg)
            transcript = msg.meeting.transcript

        result = await self.analyzer.acall(transcript=transcript)
        return {
            "transcript":      transcript,
            "tldr":            result.get("tldr", ""),
            "decisions":       result.get("decisions", []),
            "action_items":    self._zip_action_items(result),
            "open_questions":  result.get("open_questions", []),
            "sentiment":       result.get("sentiment", ""),
            "follow_up_meeting": result.get("follow_up_meeting", False),
        }


assistant = MeetingAssistant()
```

---

## Examples

???+ example

    === "From audio file"

        ```python
        assistant = MeetingAssistant()

        result = assistant(audio=open("standup.mp3", "rb").read())

        print("TL;DR:", result["tldr"])

        print("\nDecisions:")
        for d in result["decisions"]:
            print(f"  - {d}")

        print("\nAction Items:")
        for item in result["action_items"]:
            deadline = item["deadline"] or "TBD"
            print(f"  [{item['owner']}] {item['task']} → {deadline}")

        print("\nOpen Questions:")
        for q in result["open_questions"]:
            print(f"  ? {q}")

        print(f"\nSentiment:        {result['sentiment']}")
        print(f"Follow-up needed: {result['follow_up_meeting']}")
        ```

    === "From transcript (no audio)"

        Seed a plain transcript to skip STT — useful for testing or when the transcript already exists.

        ```python
        assistant = MeetingAssistant()

        transcript = """
        Sarah: Alright, let's get started. Main agenda: Q3 roadmap.
        Tom: I think we should prioritize the API rate limiting feature.
             We've had three customer complaints this week alone.
        Sarah: Agreed. That's decided then — API rate limiting goes to the top of Q3.
        Tom: I'll write the technical spec by Friday.
        Sarah: What about the mobile app redesign?
        Lisa: We still haven't decided on the design system. Flutter vs React Native.
        Tom: Can we get a prototype from the design team first before deciding?
        Sarah: Good point. Lisa, can you coordinate that?
        Lisa: Sure, I'll reach out to design. Target date — end of next week?
        Sarah: Perfect. Open question: design system decision is blocked on prototype.
        Tom: Also, we need to align with the backend team on the new auth flow. Not resolved.
        Sarah: Let's schedule a follow-up with them next Tuesday. I'll send the invite.
        """

        result = assistant(transcript=transcript)

        print("TL;DR:", result["tldr"])

        print("\nAction Items:")
        for item in result["action_items"]:
            deadline = item["deadline"] or "TBD"
            print(f"  [{item['owner']}] {item['task']} → {deadline}")
        ```

        ```
        TL;DR: Team aligned on Q3 priorities — API rate limiting goes first,
               mobile redesign blocked on design prototype.

        Action Items:
          [Tom]   Write technical spec for API rate limiting → Friday
          [Lisa]  Coordinate prototype with design team → end of next week
          [Sarah] Send calendar invite for backend alignment → Tuesday
        ```

    === "Async"

        ```python
        import asyncio

        async def main():
            assistant = MeetingAssistant()
            result    = await assistant.acall(audio=open("standup.mp3", "rb").read())

            print("TL;DR:", result["tldr"])
            for item in result["action_items"]:
                print(f"  [{item['owner']}] {item['task']}")

        asyncio.run(main())
        ```

    === "Async batch"

        ```python
        import asyncio
        import msgflux.nn.functional as F

        async def main():
            assistant  = MeetingAssistant()
            recordings = [open(f, "rb").read() for f in ["week1.mp3", "week2.mp3"]]

            results = await F.amap_gather(
                assistant,
                kwargs_list=[{"audio": a} for a in recordings],
            )

            for i, r in enumerate(results, 1):
                print(f"Meeting {i}: {r['tldr']}")

        asyncio.run(main())
        ```

---

## Extending

### Narrating the notes

Add a `nn.Speaker` step after the analyzer to produce an audio summary the team can listen to:

```python
class Narrator(nn.Speaker):
    model           = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
    response_format = "mp3"
    config          = {"voice": "nova"}
```

Call it with `result["tldr"]` for a quick audio notification, or build a formatted script from all sections for a full recap.

### Triggering a follow-up calendar event

Act on `follow_up_meeting` directly after the pipeline returns:

```python
result = assistant(audio=audio_bytes)

if result["follow_up_meeting"]:
    create_calendar_event(
        title="Follow-up: " + result["tldr"],
        attendees=list({item["owner"] for item in result["action_items"]}),
    )
```

### Posting to Slack

```python
import httpx

def post_to_slack(result: dict, webhook_url: str) -> None:
    lines = [f"*{result['tldr']}*", ""]
    for item in result["action_items"]:
        deadline = item["deadline"] or "TBD"
        lines.append(f"• [{item['owner']}] {item['task']} — {deadline}")
    httpx.post(webhook_url, json={"text": "\n".join(lines)})
```

---

## Complete Script

```python
from typing import List, Literal

import msgflux as mf
import msgflux.nn as nn
from msgflux import Signature, InputField, OutputField

mf.load_dotenv()


# Models
chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model  = mf.Model.speech_to_text("openai/whisper-1")


# Signature
class MeetingAnalysis(Signature):
    """Extract structured notes from a meeting transcript."""
    transcript:       str                             = InputField(desc="Full verbatim transcript of the meeting")
    tldr:             str                             = OutputField(desc="One-sentence summary of what the meeting accomplished")
    decisions:        List[str]                       = OutputField(desc="Decisions made and agreed upon")
    action_owners:    List[str]                       = OutputField(desc="Owner for each action item")
    action_tasks:     List[str]                       = OutputField(desc="Task description for each action item")
    action_deadlines: List[str]                       = OutputField(desc="Deadline per item, empty string if none")
    open_questions:   List[str]                       = OutputField(desc="Unresolved questions needing follow-up")
    sentiment:        Literal["positive", "neutral", "tense"] = OutputField(desc="Overall tone of the meeting")
    follow_up_meeting: bool                           = OutputField(desc="True if a follow-up was explicitly agreed")


# Agents
class MeetingTranscriber(nn.Transcriber):
    """Transcribes meeting audio into msg.meeting.transcript."""
    model          = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode  = "meeting.transcript"


class MeetingAnalyzer(nn.Agent):
    """Extracts structured notes from a transcript."""
    model     = chat_model
    signature = MeetingAnalysis
    config    = {"verbose": True}


# Pipeline
class MeetingAssistant(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = MeetingTranscriber()
        self.analyzer    = MeetingAnalyzer()

    def _zip_action_items(self, result: dict) -> list:
        return [
            {"owner": o, "task": t, "deadline": d or None}
            for o, t, d in zip(
                result.get("action_owners",    []),
                result.get("action_tasks",     []),
                result.get("action_deadlines", []),
            )
        ]

    def forward(self, audio: bytes | None = None, transcript: str | None = None) -> dict:
        if audio:
            msg = mf.Message()
            msg.audio_content = audio
            self.transcriber(msg)
            transcript = msg.meeting.transcript

        result = self.analyzer(transcript=transcript)
        return {
            "transcript":        transcript,
            "tldr":              result.get("tldr", ""),
            "decisions":         result.get("decisions", []),
            "action_items":      self._zip_action_items(result),
            "open_questions":    result.get("open_questions", []),
            "sentiment":         result.get("sentiment", ""),
            "follow_up_meeting": result.get("follow_up_meeting", False),
        }

    async def aforward(self, audio: bytes | None = None, transcript: str | None = None) -> dict:
        if audio:
            msg = mf.Message()
            msg.audio_content = audio
            await self.transcriber.acall(msg)
            transcript = msg.meeting.transcript

        result = await self.analyzer.acall(transcript=transcript)
        return {
            "transcript":        transcript,
            "tldr":              result.get("tldr", ""),
            "decisions":         result.get("decisions", []),
            "action_items":      self._zip_action_items(result),
            "open_questions":    result.get("open_questions", []),
            "sentiment":         result.get("sentiment", ""),
            "follow_up_meeting": result.get("follow_up_meeting", False),
        }
```

---

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — signatures and structured output
- [nn.Transcriber](../learn/nn/transcriber.md) — speech-to-text integration
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts
- [Functional API](../learn/nn/functional.md) — `amap_gather` and parallel execution
