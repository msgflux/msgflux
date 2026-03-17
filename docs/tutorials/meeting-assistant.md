# Meeting Assistant

Turn a meeting recording into structured notes: transcribe the audio, then extract decisions, action items, open questions, and a TL;DR — all with typed outputs from a single `Signature`.

## What You'll Build

```
audio file / bytes
       │
       ▼
  Transcriber ──── nn.Transcriber (Whisper)
       │  transcript: str
       ▼
  MeetingAnalyzer ─ Signature:
                      transcript →
                        tldr: str
                        decisions: list[str]
                        action_items: list[dict]
                        open_questions: list[str]
                        sentiment: Literal['positive','neutral','tense']
                        follow_up_meeting: bool
       │
       ▼
  Structured notes on msg
```

---

## Setup

```bash
pip install msgflux[openai]
```

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Step 1 — Transcriber

`nn.Transcriber` wraps Whisper (or any STT model). Pass an audio file path or raw `bytes`:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Signature, InputField, OutputField
from typing import Literal, List


stt   = mf.Model.speech_to_text("openai/whisper-1")
model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class MeetingTranscriber(nn.Transcriber):
    """Transcribes meeting audio to text."""
    model = stt
    response_mode = "transcript"


transcriber = MeetingTranscriber()

# From a file path:
msg = Message(audio_path="/path/to/meeting.mp3")
transcriber(msg)
print(msg.transcript[:300])
```

For testing without an audio file, you can seed `msg.transcript` directly and skip the transcription step.

---

## Step 2 — Analyzer Signature

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
    action_items: List[dict] = OutputField(
        desc=(
            "Action items with keys: 'owner' (person responsible), "
            "'task' (what needs to be done), 'deadline' (due date or None)"
        )
    )
    open_questions: List[str] = OutputField(
        desc="Questions that were raised but not answered — need follow-up"
    )
    sentiment: Literal["positive", "neutral", "tense"] = OutputField(
        desc="Overall tone of the meeting"
    )
    follow_up_meeting: bool = OutputField(
        desc="True if the group explicitly agreed to schedule a follow-up meeting"
    )
```

---

## Step 3 — Analyzer Agent

```python
class MeetingAnalyzer(nn.Agent):
    """Extracts structured meeting notes from a transcript."""
    model = model
    signature = MeetingAnalysis
    config = {"verbose": True}
```

---

## Step 4 — Full Pipeline Module

```python
class MeetingAssistant(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = MeetingTranscriber()
        self.analyzer    = MeetingAnalyzer()

    def forward(self, msg):
        # Skip transcription if transcript already provided
        if not msg.get("transcript"):
            self.transcriber(msg)

        self.analyzer(msg)
        return msg


assistant = MeetingAssistant()

msg = Message()
msg.audio_path = "/path/to/standup.mp3"
assistant(msg)

print("TL;DR:", msg.tldr)
print("\nDecisions:")
for d in msg.decisions:
    print(f"  • {d}")

print("\nAction Items:")
for item in msg.action_items:
    deadline = item.get("deadline") or "no deadline"
    print(f"  [{item['owner']}] {item['task']} — {deadline}")

print("\nOpen Questions:")
for q in msg.open_questions:
    print(f"  ? {q}")

print(f"\nSentiment: {msg.sentiment}")
print(f"Follow-up needed: {msg.follow_up_meeting}")
```

---

## Complete Example

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Signature, InputField, OutputField
from typing import Literal, List


# ── Models ────────────────────────────────────────────────────────────────────

stt   = mf.Model.speech_to_text("openai/whisper-1")
model = mf.Model.chat_completion("openai/gpt-4.1-mini")


# ── Signature ─────────────────────────────────────────────────────────────────

class MeetingAnalysis(Signature):
    """Extract structured notes from a meeting transcript."""

    transcript: str = InputField(desc="Full verbatim transcript of the meeting")

    tldr: str = OutputField(desc="One-sentence summary of what the meeting accomplished")
    decisions: List[str] = OutputField(
        desc="Decisions made and agreed upon during the meeting"
    )
    action_items: List[dict] = OutputField(
        desc=(
            "Action items: [{'owner': str, 'task': str, 'deadline': str | None}, ...]"
        )
    )
    open_questions: List[str] = OutputField(
        desc="Unanswered questions that need follow-up"
    )
    sentiment: Literal["positive", "neutral", "tense"] = OutputField(
        desc="Overall tone of the meeting"
    )
    follow_up_meeting: bool = OutputField(
        desc="True if a follow-up meeting was explicitly agreed upon"
    )


# ── Modules ───────────────────────────────────────────────────────────────────

class MeetingTranscriber(nn.Transcriber):
    """Transcribes meeting audio to text."""
    model = stt
    response_mode = "transcript"


class MeetingAnalyzer(nn.Agent):
    """Extracts structured meeting notes from a transcript."""
    model = model
    signature = MeetingAnalysis
    config = {"verbose": True}


class MeetingAssistant(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = MeetingTranscriber()
        self.analyzer    = MeetingAnalyzer()

    def forward(self, msg):
        if not msg.get("transcript"):
            self.transcriber(msg)
        self.analyzer(msg)
        return msg


# ── Run ───────────────────────────────────────────────────────────────────────

assistant = MeetingAssistant()

# ── Option A: from audio file ─────────────────────────────────────────────────
# msg = Message(audio_path="/path/to/meeting.mp3")

# ── Option B: from raw transcript (for testing) ───────────────────────────────
msg = Message()
msg.transcript = """
Sarah: Alright, let's get started. Main agenda: Q3 roadmap.
Tom: I think we should prioritize the API rate limiting feature. We've had three customer
     complaints this week alone.
Sarah: Agreed. That's decided then — API rate limiting goes to the top of Q3.
Tom: I'll write the technical spec by Friday.
Sarah: Great. What about the mobile app redesign?
Lisa: We still haven't decided on the design system. Flutter vs React Native.
Tom: Can we get a prototype from the design team first before deciding?
Sarah: Good point. Lisa, can you coordinate that?
Lisa: Sure, I'll reach out to design. Target date — end of next week?
Sarah: Perfect. So open question: design system decision is blocked on prototype.
Tom: Also, we need to align with the backend team on the new auth flow. That's not resolved.
Sarah: Let's schedule a follow-up with them next Tuesday. I'll send the invite.
Tom: Works for me.
Sarah: Great meeting, everyone. Productive session.
"""

assistant(msg)

print("=" * 60)
print("TL;DR:", msg.tldr)

print("\nDecisions:")
for d in msg.decisions:
    print(f"  ✓ {d}")

print("\nAction Items:")
for item in msg.action_items:
    deadline = item.get("deadline") or "TBD"
    print(f"  [{item['owner']}] {item['task']} → {deadline}")

print("\nOpen Questions:")
for q in msg.open_questions:
    print(f"  ? {q}")

print(f"\nSentiment:       {msg.sentiment}")
print(f"Follow-up needed: {msg.follow_up_meeting}")
```

Expected output:

```
TL;DR: Team aligned on Q3 priorities: API rate limiting goes first,
       mobile redesign blocked on design prototype.

Decisions:
  ✓ API rate limiting is the top Q3 priority
  ✓ Design prototype needed before choosing Flutter vs React Native
  ✓ Follow-up meeting with backend team on Tuesday

Action Items:
  [Tom]   Write technical spec for API rate limiting → Friday
  [Lisa]  Coordinate prototype with design team → end of next week
  [Sarah] Send calendar invite for backend alignment meeting → Tuesday

Open Questions:
  ? Design system choice (Flutter vs React Native) — blocked on prototype
  ? Auth flow alignment with backend team — unresolved

Sentiment:       positive
Follow-up needed: True
```

---

## Async Version

```python
import asyncio


class MeetingAssistant(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = MeetingTranscriber()
        self.analyzer    = MeetingAnalyzer()

    async def aforward(self, msg):
        if not msg.get("transcript"):
            await self.transcriber.acall(msg)
        await self.analyzer.acall(msg)
        return msg


async def main():
    assistant = MeetingAssistant()
    msg = Message(audio_path="/path/to/meeting.mp3")
    await assistant.acall(msg)
    print(msg.tldr)

asyncio.run(main())
```

---

## Extending the Pipeline

**Slack/email delivery**: add a module after `MeetingAnalyzer` that formats and posts the notes.

**Speaker diarization**: use `response_format="verbose_json"` on the transcriber to get per-speaker timestamps, then pass that as `transcript` to the analyzer.

**Calendar integration**: if `msg.follow_up_meeting` is `True`, a tool-equipped agent can create the calendar event automatically.
