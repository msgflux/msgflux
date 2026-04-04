# Meeting Action Items Tracker

<span class="tag tag-blue">Intermediate</span>

---

## The Problem

Meetings generate commitments that never become tasks. Someone says "let's look into that", someone else says "I'll handle it" — and three days later, no one remembers who was supposed to do what. The recording exists. The transcript could be parsed. But without a quality bar, everything sounds like an action item.

The harder problem is noise. A model without guidance treats "we should probably revisit this at some point" the same as "Ana will send the contract by Friday." Both are future-oriented statements. Only one is a real action item.

---

## The Plan

We will build a pipeline that takes a meeting transcript — or an audio recording — and produces a structured checklist of action items, each with an assignee and a deadline when one was stated.

The transcript is the primary input; if audio is provided, it is transcribed first and the rest of the pipeline runs identically. An extractor reads the transcript and identifies action items. Few-shot examples teach it the distinction that matters: a committed action ("Pedro will send the proposal by Thursday") versus a vague intention ("we'll figure that out later") versus an implicit assignment ("so that's on Ana's side"). The extractor captures all three types differently — confirmed, vague, and implicit — so the output reflects the actual level of commitment.

A formatter agent turns the extracted items into a clean checklist, grouping by assignee and flagging items with no deadline or no owner.

---

## Architecture

```
Meeting input (text or audio)
        │
        ├── audio? → Transcriber (Whisper) → transcript
        │
        ▼
   Extractor ─── mf.Example × N (labeled transcript excerpts)
        │
        │  action_items: [{assignee, task, deadline, confidence}]
        ▼
   Formatter
        │
        ▼
   msg.checklist  (markdown checklist grouped by assignee)
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
model     = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model = mf.Model.speech_to_text("openai/whisper-1")
```

---

## Step 2 — Signatures

`ExtractItems` uses five parallel lists — one per attribute — so each field has a single,
well-defined type. This avoids the need for nested dicts and keeps every type expressible in the
OpenAI structured output schema. The lists are correlated by index: `tasks[i]` belongs to
`assignees[i]`, `deadlines[i]`, and so on.

```python
class ExtractItems(Signature):
    """Extract action items from a meeting transcript."""

    transcript: str = InputField(
        desc="Full meeting transcript"
    )

    tasks:            List[str]   = OutputField(desc="Task description for each item")
    assignees:        List[str]   = OutputField(desc="Assignee name per item, or empty string if unknown")
    deadlines:        List[str]   = OutputField(desc="Deadline per item, or empty string if none")
    confidences:      List[float] = OutputField(desc="Confidence score 0.0-1.0 per item")
    commitment_types: List[Literal["confirmed", "vague", "implicit"]] = OutputField(desc="Commitment type per item")
```

`FormatChecklist` receives the same five parallel lists and produces a human-readable checklist
grouped by assignee.

```python
class FormatChecklist(Signature):
    """Format extracted action items into a markdown checklist grouped by assignee."""

    tasks:            List[str]   = InputField(desc="Task descriptions")
    assignees:        List[str]   = InputField(desc="Assignees (empty string if unassigned)")
    deadlines:        List[str]   = InputField(desc="Deadlines (empty string if none)")
    confidences:      List[float] = InputField(desc="Confidence scores")
    commitment_types: List[Literal["confirmed", "vague", "implicit"]] = InputField(desc="Commitment type per item")

    checklist: str = OutputField(desc="Markdown checklist grouped by assignee")
    summary: str = OutputField(
        desc="One sentence: total items, how many confirmed vs. vague/implicit, assignees involved."
    )
```

---

## Step 3 — Few-shot Examples

Four labeled examples teach the extractor the distinction that matters. The third example is the
critical one: an implicit agreement extracted from a manager's question and a one-word confirmation.
Without this reference, the model often treats it as a clarification rather than an assignment.
The fourth example — a calendar placeholder — shows what should produce zero items.

```python
examples = [
    mf.Example(
        inputs="Pedro: I'll send the updated proposal to the client by Thursday.",
        labels={
            "tasks":            ["Send updated proposal to client"],
            "assignees":        ["Pedro"],
            "deadlines":        ["Thursday"],
            "confidences":      [0.98],
            "commitment_types": ["confirmed"],
        },
        title="Named assignment with deadline",
    ),
    mf.Example(
        inputs="Ana: we should probably revisit the pricing model at some point.",
        labels={
            "tasks":            ["Revisit pricing model"],
            "assignees":        [""],
            "deadlines":        [""],
            "confidences":      [0.45],
            "commitment_types": ["vague"],
        },
        title="Vague intention — no owner, no deadline",
    ),
    mf.Example(
        inputs="Manager: so the onboarding flow is on your side, right Lucas?  Lucas: yeah.",
        labels={
            "tasks":            ["Handle the onboarding flow"],
            "assignees":        ["Lucas"],
            "deadlines":        [""],
            "confidences":      [0.75],
            "commitment_types": ["implicit"],
        },
        title="Implicit assignment — one-word confirmation",
    ),
    mf.Example(
        inputs="Let's circle back on the infrastructure cost review next quarter.",
        labels={
            "tasks":            [],
            "assignees":        [],
            "deadlines":        [],
            "confidences":      [],
            "commitment_types": [],
        },
        title="No action item — calendar placeholder only",
    ),
]
```

---

## Step 4 — Extractor

`Extractor` is bound to `ExtractItems` and carries the four labeled examples. The examples are
injected into the system prompt before the transcript — the model sees the pattern before it
processes any real input.

```python
class Extractor(nn.Agent):
    """Reads the transcript and identifies action items with commitment type."""
    model = model
    signature = ExtractItems
    examples = examples
    config = {"verbose": True}
```

---

## Step 5 — Formatter

`Formatter` receives only the structured list — not the raw transcript. Keeping the two agents
separate means you can swap the formatting logic (output format, language, grouping strategy)
without touching extraction.

```python
class Formatter(nn.Agent):
    """Turns raw action items into a checklist grouped by assignee."""
    model = model
    signature = FormatChecklist
```

---

## Step 6 — Transcriber

`AudioTranscriber` reads `audio_content` from the message and writes the transcription to
`meeting.transcript`. The rest of the pipeline reads from that field, so text and audio inputs
flow through identical downstream logic.

```python
class AudioTranscriber(nn.Transcriber):
    """Transcribes meeting audio into msg.meeting.transcript."""
    model = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode = "meeting.transcript"
```

---

## Step 7 — MeetingTracker

`MeetingTracker` is the entry point. It accepts either a plain transcript string or raw audio
bytes — never both. When audio is provided, it transcribes first and the rest of the pipeline
runs identically.

```python
class MeetingTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = AudioTranscriber()
        self.extractor   = Extractor()
        self.formatter   = Formatter()

    def _zip_items(self, extracted: dict) -> list:
        return [
            {
                "task":            t,
                "assignee":        a,
                "deadline":        d,
                "confidence":      c,
                "commitment_type": ct,
            }
            for t, a, d, c, ct in zip(
                extracted["tasks"],
                extracted["assignees"],
                extracted["deadlines"],
                extracted["confidences"],
                extracted["commitment_types"],
            )
        ]

    def forward(
        self,
        transcript: str | None = None,
        audio: bytes | None = None,
    ) -> dict:
        if audio:
            msg = mf.Message()
            msg.audio_content = audio
            self.transcriber(msg)
            transcript = msg.meeting.transcript

        extracted    = self.extractor(transcript=transcript)
        action_items = self._zip_items(extracted)
        formatted    = self.formatter(
            tasks=extracted["tasks"],
            assignees=extracted["assignees"],
            deadlines=extracted["deadlines"],
            confidences=extracted["confidences"],
            commitment_types=extracted["commitment_types"],
        )

        return {
            "transcript":   transcript,
            "action_items": action_items,
            "checklist":    formatted["checklist"],
            "summary":      formatted["summary"],
        }

    async def aforward(
        self,
        transcript: str | None = None,
        audio: bytes | None = None,
    ) -> dict:
        if audio:
            msg = mf.Message()
            msg.audio_content = audio
            await self.transcriber.acall(msg)
            transcript = msg.meeting.transcript

        extracted    = await self.extractor.acall(transcript=transcript)
        action_items = self._zip_items(extracted)
        formatted    = await self.formatter.acall(
            tasks=extracted["tasks"],
            assignees=extracted["assignees"],
            deadlines=extracted["deadlines"],
            confidences=extracted["confidences"],
            commitment_types=extracted["commitment_types"],
        )

        return {
            "transcript":   transcript,
            "action_items": action_items,
            "checklist":    formatted["checklist"],
            "summary":      formatted["summary"],
        }


tracker = MeetingTracker()
```

---

## Examples

???+ example

    === "Text transcript"

        ```python
        tracker = MeetingTracker()

        transcript = """
        Sarah: Alright, so the biggest blocker right now is the API rate limit issue.
        Tom: I'll open a ticket with the provider today and follow up by Wednesday.
        Sarah: Perfect. Also, we need to update the onboarding docs before the launch.
        Tom: That should be on the content team's side.
        Sarah: Right, Maria — can you handle that?
        Maria: Yeah, I'll have a draft ready by end of next week.
        Sarah: Great. And we should probably think about adding retry logic at some point.
        Tom: Agreed. No rush though.
        Sarah: One more thing — the security audit report. Pedro said he'd share it today, right?
        Tom: He confirmed it this morning, should arrive before EOD.
        """

        result = tracker.forward(transcript=transcript)

        print(result["summary"])
        print()
        print(result["checklist"])
        ```

        ```
        5 action items: 3 confirmed, 1 implicit, 1 vague — assignees: Tom, Maria, Pedro.

        ## Tom
        - [ ] Open ticket with API provider and follow up *(deadline: Wednesday)*
        ## Maria
        - [ ] Draft onboarding docs update *(deadline: end of next week)*
        ## Pedro
        - [ ] Share security audit report *(deadline: today)*
        ## Unassigned
        - [ ] Add retry logic (vague) ⚠ no deadline
        ```

    === "Audio recording"

        ```python
        tracker = MeetingTracker()

        result = tracker.forward(audio=open("meeting.mp3", "rb").read())

        print(result["summary"])
        print(result["checklist"])
        ```

    === "Filtering by commitment type"

        Show only confirmed items:

        ```python
        result = tracker.forward(transcript=transcript)

        confirmed = [
            item for item in result["action_items"]
            if item["commitment_type"] == "confirmed"
        ]

        for item in confirmed:
            owner    = item["assignee"] or "Unassigned"
            deadline = item["deadline"] or "no deadline"
            print(f"[{owner}] {item['task']} — {deadline}")
        ```

    === "Async"

        ```python
        import asyncio

        async def main():
            tracker = MeetingTracker()
            result  = await tracker.acall(transcript=transcript)
            print(result["checklist"])

        asyncio.run(main())
        ```

---

## Extending

### Raising the confidence threshold

Filter out low-confidence items before passing them to the formatter. Items below the threshold
are still returned in `action_items` so callers can inspect them — they just do not appear in
the checklist:

```python
MIN_CONFIDENCE = 0.6

class MeetingTracker(nn.Module):
    ...
    def forward(self, transcript: str | None = None, audio: bytes | None = None) -> dict:
        ...
        extracted   = self.extractor(transcript=transcript)
        all_items   = extracted["action_items"]
        filtered    = [i for i in all_items if i.get("confidence", 1.0) >= MIN_CONFIDENCE]
        formatted   = self.formatter(action_items=filtered)

        return {
            "transcript":        transcript,
            "action_items":      all_items,
            "action_items_used": filtered,
            "checklist":         formatted["checklist"],
            "summary":           formatted["summary"],
        }
```

### Sending the checklist to Slack

Add a plain function after the pipeline returns:

```python
import httpx

def post_to_slack(checklist: str, webhook_url: str) -> None:
    httpx.post(webhook_url, json={"text": checklist})

result = tracker.forward(transcript=transcript)
post_to_slack(result["checklist"], webhook_url="https://hooks.slack.com/...")
```

### Processing multiple recordings concurrently

Use `F.amap_gather` to run the full pipeline on several audio files at the same time:

```python
import asyncio
import msgflux.nn.functional as F

async def main():
    tracker  = MeetingTracker()
    audios   = [open(f, "rb").read() for f in ["week1.mp3", "week2.mp3", "week3.mp3"]]
    results  = await F.amap_gather(
        tracker.acall,
        kwargs_list=[{"audio": a} for a in audios],
    )
    for i, r in enumerate(results, 1):
        print(f"Meeting {i}: {r['summary']}")

asyncio.run(main())
```

---

## Complete Script

```python
import asyncio
import msgflux as mf
import msgflux.nn as nn
from msgflux import Signature, InputField, OutputField
from typing import List, Literal

mf.load_dotenv()
model     = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model = mf.Model.speech_to_text("openai/whisper-1")


# ── Signatures ────────────────────────────────────────────────────────────────

class ExtractItems(Signature):
    """Extract action items from a meeting transcript."""
    transcript:       str         = InputField(desc="Full meeting transcript")
    tasks:            List[str]   = OutputField(desc="Task description for each item")
    assignees:        List[str]   = OutputField(desc="Assignee name per item, or empty string if unknown")
    deadlines:        List[str]   = OutputField(desc="Deadline per item, or empty string if none")
    confidences:      List[float] = OutputField(desc="Confidence score 0.0-1.0 per item")
    commitment_types: List[Literal["confirmed", "vague", "implicit"]] = OutputField(desc="Commitment type per item")


class FormatChecklist(Signature):
    """Format extracted action items into a markdown checklist grouped by assignee."""
    tasks:            List[str]   = InputField(desc="Task descriptions")
    assignees:        List[str]   = InputField(desc="Assignees (empty string if unassigned)")
    deadlines:        List[str]   = InputField(desc="Deadlines (empty string if none)")
    confidences:      List[float] = InputField(desc="Confidence scores")
    commitment_types: List[Literal["confirmed", "vague", "implicit"]] = InputField(desc="Commitment type per item")
    checklist: str = OutputField(desc="Markdown checklist grouped by assignee")
    summary: str = OutputField(
        desc="One sentence: total items, how many confirmed vs. vague/implicit, assignees involved."
    )


# ── Few-shot examples ─────────────────────────────────────────────────────────

examples = [
    mf.Example(
        inputs="Pedro: I'll send the updated proposal to the client by Thursday.",
        labels={
            "tasks":            ["Send updated proposal to client"],
            "assignees":        ["Pedro"],
            "deadlines":        ["Thursday"],
            "confidences":      [0.98],
            "commitment_types": ["confirmed"],
        },
        title="Named assignment with deadline",
    ),
    mf.Example(
        inputs="Ana: we should probably revisit the pricing model at some point.",
        labels={
            "tasks":            ["Revisit pricing model"],
            "assignees":        [""],
            "deadlines":        [""],
            "confidences":      [0.45],
            "commitment_types": ["vague"],
        },
        title="Vague intention — no owner, no deadline",
    ),
    mf.Example(
        inputs="Manager: so the onboarding flow is on your side, right Lucas?  Lucas: yeah.",
        labels={
            "tasks":            ["Handle the onboarding flow"],
            "assignees":        ["Lucas"],
            "deadlines":        [""],
            "confidences":      [0.75],
            "commitment_types": ["implicit"],
        },
        title="Implicit assignment — one-word confirmation",
    ),
    mf.Example(
        inputs="Let's circle back on the infrastructure cost review next quarter.",
        labels={
            "tasks":            [],
            "assignees":        [],
            "deadlines":        [],
            "confidences":      [],
            "commitment_types": [],
        },
        title="No action item — calendar placeholder only",
    ),
]


# ── Agents ────────────────────────────────────────────────────────────────────

class Extractor(nn.Agent):
    """Reads the transcript and identifies action items with commitment type."""
    model     = model
    signature = ExtractItems
    examples  = examples
    config    = {"verbose": True}


class Formatter(nn.Agent):
    """Turns raw action items into a checklist grouped by assignee."""
    model     = model
    signature = FormatChecklist


class AudioTranscriber(nn.Transcriber):
    """Transcribes meeting audio into msg.meeting.transcript."""
    model          = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode  = "meeting.transcript"


# ── Pipeline ──────────────────────────────────────────────────────────────────

class MeetingTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = AudioTranscriber()
        self.extractor   = Extractor()
        self.formatter   = Formatter()

    def _zip_items(self, extracted: dict) -> list:
        return [
            {
                "task":            t,
                "assignee":        a,
                "deadline":        d,
                "confidence":      c,
                "commitment_type": ct,
            }
            for t, a, d, c, ct in zip(
                extracted["tasks"],
                extracted["assignees"],
                extracted["deadlines"],
                extracted["confidences"],
                extracted["commitment_types"],
            )
        ]

    def forward(
        self,
        transcript: str | None = None,
        audio: bytes | None = None,
    ) -> dict:
        if audio:
            msg = mf.Message()
            msg.audio_content = audio
            self.transcriber(msg)
            transcript = msg.meeting.transcript

        extracted    = self.extractor(transcript=transcript)
        action_items = self._zip_items(extracted)
        formatted    = self.formatter(
            tasks=extracted["tasks"],
            assignees=extracted["assignees"],
            deadlines=extracted["deadlines"],
            confidences=extracted["confidences"],
            commitment_types=extracted["commitment_types"],
        )

        return {
            "transcript":   transcript,
            "action_items": action_items,
            "checklist":    formatted["checklist"],
            "summary":      formatted["summary"],
        }

    async def aforward(
        self,
        transcript: str | None = None,
        audio: bytes | None = None,
    ) -> dict:
        if audio:
            msg = mf.Message()
            msg.audio_content = audio
            await self.transcriber.acall(msg)
            transcript = msg.meeting.transcript

        extracted    = await self.extractor.acall(transcript=transcript)
        action_items = self._zip_items(extracted)
        formatted    = await self.formatter.acall(
            tasks=extracted["tasks"],
            assignees=extracted["assignees"],
            deadlines=extracted["deadlines"],
            confidences=extracted["confidences"],
            commitment_types=extracted["commitment_types"],
        )

        return {
            "transcript":   transcript,
            "action_items": action_items,
            "checklist":    formatted["checklist"],
            "summary":      formatted["summary"],
        }


tracker = MeetingTracker()

transcript = """
Sarah: Alright, so the biggest blocker right now is the API rate limit issue.
Tom: I'll open a ticket with the provider today and follow up by Wednesday.
Sarah: Perfect. Also, we need to update the onboarding docs before the launch.
Tom: That should be on the content team's side.
Sarah: Right, Maria — can you handle that?
Maria: Yeah, I'll have a draft ready by end of next week.
Sarah: Great. And we should probably think about adding retry logic at some point.
Tom: Agreed. No rush though.
Sarah: One more thing — the security audit report. Pedro said he'd share it today, right?
Tom: He confirmed it this morning, should arrive before EOD.
"""

result = tracker.forward(transcript=transcript)
print(result["summary"])
print()
print(result["checklist"])
```

---

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — signatures, few-shot examples, and structured output
- [nn.Transcriber](../learn/nn/transcriber.md) — speech-to-text integration
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts
- [Few-shot Examples](../learn/nn/agent/system-prompt.md) — `mf.Example` and prompt construction
