# Deal Briefing Generator

<span class="tag tag-purple">Intermediate</span><span class="tag tag-gray">Signature</span><span class="tag tag-gray">Few-shot</span><span class="tag tag-gray">Multimodal</span>

Sales reps finish a call and move on to the next one. Context fades fast — the specific objection raised, the budget number mentioned in passing, the follow-up that was implicitly agreed to. Voice recordings exist but no one has time to re-listen before the next meeting. The rep walks in underprepared, asks questions already answered in the previous call, and loses credibility.

The information is there. It just never gets turned into anything actionable.

---

## The Problem

The typical post-call workflow looks like this.

```
Call recording
       │
       ▼
│             SalesRep                     │
│                                          │
│   re-listen  ←──→  take notes manually  │
       │ scattered notes
       ▼
  next meeting (underprepared)
```

- Re-listening takes as long as the call itself. No one does it.
- Notes taken during the call miss what was said while writing.
- Pain points, objections, and budget signals end up buried or forgotten.
- The agreed next step gets lost — or is remembered differently by each side.

You are one missed signal away from a lost deal.

---

## The Plan

We will build a pipeline that takes a sales call recording and produces a structured briefing the rep can read — or listen to — before the next meeting.

The call is transcribed first. From there, an extractor identifies the key signals a rep actually needs: what pain points the prospect raised, what objections came up, what was agreed as a next step, and whether a date or budget was mentioned. Few-shot examples anchor the extraction to a consistent format, showing the model what counts as a real pain point versus a vague comment, and what a committed next step looks like versus a vague "let's reconnect."

A drafting agent turns those signals into a concise briefing with clearly labeled sections. The same briefing is then narrated via text-to-speech so the rep can listen on the way to the next meeting instead of reading.

---

## Architecture

```
Call recording (audio)
        │
        ▼
   CallTranscriber (Whisper)
        │ transcript text
        ▼
   Extractor ─── mf.Example × 4 (labeled call excerpts)
        │
        │  pain_points, objections, next_step, budget, timeline
        ▼
   Drafter
        │ briefing (markdown)
        ▼                      ▼
   briefing text          Narrator (TTS)
                               │
                               ▼
                          briefing.mp3
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
from typing import List

mf.load_dotenv()

chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
stt_model  = mf.Model.speech_to_text("openai/whisper-1")
tts_model  = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
```

---

## Step 2 — Signatures

`ExtractSignals` pulls the five signals a rep needs from the raw transcript. Empty string is the contract for "not mentioned" — the drafter handles gaps without hallucinating information.

```python
class ExtractSignals(Signature):
    """Extract sales signals from a call transcript."""

    transcript: str = InputField(
        desc="Full sales call transcript with speaker labels"
    )

    prospect_name: str = OutputField(
        desc="Name of the prospect or their company if mentioned, empty string if not"
    )
    pain_points: List[str] = OutputField(
        desc=(
            "Specific, concrete problems the prospect expressed — measurable issues, "
            "not vague comments like 'it's a bit slow'"
        )
    )
    objections: List[str] = OutputField(
        desc="Explicit concerns, hesitations, or blockers the prospect raised about moving forward"
    )
    next_step: str = OutputField(
        desc=(
            "The specific action committed at the end of the call with a clear owner. "
            "Empty string if no concrete step was agreed — do not convert vague intentions into next steps"
        )
    )
    budget: str = OutputField(
        desc="Budget or pricing information mentioned, empty string if not discussed"
    )
    timeline: str = OutputField(
        desc="Timeline, deadline, or urgency signals mentioned, empty string if not"
    )
```

`DraftBriefing` receives the structured signals and produces a markdown briefing ready for the rep to read or send.

```python
class DraftBriefing(Signature):
    """Draft a pre-meeting briefing from extracted sales signals."""

    prospect_name: str       = InputField(desc="Prospect or company name")
    pain_points:   List[str] = InputField(desc="Confirmed pain points from the call")
    objections:    List[str] = InputField(desc="Objections raised by the prospect")
    next_step:     str       = InputField(desc="Agreed next step, empty if none")
    budget:        str       = InputField(desc="Budget signals, empty if not mentioned")
    timeline:      str       = InputField(desc="Timeline signals, empty if not mentioned")

    briefing: str = OutputField(
        desc=(
            "Concise markdown briefing with the following sections: "
            "## Context, ## Pain Points, ## Objections to Address, "
            "## Agreed Next Step, ## Talking Points. "
            "Skip a section entirely if there is no data for it. "
            "Keep the total under 300 words — this is a quick reference, not a report."
        )
    )
```

---

## Step 3 — Few-shot Examples

Four labeled excerpts anchor the extractor. The critical distinctions: a measurable pain point vs. a vague comment, and a committed next step vs. a loose intention.

```python
examples = [
    mf.Example(
        inputs=(
            "Rep: So what's the biggest challenge right now?\n"
            "Prospect: Honestly, the monthly reporting. Our team spends three full days "
            "pulling data from four different systems. It's completely manual.\n"
            "Rep: Got it. And timeline-wise, is there pressure to fix this?\n"
            "Prospect: We're presenting to the board in Q2, so ideally something is in place by then.\n"
            "Rep: Makes sense. I'll send over a proposal by end of week — does that work?\n"
            "Prospect: Perfect."
        ),
        labels={
            "prospect_name": "",
            "pain_points":   ["Monthly reporting takes 3 days across 4 systems — fully manual"],
            "objections":    [],
            "next_step":     "Rep sends proposal by end of week",
            "budget":        "",
            "timeline":      "Q2 (board presentation deadline)",
        },
        title="Concrete pain point + committed next step",
    ),
    mf.Example(
        inputs=(
            "Prospect: We're paying around $60k a year for our current platform. "
            "If you can come in under that we'd definitely consider switching.\n"
            "Rep: Understood. What's the main reason you're looking?\n"
            "Prospect: It's fine, honestly. Just exploring options.\n"
            "Rep: Let's stay in touch then.\n"
            "Prospect: Sure, reach out in a few months."
        ),
        labels={
            "prospect_name": "",
            "pain_points":   [],
            "objections":    [],
            "next_step":     "",
            "budget":        "Current spend ~$60k/year — open to switching if lower",
            "timeline":      "A few months",
        },
        title="Budget signal + vague intention (no committed next step)",
    ),
    mf.Example(
        inputs=(
            "Prospect: It's a bit slow sometimes, I guess.\n"
            "Rep: In what way?\n"
            "Prospect: Just, you know, here and there. Nothing major.\n"
            "Rep: Okay. And what about the integration side — any concerns?\n"
            "Prospect: We'd need to make sure it works with Salesforce. "
            "Our IT team would have to sign off before anything moves forward."
        ),
        labels={
            "prospect_name": "",
            "pain_points":   [],
            "objections":    [
                "Vague performance concern — no specifics given",
                "Salesforce integration required — IT sign-off needed before any decision",
            ],
            "next_step":     "",
            "budget":        "",
            "timeline":      "",
        },
        title="Vague complaint (not a pain point) + gating objection",
    ),
    mf.Example(
        inputs=(
            "Rep: Great talking with you, Ana from Nexus Solutions. "
            "So to confirm — you'll share our pricing deck with your CFO this week, "
            "and we reconnect Thursday at 3pm?\n"
            "Prospect: Exactly. And if the numbers work, we could potentially start onboarding in January."
        ),
        labels={
            "prospect_name": "Ana / Nexus Solutions",
            "pain_points":   [],
            "objections":    [],
            "next_step":     "Ana shares pricing deck with CFO this week — follow-up call Thursday 3pm",
            "budget":        "",
            "timeline":      "Potential onboarding start: January",
        },
        title="Named prospect + double-confirmed next step + timeline",
    ),
]
```

---

## Step 4 — Agents

```python
class CallTranscriber(nn.Transcriber):
    """Transcribes call audio into msg.call.transcript."""
    model          = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode  = "call.transcript"


class Extractor(nn.Agent):
    """Extracts pain points, objections, next step, budget and timeline from a call transcript."""
    model     = chat_model
    signature = ExtractSignals
    examples  = examples
    config    = {"verbose": True}


class Drafter(nn.Agent):
    """Drafts a concise pre-meeting briefing from extracted sales signals."""
    model     = chat_model
    signature = DraftBriefing


class Narrator(nn.Speaker):
    """Narrates the briefing as an audio file the rep can listen to."""
    model           = tts_model
    response_format = "mp3"
    config          = {"voice": "nova"}
```

---

## Step 5 — Pipeline

`DealBriefingGenerator` orchestrates the four steps. `narrate=True` is the default — set it to `False` when the audio file is not needed (e.g. in batch processing or tests).

```python
class DealBriefingGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = CallTranscriber()
        self.extractor   = Extractor()
        self.drafter     = Drafter()
        self.narrator    = Narrator()

    def forward(self, audio: bytes, narrate: bool = True) -> dict:
        msg = mf.Message()
        msg.audio_content = audio
        self.transcriber(msg)
        transcript = msg.call.transcript

        signals  = self.extractor(transcript=transcript)
        briefing_result = self.drafter(
            prospect_name=signals.get("prospect_name", ""),
            pain_points=signals.get("pain_points", []),
            objections=signals.get("objections", []),
            next_step=signals.get("next_step", ""),
            budget=signals.get("budget", ""),
            timeline=signals.get("timeline", ""),
        )
        briefing = briefing_result["briefing"]

        result = {
            "transcript": transcript,
            "signals":    signals,
            "briefing":   briefing,
            "audio_path": None,
        }

        if narrate:
            result["audio_path"] = self.narrator(briefing)

        return result

    async def aforward(self, audio: bytes, narrate: bool = True) -> dict:
        msg = mf.Message()
        msg.audio_content = audio
        await self.transcriber.acall(msg)
        transcript = msg.call.transcript

        signals  = await self.extractor.acall(transcript=transcript)
        briefing_result = await self.drafter.acall(
            prospect_name=signals.get("prospect_name", ""),
            pain_points=signals.get("pain_points", []),
            objections=signals.get("objections", []),
            next_step=signals.get("next_step", ""),
            budget=signals.get("budget", ""),
            timeline=signals.get("timeline", ""),
        )
        briefing = briefing_result["briefing"]

        result = {
            "transcript": transcript,
            "signals":    signals,
            "briefing":   briefing,
            "audio_path": None,
        }

        if narrate:
            result["audio_path"] = await self.narrator.acall(briefing)

        return result


generator = DealBriefingGenerator()
```

---

## Examples

???+ example

    === "From audio file"

        ```python
        generator = DealBriefingGenerator()

        result = generator(audio=open("call.mp3", "rb").read())

        print(result["briefing"])
        print("Audio saved to:", result["audio_path"])
        ```

    === "Text-only (no narration)"

        ```python
        generator = DealBriefingGenerator()

        result = generator(audio=open("call.mp3", "rb").read(), narrate=False)

        print(result["briefing"])

        # Inspect extracted signals directly
        signals = result["signals"]
        print("Next step:", signals["next_step"])
        print("Pain points:", signals["pain_points"])
        ```

    === "Batch — process a call queue"

        ```python
        import asyncio
        import msgflux.nn.functional as F

        async def main():
            generator = DealBriefingGenerator()

            recordings = [
                open("call_001.mp3", "rb").read(),
                open("call_002.mp3", "rb").read(),
                open("call_003.mp3", "rb").read(),
            ]

            results = await F.amap_gather(
                generator,
                kwargs_list=[{"audio": a, "narrate": False} for a in recordings],
            )

            for i, r in enumerate(results, 1):
                step = r["signals"].get("next_step") or "no next step recorded"
                print(f"Call {i}: {step}")

        asyncio.run(main())
        ```

    === "Async single call"

        ```python
        import asyncio

        async def main():
            generator = DealBriefingGenerator()
            result    = await generator.acall(audio=open("call.mp3", "rb").read())
            print(result["briefing"])

        asyncio.run(main())
        ```

---

## Extending

### Routing by deal stage

Use the extracted signals to route the briefing to different workflows depending on where the prospect is in the funnel:

```python
signals = result["signals"]

if signals["next_step"] and signals["budget"]:
    send_to_crm(result["briefing"], stage="proposal")
elif signals["pain_points"] and not signals["next_step"]:
    send_to_crm(result["briefing"], stage="discovery")
else:
    send_to_crm(result["briefing"], stage="nurture")
```

### Pushing the briefing to Slack

Deliver the briefing to the rep's channel immediately after the call ends:

```python
import httpx

def post_to_slack(briefing: str, webhook_url: str) -> None:
    httpx.post(webhook_url, json={"text": briefing})

result = generator(audio=audio_bytes)
post_to_slack(result["briefing"], webhook_url="https://hooks.slack.com/...")
```

### Adding a competitor mention detector

Extend `ExtractSignals` with a new output field — the extraction step already reads the full transcript, so no extra model call is needed:

```python
competitor_mentions: List[str] = OutputField(
    desc="Competitor names or products mentioned by the prospect, empty list if none"
)
```

---

## Complete Script

??? example "Expand full script"
    ```python
    # /// script
    # dependencies = []
    # ///

    from typing import List

    import msgflux as mf
    import msgflux.nn as nn
    import msgflux.nn.functional as F
    from msgflux import Signature, InputField, OutputField

    mf.load_dotenv()


    # Models
    chat_model = mf.Model.chat_completion("openai/gpt-4.1-mini")
    stt_model  = mf.Model.speech_to_text("openai/whisper-1")
    tts_model  = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")


    # Signatures
    class ExtractSignals(Signature):
        """Extract sales signals from a call transcript."""
        transcript:    str       = InputField(desc="Full sales call transcript with speaker labels")
        prospect_name: str       = OutputField(desc="Prospect or company name, empty string if not mentioned")
        pain_points:   List[str] = OutputField(desc="Specific, concrete problems the prospect expressed — not vague comments")
        objections:    List[str] = OutputField(desc="Explicit concerns or blockers the prospect raised")
        next_step:     str       = OutputField(desc="Specific action committed at the end of the call, empty string if none")
        budget:        str       = OutputField(desc="Budget or pricing signals, empty string if not discussed")
        timeline:      str       = OutputField(desc="Timeline or urgency signals, empty string if not mentioned")


    class DraftBriefing(Signature):
        """Draft a pre-meeting briefing from extracted sales signals."""
        prospect_name: str       = InputField(desc="Prospect or company name")
        pain_points:   List[str] = InputField(desc="Confirmed pain points")
        objections:    List[str] = InputField(desc="Objections raised")
        next_step:     str       = InputField(desc="Agreed next step, empty if none")
        budget:        str       = InputField(desc="Budget signals, empty if not mentioned")
        timeline:      str       = InputField(desc="Timeline signals, empty if not mentioned")
        briefing:      str       = OutputField(
            desc=(
                "Concise markdown briefing: ## Context, ## Pain Points, ## Objections to Address, "
                "## Agreed Next Step, ## Talking Points. Skip empty sections. Under 300 words."
            )
        )


    # Few-shot examples
    examples = [
        mf.Example(
            inputs=(
                "Rep: So what's the biggest challenge right now?\n"
                "Prospect: Honestly, the monthly reporting. Our team spends three full days "
                "pulling data from four different systems. It's completely manual.\n"
                "Rep: Got it. And timeline-wise, is there pressure to fix this?\n"
                "Prospect: We're presenting to the board in Q2, so ideally something is in place by then.\n"
                "Rep: Makes sense. I'll send over a proposal by end of week — does that work?\n"
                "Prospect: Perfect."
            ),
            labels={
                "prospect_name": "",
                "pain_points":   ["Monthly reporting takes 3 days across 4 systems — fully manual"],
                "objections":    [],
                "next_step":     "Rep sends proposal by end of week",
                "budget":        "",
                "timeline":      "Q2 (board presentation deadline)",
            },
            title="Concrete pain point + committed next step",
        ),
        mf.Example(
            inputs=(
                "Prospect: We're paying around $60k a year for our current platform. "
                "If you can come in under that we'd definitely consider switching.\n"
                "Rep: Understood. What's the main reason you're looking?\n"
                "Prospect: It's fine, honestly. Just exploring options.\n"
                "Rep: Let's stay in touch then.\n"
                "Prospect: Sure, reach out in a few months."
            ),
            labels={
                "prospect_name": "",
                "pain_points":   [],
                "objections":    [],
                "next_step":     "",
                "budget":        "Current spend ~$60k/year — open to switching if lower",
                "timeline":      "A few months",
            },
            title="Budget signal + vague intention (no committed next step)",
        ),
        mf.Example(
            inputs=(
                "Prospect: It's a bit slow sometimes, I guess.\n"
                "Rep: In what way?\n"
                "Prospect: Just, you know, here and there. Nothing major.\n"
                "Rep: Okay. And what about the integration side — any concerns?\n"
                "Prospect: We'd need to make sure it works with Salesforce. "
                "Our IT team would have to sign off before anything moves forward."
            ),
            labels={
                "prospect_name": "",
                "pain_points":   [],
                "objections":    [
                    "Vague performance concern — no specifics given",
                    "Salesforce integration required — IT sign-off needed before any decision",
                ],
                "next_step":     "",
                "budget":        "",
                "timeline":      "",
            },
            title="Vague complaint (not a pain point) + gating objection",
        ),
        mf.Example(
            inputs=(
                "Rep: Great talking with you, Ana from Nexus Solutions. "
                "So to confirm — you'll share our pricing deck with your CFO this week, "
                "and we reconnect Thursday at 3pm?\n"
                "Prospect: Exactly. And if the numbers work, we could potentially start onboarding in January."
            ),
            labels={
                "prospect_name": "Ana / Nexus Solutions",
                "pain_points":   [],
                "objections":    [],
                "next_step":     "Ana shares pricing deck with CFO this week — follow-up call Thursday 3pm",
                "budget":        "",
                "timeline":      "Potential onboarding start: January",
            },
            title="Named prospect + double-confirmed next step + timeline",
        ),
    ]


    # Agents
    class CallTranscriber(nn.Transcriber):
        """Transcribes call audio into msg.call.transcript."""
        model          = stt_model
        message_fields = {"task_multimodal": {"audio": "audio_content"}}
        response_mode  = "call.transcript"


    class Extractor(nn.Agent):
        """Extracts pain points, objections, next step, budget and timeline from a call transcript."""
        model     = chat_model
        signature = ExtractSignals
        examples  = examples
        config    = {"verbose": True}


    class Drafter(nn.Agent):
        """Drafts a concise pre-meeting briefing from extracted sales signals."""
        model     = chat_model
        signature = DraftBriefing


    class Narrator(nn.Speaker):
        """Narrates the briefing as an audio file the rep can listen to."""
        model           = tts_model
        response_format = "mp3"
        config          = {"voice": "nova"}


    # Pipeline
    class DealBriefingGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.transcriber = CallTranscriber()
            self.extractor   = Extractor()
            self.drafter     = Drafter()
            self.narrator    = Narrator()

        def forward(self, audio: bytes, narrate: bool = True) -> dict:
            msg = mf.Message()
            msg.audio_content = audio
            self.transcriber(msg)
            transcript = msg.call.transcript

            signals = self.extractor(transcript=transcript)
            briefing_result = self.drafter(
                prospect_name=signals.get("prospect_name", ""),
                pain_points=signals.get("pain_points", []),
                objections=signals.get("objections", []),
                next_step=signals.get("next_step", ""),
                budget=signals.get("budget", ""),
                timeline=signals.get("timeline", ""),
            )
            briefing = briefing_result["briefing"]

            result = {
                "transcript": transcript,
                "signals":    signals,
                "briefing":   briefing,
                "audio_path": None,
            }

            if narrate:
                result["audio_path"] = self.narrator(briefing)

            return result

        async def aforward(self, audio: bytes, narrate: bool = True) -> dict:
            msg = mf.Message()
            msg.audio_content = audio
            await self.transcriber.acall(msg)
            transcript = msg.call.transcript

            signals = await self.extractor.acall(transcript=transcript)
            briefing_result = await self.drafter.acall(
                prospect_name=signals.get("prospect_name", ""),
                pain_points=signals.get("pain_points", []),
                objections=signals.get("objections", []),
                next_step=signals.get("next_step", ""),
                budget=signals.get("budget", ""),
                timeline=signals.get("timeline", ""),
            )
            briefing = briefing_result["briefing"]

            result = {
                "transcript": transcript,
                "signals":    signals,
                "briefing":   briefing,
                "audio_path": None,
            }

            if narrate:
                result["audio_path"] = await self.narrator.acall(briefing)

            return result


    # Run
    if __name__ == "__main__":
        import sys

        generator = DealBriefingGenerator()
        audio     = open(sys.argv[1], "rb").read() if len(sys.argv) > 1 else open("call.mp3", "rb").read()
        narrate   = "--no-narrate" not in sys.argv

        result = generator(audio=audio, narrate=narrate)

        print("\n" + "=" * 60)
        print(result["briefing"])
        print("=" * 60)

        if result["audio_path"]:
            print(f"\nNarration saved to: {result['audio_path']}")
    ```

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — signatures, few-shot examples, and structured output
- [nn.Transcriber](../learn/nn/transcriber.md) — speech-to-text integration
- [nn.Speaker](../learn/nn/speaker.md) — text-to-speech synthesis
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts
- [Few-shot Examples](../learn/nn/agent/system-prompt.md) — `mf.Example` and prompt construction
- [Functional API](../learn/nn/functional.md) — `amap_gather` and parallel execution
