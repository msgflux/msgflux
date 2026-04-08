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
    transcript:       str                                              = InputField(desc="Full meeting transcript")
    tasks:            List[str]                                        = OutputField(desc="Task description for each item")
    assignees:        List[str]                                        = OutputField(desc="Assignee name per item, or empty string if unknown")
    deadlines:        List[str]                                        = OutputField(desc="Deadline per item, or empty string if none")
    confidences:      List[float]                                      = OutputField(desc="Confidence score 0.0-1.0 per item")
    commitment_types: List[Literal["confirmed", "vague", "implicit"]]  = OutputField(desc="Commitment type per item")


class FormatChecklist(Signature):
    """Format extracted action items into a markdown checklist grouped by assignee."""
    tasks:            List[str]                                        = InputField(desc="Task descriptions")
    assignees:        List[str]                                        = InputField(desc="Assignees (empty string if unassigned)")
    deadlines:        List[str]                                        = InputField(desc="Deadlines (empty string if none)")
    confidences:      List[float]                                      = InputField(desc="Confidence scores")
    commitment_types: List[Literal["confirmed", "vague", "implicit"]]  = InputField(desc="Commitment type per item")
    checklist: str = OutputField(desc="Markdown checklist grouped by assignee")
    summary:   str = OutputField(desc="One sentence: total items, confirmed vs. vague/implicit, assignees involved.")


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

class AudioTranscriber(nn.Transcriber):
    """Transcribes meeting audio into msg.meeting.transcript."""
    model          = stt_model
    message_fields = {"task_multimodal": {"audio": "audio_content"}}
    response_mode  = "meeting.transcript"


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


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
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

    print("=== SUMMARY ===")
    print(result["summary"])
    print()
    print("=== CHECKLIST ===")
    print(result["checklist"])
    print()
    print("=== ACTION ITEMS ===")
    for item in result["action_items"]:
        print(item)
