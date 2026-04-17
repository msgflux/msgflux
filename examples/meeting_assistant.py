# /// script
# dependencies = []
# ///

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
    transcript:       str                                     = InputField(desc="Full verbatim transcript of the meeting")
    tldr:             str                                     = OutputField(desc="One-sentence summary of what the meeting accomplished")
    decisions:        List[str]                               = OutputField(desc="Decisions made and agreed upon")
    action_owners:    List[str]                               = OutputField(desc="Owner for each action item")
    action_tasks:     List[str]                               = OutputField(desc="Task description for each action item")
    action_deadlines: List[str]                               = OutputField(desc="Deadline per item, empty string if none")
    open_questions:   List[str]                               = OutputField(desc="Unresolved questions needing follow-up")
    sentiment:        Literal["positive", "neutral", "tense"] = OutputField(desc="Overall tone of the meeting")
    follow_up_meeting: bool                                   = OutputField(desc="True if a follow-up was explicitly agreed")


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


TRANSCRIPT = """
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


if __name__ == "__main__":
    import sys

    assistant = MeetingAssistant()
    mode      = sys.argv[1] if len(sys.argv) > 1 else "text"

    if mode == "audio":
        audio  = open(sys.argv[2], "rb").read()
        result = assistant(audio=audio)
    else:
        result = assistant(transcript=TRANSCRIPT)

    print("=" * 60)
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
    print("=" * 60)
