# YouTube Video Cut Detector

Build a pipeline that fetches a YouTube transcript and uses a reasoning agent to identify the best moments for short-form clips — combining `Signature` with `generation_schema=ChainOfThought` to get both structured outputs and step-by-step transparency.

## What You'll Build

```
YouTube URL
    │
    ▼
TranscriptFetcher ──────────────► segments: [{start, text, duration}]
                                              │
                                              ▼
                                   Transcript Formatter
                                   (adds [MM:SS] timestamps)
                                              │
                                              ▼
                                       CutAnalyzer
                               signature + ChainOfThought
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                          reasoning                       final_answer
                      (step-by-step analysis)     (cuts: start, end, score…)
```

The key pattern: `generation_schema=ChainOfThought` fuses with the `Signature`, making the model reason first and then fill in the structured output fields — one model call, two layers of information.

---

## Setup

```bash
pip install msgflux[openai] youtube-transcript-api
```

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Step 1 — Fetch the Transcript

`youtube-transcript-api` returns a list of segments with text, start time, and duration. Wrap it in a small helper that accepts a full YouTube URL:

```python
import re
from youtube_transcript_api import YouTubeTranscriptApi

def fetch_transcript(url: str) -> list[dict]:
    """Download the transcript for a YouTube video URL."""
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    video_id = match.group(1)
    return YouTubeTranscriptApi.get_transcript(video_id)
```

Each returned segment looks like:

```python
{"text": "and today we are going to talk about", "start": 12.48, "duration": 2.8}
```

---

## Step 2 — Format Timestamps for the Model

Convert the raw segments into a readable transcript with `[MM:SS]` markers so the model can reference exact moments:

```python
def format_transcript(segments: list[dict], max_chars: int = 12_000) -> str:
    """Render segments as a timestamped transcript string."""
    lines = []
    for seg in segments:
        minutes, seconds = divmod(int(seg["start"]), 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {seg['text']}")
    return "\n".join(lines)[:max_chars]
```

!!! tip
    `max_chars=12_000` keeps the prompt inside context limits for most models.
    For very long videos, pass a smaller value or chunk the transcript.

---

## Step 3 — Define the Signature

A `Signature` declares exactly what the agent should produce. Each `OutputField` description guides the model toward well-structured, purpose-built values:

```python
import msgflux as mf
from msgflux import Signature, InputField, OutputField

class VideoCutSignature(Signature):
    """Analyze a video transcript and identify the most engaging moments suitable for viral short clips."""

    transcript: str = InputField(
        desc="Full transcript with [MM:SS] timestamps, one line per segment"
    )
    max_cuts: int = InputField(
        desc="Maximum number of clips to return"
    )

    cuts: list[dict] = OutputField(
        desc=(
            "Ordered list of clip intervals. Each item must contain: "
            "start_seconds (int), end_seconds (int), "
            "title (str — punchy clip title), "
            "hook (str — the opening line that grabs attention), "
            "viral_score (int 1–10)"
        )
    )
    strategy: str = OutputField(
        desc="One-paragraph summary of the cutting strategy: why these moments work and how they fit together"
    )
```

---

## Step 4 — Combine Signature with ChainOfThought

Passing `generation_schema=ChainOfThought` alongside `signature` tells msgFlux to **fuse** the two schemas: the model first produces a `reasoning` field (step-by-step analysis), then fills `final_answer` with the signature's structured fields.

```python
import msgflux.nn as nn
from msgflux.generation.reasoning import ChainOfThought

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class CutAnalyzer(nn.Agent):
    """Analyzes video transcripts and identifies the best cut intervals for short clips."""
    model = model
    signature = VideoCutSignature
    generation_schema = ChainOfThought
    config = {"verbose": True}
```

The fused output structure that msgFlux builds internally:

```
Output
  ├── reasoning   — "Let's think step by step …" (from ChainOfThought)
  └── final_answer
        ├── cuts     — list[dict]  (from VideoCutSignature)
        └── strategy — str         (from VideoCutSignature)
```

One call, two layers: the reasoning trace is available for debugging while `final_answer` carries the clean structured result.

---

## Step 5 — Compose the Pipeline

Wire everything into a single `Module`:

```python
from msgflux import Message


class VideoCutPipeline(nn.Module):
    """Fetches a YouTube transcript and detects the best cut intervals."""

    def __init__(self, max_cuts: int = 5):
        super().__init__()
        self.max_cuts = max_cuts
        self.analyzer = CutAnalyzer()

    def forward(self, msg: Message) -> Message:
        # 1. Download transcript
        segments = fetch_transcript(msg.url)

        # 2. Format with timestamps
        msg.transcript = format_transcript(segments)
        msg.max_cuts = self.max_cuts

        # 3. Analyze with Signature + ChainOfThought
        self.analyzer(msg)

        return msg
```

Usage:

```python
pipeline = VideoCutPipeline(max_cuts=5)

msg = Message(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
pipeline(msg)

# Reasoning trace (from ChainOfThought)
print("Reasoning:\n", msg.reasoning)

# Structured output (from VideoCutSignature)
print("\nStrategy:\n", msg.strategy)
print(f"\nTop {len(msg.cuts)} cuts:")
for i, clip in enumerate(msg.cuts, 1):
    start = clip["start_seconds"]
    end = clip["end_seconds"]
    print(f"  {i}. [{start}s → {end}s] {clip['title']}  (score: {clip['viral_score']}/10)")
    print(f"     Hook: {clip['hook']}")
```

Sample output:

```
Reasoning:
  Let's think step by step in order to find the best moments...
  The video opens with a strong hook at 0:12 that immediately establishes tension.
  Around 2:45 there is a key insight that would resonate well as a standalone clip...

Strategy:
  The video has a strong narrative arc with three distinct peaks. The first clip
  capitalises on the opening hook, the second captures the core insight, and the
  remaining three cover the most quotable exchanges in the second half.

Top 5 cuts:
  1. [12s → 55s] The Opening Hook  (score: 9/10)
     Hook: "Today I'm going to show you something you've never seen before."
  2. [165s → 210s] The Core Insight  (score: 8/10)
     Hook: "Here's why everything you thought you knew is wrong."
  ...
```

---

## Complete Example

```python
import re
import msgflux as mf
import msgflux.nn as nn
from msgflux import Message, Signature, InputField, OutputField
from msgflux.generation.reasoning import ChainOfThought
from youtube_transcript_api import YouTubeTranscriptApi


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_transcript(url: str) -> list[dict]:
    """Download the transcript for a YouTube video URL."""
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    return YouTubeTranscriptApi.get_transcript(match.group(1))


def format_transcript(segments: list[dict], max_chars: int = 12_000) -> str:
    """Render segments as a timestamped transcript string."""
    lines = []
    for seg in segments:
        minutes, seconds = divmod(int(seg["start"]), 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {seg['text']}")
    return "\n".join(lines)[:max_chars]


# ── Signature ─────────────────────────────────────────────────────────────────

class VideoCutSignature(Signature):
    """Analyze a video transcript and identify the most engaging moments suitable for viral short clips."""

    transcript: str = InputField(
        desc="Full transcript with [MM:SS] timestamps, one line per segment"
    )
    max_cuts: int = InputField(
        desc="Maximum number of clips to return"
    )

    cuts: list[dict] = OutputField(
        desc=(
            "Ordered list of clip intervals. Each item must contain: "
            "start_seconds (int), end_seconds (int), "
            "title (str — punchy clip title), "
            "hook (str — the opening line that grabs attention), "
            "viral_score (int 1–10)"
        )
    )
    strategy: str = OutputField(
        desc="One-paragraph summary of the cutting strategy: why these moments work and how they fit together"
    )


# ── Model + Agent ─────────────────────────────────────────────────────────────

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class CutAnalyzer(nn.Agent):
    """Analyzes video transcripts and identifies the best cut intervals for short clips."""
    model = model
    signature = VideoCutSignature
    generation_schema = ChainOfThought
    config = {"verbose": True}


# ── Pipeline ──────────────────────────────────────────────────────────────────

class VideoCutPipeline(nn.Module):
    """Fetches a YouTube transcript and detects the best cut intervals."""

    def __init__(self, max_cuts: int = 5):
        super().__init__()
        self.max_cuts = max_cuts
        self.analyzer = CutAnalyzer()

    def forward(self, msg: Message) -> Message:
        segments = fetch_transcript(msg.url)
        msg.transcript = format_transcript(segments)
        msg.max_cuts = self.max_cuts
        self.analyzer(msg)
        return msg


# ── Run ───────────────────────────────────────────────────────────────────────

pipeline = VideoCutPipeline(max_cuts=5)

msg = Message(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
pipeline(msg)

print("Reasoning:\n", msg.reasoning)
print("\nStrategy:\n", msg.strategy)
print(f"\nTop {len(msg.cuts)} cuts:")
for i, clip in enumerate(msg.cuts, 1):
    start = clip["start_seconds"]
    end   = clip["end_seconds"]
    print(f"  {i}. [{start}s → {end}s] {clip['title']}  (score: {clip['viral_score']}/10)")
    print(f"     Hook: {clip['hook']}")
```

---

## Async Version

Process multiple videos in parallel with `ascatter_gather`:

```python
import asyncio
import re
import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux import Message, Signature, InputField, OutputField
from msgflux.generation.reasoning import ChainOfThought
from youtube_transcript_api import YouTubeTranscriptApi

# ... (definitions above) ...


class AsyncVideoCutPipeline(nn.Module):
    def __init__(self, max_cuts: int = 5):
        super().__init__()
        self.max_cuts = max_cuts
        self.analyzer = CutAnalyzer()

    async def aforward(self, msg: Message) -> Message:
        segments = fetch_transcript(msg.url)
        msg.transcript = format_transcript(segments)
        msg.max_cuts = self.max_cuts
        await self.analyzer.acall(msg)
        return msg


async def main():
    urls = [
        "https://www.youtube.com/watch?v=VIDEO_ID_1",
        "https://www.youtube.com/watch?v=VIDEO_ID_2",
        "https://www.youtube.com/watch?v=VIDEO_ID_3",
    ]

    pipeline = AsyncVideoCutPipeline(max_cuts=5)
    messages = [Message(url=url) for url in urls]

    results = await F.ascatter_gather(
        [pipeline.acall] * len(messages),
        args_list=[(msg,) for msg in messages],
    )

    for msg in results:
        print(f"\n{msg.url}")
        for clip in msg.cuts:
            print(f"  [{clip['start_seconds']}s → {clip['end_seconds']}s] {clip['title']}")


asyncio.run(main())
```

---

## Why `generation_schema=ChainOfThought` Here?

| Without `ChainOfThought`           | With `ChainOfThought`                        |
| ----------------------------------- | -------------------------------------------- |
| Model jumps directly to output      | Model reasons over the full transcript first |
| May miss narrative context          | Considers pacing, arc, and quotability       |
| Output only in `final_answer`       | Reasoning trace available in `msg.reasoning` |
| Harder to debug wrong cuts          | Trace explains *why* each moment was chosen  |

Use `ChainOfThought` whenever the task requires weighing multiple candidates — the reasoning step consistently improves output quality on selection and ranking tasks.

---

## Next Steps

- **Add a speaker diarization step**: Use `nn.Transcriber` with a speaker-aware model to label who is speaking before cutting — great for podcast clips.
- **Rank by topic**: Add a `topic: Literal[...]` output field to the signature and filter cuts by theme.
- **Export to EDL**: Convert `msg.cuts` to an Edit Decision List and import directly into Premiere Pro or DaVinci Resolve.
