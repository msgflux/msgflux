# YouTube Cut Detector

<span class="tag tag-green">Beginner</span>

Every long-form video contains a handful of moments that could stand alone as viral short clips. Finding them manually means watching the whole thing. A transcript already has the timestamps — the only missing piece is a model that can read it and rank the moments worth cutting.

## The Problem

The naive approach is to drop the transcript into a single prompt and ask for clips. It works — until the model picks moments that look good in isolation but miss the video's narrative arc. A chorus repeated four times gets picked once. A build-up that pays off thirty seconds later gets cut in the middle.

The harder problem is transparency. When a clip doesn't land, there is no record of *why* the model chose it, making iteration guesswork.

---

## The Plan

We will build a pipeline that downloads a YouTube transcript, marks each line with a timestamp, and runs a single analysis to identify the most engaging moments worth cutting into short clips.

The key design choice is asking the model to reason over the full video before committing to any selection — considering pacing, narrative arc, build-ups, and quotability. Without this, the model treats each segment independently and misses moments that only pay off in context: a build-up gets cut in the middle; a line repeated four times gets picked at random. With step-by-step reasoning, the model builds a picture of the whole video first, then selects clips that hold up on their own. That reasoning is returned alongside the results so you can see why each moment was chosen.

The output includes start and end times, a title and hook for each clip, a viral potential score, and a summary of the overall cutting strategy.

---

## Architecture

```
YouTube URL
      │
      ▼
fetch_transcript()  →  list of FetchedTranscriptSnippet
      │
      ▼
format_transcript()  →  "[MM:SS] text\n..." string
      │
      ▼
CutAnalyzer
  signature         = VideoCutSignature
  generation_schema = ChainOfThought
      │
      ├── reasoning      ← step-by-step analysis (ChainOfThought)
      └── final_answer
            ├── start_seconds, end_seconds  ← clip boundaries
            ├── titles, hooks               ← clip metadata
            ├── viral_scores                ← ranking
            └── strategy                   ← cutting rationale
      │
      ▼
VideoCutPipeline._zip_cuts()  →  list of clip dicts
```

---

## Setup

```bash
pip install youtube-transcript-api
```

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Step 1 — Fetch and Format the Transcript

`YouTubeTranscriptApi` returns `FetchedTranscriptSnippet` objects with `.text` and `.start` attributes. `format_transcript` renders them into a single string with `[MM:SS]` markers so the model can reference exact moments.

```python
import re
from youtube_transcript_api import YouTubeTranscriptApi


def fetch_transcript(url: str) -> list:
    """Download transcript snippets for a YouTube URL."""
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    return list(YouTubeTranscriptApi().fetch(match.group(1)))


def format_transcript(snippets: list, max_chars: int = 12_000) -> str:
    """Render snippets as a timestamped string."""
    lines = []
    for s in snippets:
        minutes, seconds = divmod(int(s.start), 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {s.text}")
    return "\n".join(lines)[:max_chars]
```

!!! tip
    `max_chars=12_000` keeps the prompt inside context limits for most models.
    For very long videos, pass a smaller value or chunk the transcript.

---

## Step 2 — Signature

Five parallel output lists — one per clip attribute — keep every type native and avoid nested dicts. The lists are correlated by index: `titles[i]` belongs to `start_seconds[i]`, `end_seconds[i]`, and so on.

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Signature, InputField, OutputField
from msgflux.generation.reasoning import ChainOfThought
from typing import List

mf.load_dotenv()
model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class VideoCutSignature(Signature):
    """Analyze a video transcript and identify the most engaging moments for short clips."""

    transcript:    str       = InputField(desc="Full transcript with [MM:SS] timestamps, one line per segment")
    max_cuts:      int       = InputField(desc="Maximum number of clips to return")

    start_seconds: List[int] = OutputField(desc="Start time in seconds for each clip")
    end_seconds:   List[int] = OutputField(desc="End time in seconds for each clip")
    titles:        List[str] = OutputField(desc="Punchy clip title for each clip")
    hooks:         List[str] = OutputField(desc="Opening line that grabs attention for each clip")
    viral_scores:  List[int] = OutputField(desc="Viral potential score 1-10 for each clip")
    strategy:      str       = OutputField(desc="One-paragraph summary of the cutting strategy")
```

---

## Step 3 — CutAnalyzer

`generation_schema=ChainOfThought` fuses with `VideoCutSignature`: the model produces a `reasoning` field first, then fills `final_answer` with the signature's fields — one model call, two layers.

```python
class CutAnalyzer(nn.Agent):
    """Analyzes video transcripts and identifies the best cut intervals for short clips."""
    model             = model
    signature         = VideoCutSignature
    generation_schema = ChainOfThought
    config            = {"verbose": True}
```

The fused output structure msgFlux builds internally:

```
{
  "reasoning":    "Let's think step by step…",
  "final_answer": {
    "start_seconds": [...],
    "end_seconds":   [...],
    "titles":        [...],
    "hooks":         [...],
    "viral_scores":  [...],
    "strategy":      "…"
  }
}
```

---

## Step 4 — VideoCutPipeline

`_zip_cuts` reconstructs the per-clip dicts from the parallel lists. The analyzer is called directly with keyword args — no `Message` needed.

```python
class VideoCutPipeline(nn.Module):
    """Fetches a YouTube transcript and detects the best cut intervals."""

    def __init__(self, max_cuts: int = 5):
        super().__init__()
        self.max_cuts = max_cuts
        self.analyzer = CutAnalyzer()

    def _zip_cuts(self, result: dict) -> list:
        return [
            {
                "start_seconds": s,
                "end_seconds":   e,
                "title":         t,
                "hook":          h,
                "viral_score":   v,
            }
            for s, e, t, h, v in zip(
                result["start_seconds"], result["end_seconds"],
                result["titles"], result["hooks"], result["viral_scores"],
            )
        ]

    def forward(self, url: str) -> dict:
        transcript = format_transcript(fetch_transcript(url))
        result     = self.analyzer(transcript=transcript, max_cuts=self.max_cuts)
        final      = result.get("final_answer", result)
        return {
            "reasoning": result.get("reasoning", ""),
            "cuts":      self._zip_cuts(final),
            "strategy":  final.get("strategy", ""),
        }

    async def aforward(self, url: str) -> dict:
        transcript = format_transcript(fetch_transcript(url))
        result     = await self.analyzer.acall(transcript=transcript, max_cuts=self.max_cuts)
        final      = result.get("final_answer", result)
        return {
            "reasoning": result.get("reasoning", ""),
            "cuts":      self._zip_cuts(final),
            "strategy":  final.get("strategy", ""),
        }


pipeline = VideoCutPipeline(max_cuts=5)
```

---

## Examples

???+ example

    === "Single video"

        ```python
        pipeline = VideoCutPipeline(max_cuts=5)

        result = pipeline.forward("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        print(result["strategy"])
        print()
        for i, clip in enumerate(result["cuts"], 1):
            start, end = clip["start_seconds"], clip["end_seconds"]
            print(f"{i}. [{start}s → {end}s] {clip['title']}  (score: {clip['viral_score']}/10)")
            print(f"   Hook: {clip['hook']}")
        ```

    === "Inspecting the reasoning trace"

        ```python
        result = pipeline.forward("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        print(result["reasoning"])
        ```

        The trace explains *why* each moment was chosen — useful when a clip doesn't land and you need to adjust the instructions.

    === "Multiple videos (async)"

        ```python
        import asyncio
        import msgflux.nn.functional as F

        async def main():
            pipeline = VideoCutPipeline(max_cuts=3)
            urls = [
                "https://www.youtube.com/watch?v=VIDEO_1",
                "https://www.youtube.com/watch?v=VIDEO_2",
                "https://www.youtube.com/watch?v=VIDEO_3",
            ]
            results = await F.amap_gather(
                pipeline,
                kwargs_list=[{"url": u} for u in urls],
            )
            for url, result in zip(urls, results):
                print(f"\n{url}")
                for clip in result["cuts"]:
                    print(f"  [{clip['start_seconds']}s → {clip['end_seconds']}s] {clip['title']}")

        asyncio.run(main())
        ```

---

## Extending

### Why `ChainOfThought` here?

| Without `ChainOfThought`               | With `ChainOfThought`                         |
| --------------------------------------- | --------------------------------------------- |
| Model jumps directly to output          | Model reasons over the full transcript first  |
| May miss narrative arc and pacing       | Considers build-ups, callbacks, quotability   |
| No record of selection rationale        | Reasoning trace available for debugging       |

Use `ChainOfThought` whenever the task requires weighing multiple candidates — it consistently improves output quality on selection and ranking tasks.

### Filtering by viral score

```python
result = pipeline.forward(url)
top = [c for c in result["cuts"] if c["viral_score"] >= 8]
```

### Adding a topic label

Add a `topics` field to group cuts by theme:

```python
topics: List[str] = OutputField(desc="Topic label for each clip (e.g. 'insight', 'story', 'hook')")
```

Then filter: `[c for c in cuts if c["topic"] == "insight"]`

### Exporting timestamps for a video editor

```python
def to_edl(cuts: list) -> str:
    lines = ["TITLE: Auto Cuts", "FCM: NON-DROP FRAME", ""]
    for i, c in enumerate(cuts, 1):
        lines.append(f"{i:03d}  AX  V  C  {c['start_seconds']}s {c['end_seconds']}s")
    return "\n".join(lines)

print(to_edl(result["cuts"]))
```

---

## Complete Script

```python
import re
import asyncio
import msgflux as mf
import msgflux.nn as nn
from msgflux import Signature, InputField, OutputField
from msgflux.generation.reasoning import ChainOfThought
from youtube_transcript_api import YouTubeTranscriptApi
from typing import List

mf.load_dotenv()
model = mf.Model.chat_completion("openai/gpt-4.1-mini")


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_transcript(url: str) -> list:
    """Download transcript snippets for a YouTube URL."""
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError(f"Could not extract video ID from URL: {url}")
    return list(YouTubeTranscriptApi().fetch(match.group(1)))


def format_transcript(snippets: list, max_chars: int = 12_000) -> str:
    """Render snippets as a timestamped string."""
    lines = []
    for s in snippets:
        minutes, seconds = divmod(int(s.start), 60)
        lines.append(f"[{minutes:02d}:{seconds:02d}] {s.text}")
    return "\n".join(lines)[:max_chars]


# ── Signature ─────────────────────────────────────────────────────────────────

class VideoCutSignature(Signature):
    """Analyze a video transcript and identify the most engaging moments for short clips."""
    transcript:    str       = InputField(desc="Full transcript with [MM:SS] timestamps, one line per segment")
    max_cuts:      int       = InputField(desc="Maximum number of clips to return")
    start_seconds: List[int] = OutputField(desc="Start time in seconds for each clip")
    end_seconds:   List[int] = OutputField(desc="End time in seconds for each clip")
    titles:        List[str] = OutputField(desc="Punchy clip title for each clip")
    hooks:         List[str] = OutputField(desc="Opening line that grabs attention for each clip")
    viral_scores:  List[int] = OutputField(desc="Viral potential score 1-10 for each clip")
    strategy:      str       = OutputField(desc="One-paragraph summary of the cutting strategy")


# ── Agent ─────────────────────────────────────────────────────────────────────

class CutAnalyzer(nn.Agent):
    """Analyzes video transcripts and identifies the best cut intervals for short clips."""
    model             = model
    signature         = VideoCutSignature
    generation_schema = ChainOfThought
    config            = {"verbose": True}


# ── Pipeline ──────────────────────────────────────────────────────────────────

class VideoCutPipeline(nn.Module):
    """Fetches a YouTube transcript and detects the best cut intervals."""

    def __init__(self, max_cuts: int = 5):
        super().__init__()
        self.max_cuts = max_cuts
        self.analyzer = CutAnalyzer()

    def _zip_cuts(self, result: dict) -> list:
        return [
            {
                "start_seconds": s,
                "end_seconds":   e,
                "title":         t,
                "hook":          h,
                "viral_score":   v,
            }
            for s, e, t, h, v in zip(
                result["start_seconds"], result["end_seconds"],
                result["titles"], result["hooks"], result["viral_scores"],
            )
        ]

    def forward(self, url: str) -> dict:
        transcript = format_transcript(fetch_transcript(url))
        result     = self.analyzer(transcript=transcript, max_cuts=self.max_cuts)
        final      = result.get("final_answer", result)
        return {
            "reasoning": result.get("reasoning", ""),
            "cuts":      self._zip_cuts(final),
            "strategy":  final.get("strategy", ""),
        }

    async def aforward(self, url: str) -> dict:
        transcript = format_transcript(fetch_transcript(url))
        result     = await self.analyzer.acall(transcript=transcript, max_cuts=self.max_cuts)
        final      = result.get("final_answer", result)
        return {
            "reasoning": result.get("reasoning", ""),
            "cuts":      self._zip_cuts(final),
            "strategy":  final.get("strategy", ""),
        }


# ── Run ───────────────────────────────────────────────────────────────────────

pipeline = VideoCutPipeline(max_cuts=5)
result   = pipeline.forward("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

print(result["strategy"])
print()
for i, clip in enumerate(result["cuts"], 1):
    start, end = clip["start_seconds"], clip["end_seconds"]
    print(f"{i}. [{start}s → {end}s] {clip['title']}  (score: {clip['viral_score']}/10)")
    print(f"   Hook: {clip['hook']}")
```

---

## Further Reading

- [nn.Agent](../learn/nn/agent/index.md) — signatures and structured output
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts
- [Generation Schemas](../learn/nn/agent/generation-schemas.md) — `ChainOfThought` and reasoning traces
- [Functional API](../learn/nn/functional.md) — `amap_gather` for parallel execution
