# Ad Focus Group Simulator

Build a creative pipeline that simulates a focus group evaluating product advertisements. Multiple AI personas review the same ad from different perspectives, then an iterative refinement loop improves the copy based on their feedback — and finally an image model generates the visual.

The entire flow is orchestrated with `Inline`, the msgFlux DSL for composing pipelines.

## What You'll Build

```
                    ┌──> Teenager ──────┐
Author's brief ──> │    Professional ──┼──> evaluations
                    └──> Budget Shopper ┘        │
                                                 ▼
                                          ┌─── Refiner ◄─── evaluations
                                          │      │
                                          │      ▼
                                          │  refined_text
                                          │      │
                                          │      ▼
                                          │  [Teenager, Professional, Budget Shopper]
                                          │      │
                                          │      ▼
                                          └── score >= 8 ?  ──── done
                                                 │
                                                 ▼
                                          Image Generator
                                                 │
                                                 ▼
                                             poster.png
```

Three levels of complexity, each building on the previous:

| Level | What happens |
|-------|-------------|
| **1 — Evaluate** | Multiple customer personas rate the same ad in parallel |
| **2 — Refine** | A loop refines the text until the panel is satisfied |
| **3 — Generate** | A creative director writes the brief, a copywriter proposes, and an image model produces the visual |

---

## Setup

```bash
pip install msgflux[openai]
```

Create a `.env` file at the project root with your API keys:

```bash title=".env"
OPENAI_API_KEY=sk-...
```

Then load it at the top of your script with `mf.load_dotenv()` — this reads the `.env` file and injects the variables into `os.environ`:

```python
import msgflux as mf

mf.load_dotenv()
```

---

## Level 1 — Parallel Evaluation

The simplest version: define customer personas as Agents and evaluate a product ad in parallel using `bcast_gather` inside an Inline pipeline.

### Step 1: Define the Evaluation Signature

A class-based `Signature` defines the structured contract shared by all evaluators. The docstring becomes the agent's task instructions, `InputField` maps the input, and each `OutputField` describes exactly what the model should produce. The framework handles parsing and type validation — no manual JSON formatting needed:

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Signature, InputField, OutputField

mf.load_dotenv()

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class AdEvaluation(Signature):
    """Evaluate the advertisement from your perspective. Consider the tone,
    clarity, and appeal. Be honest and specific in your opinion."""

    ad_text: str = InputField(desc="The advertisement text to evaluate")
    opinion: str = OutputField(desc="Your honest reaction to the ad in 2-3 sentences")
    score: int = OutputField(desc="Overall score from 1 (terrible) to 10 (perfect)")
```

The Signature does three things at once:

1. **Instructions**: the docstring becomes the agent's task instructions
2. **Input mapping**: `InputField` tells the agent to read `ad_text` from the message
3. **Expected output**: each `OutputField` constrains the model to return typed fields — the framework builds the output schema automatically

### Step 2: Define the Personas

All three personas share the same `AdEvaluation` signature, so they produce the same structured output. What differentiates them is `system_message` — the persona's personality and evaluation criteria:

```python
class Teenager(nn.Agent):
    model = model
    system_message = """You are a 17-year-old social media native. You care about
    aesthetics, trends, memes, and authenticity. Focus on whether the ad feels
    authentic or corporate and whether you would share it on social media."""
    signature = AdEvaluation
    response_mode = "eval_teenager"


class Professional(nn.Agent):
    model = model
    system_message = """You are a 35-year-old working professional. You value clarity,
    time-saving, and quality. Focus on whether the value proposition is clear
    and whether the ad respects your time."""
    signature = AdEvaluation
    response_mode = "eval_professional"


class BudgetShopper(nn.Agent):
    model = model
    system_message = """You are a budget-conscious parent. You look for deals,
    compare prices, and distrust hype. Focus on whether the ad mentions price
    or value and whether it feels honest or manipulative."""
    signature = AdEvaluation
    response_mode = "eval_budget"
```

The `response_mode` writes the entire structured result to a specific field on the message (e.g., `msg.eval_teenager`), keeping the three evaluations separate. Since the output structure is identical, `compute_score` can read `.score` from any of them uniformly.

### Step 3: Build the Evaluation Pipeline

The `[teenager, professional, budget]` syntax runs all three in parallel on the same message. Each agent reads `msg.ad_text` and writes its structured output to its own `msg.eval_*` field:

```python
from msgflux import Inline, Message

evaluate = Inline(
    "[teenager, professional, budget]",
    {
        "teenager": Teenager(),
        "professional": Professional(),
        "budget": BudgetShopper(),
    },
)
```

### Step 4: Run It

```python
msg = Message()
msg.ad_text = """
Introducing CloudBrew — the smart coffee maker that learns your taste.
Wake up to your perfect cup, every morning, no buttons needed.
Pre-order now for 40% off. Free shipping.
"""

evaluate(msg)

print("Teenager:", msg.eval_teenager)
# {'opinion': 'Feels clean but a bit corporate...', 'score': 6}

print("Professional:", msg.eval_professional)
# {'opinion': 'Clear value prop, concise...', 'score': 8}

print("Budget Shopper:", msg.eval_budget)
# {'opinion': '40% off and free shipping are strong...', 'score': 7}
```

The three evaluations run concurrently — total time is roughly one API call, not three. Each result is a structured `dotdict` with typed fields, not a raw JSON string.

---

## Level 2 — Iterative Refinement

Now we close the loop: a Refiner agent reads the evaluations and improves the ad text, then the panel re-evaluates. This repeats until the average score is high enough.

### Step 5: Add the Refiner and Scorer

The Refiner reads the original text and all evaluations, then produces a new version. The Scorer accesses the `.score` field directly from each structured evaluation — no JSON parsing needed:

```python
class Refiner(nn.Agent):
    """Rewrites ad copy based on focus group feedback."""

    model = model
    instructions = """
    You are a senior copywriter. You receive the original ad text and feedback
    from three customer personas (teenager, professional, budget shopper).

    Rewrite the ad to address their concerns while keeping the core message.
    Return only the new ad text — nothing else.
    """
    message_fields = {"task_inputs": "refinement_input"}
    response_mode = "ad_text"


def compute_score(msg):
    """Extract scores from structured evaluations and compute the average."""
    scores = []
    for field in ["eval_teenager", "eval_professional", "eval_budget"]:
        evaluation = msg.get(field)
        if isinstance(evaluation, dict):
            scores.append(evaluation.get("score", 5))
        else:
            scores.append(5)
    msg.avg_score = sum(scores) / len(scores) if scores else 0
    msg.iteration = msg.get("iteration", 0) + 1
```

### Step 6: Prepare Refinement Input

A helper module assembles the input for the Refiner by combining the current text with all evaluations:

```python
def prepare_refinement(msg):
    """Combine ad text + evaluations into a single refinement prompt."""
    msg.refinement_input = (
        f"Current ad:\n{msg.ad_text}\n\n"
        f"Teenager feedback: {msg.eval_teenager.opinion} (score: {msg.eval_teenager.score})\n\n"
        f"Professional feedback: {msg.eval_professional.opinion} (score: {msg.eval_professional.score})\n\n"
        f"Budget shopper feedback: {msg.eval_budget.opinion} (score: {msg.eval_budget.score})\n\n"
        f"Rewrite the ad addressing this feedback."
    )
```

### Step 7: Compose the Refinement Loop

The Inline DSL's `@{condition}: body;` syntax creates a while loop. The pipeline evaluates first, then enters the loop — each iteration refines, re-evaluates, and checks the score:

```python
pipeline = Inline(
    "[teenager, professional, budget] -> scorer -> @{avg_score < 8 & iteration < 3}: prepare -> refiner -> [teenager, professional, budget] -> scorer;",
    {
        "teenager": Teenager(),
        "professional": Professional(),
        "budget": BudgetShopper(),
        "refiner": Refiner(),
        "scorer": compute_score,
        "prepare": prepare_refinement,
    },
)

msg = Message()
msg.ad_text = """
CloudBrew. Smart coffee. Your taste. Pre-order.
"""

pipeline(msg)

print(f"Iterations: {msg.iteration}")
print(f"Final score: {msg.avg_score:.1f}")
print(f"Final ad:\n{msg.ad_text}")
```

The loop runs at most 3 times or until the average score reaches 8+. Each iteration, the Refiner reads what all three personas said and adapts the copy.

---

## Level 3 — Full Creative Pipeline

The final version adds two stages: a Creative Director who writes the brief from a product description, and a MediaMaker that generates the poster image from the refined text.

### Step 8: Creative Director and Image Generator

```python
class CreativeDirector(nn.Agent):
    """Writes an initial ad brief from a product description."""

    model = model
    instructions = """
    You are a creative director at an ad agency. Given a product description,
    write compelling ad copy (3-5 sentences). Include:
    - A catchy headline
    - Key benefits
    - A call to action

    Return only the ad text.
    """
    message_fields = {"task_inputs": "product_description"}
    response_mode = "ad_text"


class PosterMaker(nn.MediaMaker):
    """Generates a poster image from ad text."""

    model = mf.Model.text_to_image("openai/gpt-image-1")
    message_fields = {"task_inputs": "poster_prompt"}
    response_mode = "poster"
    config = {"size": "1536x1024", "quality": "high"}


def build_poster_prompt(msg):
    """Turn the final ad text into an image generation prompt."""
    msg.poster_prompt = (
        f"Professional product advertisement poster. Clean, modern design. "
        f"The ad copy reads: {msg.ad_text}. "
        f"Minimalist layout, premium feel, warm lighting, brand-quality photography."
    )
```

### Step 9: Complete Pipeline

The full pipeline chains all three levels: brief, evaluate, refine loop, generate image. Reading left to right: the director writes the first draft, the panel evaluates, the loop refines, then the image is generated:

```python
full_pipeline = Inline(
    "director -> [teenager, professional, budget] -> scorer -> @{avg_score < 8 & iteration < 3}: prepare -> refiner -> [teenager, professional, budget] -> scorer; -> poster_prompt -> poster_maker",
    {
        "director": CreativeDirector(),
        "teenager": Teenager(),
        "professional": Professional(),
        "budget": BudgetShopper(),
        "refiner": Refiner(),
        "scorer": compute_score,
        "prepare": prepare_refinement,
        "poster_prompt": build_poster_prompt,
        "poster_maker": PosterMaker(),
    },
)

msg = Message()
msg.product_description = """
CloudBrew is a Wi-Fi-enabled coffee maker with a built-in taste profile system.
It learns your preferences over time and adjusts brew strength, temperature,
and grind size automatically. Compatible with any ground coffee or pods.
Retail price: $149. Launch promotion: 40% off pre-orders with free shipping.
"""

full_pipeline(msg)

print(f"Iterations: {msg.iteration}")
print(f"Final score: {msg.avg_score:.1f}")
print(f"Final ad:\n{msg.ad_text}")

with open("poster.png", "wb") as f:
    f.write(msg.poster)

print("Poster saved to poster.png")
```

---

## Complete Example

```python
import msgflux as mf
import msgflux.nn as nn
from msgflux import Inline, Message, Signature, InputField, OutputField

mf.load_dotenv()


# ── Model ────────────────────────────────────────────────────────────────────

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


# ── Evaluation Signature ─────────────────────────────────────────────────────

class AdEvaluation(Signature):
    """Evaluate the advertisement from your perspective. Consider the tone,
    clarity, and appeal. Be honest and specific in your opinion."""

    ad_text: str = InputField(desc="The advertisement text to evaluate")
    opinion: str = OutputField(desc="Your honest reaction to the ad in 2-3 sentences")
    score: int = OutputField(desc="Overall score from 1 (terrible) to 10 (perfect)")


# ── Personas ─────────────────────────────────────────────────────────────────

class Teenager(nn.Agent):
    model = model
    system_message = """You are a 17-year-old social media native. You care about
    aesthetics, trends, memes, and authenticity. Focus on whether the ad feels
    authentic or corporate and whether you would share it on social media."""
    signature = AdEvaluation
    response_mode = "eval_teenager"


class Professional(nn.Agent):
    model = model
    system_message = """You are a 35-year-old working professional. You value clarity,
    time-saving, and quality. Focus on whether the value proposition is clear
    and whether the ad respects your time."""
    signature = AdEvaluation
    response_mode = "eval_professional"


class BudgetShopper(nn.Agent):
    model = model
    system_message = """You are a budget-conscious parent. You look for deals,
    compare prices, and distrust hype. Focus on whether the ad mentions price
    or value and whether it feels honest or manipulative."""
    signature = AdEvaluation
    response_mode = "eval_budget"


# ── Creative Agents ──────────────────────────────────────────────────────────

class CreativeDirector(nn.Agent):
    """Writes an initial ad brief from a product description."""

    model = model
    instructions = """
    You are a creative director at an ad agency. Given a product description,
    write compelling ad copy (3-5 sentences). Include:
    - A catchy headline
    - Key benefits
    - A call to action

    Return only the ad text.
    """
    message_fields = {"task_inputs": "product_description"}
    response_mode = "ad_text"


class Refiner(nn.Agent):
    """Rewrites ad copy based on focus group feedback."""

    model = model
    instructions = """
    You are a senior copywriter. You receive the original ad text and feedback
    from three customer personas (teenager, professional, budget shopper).

    Rewrite the ad to address their concerns while keeping the core message.
    Return only the new ad text — nothing else.
    """
    message_fields = {"task_inputs": "refinement_input"}
    response_mode = "ad_text"


# ── Image Generation ─────────────────────────────────────────────────────────

class PosterMaker(nn.MediaMaker):
    """Generates a poster image from ad text."""

    model = mf.Model.text_to_image("openai/gpt-image-1")
    message_fields = {"task_inputs": "poster_prompt"}
    response_mode = "poster"
    config = {"size": "1536x1024", "quality": "high"}


# ── Helper Functions ─────────────────────────────────────────────────────────

def compute_score(msg):
    """Extract scores from structured evaluations and compute the average."""
    scores = []
    for field in ["eval_teenager", "eval_professional", "eval_budget"]:
        evaluation = msg.get(field)
        if isinstance(evaluation, dict):
            scores.append(evaluation.get("score", 5))
        else:
            scores.append(5)
    msg.avg_score = sum(scores) / len(scores) if scores else 0
    msg.iteration = msg.get("iteration", 0) + 1


def prepare_refinement(msg):
    """Combine ad text + evaluations into a single refinement prompt."""
    msg.refinement_input = (
        f"Current ad:\n{msg.ad_text}\n\n"
        f"Teenager feedback: {msg.eval_teenager.opinion} (score: {msg.eval_teenager.score})\n\n"
        f"Professional feedback: {msg.eval_professional.opinion} (score: {msg.eval_professional.score})\n\n"
        f"Budget shopper feedback: {msg.eval_budget.opinion} (score: {msg.eval_budget.score})\n\n"
        f"Rewrite the ad addressing this feedback."
    )


def build_poster_prompt(msg):
    """Turn the final ad text into an image generation prompt."""
    msg.poster_prompt = (
        f"Professional product advertisement poster. Clean, modern design. "
        f"The ad copy reads: {msg.ad_text}. "
        f"Minimalist layout, premium feel, warm lighting, brand-quality photography."
    )


# ── Pipeline ─────────────────────────────────────────────────────────────────

pipeline = Inline(
    "director -> [teenager, professional, budget] -> scorer"
    " -> @{avg_score < 8 & iteration < 3}:"
    " prepare -> refiner -> [teenager, professional, budget] -> scorer;"
    " -> poster_prompt -> poster_maker",
    {
        "director": CreativeDirector(),
        "teenager": Teenager(),
        "professional": Professional(),
        "budget": BudgetShopper(),
        "refiner": Refiner(),
        "scorer": compute_score,
        "prepare": prepare_refinement,
        "poster_prompt": build_poster_prompt,
        "poster_maker": PosterMaker(),
    },
)


# ── Run ──────────────────────────────────────────────────────────────────────

msg = Message()
msg.product_description = """
CloudBrew is a Wi-Fi-enabled coffee maker with a built-in taste profile system.
It learns your preferences over time and adjusts brew strength, temperature,
and grind size automatically. Compatible with any ground coffee or pods.
Retail price: $149. Launch promotion: 40% off pre-orders with free shipping.
"""

pipeline(msg)

print(f"Iterations: {msg.iteration}")
print(f"Final score: {msg.avg_score:.1f}")
print(f"\nFinal ad:\n{msg.ad_text}")

with open("poster.png", "wb") as f:
    f.write(msg.poster)

print("\nPoster saved to poster.png")
```

---

## How It Works

The pipeline is a single `Inline` expression that reads naturally from left to right:

```
director → [teenager, professional, budget] → scorer → @{loop} → poster_prompt → poster_maker
```

| Stage | What happens | Inline syntax |
|-------|-------------|---------------|
| **Brief** | CreativeDirector writes the first draft from `product_description` | `director` |
| **Evaluate** | Three personas rate the ad in parallel | `[teenager, professional, budget]` |
| **Score** | `compute_score` reads `.score` from each structured evaluation | `scorer` |
| **Refine loop** | While score < 8 and iteration < 3: refine and re-evaluate | `@{avg_score < 8 & iteration < 3}: ...;` |
| **Image prompt** | `build_poster_prompt` turns the final text into an image prompt | `poster_prompt` |
| **Generate** | MediaMaker calls the image model and writes bytes to `msg.poster` | `poster_maker` |

Every module reads from and writes to the same `Message` object. The parallel bracket `[...]` runs all three evaluators concurrently via `bcast_gather` — each writes to a disjoint field (`eval_teenager`, `eval_professional`, `eval_budget`), so there are no race conditions.

All three personas share a single `AdEvaluation` Signature that defines the structured output (`opinion`, `score`). The Signature docstring becomes the task instructions, and each `OutputField` description constrains what the model produces. What differentiates the personas is `system_message` — each agent gets its own personality and evaluation criteria while producing the same typed output. The `response_mode` then writes each result to a separate path on the message.

The `@{condition}: body;` while loop is the key to iterative refinement. Each pass through the loop:

1. `prepare` — assembles the feedback into a single prompt
2. `refiner` — rewrites `msg.ad_text` based on the feedback
3. `[teenager, professional, budget]` — re-evaluates the new version
4. `scorer` — recomputes `msg.avg_score` and increments `msg.iteration`

When the score reaches 8+ or 3 iterations pass, the loop exits and the pipeline continues to image generation.
