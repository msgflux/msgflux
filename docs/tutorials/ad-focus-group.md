# Ad Focus Group Simulator

<span class="tag tag-orange">Advanced</span>

Testing ad copy against real customers is slow and expensive. This tutorial builds a synthetic focus group — three AI personas review the same ad in parallel, a refinement loop iterates until the panel approves, and an image model generates the final poster.

## The Problem

The typical workflow for evaluating ad copy looks like this.

```
Product brief
       │
       ▼
│           SingleReviewerAgent            │
│                                          │
│   read ad  ←──→  rate it                │
       │ one opinion, one score
       ▼
  revision (or ship it)
```

- A single agent with no persona produces generic feedback. "The copy is clear and compelling" — every time.
- There is no iterative loop. Feedback and rewriting are separate manual steps.
- There is no image. Someone has to brief a designer separately.
- When the copy is bad, you do not know *why* — which audience it failed with and on what dimension.

You are guessing until it ships.

---

## The Plan

We will build a pipeline that evaluates ad copy against three distinct customer personas in parallel, refines the text in a loop until the panel approves, and finally generates a poster image from the accepted copy.

The pipeline uses the **declarative API**: the evaluation–refinement–generation sequence is expressed as a single pipeline declaration. Parallel evaluation and iterative refinement are composed naturally — no manual orchestration, no callbacks. State flows through a shared message that every module reads from and writes to.

There are three levels of complexity, each building on the previous:

| Level | What it adds |
|-------|-------------|
| **1 — Evaluate** | Three personas rate the same ad simultaneously |
| **2 — Refine** | A while loop refines the copy until the panel score reaches 8+ |
| **3 — Generate** | A Creative Director writes the first draft; an image model produces the poster |

---

## Architecture

```
product_description
         │
         ▼
         │ msg.ad_text                                        │
         ▼                                                    │
│  Teenager  Professional  Budget│                            │
│  eval_teenager  eval_professional  eval_budget              │
                 │ msg.avg_score, msg.iteration               │
                 ▼                                            │
     @{ avg_score < 8 & iteration < 3 }                      │
                 │                                            │
          prepare_refinement  ←── assembles refinement_input  │
                 │                                            │
                 │                                            │
       [re-evaluate + rescore]                                │
                 │ loop exits when score ≥ 8 or iter ≥ 3     │
                 ▼                                            │
       build_poster_prompt ── msg.poster_prompt               │
                 │                                            │
```

The three personas share one `AdEvaluation` Signature and produce identical structured output. `message_fields` maps the shared `ad_text` field from the message to each persona's input; `response_mode` writes each evaluation to its own namespace (`eval_teenager`, `eval_professional`, `eval_budget`). The Refiner overwrites `msg.ad_text` on each pass — so the panel always re-evaluates the latest version.

---

## Setup

--8<-- "docs/_includes/init_chat_completion_model.md"

---

## Level 1 — Parallel Evaluation

The simplest version: define customer personas as Agents and evaluate a product ad in parallel.

### Step 1 — Evaluation Signature

All three personas share the same `AdEvaluation` Signature. The docstring becomes the agent's task instructions, `InputField` maps the input field, and each `OutputField` constrains what the model returns. The framework builds the output schema automatically — no manual JSON formatting needed.

```python
import msgflux as mf
import msgflux.nn as nn

mf.load_dotenv()

model = mf.Model.chat_completion("openai/gpt-4.1-mini")


class AdEvaluation(mf.Signature):
    """Evaluate the advertisement from your perspective. Consider the tone,
    clarity, and appeal. Be honest and specific in your opinion."""

    ad_text: str = mf.InputField(desc="The advertisement text to evaluate")

    opinion: str = mf.OutputField(desc="Your honest reaction to the ad in 2-3 sentences")
    score: int = mf.OutputField(desc="Overall score from 1 (terrible) to 10 (perfect)")
```

### Step 2 — Personas

All three agents share the same Signature — so they produce the same structured output. What differentiates them is `system_message`. `message_fields` maps the signature's `ad_text` InputField to `msg.ad_text`, and `response_mode` writes each evaluation to a separate namespace so the results never overwrite each other.

```python
class Teenager(nn.Agent):
    model = model
    system_message = """You are a 17-year-old social media native. You care about
    aesthetics, trends, memes, and authenticity. Focus on whether the ad feels
    authentic or corporate and whether you would share it on social media."""
    signature = AdEvaluation
    message_fields = {"task": {"ad_text": "ad_text"}}
    response_mode = "eval_teenager"
    config = {"verbose": True}


class Professional(nn.Agent):
    model = model
    system_message = """You are a 35-year-old working professional. You value clarity,
    time-saving, and quality. Focus on whether the value proposition is clear
    and whether the ad respects your time."""
    signature = AdEvaluation
    message_fields = {"task": {"ad_text": "ad_text"}}
    response_mode = "eval_professional"
    config = {"verbose": True}


class BudgetShopper(nn.Agent):
    model = model
    system_message = """You are a budget-conscious parent. You look for deals,
    compare prices, and distrust hype. Focus on whether the ad mentions price
    or value and whether it feels honest or manipulative."""
    signature = AdEvaluation
    message_fields = {"task": {"ad_text": "ad_text"}}
    response_mode = "eval_budget"
    config = {"verbose": True}
```

### Step 3 — Evaluation Pipeline

`[teenager, professional, budget]` runs all three agents in parallel on the same message via `bcast_gather`. Each reads `msg.ad_text` and writes its structured result to its own namespace.

```python
evaluate = Inline(
    "[teenager, professional, budget]",
    {
        "teenager": Teenager(),
        "professional": Professional(),
        "budget": BudgetShopper(),
    },
)

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

The three evaluations run concurrently — total time is roughly one API call, not three. Each result is a structured `dotdict` with typed fields accessible via `msg.eval_teenager.score`, `msg.eval_professional.opinion`, etc.

---

## Level 2 — Iterative Refinement

Now we close the loop: a `Refiner` reads all three evaluations and rewrites the ad copy. The panel re-evaluates and the cycle repeats until the average score reaches 8 or three iterations pass.

### Step 4 — Refiner and Scorer

The `Refiner` reads `msg.refinement_input` (assembled by `prepare_refinement`) and overwrites `msg.ad_text` via `response_mode` — so the personas always re-evaluate the latest version on the next pass.

`compute_score` is a plain Python function used as an Inline step. It reads `.score` directly from each structured evaluation namespace and writes the average back to `msg.avg_score`.

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
    message_fields = {"task": "refinement_input"}
    response_mode = "ad_text"


def compute_score(msg):
    scores = [
        msg.eval_teenager.get("score", 5),
        msg.eval_professional.get("score", 5),
        msg.eval_budget.get("score", 5),
    ]
    msg.avg_score = sum(scores) / len(scores)
    msg.iteration = msg.get("iteration", 0) + 1


def prepare_refinement(msg):
    msg.refinement_input = (
        f"Current ad:\n{msg.ad_text}\n\n"
        f"Teenager feedback: {msg.eval_teenager.opinion} "
        f"(score: {msg.eval_teenager.score})\n\n"
        f"Professional feedback: {msg.eval_professional.opinion} "
        f"(score: {msg.eval_professional.score})\n\n"
        f"Budget shopper feedback: {msg.eval_budget.opinion} "
        f"(score: {msg.eval_budget.score})\n\n"
        "Rewrite the ad addressing this feedback."
    )
```

### Step 5 — Refinement Loop

The `@{condition}: body;` syntax creates a while loop. The pipeline evaluates first, then enters the loop — each iteration calls `prepare`, rewrites the copy with `refiner`, re-evaluates with the panel, and recomputes the score. The loop exits when `avg_score >= 8` or `iteration >= 3`.

```python
pipeline = mf.Inline(
    "[teenager, professional, budget] -> scorer"
    " -> @{avg_score < 8 & iteration < 3}:"
    " prepare -> refiner -> [teenager, professional, budget] -> scorer;",
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
msg.ad_text = "CloudBrew. Smart coffee. Your taste. Pre-order."

pipeline(msg)

print(f"Iterations: {msg.iteration}")
print(f"Final score: {msg.avg_score:.1f}")
print(f"Final ad:\n{msg.ad_text}")
```

---

## Level 3 — Full Creative Pipeline

The final version adds two stages at the front and back: a `CreativeDirector` that writes the first draft from a product description, and a `PosterMaker` that generates the poster image from the approved copy.

### Step 6 — Creative Director and Image Generator

`CreativeDirector` reads `msg.product_description` and writes the first draft to `msg.ad_text`. `PosterMaker` is a `nn.MediaMaker` — it calls the image model with `msg.poster_prompt` and writes the raw bytes to `msg.poster`. `build_poster_prompt` is the Inline step that transforms the final ad text into an image prompt.

```python
class CreativeDirector(nn.Agent):
    """Writes the first ad draft from a product description."""

    model = model
    instructions = """
    You are a creative director at an ad agency. Given a product description,
    write compelling ad copy (3-5 sentences). Include:
    - A catchy headline
    - Key benefits
    - A call to action

    Return only the ad text.
    """
    message_fields = {"task": "product_description"}
    response_mode = "ad_text"


class PosterMaker(nn.MediaMaker):
    """Generates a poster image from the final ad text."""

    model = mf.Model.text_to_image("openai/gpt-image-1")
    message_fields = {"task": "poster_prompt"}
    response_mode = "poster"
    config = {"size": "1536x1024", "quality": "high"}


def build_poster_prompt(msg):
    msg.poster_prompt = (
        f"Professional product advertisement poster. Clean, modern design. "
        f"The ad copy reads: {msg.ad_text}. "
        f"Minimalist layout, premium feel, warm lighting, brand-quality photography."
    )
```

### Step 7 — Full Pipeline

The complete expression reads left to right: the director writes the first draft, the panel evaluates, the loop refines, then the image is generated.

```python
full_pipeline = Inline(
    "director"
    " -> [teenager, professional, budget] -> scorer"
    " -> @{avg_score < 8 & iteration < 3}:"
    " prepare -> refiner -> [teenager, professional, budget] -> scorer;"
    " -> poster_prompt -> poster_maker",
    {
        "director":     CreativeDirector(),
        "teenager":     Teenager(),
        "professional": Professional(),
        "budget":       BudgetShopper(),
        "refiner":      Refiner(),
        "scorer":       compute_score,
        "prepare":      prepare_refinement,
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
print(f"\nFinal ad:\n{msg.ad_text}")

with open("poster.png", "wb") as f:
    f.write(msg.poster)

print("\nPoster saved to poster.png")
```

---

## Examples

???+ example

    === "Level 1 — Evaluate only"

        ```python
        evaluate = Inline(
            "[teenager, professional, budget]",
            {
                "teenager": Teenager(),
                "professional": Professional(),
                "budget": BudgetShopper(),
            },
        )

        msg = Message()
        msg.ad_text = """
        BrewBot: the coffee maker that knows you.
        Personalized coffee, no buttons, no hassle. Free shipping today.
        """

        evaluate(msg)

        for persona, field in [
            ("Teenager", "eval_teenager"),
            ("Professional", "eval_professional"),
            ("Budget Shopper", "eval_budget"),
        ]:
            ev = msg.get(field)
            print(f"{persona}: {ev['opinion']} (score: {ev['score']})")
        ```

    === "Level 2 — Refine until approved"

        ```python
        pipeline = Inline(
            "[teenager, professional, budget] -> scorer"
            " -> @{avg_score < 8 & iteration < 3}:"
            " prepare -> refiner -> [teenager, professional, budget] -> scorer;",
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
        msg.ad_text = "BrewBot. Smart coffee. Order now."

        pipeline(msg)

        print(f"Iterations: {msg.iteration}  |  Final score: {msg.avg_score:.1f}")
        print(f"\nFinal ad:\n{msg.ad_text}")
        ```

    === "Level 3 — Full pipeline"

        ```python
        msg = Message()
        msg.product_description = """
        BrewBot is a compact, Wi-Fi-connected espresso machine for home baristas.
        It syncs with a mobile app, supports 12 brew profiles, and uses any ground
        coffee or pods. Price: $199. Launch discount: 25% off, free shipping.
        """

        full_pipeline(msg)

        print(f"Final score: {msg.avg_score:.1f}")
        print(f"Final ad:\n{msg.ad_text}")

        with open("brewbot_poster.png", "wb") as f:
            f.write(msg.poster)
        ```

---

## Extending

### Adding a new persona

Define a new `Agent` with the same `AdEvaluation` signature and add it to every Inline dictionary that includes the evaluation group:

```python
class SeniorCitizen(nn.Agent):
    model = model
    system_message = """You are a 68-year-old retiree. You value simplicity,
    reliability, and good support. Focus on whether the ad is easy to understand
    and whether the product seems trustworthy."""
    signature = AdEvaluation
    message_fields = {"task": {"ad_text": "ad_text"}}
    response_mode = "eval_senior"
    config = {"verbose": True}
```

Then add `"senior": SeniorCitizen()` to the Inline registry and include `senior` in the `[...]` group. Update `compute_score` to read from `eval_senior`.

### Raising the quality bar

Change the loop condition to require a higher average score or more iterations:

```python
" -> @{avg_score < 9 & iteration < 5}: ..."
```

### Logging iteration history

Capture each iteration's ad text and scores before the Refiner overwrites them:

```python
def save_iteration(msg):
    history = msg.get("history", [])
    history.append({
        "iteration": msg.iteration,
        "ad_text": msg.ad_text,
        "avg_score": msg.avg_score,
    })
    msg.history = history
```

Add `"save": save_iteration` to the registry and insert `save ->` before `prepare` in the loop body.

---

## Complete Script

??? example "Expand full script"
    ```python
    # /// script
    # dependencies = []
    # ///

    import msgflux as mf
    import msgflux.nn as nn

    mf.load_dotenv()


    model = mf.Model.chat_completion("openai/gpt-4.1-mini")



    class AdEvaluation(mf.Signature):
        """Evaluate the advertisement from your perspective. Consider the tone,
        clarity, and appeal. Be honest and specific in your opinion."""

        ad_text: str = mf.InputField(desc="The advertisement text to evaluate")

        opinion: str = mf.OutputField(desc="Your honest reaction to the ad in 2-3 sentences")
        score: int = mf.OutputField(desc="Overall score from 1 (terrible) to 10 (perfect)")



    class Teenager(nn.Agent):
        model = model
        system_message = """You are a 17-year-old social media native. You care about
        aesthetics, trends, memes, and authenticity. Focus on whether the ad feels
        authentic or corporate and whether you would share it on social media."""
        signature = AdEvaluation
        message_fields = {"task": {"ad_text": "ad_text"}}
        response_mode = "eval_teenager"


    class Professional(nn.Agent):
        model = model
        system_message = """You are a 35-year-old working professional. You value clarity,
        time-saving, and quality. Focus on whether the value proposition is clear
        and whether the ad respects your time."""
        signature = AdEvaluation
        message_fields = {"task": {"ad_text": "ad_text"}}
        response_mode = "eval_professional"


    class BudgetShopper(nn.Agent):
        model = model
        system_message = """You are a budget-conscious parent. You look for deals,
        compare prices, and distrust hype. Focus on whether the ad mentions price
        or value and whether it feels honest or manipulative."""
        signature = AdEvaluation
        message_fields = {"task": {"ad_text": "ad_text"}}
        response_mode = "eval_budget"



    class CreativeDirector(nn.Agent):
        """Writes the first ad draft from a product description."""

        model = model
        instructions = """
        You are a creative director at an ad agency. Given a product description,
        write compelling ad copy (3-5 sentences). Include:
        - A catchy headline
        - Key benefits
        - A call to action

        Return only the ad text.
        """
        message_fields = {"task": "product_description"}
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
        message_fields = {"task": "refinement_input"}
        response_mode = "ad_text"



    def compute_score(msg):
        scores = [
            msg.eval_teenager.get("score", 5),
            msg.eval_professional.get("score", 5),
            msg.eval_budget.get("score", 5),
        ]
        msg.avg_score = sum(scores) / len(scores)
        msg.iteration = msg.get("iteration", 0) + 1


    def prepare_refinement(msg):
        msg.refinement_input = (
            f"Current ad:\n{msg.ad_text}\n\n"
            f"Teenager feedback: {msg.eval_teenager.opinion} "
            f"(score: {msg.eval_teenager.score})\n\n"
            f"Professional feedback: {msg.eval_professional.opinion} "
            f"(score: {msg.eval_professional.score})\n\n"
            f"Budget shopper feedback: {msg.eval_budget.opinion} "
            f"(score: {msg.eval_budget.score})\n\n"
            "Rewrite the ad addressing this feedback."
        )



    pipeline = mf.Inline(
        "director"
        " -> [teenager, professional, budget] -> scorer"
        " -> @{avg_score < 8 & iteration < 3}:"
        " prepare -> refiner -> [teenager, professional, budget] -> scorer;",
        {
            "director":     CreativeDirector(),
            "teenager":     Teenager(),
            "professional": Professional(),
            "budget":       BudgetShopper(),
            "refiner":      Refiner(),
            "scorer":       compute_score,
            "prepare":      prepare_refinement,
        },
    )


    msg = Message()
    msg.product_description = """
    CloudBrew is a Wi-Fi-enabled coffee maker with a built-in taste profile system.
    It learns your preferences over time and adjusts brew strength, temperature,
    and grind size automatically. Compatible with any ground coffee or pods.
    Retail price: $149. Launch promotion: 40% off pre-orders with free shipping.
    """

    pipeline(msg)

    print("\n" + "=" * 60)
    print("FOCUS GROUP RESULTS")
    print("=" * 60)
    print(f"\nIterations: {msg.iteration}  |  Final score: {msg.avg_score:.1f}/10")
    print(f"\nFinal ad:\n{msg.ad_text}")
    print("\n--- Evaluations ---")
    for label, field in [
        ("Teenager",       "eval_teenager"),
        ("Professional",   "eval_professional"),
        ("Budget Shopper", "eval_budget"),
    ]:
        ev = msg.get(field)
        print(f"\n{label} ({ev['score']}/10): {ev['opinion']}")
    ```

## Further Reading

- [Inline DSL](../learn/inline.md) — pipeline syntax, parallel steps, and while loops
- [nn.Agent](../learn/nn/agent/index.md) — `message_fields`, `response_mode`, and structured output
- [Signatures](../learn/nn/agent/signatures.md) — typed input/output contracts
- [nn.MediaMaker](../learn/nn/mediamaker.md) — image and video generation
