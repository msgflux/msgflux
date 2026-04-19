# Verifiers

## LLM-as-a-Verifier

`LLMAsVerifier` is a logprob-aware verification pattern adapted from
[LLM-as-a-Verifier](https://llm-as-a-verifier.notion.site/).

- it decomposes evaluation into explicit criteria
- it can repeat the verification multiple times
- it asks the model to emit a discrete score token
- it computes the final score from the expected value of the score-token
  distribution, instead of trusting only the chosen token

Use it when you want a reusable verifier that can score, compare, and select
model outputs with a stable signal.

## Best Use Cases

`LLMAsVerifier` is most useful when you need to compare or filter candidates,
not when you already have a cheap deterministic validator.

### Trajectory Selection

This is the most natural fit for the technique.

- compare alternative reasoning trajectories
- compare multiple agent runs for the same task
- select the strongest final trajectory before returning it

### Candidate Reranking

Use it to rerank multiple drafts of the same task.

- final answers
- summaries
- plans
- retrieval-grounded responses

### Patch Selection

It works well for code generation when you want to compare multiple candidate
patches before running heavier validation.

- choose the patch that best satisfies the task
- prefer the patch that looks more correct or complete
- filter obviously weak candidates before expensive execution

### Tool-Using Agent Verification

It is useful for checking whether a final answer is consistent with the tool
results and the execution trace.

- verify completion quality
- verify grounding in tool outputs
- detect unresolved errors hidden by a confident final answer

### Synthetic Data Filtering

Use it to filter generated examples before storing them in datasets, evals, or
distillation corpora.

- reject inconsistent examples
- reject weak labels
- keep only high-confidence candidates

### Optimizer Feedback

This is a strong fit for future optimizer integrations.

- provide a reusable reward signal
- score prompt variants
- compare sampled candidates during search or optimization

## When Not to Use

Prefer a cheaper validator when the task already has a deterministic check.

- exact-match tasks
- schema validation
- unit tests
- regex-based extraction checks
- simple business rules

## Built-In Presets

`LLMAsVerifier` ships with preset constructors for common evaluation tasks. Each
preset returns a regular `LLMAsVerifier`, so you can still override `criteria`,
`ground_truth_note`, `extra_instructions`, `n_verifications`, and the model
request kwargs.

```python
from msgflux.generation.verifiers import LLMAsVerifier

trajectory = LLMAsVerifier.trajectory_analysis(
    model="openai/gpt-4.1-mini",
)

reranker = LLMAsVerifier.answer_reranking(
    model="openai/gpt-4.1-mini",
)

grounded = LLMAsVerifier.grounded_answer_verification(
    model="openai/gpt-4.1-mini",
)

patches = LLMAsVerifier.patch_selection(
    model="openai/gpt-4.1-mini",
)

tools = LLMAsVerifier.tool_trace_verification(
    model="openai/gpt-4.1-mini",
)

filtering = LLMAsVerifier.synthetic_data_filtering(
    model="openai/gpt-4.1-mini",
)
```

### `trajectory_analysis`

Use for agent runs and reasoning trajectories.

- checks whether the task was actually completed
- checks whether verification was meaningful
- checks unresolved error signals

### `answer_reranking`

Use for comparing multiple final drafts of the same task.

- correctness
- instruction following
- completeness
- clarity

### `grounded_answer_verification`

Use for RAG and other context-grounded tasks.

- grounding in context
- unsupported claims
- answer completeness

### `patch_selection`

Use for comparing candidate patches or code changes.

- requirement coverage
- correctness risk
- regression risk
- minimality

### `tool_trace_verification`

Use for tool-using agents when you want to compare the final answer against the
trace and tool outputs.

- tool grounding
- unresolved errors
- final answer quality
- action efficiency

### `synthetic_data_filtering`

Use for generated examples before adding them to datasets, evals, or
distillation corpora.

- consistency
- label quality
- ambiguity
- usefulness

## Single Candidate

```python
# pip install msgflux[openai]
import msgflux as mf
from msgflux.generation.verifiers import LLMAsVerifier, VerificationCriterion

# mf.load_dotenv()

criterion = VerificationCriterion(
    id="correctness",
    name="Correctness",
    description="Assess whether the candidate fully answers the task.",
)

verifier = LLMAsVerifier(
    model="openai/gpt-4.1-mini",
    criteria=[criterion],
    n_verifications=2,
)

result = verifier(
    task="What is 2 + 2?",
    candidates={"answer": "The answer is 4."},
)

print(result.verdict)  # pass
print(result.score)    # normalized score in [0, 1]
print(result.scores)   # {"answer": ...}
```

`LLMAsVerifier` requests `logprobs=True` and `top_logprobs` automatically on the
model call. If the provider returns token logprobs, the verifier uses them to
compute the score distribution. If not, it falls back to parsing the emitted
score from text. Set `strict_logprobs=True` to require logprob-based extraction.

## Pairwise Comparison

```python
result = verifier(
    task="Pick the better final answer.",
    candidates={
        "paris_answer": "The capital of France is Paris.",
        "lyon_answer": "The capital of France is Lyon.",
    },
)

print(result.verdict)  # "paris_answer", "lyon_answer", or "tie"
print(result.winner)   # winner label or None
print(result.scores)   # {"paris_answer": ..., "lyon_answer": ...}
```

Pairwise mode is the natural building block for trajectory comparison and
candidate selection.

## API Shape

Use `__call__` or `acall` when you want to verify `1` or `2` candidates:

```python
single = verifier(
    task="What is 2 + 2?",
    candidates={"answer": "The answer is 4."},
)

pairwise = verifier(
    task="Pick the better final answer.",
    candidates={
        "draft_1": "The answer is 4.",
        "draft_2": "The answer is 5.",
    },
)
```

Use `select_best` or `aselect_best` when you have more than `2` candidates:

```python
tournament = verifier.select_best(
    task="Pick the best final answer.",
    candidates={
        "draft_1": "The answer is 4.",
        "draft_2": "It is probably 4.",
        "draft_3": "The answer is 5.",
    },
)
```

## Round-Robin Selection

For tasks with multiple candidates, use `select_best`:

```python
tournament = verifier.select_best(
    task="Pick the best final answer.",
    candidates={
        "draft_1": "Answer one",
        "draft_2": "Answer two",
        "draft_3": "Answer three",
    },
)

print(tournament.winner)
print(tournament.ranking)
print(tournament.wins)
```

This runs pairwise comparisons across all candidates and selects the winner by
round-robin wins, using average verifier score as a tiebreaker.

## Custom Prompting

You can replace the default prompt with `prompt_builder`. The builder receives a
`VerificationPromptInput` containing the task, candidates, criterion,
score scale, context, and optional verifier instructions.

```python
from msgflux.generation.verifiers import (
    LLMAsVerifier,
    VerificationCriterion,
    VerificationPromptInput,
)


def build_prompt(data: VerificationPromptInput) -> str:
    label, candidate = next(iter(data.candidates.items()))
    return (
        "Evaluate the answer.\n\n"
        f"Task:\n{data.task}\n\n"
        f"Criterion:\n{data.criterion.description}\n\n"
        f"Candidate ({label}):\n{candidate}\n\n"
        f"<score>{data.score_scale.score_format}</score>"
    )


verifier = LLMAsVerifier(
    model="openai/gpt-4.1-mini",
    criteria=[
        VerificationCriterion(
            id="faithfulness",
            name="Faithfulness",
            description="Check whether the answer is grounded in the context.",
        )
    ],
    prompt_builder=build_prompt,
)
```
