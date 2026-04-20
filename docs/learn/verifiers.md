# Verifiers

## LLM-as-a-Verifier

`LLMAsVerifier` is a logprob-aware verification pattern adapted from
[LLM-as-a-Verifier](https://llm-as-a-verifier.notion.site/).

For the internal execution model and design choices, see
[Anatomy: LLM-as-a-Verifier](../anatomy/llm-as-a-verifier.md).

- it decomposes evaluation into explicit criteria
- it can repeat the verification multiple times
- it asks the model to emit a discrete score token
- it computes the final score from the expected value of the score-token
  distribution, instead of trusting only the chosen token

Use it when you want a reusable verifier that can score, compare, and select
model outputs with a stable signal.

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

## Verbose Debugging

Set `verbose=True` when you want the verifier to return the final prompt and raw
model output for each attempt.

```python
verifier = LLMAsVerifier(
    model="openai/gpt-4.1-mini",
    criteria=[criterion],
    verbose=True,
)

result = verifier(
    task="What is 2 + 2?",
    candidates={"answer": "The answer is 4."},
)

print(result.metadata["raw_outputs"][0]["prompt"])
print(result.metadata["raw_outputs"][0]["response_text"])
```

The same data also remains available in `criteria_results[*].attempts[*]`:

```python
attempt = result.criteria_results[0].attempts[0]
print(attempt.prompt_text)
print(attempt.response_text)
```

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

terminal = LLMAsVerifier.terminal_bench(
    model="openai/gpt-4.1-mini",
)

swe = LLMAsVerifier.swe_bench_verified(
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

### `terminal_bench`

Use for terminal-task trajectory selection in the style of Terminal-Bench.

- specification adherence
- output match
- unresolved error signals
- terminal output treated as primary ground truth

### `swe_bench_verified`

Use for patch and trajectory evaluation in the style of SWE-bench Verified.

- root-cause analysis
- code review quality
- empirical verification from executed commands
- narration treated as weaker evidence than the actual patch and outputs

## Score Scale And Granularity

The default verifier already uses a 20-level letter scale (`A..T`). If you want
to set that explicitly, or choose a different number of score levels, pass
`score_scale=ScoreScale.letter(granularity=...)`.

```python
from msgflux.generation.verifiers import LLMAsVerifier, ScoreScale

verifier = LLMAsVerifier.answer_reranking(
    model="openai/gpt-4.1-mini",
    score_scale=ScoreScale.letter(granularity=20),
)
```

- `granularity=20` gives the usual `A..T` scale
- supported range is `2..26`
- `top_logprobs` follows the scale size automatically unless you override it

Use `ScoreScale.letter(granularity=20)` to request a 20-level score scale.
The runtime keeps the scale letter-based because letter tokens are more stable
for `logprobs` extraction than numeric labels such as `1..20`.

## Trajectory Formatting Helpers

For benchmark-style presets, pass the full trajectory as the candidate
evidence, not only the final answer. msgFlux ships two helpers for that:

- `format_terminal_trajectory(...)`
- `format_swe_bench_trajectory(...)`

Keep the task or issue description in `task`. Use the helpers to build each
candidate string from commands, outputs, patch text, and final answer.

```python
from msgflux.generation.verifiers import (
    LLMAsVerifier,
    format_terminal_trajectory,
)

candidate = format_terminal_trajectory(
    summary="Installed the binary and verified the version output.",
    metadata={"expected_output": "tool 1.2.0"},
    steps=[
        {
            "command": "cp ./tool /usr/local/bin/tool",
            "exit_code": 0,
        },
        {
            "command": "tool --version",
            "output": "tool 1.2.0",
            "exit_code": 0,
        },
    ],
    final_answer="Installation complete.",
)

verifier = LLMAsVerifier.terminal_bench(model="openai/gpt-4.1-mini")
result = verifier(
    task="Install the binary to /usr/local/bin/tool and verify `tool --version`.",
    candidates={"run_a": candidate},
)
```

```python
from msgflux.generation.verifiers import (
    LLMAsVerifier,
    format_swe_bench_trajectory,
)

candidate = format_swe_bench_trajectory(
    summary="Reproduced the bug, patched the parser, and reran the focused test.",
    steps=[
        {
            "command": "pytest tests/test_parser.py -q",
            "output": "1 failed, 4 passed",
            "exit_code": 1,
        },
        {
            "command": "pytest tests/test_parser.py -q",
            "output": "5 passed",
            "exit_code": 0,
        },
    ],
    patch=patch_text,
    final_answer="Patched parser and verified the focused test.",
)

verifier = LLMAsVerifier.swe_bench_verified(model="openai/gpt-4.1-mini")
result = verifier(
    task="Fix the parser bug described in the issue.",
    candidates={"candidate_patch": candidate},
)
```

## Best Use Cases

`LLMAsVerifier` is most useful when you need to compare or filter candidates,
not when the task already has a deterministic validator.

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

Prefer deterministic validation when the task already has a deterministic check.

- exact-match tasks
- schema validation
- unit tests
- regex-based extraction checks
- simple business rules
