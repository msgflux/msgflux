# LLM-as-a-Verifier

`src/msgflux/generation/verifiers/llm_as_a_verifier.py` is the first verifier
module in msgFlux.

At a high level, it turns a verification task into a reusable runtime with four
main jobs:

- build a verifier prompt from task, criteria, candidates, and context
- request discrete score tokens from the model
- estimate a stable score from `logprobs` instead of trusting only the chosen
  token
- aggregate repeated attempts and multiple criteria into a final verdict

It is not tied to `Agent`. It is a callable verification layer that can be used
from plain generation code, tool workflows, candidate reranking, or future
optimizers.

## Mental Model

The verifier pipeline looks like this:

```text
task + candidates + criteria
  -> prompt builder
  -> request batch
  -> concurrent model calls
  -> per-attempt evidence extraction
     -> text token
     -> logprob distribution
  -> criterion aggregation
  -> final verifier result
```

The important part is that the verifier does not ask the model for an open-ended
score. It asks for a small discrete token from a known scale, then uses the
token distribution to estimate the final normalized score.

## Why The Output Is A Score Token

The verifier uses a discrete score scale such as `A..T`.

That choice is deliberate:

- it keeps the scoring space bounded
- it makes the provider response easier to parse
- it makes `logprobs` directly useful
- it lets the runtime compute an expected score instead of a brittle hard label

So the model is not asked to emit a floating-point score. It is asked to emit a
single token inside a known tag:

```text
<score>A</score>
```

or, in pairwise mode:

```text
<score_A>A</score_A>
<score_B>H</score_B>
```

## Prompt Construction

`default_prompt_builder(...)` assembles the verifier prompt from
`VerificationPromptInput`.

That input contains:

- `task`
- `criterion`
- `candidates`
- `score_scale`
- optional `context`
- optional verifier-level instructions

The default builder always explains:

- the criterion being judged
- the valid score-token range
- the exact tag format expected at the end

This is why the verifier can support custom prompting without changing the rest
of the execution pipeline. The builder is only responsible for prompt text; the
runtime remains responsible for scoring and aggregation.

## Request Batching And Concurrency

The verifier builds a `VerificationRequest` for each independent
criterion/repetition pair.

That means the number of model calls in a single verification is:

```text
len(criteria) * n_verifications
```

Those requests are independent, so the verifier fans them out concurrently:

- `__call__` uses `F.map_gather(...)`
- `acall` uses `F.amap_gather(...)`

This matters because a verifier with multiple criteria and repeated attempts can
otherwise accumulate latency linearly.

The runtime still keeps the final result deterministic:

- requests are built in criterion order, then repetition order
- gather responses preserve input order
- attempts are regrouped per criterion after execution

So the work is concurrent, but the aggregation order remains stable.

## Why `select_best(...)` Is Separate

`__call__` and `acall` support only one or two candidates.

That boundary is intentional:

- one candidate -> verification
- two candidates -> pairwise comparison
- more than two candidates -> tournament selection

`select_best(...)` and `aselect_best(...)` implement the tournament layer on top
of the base verifier. They run pairwise matches, accumulate wins, and use
average score as a tiebreaker.

The tournament layer can also fan out its pairwise matches concurrently. That
means a multi-candidate selection now has two concurrency layers:

- concurrent attempts inside each verifier call
- concurrent pairwise matches across the tournament

This keeps the core verifier simple instead of making `__call__` return
different shapes for different candidate counts.

## Scoring From Logprobs

The central scoring path is:

```text
model response
  -> locate score token
  -> collect chosen token + top_logprobs alternatives
  -> convert logprobs to probabilities
  -> compute expected value on the discrete score scale
  -> normalize to [0, 1]
```

This is why `LLMAsVerifier` always requests:

- `logprobs=True`
- `top_logprobs=<scale size or override>`

If the provider returns a good token distribution, the verifier uses that
distribution directly.

If not, and `strict_logprobs=False`, it falls back to parsing the score from the
response text.

## Why The Parser Has Heuristics

Provider tokenization around XML-like tags is not always clean.

The verifier therefore uses a layered extraction strategy:

1. parse the text tag if possible
2. try to find the matching score entry directly in `logprobs`
3. fall back to finding the expected score token near the closing tag
4. if strict mode is disabled, fall back to text-only extraction

This is also why the parser tolerates malformed openings such as:

```text
<20A</score>
```

The runtime treats the text and token stream as evidence that must be reconciled,
not as perfectly formatted output.

## Aggregation Model

There are two aggregation stages.

### Attempt Aggregation

Each criterion can run multiple repetitions.

Those repeated attempts are averaged into a single score per criterion and
candidate.

### Criterion Aggregation

Each criterion carries a weight.

The final score per candidate is the weighted average of all criterion scores.

That is what makes the verifier reusable across different presets: the
execution path stays the same while the criteria set changes.

## Verdict Semantics

The final verdict depends on candidate count.

For a single candidate:

- `pass`
- `fail`
- `uncertain`

For two candidates:

- winner label
- `tie`

This is why `VerifierResult` exposes both:

- `verdict`
- optional `winner`

## Verbose Mode

When `verbose=True`, the verifier stores enough information to debug prompt and
scoring behavior without rerunning the call.

Per attempt, it keeps:

- final prompt text
- raw response text
- per-candidate evidence
- extracted token probabilities

That data is exposed both in:

- `criteria_results[*].attempts[*]`
- `result.metadata["raw_outputs"]`

Tournament mode mirrors that same information in each match entry of
`TournamentResult.metadata["raw_outputs"]`.

## Presets

The preset constructors are thin wrappers around the same runtime:

- `trajectory_analysis`
- `answer_reranking`
- `grounded_answer_verification`
- `patch_selection`
- `tool_trace_verification`
- `synthetic_data_filtering`

Each preset only supplies:

- a default criteria set
- optional default instructions or ground-truth notes

The runtime path stays identical.

This is important for maintenance because it means new verifier presets should
usually be data/config additions, not new execution engines.

## Main Design Choices

The current design makes a few explicit tradeoffs:

- verifier core is independent from `Agent`
- prompt building is pluggable, scoring is centralized
- discrete score tokens are preferred over free-form numeric scores
- `logprobs` are first-class evidence
- tournament selection is built on top of pairwise verification
- verbose debugging is part of the runtime, not an external wrapper

These choices make the verifier easier to reuse for reranking, trajectory
selection, patch selection, and future optimizer integrations.
