# Evaluating and comparing programs

`Evaluate` scores a program on a dataset you keep in one place. Construct it once with the dataset and the metric, then call it for the baseline, the compiled program and any later revision. Because every call reuses the same examples, scores are comparable across programs.

The evaluator is part of the base package. It needs no extra dependency and does not import the GEPA search backend.

## Evaluate one program

```python
import msgflux as mf
from msgflux import nn
from msgflux.models.response import ModelResponse
from msgflux.optim import Evaluate


class TriageModel:
    """A deterministic chat model so this example runs without API keys."""

    model_type = "chat_completion"

    def __call__(self, *, messages, system_prompt, **_kwargs):
        ticket = messages[-1]["content"].lower()
        queue = "general"
        if "invoice" in ticket and "invoice" in system_prompt:
            queue = "billing"
        if "password" in ticket and "password" in system_prompt:
            queue = "access"
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add(queue)
        return response


def build_agent(instruction):
    return nn.Agent(name="triage", model=TriageModel(), system_prompt=instruction)


def queue_accuracy(example, prediction):
    return float(prediction == example.labels), {
        "feedback": f"Expected {example.labels}, predicted {prediction}."
    }


holdout = [
    mf.Example("Invoice is missing", "billing"),
    mf.Example("Reset my password", "access"),
    mf.Example("Where is my badge?", "general"),
]

evaluator = Evaluate(dataset=holdout, metric=queue_accuracy)
report = evaluator(build_agent("Route invoice tickets to billing."))

print(f"{report.mean_score:.2f}")
print([row.score for row in report.rows])
print(report.rows[1].feedback)
```

Run the script to see the accuracy, the per-example scores and the metric's feedback:

```text
0.67
[1.0, 0.0, 1.0]
{'feedback': 'Expected access, predicted general.'}
```

The model routes invoice tickets once the instruction mentions `invoice`, but password tickets still fall back to `general`. The second example therefore scores zero and carries the diagnostic that explains why.

`Evaluate.__call__` returns an `EvaluationResult`:

| Attribute | Description |
|---|---|
| `rows` | One `EvaluationRow` per example, in dataset order |
| `mean_score` | Arithmetic mean of the row scores |
| `scores` | Row scores as a tuple |
| `predictions` | Row predictions as a tuple |
| `mean_scores` | Mean of each named metric, or `{}` for a plain metric |
| `primary_metric` | Name of the optimizing metric, or `None` |
| `for_metric(name)` | The same rows projected onto one named metric |

Each `EvaluationRow` exposes `example`, the `prediction` the program produced, the numeric `score` and the `feedback` dictionary returned by the metric. A metric that returns a bare number produces an empty feedback dictionary. Metrics must return a finite score; `nan` and `inf` raise `ValueError`. `row.metric_values` holds the score and feedback of every named metric when the evaluator was given a `MetricSet`, and is empty otherwise.

## Compare a baseline with a candidate

`evaluate()` still evaluates one program at a time and keeps its original signature. Use `comparison = evaluator.compare(baseline, candidate)` to evaluate both programs on the same examples and pair the rows. Continue in the same script, after the definitions above:

```python
comparison = evaluator.compare(
    build_agent("Route the ticket."),
    build_agent("Route invoice tickets to billing and password tickets to access."),
)

print(
    f"{comparison.baseline_mean_score:.2f} -> {comparison.candidate_mean_score:.2f}"
)
for row in comparison.rows:
    print(f"{row.example.inputs}: {row.outcome} ({row.delta:+.2f})")
print(f"Mean delta: {comparison.mean_delta:+.2f}")
```

The call returns a `ComparisonResult` whose rows hold the pre- and post-optimization evaluation of each example. The output is deterministic:

```text
0.33 -> 1.00
Invoice is missing: improved (+1.00)
Reset my password: improved (+1.00)
Where is my badge?: unchanged (+0.00)
Mean delta: +0.67
```

The written instruction fixes both failing queues, so two examples improve and the third was already correct. A paired report shows that kind of per-example movement, which a pair of means hides.

`ComparisonRow` provides:

| Attribute | Description |
|---|---|
| `baseline` | The `EvaluationRow` of the baseline program |
| `candidate` | The `EvaluationRow` of the candidate program |
| `example` | The example shared by both rows |
| `delta` | `candidate.score - baseline.score` |
| `outcome` | `"improved"`, `"regressed"` or `"unchanged"` |

`ComparisonResult` aggregates the pairs:

| Attribute | Description |
|---|---|
| `rows` | One `ComparisonRow` per example, in dataset order |
| `baseline` / `candidate` | The two `EvaluationResult` objects |
| `baseline_mean_score` / `candidate_mean_score` | Mean score of each program |
| `deltas` | Per-example deltas as a tuple |
| `mean_delta` | Mean of the per-example deltas |
| `improved` / `regressed` / `unchanged` | Rows grouped by `outcome` |
| `primary_metric` | Name of the optimizing metric, or `None` |
| `for_metric(name)` | A scalar `ComparisonResult` built from one named metric |

!!! note "Pairing uses inputs and labels"
    `Example` is a mutable dataclass, and its generated equality also compares
    derived transform flags. Two examples built from the same inputs and labels
    can therefore compare unequal. `ComparisonResult.pair()` compares the
    `inputs` and `labels` of each row instead, through
    `msgflux.optim.evaluation.example_identity()`, so independently built
    examples still pair.

## Optimize one objective and track other metrics

A better router may send tickets to specialists who spend more time on them. Use routing accuracy to select a prompt and a second metric to track estimated handling time. This example reuses the agents, held-out tickets and accuracy metric defined above.

```python
from msgflux.optim import Evaluate, MetricSet

minutes_by_queue = {"general": 1, "access": 2, "billing": 5}


def efficiency(example, prediction):
    minutes = minutes_by_queue[prediction]
    return 1 - minutes / 10, {"estimated_minutes": minutes}


metrics = MetricSet(
    {"accuracy": queue_accuracy, "efficiency": efficiency}, primary="accuracy"
)
evaluator = Evaluate(dataset=holdout, metric=metrics)
comparison = evaluator.compare(
    build_agent("Route the ticket."),
    build_agent("Route invoice tickets to billing and password tickets to access."),
)
for name in metrics.metrics:
    view = comparison.for_metric(name)
    print(name, round(view.baseline_mean_score, 2), round(view.candidate_mean_score, 2),
          len(view.improved), len(view.regressed))
```

```text
accuracy 0.33 1.0 2 0
efficiency 0.9 0.73 0 2
```

Two tickets reached the correct specialist and took longer under the example's time estimates. Inspect `comparison.for_metric("efficiency").regressed` for those tickets and their feedback. Higher scores indicate better outcomes, so the efficiency score decreases with the estimated time.

Pass the same `MetricSet` as `metric=` to `GEPA.compile`. During search, GEPA calls only the primary metric and sends its feedback to the reflector. During evaluation, each program runs once per example and all metrics score that prediction. `result.mean_scores` gives the mean for each name. `result.for_metric(name)` and `comparison.for_metric(name)` reuse stored predictions and scores. Pairing independently evaluated results requires the same names and primary objective.

## Reuse one evaluator

The constructor arguments are optional so a single instance can serve either role:

```python
import msgflux as mf
from msgflux.optim import Evaluate

holdout = [mf.Example("Invoice is missing", "billing")]


def constant(example, prediction):
    return 1.0


# Dataset stored, metric supplied per call.
report = Evaluate(dataset=holdout)(str.lower, metric=constant)

# Metric stored, dataset supplied per call.
report = Evaluate(metric=constant)(str.lower, dataset=holdout)

# Both stored, so the call only names the program.
evaluator = Evaluate(dataset=holdout, metric=constant)
report = evaluator(str.lower)
print(len(evaluator.dataset), report.mean_score)
```

```text
1 1.0
```

`dataset` is copied into a tuple at construction time, so later mutations of the caller's list do not change the evaluation. `evaluator.dataset` and `evaluator.metric` expose the stored values; both are `None` when omitted. Calling the evaluator without either a stored or a supplied dataset or metric raises `ValueError`. A dataset that is not a sequence, an empty dataset, a non-callable metric and a non-callable program raise an error before any example is scored.

## How examples reach a program

`Evaluate` and `evaluate()` use the same call contract as the optimizers. A string in `Example.inputs` is passed positionally, and a mapping is expanded into keyword arguments. A returned `ModelResponse` is consumed, so the metric receives the response value rather than the response object. Anything else is passed through unchanged.

```python
import msgflux as mf
from msgflux.optim import Evaluate

tickets = [
    mf.Example({"ticket": "Invoice is missing"}, "billing"),
    mf.Example({"ticket": "Reset my password"}, "access"),
]

report = Evaluate(dataset=tickets, metric=lambda ex, p: float(p == ex.labels))(
    lambda ticket: "access" if "password" in ticket else "general"
)
print(len(report.rows), report.predictions)
```

```text
2 ('general', 'access')
```

Because the evaluator calls the supplied program once per example, pass an object that is safe to reuse. When a module keeps conversation or tool state across inputs, pass a callable that builds a fresh instance per call, or call the evaluator several times with separate programs.

Comparing a compiled program only needs the artifact saved from an optimizer. This snippet reuses `build_agent`, `holdout` and `queue_accuracy` from the first section and expects the `CompilationResult.save("triage.optim.json")` file from [Optimizers and evaluation](gepa.md):

```python
from msgflux.optim import Evaluate, OptimizationArtifact

artifact = OptimizationArtifact.load("triage.optim.json")


def compiled():
    program = build_agent("Route the ticket.")
    artifact.apply(program)
    return program


comparison = Evaluate(dataset=holdout, metric=queue_accuracy).compare(
    build_agent("Route the ticket."), compiled()
)
```

Both programs must run with the same models and inference settings for the deltas to describe the instruction change alone.

## Where it fits

`Evaluate` reports what a metric measured; it does not search. Use it after `GEPA.compile()` or `optimize_text()` to compare the selected program with the initial one, as described in [Optimizers and evaluation](gepa.md). A tracker or experiment log can consume `comparison.rows` directly, since each row carries the example, both scores and both feedback dictionaries.
