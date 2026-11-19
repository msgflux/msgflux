# Evaluating and comparing programs

`Evaluate` scores a program on a dataset you keep in one place. Construct it once with the dataset and the metric, then call it for the baseline, the compiled program and any later revision. Because every call reuses the same examples, scores are comparable across programs.

The evaluator is part of the base package. It needs no extra dependency and does not import the GEPA search backend.

## Evaluate one program

```python
import msgflux as mf
from msgflux import nn
from msgflux.optim import Evaluate

mf.load_dotenv()
model = mf.Model.chat_completion("openai/gpt-6-luna")


def build_agent(instruction):
    return nn.Agent(name="triage", model=model, system_prompt=instruction)


def queue_accuracy(example, prediction):
    return float(str(prediction).strip().casefold() == example.labels), {
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

Set `OPENAI_API_KEY` before running this example. The script prints the overall accuracy, each ticket's score and the feedback from the second ticket. The expected label is fixed. The live agent's answer can vary.

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

The call returns a `ComparisonResult`. Each row pairs the two responses to one ticket so you can inspect which change improved or regressed a prediction. For a measured comparison on 68 held-out examples with published per-example scores, see [the Facility Support reproduction](facility-support.md).

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

Use routing accuracy to select a prompt and a second metric to measure whether the response follows the one-label output contract. Both metrics inspect the same prediction.

```python
from msgflux.optim import MetricSet

def valid_queue(example, prediction):
    return float(str(prediction).strip().casefold() in {"billing", "access", "general"})

metrics = MetricSet({"accuracy": queue_accuracy, "valid_queue": valid_queue}, primary="accuracy")
evaluator = Evaluate(dataset=holdout, metric=metrics)
comparison = evaluator.compare(
    build_agent("Route the ticket."),
    build_agent("Return only billing for invoices, access for credentials, otherwise general."),
)
for name in metrics.metrics:
    view = comparison.for_metric(name)
    print(name, view.baseline_mean_score, view.candidate_mean_score, len(view.regressed))
```

Pass the same `MetricSet` as `metric=` to `GEPA.compile`. During search, GEPA calls only the primary metric and sends its feedback to the reflector. During evaluation, each program runs once per example and all metrics score that prediction. `result.mean_scores` gives the mean for each name. `result.for_metric(name)` and `comparison.for_metric(name)` reuse stored predictions and scores. Pairing independently evaluated results requires the same names and primary objective.

## Reuse one evaluator

The constructor arguments are optional so a single instance can serve either role:

```python
agent = build_agent("Return only the queue label.")
report = Evaluate(dataset=holdout)(agent, metric=queue_accuracy)
report = Evaluate(metric=queue_accuracy)(agent, dataset=holdout)
report = Evaluate(dataset=holdout, metric=queue_accuracy)(agent)
```

Each form evaluates the agent against the same labeled tickets.

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

agent = build_agent("Return the ticket queue as a single word.")
report = Evaluate(dataset=tickets, metric=queue_accuracy)(lambda ticket: agent(ticket))
print(report.predictions)
```

Because the evaluator calls the supplied program once per example, pass an object that is safe to reuse. When a module keeps conversation or tool state across inputs, pass a callable that builds a fresh instance per call, or call the evaluator several times with separate programs.

## Compare a saved artifact

The tool-description example writes `receipts.optim.json`. From the repository root, reuse its original factory, metric and held-out reference. The source digest in the artifact checks that the instruction and lookup description match the originals before applying the selected texts.

```python
from examples.optimize_tool_description import build_agent as build_receipt_agent
from examples.optimize_tool_description import exact_receipt, receipt
from msgflux.optim import OptimizationArtifact

baseline = build_receipt_agent(model)
candidate = build_receipt_agent(model)
OptimizationArtifact.load("receipts.optim.json").apply(candidate)
comparison = Evaluate(dataset=[receipt("KR17")], metric=exact_receipt).compare(
    baseline, candidate
)
print(comparison.mean_delta)
```

Use the same task model and inference settings to attribute differences to the text update. [Tracking optimization with MLflow](mlflow.md).
