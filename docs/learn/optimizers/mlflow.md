# Tracking optimization with MLflow

Use `msgflux.optim.mlflow.log_optimization` to attach a compiled artifact and its paired holdout evaluation to an active MLflow run. The calling application controls the tracking URI, experiment and run lifetime. Install the `mlflow` extra for tracking and the `optimizers` extra for GEPA.

## Run locally

Set `OPENAI_API_KEY` for Luna, then run this from a checkout:

```bash
MLFLOW_TRACKING_URI=sqlite:///receipts.db uv run --extra optimizers --extra mlflow \
  python examples/optimize_tool_description.py --mlflow --train-usage-guidance
```

The example prints the run ID, writes the selected texts to `receipts.optim.json` and logs a copy in the experiment named `joint-tool-description`. The task model and the reflector both default to Luna. You can use a named OpenCode Go account for inference by adding `--task-model opencode-go/deepseek-v4.1-flash --account work`. Use the same tracking URI when reading the run back. To use a tracking server, set `MLFLOW_TRACKING_URI` to its URL.

## Log a comparison

Given an optimizer result `fit`, its original factory `build_program` and a held-out dataset, run the two programs with the same metric. Logging reuses that evaluation.

```python
import mlflow
from msgflux.optim import Evaluate
from msgflux.optim.mlflow import log_optimization

comparison = Evaluate(dataset=holdout, metric=metric).compare(build_program(), fit.build())
with mlflow.start_run(run_name="receipt-lookup") as run:
    log_optimization(
        fit, comparison, dataset_id="archived-receipts-v1",
        task_model="opencode-go/deepseek-v4.1-flash",
        reflection_model="openai/gpt-6-luna",
    )
    run_id = run.info.run_id
```

| Recorded item | Meaning |
|---|---|
| `search.baseline`, `search.best`, `search.metric_calls` | Initial and selected validation scores and the evaluation count reported by the optimizer |
| `holdout.baseline`, `holdout.candidate`, `holdout.delta` | Mean baseline score, selected score and paired change |
| `holdout.<name>.{baseline,candidate,delta}` | Additional series for each metric in a `MetricSet` |
| Run parameters | Caller-supplied dataset ID, task model and reflection model, plus the primary metric name when using a `MetricSet` |
| `optimization.json` | Deployable v1 single-text or v2 program artifact |
| `metric-names.json` | Mapping from encoded metric keys to original names, written when needed |

## Restore the selected texts

```python
from msgflux.optim import OptimizationArtifact

path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="optimization.json")
program = build_program()
OptimizationArtifact.load(path).apply(program)
```

Reconstruct the same source configuration and target names in the factory. Applying the artifact checks each target's source digest. The program can then serve requests with the selected texts. [Optimizers and evaluation](gepa.md).
