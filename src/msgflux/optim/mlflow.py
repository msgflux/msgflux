"""Log msgFlux search, paired evaluation and deployment artifacts to MLflow."""

import hashlib
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from msgflux.optim.evaluation import ComparisonResult
from msgflux.optim.optimizer import CompilationResult


def _metric_key(name: str) -> str:
    # MetricSet accepts arbitrary names. Keep common names readable and retain
    # a lookup for names that exceed MLflow's character or length limits.
    if re.fullmatch(r"[A-Za-z0-9_-]{1,200}", name) and not name.startswith("encoded_"):
        return name
    return "encoded_" + hashlib.sha256(name.encode()).hexdigest()


def log_optimization(
    fit: CompilationResult,
    comparison: ComparisonResult,
    *,
    dataset_id: str,
    task_model: str,
    reflection_model: str,
) -> None:
    """Record one compilation and its holdout comparison in the active run."""
    try:
        import mlflow  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if exc.name != "mlflow":
            raise
        raise ImportError("Install msgflux[mlflow] to log optimizer results") from exc

    if mlflow.active_run() is None:
        raise RuntimeError("Start an active MLflow run before logging an optimization")

    metrics = {
        "search.baseline": fit.search.baseline_score,
        "search.best": fit.search.best_score,
        "search.metric_calls": fit.search.metric_calls,
        "holdout.baseline": comparison.baseline_mean_score,
        "holdout.candidate": comparison.candidate_mean_score,
        "holdout.delta": comparison.mean_delta,
    }
    aliases = {}
    for name in comparison.baseline.mean_scores:
        key = _metric_key(name)
        if key != name:
            aliases[key] = name
        projected = comparison.for_metric(name)
        metrics[f"holdout.{key}.baseline"] = projected.baseline_mean_score
        metrics[f"holdout.{key}.candidate"] = projected.candidate_mean_score
        metrics[f"holdout.{key}.delta"] = projected.mean_delta

    params = {
        "dataset_id": dataset_id,
        "task_model": task_model,
        "reflection_model": reflection_model,
    }
    if comparison.primary_metric is not None:
        params["primary_metric"] = comparison.primary_metric
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    if aliases:
        mlflow.log_dict(aliases, "metric-names.json")
    with TemporaryDirectory() as directory:
        path = Path(directory) / "optimization.json"
        fit.save(path)
        mlflow.log_artifact(str(path))
