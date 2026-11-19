"""Read back a compiled program and paired evaluation from a real tracking store."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from msgflux import Example, nn
from msgflux.optim import Evaluate, MetricSet, OptimizationArtifact, TextTarget
from msgflux.optim.mlflow import log_optimization
from tests.optim.test_optimizer import Workflow
from tests.optim.test_trainable_targets import SelectText


class StyledWorkflow(Workflow):
    def __init__(self):
        super().__init__()
        self.style = nn.Parameter("Brief", "style")

    def forward(self, question):
        return f"{self.style.data}: {self.router(question)}"


@pytest.fixture
def tracking(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sqlalchemy")
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'tracking.db'}")
    try:
        experiment = mlflow.create_experiment(
            "receipts", (tmp_path / "artifacts").as_uri()
        )
        yield mlflow, experiment
    finally:
        mlflow.set_tracking_uri(previous)


@pytest.mark.parametrize("joint", [False, True], ids=["single-scalar", "joint-named"])
def test_log_and_restore(tracking, joint):
    mlflow, experiment = tracking
    factory = StyledWorkflow if joint else Workflow
    target = None if joint else TextTarget(module="router")
    examples = [
        Example("question", "Brief edited: Before edited" if joint else "Before edited")
    ]

    def exact(example, prediction):
        return float(prediction == example.labels)

    metric = (
        MetricSet({"accuracy": exact, "format (#1)": exact}, primary="accuracy")
        if joint
        else exact
    )
    fit = SelectText().compile(
        factory, trainset=examples, valset=examples, metric=metric, target=target
    )
    comparison = Evaluate(dataset=examples, metric=metric).compare(
        factory(), fit.build()
    )
    params = {
        "dataset_id": "test-v1",
        "task_model": "echo",
        "reflection_model": "fixture",
    }

    with pytest.raises(RuntimeError, match="active MLflow run"):
        log_optimization(fit, comparison, **params)
    with mlflow.start_run(experiment_id=experiment) as run:
        log_optimization(fit, comparison, **params)
        assert mlflow.active_run().info.run_id == run.info.run_id

    saved = mlflow.get_run(run.info.run_id)
    assert saved.info.status == "FINISHED"
    assert saved.data.params == (
        params | {"primary_metric": "accuracy"} if joint else params
    )
    scores = saved.data.metrics
    assert scores["search.baseline"] == fit.search.baseline_score
    assert scores["search.best"] == fit.search.best_score
    assert scores["search.metric_calls"] == fit.search.metric_calls
    assert scores["holdout.baseline"] == 0
    assert scores["holdout.candidate"] == scores["holdout.delta"] == 1
    client = mlflow.MlflowClient()
    if joint:
        aliases = json.loads(
            Path(
                client.download_artifacts(run.info.run_id, "metric-names.json")
            ).read_text()
        )
        (encoded,) = aliases
        assert aliases[encoded] == "format (#1)"
        for key in ["accuracy", encoded]:
            assert scores[f"holdout.{key}.baseline"] == 0
            assert (
                scores[f"holdout.{key}.candidate"]
                == scores[f"holdout.{key}.delta"]
                == 1
            )

    artifact = OptimizationArtifact.load(
        client.download_artifacts(run.info.run_id, "optimization.json")
    )
    assert artifact.version == (2 if joint else 1)
    restored = factory()
    artifact.apply(restored)
    assert restored("question") == examples[0].labels


def test_tracking_is_optional(monkeypatch):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, msgflux.optim.mlflow; assert 'mlflow' not in sys.modules",
        ],
        check=True,
    )
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(ImportError, match=r"msgflux\[mlflow\]"):
        log_optimization(None, None, dataset_id="", task_model="", reflection_model="")
