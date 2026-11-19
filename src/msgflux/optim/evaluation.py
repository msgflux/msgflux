"""Evaluate programs against labeled examples using the compilation metric."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from types import MappingProxyType
from typing import Any, Literal

from msgflux.core.examples import Example
from msgflux.models.response import ModelResponse

Score = float | tuple[float, dict[str, Any]]
Metric = Callable[[Example, Any], Score]
MetricValue = tuple[float, dict[str, Any]]
Outcome = Literal["improved", "regressed", "unchanged"]


def normalize_score(value: Score) -> tuple[float, dict[str, Any]]:
    score, feedback = value if isinstance(value, tuple) else (value, {})
    if not isinstance(feedback, dict):
        raise TypeError("Metric feedback must be a dictionary")
    if not isinstance(score, (float, int)) or not math.isfinite(score):
        raise ValueError("The metric must return a finite numeric score")
    return float(score), feedback


def predict(program, example):
    inputs = example.inputs
    prediction = program(**inputs) if isinstance(inputs, Mapping) else program(inputs)
    return prediction.consume() if isinstance(prediction, ModelResponse) else prediction


def resolve_dataset(dataset: Sequence[Example]) -> tuple[Example, ...]:
    """Freeze a dataset into an immutable sequence with at least one example."""
    if isinstance(dataset, (str, bytes, Mapping)) or not isinstance(dataset, Sequence):
        raise TypeError("The dataset must be a sequence of examples")
    examples = tuple(dataset)
    if not examples:
        raise ValueError("Evaluation requires at least one example")
    return examples


def example_identity(example: Example) -> tuple[Any, Any]:
    """Return the inputs and labels that identify a paired evaluation row.

    `Example` is a mutable dataclass whose generated equality also compares the
    derived transform flags, so two examples built with the same inputs and
    labels can compare unequal. Pairing therefore compares the task data.
    """
    return example.inputs, example.labels


def resolve_metric(metric: Callable[[Example, Any], Score]):
    """Require a callable that accepts an example and a prediction."""
    if not callable(metric):
        raise TypeError("The metric must be callable as metric(example, prediction)")
    return metric


@dataclass(frozen=True)
class MetricSet:
    """Use one objective for search and all named metrics for evaluation."""

    metrics: Mapping[str, Metric]
    primary: str = field(kw_only=True)

    def __post_init__(self):
        if not isinstance(self.metrics, Mapping) or not self.metrics:
            raise ValueError("metrics must be a non-empty mapping")
        metrics = dict(self.metrics)
        for name, metric in metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Metric names must be non-empty strings")
            resolve_metric(metric)
        if self.primary not in metrics:
            raise ValueError(f"Unknown primary metric: {self.primary!r}")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    def __call__(self, example: Example, prediction: Any) -> Score:
        return self.metrics[self.primary](example, prediction)

    def _score_all(self, example: Example, prediction: Any) -> dict[str, MetricValue]:
        return {
            name: normalize_score(metric(example, prediction))
            for name, metric in self.metrics.items()
        }


@dataclass(frozen=True)
class EvaluationRow:
    example: Example
    prediction: Any
    score: float
    feedback: dict[str, Any]
    metric_values: dict[str, MetricValue] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    rows: tuple[EvaluationRow, ...]
    primary_metric: str | None = None

    @property
    def mean_scores(self) -> dict[str, float]:
        """Mean validation or holdout score for each named metric."""
        if not self.rows:
            return {}
        return {
            name: fmean(row.metric_values[name][0] for row in self.rows)
            for name in self.rows[0].metric_values
        }

    def for_metric(self, name: str) -> "EvaluationResult":
        """Project the stored predictions and feedback onto one metric."""
        if not self.rows or name not in self.rows[0].metric_values:
            raise KeyError(f"Unknown evaluation metric: {name!r}")
        rows = []
        for row in self.rows:
            score, feedback = row.metric_values[name]
            rows.append(
                EvaluationRow(
                    row.example,
                    row.prediction,
                    score,
                    feedback,
                    {name: (score, feedback)},
                )
            )
        return EvaluationResult(tuple(rows), name)

    @property
    def mean_score(self) -> float:
        return fmean(row.score for row in self.rows)

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(row.score for row in self.rows)

    @property
    def predictions(self) -> tuple[Any, ...]:
        return tuple(row.prediction for row in self.rows)


@dataclass(frozen=True)
class ComparisonRow:
    """The baseline and candidate evaluations of one held-out example."""

    baseline: EvaluationRow
    candidate: EvaluationRow

    @property
    def example(self) -> Example:
        return self.baseline.example

    @property
    def delta(self) -> float:
        return self.candidate.score - self.baseline.score

    @property
    def outcome(self) -> Outcome:
        if self.delta > 0:
            return "improved"
        if self.delta < 0:
            return "regressed"
        return "unchanged"


@dataclass(frozen=True)
class ComparisonResult:
    """Paired per-example deltas between a baseline and a candidate program."""

    rows: tuple[ComparisonRow, ...]
    primary_metric: str | None = None

    def for_metric(self, name: str) -> "ComparisonResult":
        """Compare the saved rows using one named metric."""
        return ComparisonResult.pair(
            self.baseline.for_metric(name), self.candidate.for_metric(name)
        )

    @classmethod
    def pair(
        cls, baseline: EvaluationResult, candidate: EvaluationResult
    ) -> "ComparisonResult":
        """Pair two results that share the same examples in the same order."""
        if not baseline.rows or not candidate.rows:
            raise ValueError("Comparison requires at least one example")
        if len(baseline.rows) != len(candidate.rows):
            raise ValueError("Comparison requires the same number of examples")
        rows = tuple(
            ComparisonRow(left, right)
            for left, right in zip(baseline.rows, candidate.rows, strict=True)
        )
        if any(
            example_identity(row.example) != example_identity(row.candidate.example)
            for row in rows
        ):
            raise ValueError("Comparison requires the same examples in the same order")
        if baseline.primary_metric != candidate.primary_metric or any(
            row.baseline.metric_values.keys() != row.candidate.metric_values.keys()
            for row in rows
        ):
            raise ValueError(
                "Comparison requires the same metric names and primary metric"
            )
        return cls(rows, baseline.primary_metric)

    @property
    def baseline(self) -> EvaluationResult:
        return EvaluationResult(
            tuple(row.baseline for row in self.rows), self.primary_metric
        )

    @property
    def candidate(self) -> EvaluationResult:
        return EvaluationResult(
            tuple(row.candidate for row in self.rows), self.primary_metric
        )

    @property
    def baseline_mean_score(self) -> float:
        return self.baseline.mean_score

    @property
    def candidate_mean_score(self) -> float:
        return self.candidate.mean_score

    @property
    def deltas(self) -> tuple[float, ...]:
        return tuple(row.delta for row in self.rows)

    @property
    def mean_delta(self) -> float:
        return fmean(self.deltas)

    @property
    def improved(self) -> tuple[ComparisonRow, ...]:
        return tuple(row for row in self.rows if row.outcome == "improved")

    @property
    def regressed(self) -> tuple[ComparisonRow, ...]:
        return tuple(row for row in self.rows if row.outcome == "regressed")

    @property
    def unchanged(self) -> tuple[ComparisonRow, ...]:
        return tuple(row for row in self.rows if row.outcome == "unchanged")


def evaluate(
    program: Callable,
    examples: Sequence[Example],
    metric: Callable[[Example, Any], Score],
) -> EvaluationResult:
    """Run the supplied program and retain scores and feedback for every example."""
    if not examples:
        raise ValueError("Evaluation requires at least one example")
    rows = []
    for example in examples:
        prediction = predict(program, example)
        if isinstance(metric, MetricSet):
            values = metric._score_all(example, prediction)
            score, feedback = values[metric.primary]
        else:
            values = {}
            score, feedback = normalize_score(metric(example, prediction))
        rows.append(EvaluationRow(example, prediction, score, feedback, values))
    return EvaluationResult(
        tuple(rows), metric.primary if isinstance(metric, MetricSet) else None
    )


class Evaluate:
    """Reusable evaluator that scores a program on a stored dataset.

    The dataset and metric are resolved once, so the same instance can evaluate
    the baseline, the candidate and any other program on identical inputs.
    """

    def __init__(
        self,
        *,
        dataset: Sequence[Example] | None = None,
        metric: Callable[[Example, Any], Score] | None = None,
    ) -> None:
        self._dataset = None if dataset is None else resolve_dataset(dataset)
        self._metric = None if metric is None else resolve_metric(metric)

    @property
    def dataset(self) -> tuple[Example, ...] | None:
        return self._dataset

    @property
    def metric(self) -> Callable[[Example, Any], Score] | None:
        return self._metric

    def __call__(
        self,
        program: Callable,
        *,
        dataset: Sequence[Example] | None = None,
        metric: Callable[[Example, Any], Score] | None = None,
    ) -> EvaluationResult:
        """Score one program, overriding the stored dataset or metric when given."""
        if not callable(program):
            raise TypeError("The evaluated program must be callable")
        examples = self._dataset if dataset is None else resolve_dataset(dataset)
        scorer = self._metric if metric is None else resolve_metric(metric)
        if examples is None:
            raise ValueError("Evaluate requires a dataset in the constructor or call")
        if scorer is None:
            raise ValueError("Evaluate requires a metric in the constructor or call")
        return evaluate(program, examples, scorer)

    def compare(
        self,
        baseline: Callable,
        candidate: Callable,
        *,
        dataset: Sequence[Example] | None = None,
        metric: Callable[[Example, Any], Score] | None = None,
    ) -> ComparisonResult:
        """Evaluate both programs on the same examples and pair the scores."""
        if not callable(baseline) or not callable(candidate):
            raise TypeError("Comparison requires two callable programs")
        return ComparisonResult.pair(
            self(baseline, dataset=dataset, metric=metric),
            self(candidate, dataset=dataset, metric=metric),
        )


__all__ = [
    "ComparisonResult",
    "ComparisonRow",
    "Evaluate",
    "EvaluationResult",
    "EvaluationRow",
    "MetricSet",
    "Score",
    "evaluate",
    "example_identity",
    "normalize_score",
    "predict",
    "resolve_dataset",
    "resolve_metric",
]
