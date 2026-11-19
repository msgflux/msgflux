"""Evaluate programs against labeled examples using the compilation metric."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Literal

from msgflux.core.examples import Example
from msgflux.models.response import ModelResponse

Score = float | tuple[float, dict[str, Any]]
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
class EvaluationRow:
    example: Example
    prediction: Any
    score: float
    feedback: dict[str, Any]


@dataclass(frozen=True)
class EvaluationResult:
    rows: tuple[EvaluationRow, ...]

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
        return cls(rows)

    @property
    def baseline(self) -> EvaluationResult:
        return EvaluationResult(tuple(row.baseline for row in self.rows))

    @property
    def candidate(self) -> EvaluationResult:
        return EvaluationResult(tuple(row.candidate for row in self.rows))

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
        score, feedback = normalize_score(metric(example, prediction))
        rows.append(EvaluationRow(example, prediction, score, feedback))
    return EvaluationResult(tuple(rows))


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
    "Score",
    "evaluate",
    "example_identity",
    "normalize_score",
    "predict",
    "resolve_dataset",
    "resolve_metric",
]
