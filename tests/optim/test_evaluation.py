"""Exercise the reusable evaluator and the paired baseline/candidate report."""

import subprocess
import sys
from types import MappingProxyType

import pytest

import msgflux as mf
from msgflux import nn
from msgflux.models.response import ModelResponse
from msgflux.optim import ComparisonResult, Evaluate, EvaluationResult, evaluate
from msgflux.optim.evaluation import example_identity


class EchoModel:
    """A deterministic chat model that routes tickets by keyword."""

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
    return nn.Agent(name="triage", model=EchoModel(), system_prompt=instruction)


def queue_accuracy(example, prediction):
    return float(prediction == example.labels), {
        "feedback": f"Expected {example.labels}, predicted {prediction}."
    }


def holdout_examples():
    return [
        mf.Example("Invoice is missing", "billing"),
        mf.Example("Reset my password", "access"),
        mf.Example("Where is my badge?", "general"),
    ]


def test_evaluate_reruns_one_dataset_and_metric_for_every_program():
    calls = []

    def metric(example, prediction):
        calls.append((example.inputs, prediction))
        return float(prediction == example.labels)

    evaluator = Evaluate(dataset=holdout_examples(), metric=metric)
    baseline = evaluator(build_agent("Route the ticket."))
    candidate = evaluator(
        build_agent("Route invoice tickets to billing and password tickets to access.")
    )
    assert isinstance(baseline, EvaluationResult)
    assert isinstance(candidate, EvaluationResult)
    assert calls == [
        ("Invoice is missing", "general"),
        ("Reset my password", "general"),
        ("Where is my badge?", "general"),
        ("Invoice is missing", "billing"),
        ("Reset my password", "access"),
        ("Where is my badge?", "general"),
    ]
    assert baseline.mean_score == pytest.approx(1 / 3)
    assert baseline.scores == (0.0, 0.0, 1.0)
    assert candidate.mean_score == 1.0
    # A bare numeric score is normalized to an empty feedback dictionary.
    assert candidate.rows[0].feedback == {}


def test_evaluate_snapshots_the_dataset_and_accepts_call_time_overrides():
    examples = holdout_examples()
    evaluator = Evaluate(dataset=examples, metric=queue_accuracy)
    examples.clear()
    assert [example_identity(example) for example in evaluator.dataset] == [
        example_identity(example) for example in holdout_examples()
    ]

    result = evaluator(str.upper)
    assert len(result.rows) == 3
    assert result.predictions == (
        "INVOICE IS MISSING",
        "RESET MY PASSWORD",
        "WHERE IS MY BADGE?",
    )

    single = mf.Example("Reset my password", "access")
    overrides = Evaluate()
    assert overrides.metric is None
    assert len(overrides(str.lower, dataset=[single], metric=queue_accuracy).rows) == 1
    assert (
        len(
            evaluator(
                str.lower,
                dataset=[single],
                metric=lambda example, prediction: float(prediction == example.labels),
            ).rows
        )
        == 1
    )


def test_compare_pairs_baseline_and_candidate_per_example():
    evaluator = Evaluate(dataset=holdout_examples(), metric=queue_accuracy)
    comparison = evaluator.compare(
        build_agent("Route the ticket."),
        build_agent("Route invoice tickets to billing and password tickets to access."),
    )
    assert isinstance(comparison, ComparisonResult)
    assert comparison.baseline_mean_score == pytest.approx(1 / 3)
    assert comparison.candidate_mean_score == 1.0
    assert comparison.deltas == (1.0, 1.0, 0.0)
    assert comparison.mean_delta == pytest.approx(2 / 3)
    assert [row.outcome for row in comparison.rows] == [
        "improved",
        "improved",
        "unchanged",
    ]
    assert len(comparison.improved) == 2
    assert comparison.regressed == ()
    assert len(comparison.unchanged) == 1
    first = comparison.rows[0]
    assert example_identity(first.example) == ("Invoice is missing", "billing")
    assert (first.baseline.score, first.candidate.score) == (0.0, 1.0)
    assert first.baseline.feedback == {
        "feedback": "Expected billing, predicted general."
    }
    assert first.candidate.feedback == {
        "feedback": "Expected billing, predicted billing."
    }
    assert comparison.baseline == evaluator(build_agent("Route the ticket."))


def test_compare_reports_regressions_and_keeps_both_scores():
    evaluator = Evaluate(
        dataset=[mf.Example("Invoice is missing", "billing")], metric=queue_accuracy
    )
    comparison = evaluator.compare(
        build_agent("Route invoice tickets to billing."),
        build_agent("Route the ticket."),
    )
    assert comparison.deltas == (-1.0,)
    assert comparison.rows[0].outcome == "regressed"
    assert comparison.mean_delta == -1.0
    assert [row.outcome for row in comparison.regressed] == ["regressed"]


def test_pairing_compares_inputs_and_labels_not_dataclass_equality():
    # `Example` equality also compares derived transform flags, so the report
    # pairs rows by their task inputs and labels instead.
    question = {"question": "Invoice is missing"}
    original = mf.Example(question, "billing")
    rebuilt = mf.Example(MappingProxyType(dict(question)), "billing")
    assert original != rebuilt
    assert example_identity(original) == example_identity(rebuilt)

    def program(**kwargs):
        return kwargs["question"]

    comparison = ComparisonResult.pair(
        evaluate(program, [original], queue_accuracy),
        evaluate(program, [rebuilt], queue_accuracy),
    )
    assert comparison.deltas == (0.0,)
    assert dict(comparison.rows[0].candidate.example.inputs) == question
    assert comparison.rows[0].candidate.example.labels == "billing"


def test_compare_requires_the_same_examples_in_the_same_order():
    # `compare` always feeds both programs the same dataset, so a mismatch can
    # only arrive through `ComparisonResult.pair` with independent results.
    with pytest.raises(ValueError, match="same number of examples"):
        ComparisonResult.pair(
            evaluate(str, holdout_examples(), queue_accuracy),
            evaluate(str, holdout_examples()[:1], queue_accuracy),
        )
    with pytest.raises(ValueError, match="same examples in the same order"):
        ComparisonResult.pair(
            evaluate(str, holdout_examples(), lambda *_: 0.0),
            evaluate(list, list(reversed(holdout_examples())), lambda *_: 1.0),
        )


def test_evaluate_and_compare_validate_their_inputs():
    with pytest.raises(ValueError, match="at least one example"):
        Evaluate(dataset=[], metric=queue_accuracy)
    with pytest.raises(TypeError, match="sequence of examples"):
        Evaluate(dataset="ticket", metric=queue_accuracy)
    with pytest.raises(TypeError, match="metric must be callable"):
        Evaluate(dataset=holdout_examples(), metric="accuracy")
    with pytest.raises(TypeError, match="program must be callable"):
        Evaluate(dataset=holdout_examples(), metric=queue_accuracy)(object())
    with pytest.raises(ValueError, match="requires a dataset"):
        Evaluate(metric=queue_accuracy)(str)
    with pytest.raises(ValueError, match="requires a metric"):
        Evaluate(dataset=holdout_examples())(str)
    with pytest.raises(TypeError, match="two callable programs"):
        Evaluate(dataset=holdout_examples(), metric=queue_accuracy).compare(str, None)
    with pytest.raises(ValueError, match="finite"):
        Evaluate(dataset=holdout_examples(), metric=lambda *_: float("inf"))(
            build_agent("Route the ticket.")
        )


def test_documented_offline_comparison_example():
    """Mirror docs/learn/optimizers/evaluation.md and its documented output."""
    holdout = holdout_examples()
    evaluator = Evaluate(dataset=holdout, metric=queue_accuracy)
    comparison = evaluator.compare(
        build_agent("Route the ticket."),
        build_agent("Route invoice tickets to billing and password tickets to access."),
    )
    summary = (
        f"{comparison.baseline_mean_score:.2f} -> {comparison.candidate_mean_score:.2f}"
    )
    rows = [
        (row.example.inputs, row.outcome, f"{row.delta:+.2f}")
        for row in comparison.rows
    ]
    assert summary == "0.33 -> 1.00"
    assert rows == [
        ("Invoice is missing", "improved", "+1.00"),
        ("Reset my password", "improved", "+1.00"),
        ("Where is my badge?", "unchanged", "+0.00"),
    ]
    assert f"{comparison.mean_delta:+.2f}" == "+0.67"


def test_evaluation_imports_without_gepa():
    script = """
import sys
sys.modules['gepa'] = None

from msgflux import Example
from msgflux.optim import (
    ComparisonResult,
    ComparisonRow,
    Evaluate,
    EvaluationResult,
    EvaluationRow,
    evaluate,
)

example = Example('a', 'A')
result = evaluate(str.upper, [example], lambda ex, prediction: float(prediction == ex.labels))
assert isinstance(result, EvaluationResult) and result.mean_score == 1.0
assert isinstance(result.rows[0], EvaluationRow)

evaluator = Evaluate(dataset=[example], metric=lambda ex, prediction: 1.0)
comparison = evaluator.compare(str.upper, str.lower)
assert isinstance(comparison, ComparisonResult) and isinstance(comparison.rows[0], ComparisonRow)
assert comparison.mean_delta == 0.0
assert sys.modules['gepa'] is None
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603
