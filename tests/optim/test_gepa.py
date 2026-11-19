"""Exercise the released search engine through the public optimizer interface."""

import builtins
import sys

import pytest

from msgflux.models.response import ModelResponse
from msgflux.optim import GEPA


class ReflectionModel:
    model_type = "chat_completion"

    def __init__(self):
        self.requests = []

    def __call__(self, *, messages):
        self.requests.append(messages)
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add("```\nRoute invoices to billing.\n```")
        return response


@pytest.mark.parametrize("budget", [2, 3, 6, 9, 12])
def test_real_search_keeps_a_complete_validation_score_and_honors_budget(budget):
    pytest.importorskip("gepa")
    requests = []
    reflection = ReflectionModel()

    def evaluate(candidate, example):
        requests.append((candidate, example))
        return float("billing" in candidate), {"feedback": "Expected the billing queue"}

    result = GEPA(reflection, budget).optimize_text(
        "Route tickets.",
        trainset=["invoice", "receipt"],
        valset=["payment", "statement"],
        evaluator=evaluate,
    )
    assert result.baseline_score == 0
    assert result.metric_calls == len(requests) <= budget
    if reflection.requests:
        assert "Expected the billing queue" in str(reflection.requests[0])
        assert result.best_score == 1
        assert result.best_text == "Route invoices to billing."
        assert (result.best_text, "statement") in requests
    else:
        assert budget < 6
        assert result.best_text == "Route tickets."


def test_equal_scores_keep_the_original_text():
    pytest.importorskip("gepa")
    result = GEPA(ReflectionModel(), 8).optimize_text(
        "original", trainset=[1], valset=[2], evaluator=lambda *_: 0.5
    )
    assert result.best_text == "original"
    assert result.best_score == result.baseline_score == 0.5


def test_missing_optional_dependency_has_an_actionable_error(monkeypatch):
    original_import = builtins.__import__

    def without_gepa(name, *args, **kwargs):
        if name == "gepa.optimize_anything":
            raise ModuleNotFoundError("No module named 'gepa'", name="gepa")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_gepa)
    message = "3.11-3.14" if sys.version_info >= (3, 15) else r"msgflux\[optimizers\]"
    with pytest.raises(ImportError, match=message):
        GEPA(str).optimize_text(
            "seed", trainset=[1], valset=[2], evaluator=lambda *_: 1
        )


def test_parallel_checkpoint_resume_preserves_the_selected_instruction(tmp_path):
    pytest.importorskip("gepa")

    def metric(candidate, _example):
        return float("billing" in candidate), {"feedback": "Expected the billing queue"}

    def search(budget):
        return GEPA(
            ReflectionModel(), budget, workers=2, run_dir=tmp_path
        ).optimize_text("seed", trainset=[1, 2], valset=[3, 4], evaluator=metric)

    first = search(10)
    resumed = search(18)
    assert first.metric_calls < resumed.metric_calls <= 18
    assert resumed.best_text == first.best_text
    assert resumed.best_score == first.best_score == 1


@pytest.mark.parametrize("include_context", [True, False])
def test_compile_delivers_training_context_to_the_reflection_model(include_context):
    pytest.importorskip("gepa")
    from msgflux import Example, nn

    class Echo:
        model_type = "chat_completion"

        def __call__(self, **kwargs):
            response = ModelResponse()
            response.set_response_type("text_generation")
            response.add("observed prediction")
            return response

    def factory():
        return nn.Agent(name="support", model=Echo(), system_prompt="Classify tickets")

    reflection = ReflectionModel()
    GEPA(reflection, 6).compile(
        factory,
        trainset=[Example("training ticket", "x")],
        valset=[Example("held out validation ticket", "y")],
        metric=lambda *_: (0, {"feedback": "incorrect label"}),
        include_example_context=include_context,
    )
    prompt = str(reflection.requests[0])
    assert ("training ticket" in prompt) == include_context
    assert ("observed prediction" in prompt) == include_context
    assert "incorrect label" in prompt
    assert "held out validation ticket" not in prompt
