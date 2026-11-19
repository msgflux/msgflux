"""Exercise GEPA search, artifact deployment and paired multi-metric evaluation."""

from collections import Counter

import pytest

from msgflux import Example, nn
from msgflux.models.response import ModelResponse
from msgflux.optim import Evaluate, GEPA, MetricSet, OptimizationArtifact

SEED = "Route each ticket to a queue."
TUNED = "Route invoices to billing and passwords to access."


def text(value):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(value)
    return response


class Model:
    """Deterministic in-process chat model; with no client, no network is reachable."""

    model_type = "chat_completion"

    def __init__(self, counts, rewrite=None):
        self.counts = counts
        self.rewrite, self.requests = rewrite, []

    def __call__(self, *, messages, system_prompt="", **_kwargs):
        if self.rewrite is not None:  # Reflection model: records what GEPA shows it.
            self.requests.append(messages[-1]["content"])
            return text(self.rewrite)
        self.counts["inference"] += 1
        ticket, prompt = messages[-1]["content"].lower(), system_prompt.lower()
        if "invoice" in ticket and "billing" in prompt:
            return text("billing (escalate)")
        if "password" in ticket and "access" in prompt:
            return text("access (escalate)")
        return text("general")


def metric_set(counts, *names):
    def quality(example, prediction):
        counts["quality"] += 1
        return float(example.labels in prediction), {"feedback": "Route by queue."}

    def cost(example, prediction):
        counts["cost"] += 1
        return float("(escalate)" not in prediction), {"feedback": "Keep it minimal."}

    table = {"quality": quality, "cost": cost}
    return MetricSet({name: table[name] for name in names}, primary="quality")


def test_named_metrics_end_to_end(tmp_path):
    pytest.importorskip("gepa")
    counts = Counter()

    def agent():
        return nn.Agent(name="triage", model=Model(counts), system_prompt=SEED)

    suite = metric_set(counts, "quality", "cost")
    train = [Example("Missing invoice", "billing"), Example("New password", "access")]
    val = [Example("Resend invoice", "billing"), Example("Lost password", "access")]
    holdout = [
        Example("My invoice is wrong", "billing"),
        Example("Reset my password", "access"),
    ]

    # GEPA scores only the primary metric, so cost never runs and its feedback
    # never reaches the reflection model.
    reflection = Model(counts, TUNED)
    result = GEPA(reflection, 9).compile(
        agent, trainset=train, valset=val, metric=suite
    )
    calls = result.search.metric_calls
    assert (result.search.best_score, result.search.best_text) == (1.0, TUNED)
    assert counts == {"inference": calls, "quality": calls}
    assert "Route by queue." in reflection.requests[0]
    assert "Keep it minimal." not in reflection.requests[0]

    # The selected instruction survives a save/load round trip.
    result.save(tmp_path / "triage.optim.json")
    restored = agent()
    OptimizationArtifact.load(tmp_path / "triage.optim.json").apply(restored)
    assert restored.get_task_instruction() == TUNED
    assert agent().get_task_instruction() == SEED

    # One inference per prediction, whatever the number of named metrics.
    reports = {}
    for names in (("quality",), ("quality", "cost")):
        counts.clear()
        reports[names] = Evaluate(dataset=holdout, metric=metric_set(counts, *names))(
            restored
        )
        assert list(reports[names].mean_scores) == list(names)
        assert counts == {"inference": 2, **dict.fromkeys(names, 2)}

    # The winner scores zero on cost, so best_score=1.0 could only be the primary.
    assert reports[("quality", "cost")].mean_scores == {"quality": 1.0, "cost": 0.0}

    # Paired holdout: the primary improves while the reported secondary regresses.
    counts.clear()
    paired = Evaluate(dataset=holdout, metric=suite).compare(agent(), restored)
    assert counts == {"inference": 4, "quality": 4, "cost": 4}  # two programs
    assert paired.baseline.mean_scores == {"quality": 0.0, "cost": 1.0}
    assert paired.candidate.mean_scores == {"quality": 1.0, "cost": 0.0}
    assert paired.deltas == (1.0, 1.0)
    assert [row.outcome for row in paired.for_metric("cost").rows] == ["regressed"] * 2
    cost_rows = paired.candidate.for_metric("cost")
    assert cost_rows.rows[0].feedback == {"feedback": "Keep it minimal."}
