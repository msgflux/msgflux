"""Compile a support agent and evaluate its saved instruction without API keys."""

from pathlib import Path
from tempfile import TemporaryDirectory

import msgflux as mf
from msgflux import nn
from msgflux.models.response import ModelResponse
from msgflux.optim import GEPA, OptimizationArtifact, evaluate


class TriageModel:
    """A deterministic chat model for exercising the real Agent and GEPA loop."""

    model_type = "chat_completion"

    def __call__(self, *, messages, system_prompt, **kwargs):
        ticket = messages[-1]["content"].lower()
        queue = "general"
        if "billing" in system_prompt and "invoice" in ticket:
            queue = "billing"
        if "access" in system_prompt and "password" in ticket:
            queue = "access"
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add(queue)
        return response


def build_agent():
    return nn.Agent(
        name="triage", model=TriageModel(), system_prompt="Route the ticket."
    )


def reflection(_messages):
    return "```\nRoute invoice tickets to billing and password tickets to access.\n```"


def queue_accuracy(example, prediction):
    return float(prediction == example.labels), {
        "feedback": f"Expected queue {example.labels}, predicted {prediction}."
    }


def main():
    train = [
        mf.Example("Invoice is missing", "billing"),
        mf.Example("Reset my password", "access"),
        mf.Example("Update my invoice", "billing"),
    ]
    validation = [
        mf.Example("Wrong invoice date", "billing"),
        mf.Example("Forgot password", "access"),
    ]
    held_out = [
        mf.Example("Need an invoice copy", "billing"),
        mf.Example("Change password", "access"),
    ]

    optimizer = GEPA(reflection_model=reflection, max_metric_calls=12)
    result = optimizer.compile(
        build_agent, trainset=train, valset=validation, metric=queue_accuracy
    )
    with TemporaryDirectory() as directory:
        path = Path(directory) / "triage.optim.json"
        result.save(path)
        deployed = build_agent()
        OptimizationArtifact.load(path).apply(deployed)
        report = evaluate(deployed, held_out, queue_accuracy)

    print(
        f"Validation: {result.search.baseline_score:.1f} -> {result.search.best_score:.1f}"
    )
    print(f"Holdout accuracy: {report.mean_score:.1f} ({len(report.rows)} examples)")
    return report


if __name__ == "__main__":
    main()
