# ruff: noqa: T201
"""Optimize a support router's instruction and evaluate the saved text.

Set OPENAI_API_KEY in the environment or in a local `.env`, then run from a
repository checkout:

    uv run --extra optimizers python examples/optimize_triage.py

The task model and the reflection model are the same live chat completion
model. For a two-agent program whose queue and priority prompts are optimized
together, see `examples/optimize_support_program.py`.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import msgflux as mf
from msgflux import nn
from msgflux.optim import GEPA, Evaluate, OptimizationArtifact

MODEL = "openai/gpt-6-luna"
QUEUES = ("billing", "access", "general")
SEED = (
    "Reply with exactly one queue name: billing, access or general. "
    "The ticket content decides which queue owns it."
)


def build_agent(model):
    """Build the root program: one agent whose task instruction is optimized."""
    return nn.Agent(name="triage", model=model, system_prompt=SEED)


def normalized(text):
    """Compare one label: trim the reply, ignore case and end punctuation."""
    return str(text).strip().strip(".\"' ").casefold()


def queue_accuracy(example, prediction):
    """Score the single-label contract and report every deviation from it."""
    expected = normalized(example.labels)
    answer = normalized(prediction)
    return float(answer == expected), {
        "feedback": (
            "Reply with exactly one queue name and no other text. "
            f"Expected {expected!r} from {QUEUES}; the agent replied "
            f"{str(prediction).strip()!r}."
        )
    }


def tickets():
    """Return the search, selection and holdout splits of labeled tickets."""
    train = [
        mf.Example("Invoice is missing", "billing"),
        mf.Example("Reset my password", "access"),
        mf.Example("Update my invoice", "billing"),
    ]
    validation = [
        mf.Example("Wrong invoice date", "billing"),
        mf.Example("Forgot password", "access"),
    ]
    holdout = [
        mf.Example("Need an invoice copy", "billing"),
        mf.Example("Change password", "access"),
    ]
    return train, validation, holdout


def main():
    mf.load_dotenv()
    model = mf.Model.chat_completion(MODEL)
    train, validation, holdout = tickets()

    result = GEPA(reflection_model=model, max_metric_calls=18).compile(
        lambda: build_agent(model),
        trainset=train,
        valset=validation,
        metric=queue_accuracy,
    )

    with TemporaryDirectory() as directory:
        path = Path(directory) / "triage.optim.json"
        result.save(path)
        deployed = build_agent(model)
        OptimizationArtifact.load(path).apply(deployed)
        evaluator = Evaluate(dataset=holdout, metric=queue_accuracy)
        comparison = evaluator.compare(build_agent(model), deployed)

    print("Selected instruction:", result.search.best_text)
    print(
        f"Validation: {result.search.baseline_score:.2f} -> "
        f"{result.search.best_score:.2f}"
    )
    print(
        f"Holdout: {comparison.baseline_mean_score:.2f} -> "
        f"{comparison.candidate_mean_score:.2f}"
    )
    for row in comparison.rows:
        print(f"{row.example.inputs}: {row.outcome} ({row.delta:+.2f})")
    return comparison.candidate


if __name__ == "__main__":
    main()
