# ruff: noqa: T201
"""Save, restore and compare two jointly optimized Agent instructions.

Set OPENAI_API_KEY in the environment or in a local `.env`, then run from a
repository checkout:

    uv run --extra optimizers python examples/optimize_support_program.py

For the shorter version that searches a single router and then a whole program,
see `examples/gepa_single_agent_and_program.py`. This example names both targets
explicitly and checks the joint artifact after restoring it.
"""

from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory

import msgflux as mf
from msgflux import nn
from msgflux.optim import GEPA, Evaluate, OptimizationArtifact, TextTarget

MODEL = "openai/gpt-6-luna"
QUEUES = ("billing", "access", "general")
PRIORITIES = ("urgent", "normal")
FIELDS = ("queue", "priority")
ROUTER_SEED = (
    "Reply with exactly one queue name: billing, access or general. "
    "The ticket content decides which queue owns it."
)
PRIORITY_SEED = (
    "Reply with exactly one priority name: urgent or normal. "
    "Use urgent only when work is blocked."
)
# Name each editable instruction so one artifact addresses both agents.
TARGETS = {
    "router": TextTarget(module="router"),
    "priority": TextTarget(module="priority"),
}


class SupportProgram(nn.Module):
    """A router and a priority writer whose instructions are optimized together."""

    def __init__(self, model):
        super().__init__()
        self.router = nn.Agent(name="router", model=model, system_prompt=ROUTER_SEED)
        self.priority = nn.Agent(
            name="priority", model=model, system_prompt=PRIORITY_SEED
        )

    def forward(self, ticket):
        queue = self.router(ticket)
        priority = self.priority(f"Assigned queue: {queue}\nTicket: {ticket}")
        return {"queue": queue, "priority": priority}


def normalized(text):
    """Compare one label: trim the reply, ignore case and end punctuation."""
    return str(text).strip().strip(".\"' ").casefold()


def program_accuracy(example, prediction):
    """Score both single-label fields and restate the output contract."""
    reply = prediction if isinstance(prediction, dict) else {}
    expected = {name: normalized(example.labels[name]) for name in FIELDS}
    actual = {name: normalized(reply.get(name, "")) for name in FIELDS}
    score = fmean(expected[name] == actual[name] for name in FIELDS)
    return score, {
        "feedback": (
            "Reply with exactly one queue name and no other text: "
            f"{', '.join(QUEUES)}. Answer the priority prompt with exactly one "
            f"priority name: {', '.join(PRIORITIES)}. expected={expected} "
            f"predicted={actual}"
        )
    }


def tickets():
    """Return the search, selection and holdout splits of labeled tickets."""

    def ticket(text, queue, priority):
        return mf.Example(text, {"queue": queue, "priority": priority})

    train = [
        ticket("Where is last month's invoice?", "billing", "normal"),
        ticket("Everyone is locked out of production.", "access", "urgent"),
    ]
    validation = [
        ticket("Please send a receipt for April.", "billing", "normal"),
        ticket("Our entire team just lost access.", "access", "urgent"),
    ]
    holdout = [
        ticket("Can I get a copy of my last receipt?", "billing", "normal"),
        ticket("The dashboard rejects every login.", "access", "urgent"),
    ]
    return train, validation, holdout


def main():
    mf.load_dotenv()
    model = mf.Model.chat_completion(MODEL)
    train, validation, holdout = tickets()
    # The metric scores the value returned by SupportProgram.forward, so one
    # search can improve the routing prompt and the priority prompt together.
    # module_selector="all" proposes both edits inside the same candidate.
    result = GEPA(
        reflection_model=model, max_metric_calls=24, module_selector="all"
    ).compile(
        lambda: SupportProgram(model),
        targets=TARGETS,
        trainset=train,
        valset=validation,
        metric=program_accuracy,
    )

    with TemporaryDirectory() as directory:
        path = Path(directory) / "support-program.optim.json"
        result.save(path)
        restored = SupportProgram(model)
        OptimizationArtifact.load(path).apply(restored)

    deployed = dict(result.search.best_texts)
    applied = {
        "router": restored.router.get_task_instruction(),
        "priority": restored.priority.get_task_instruction(),
    }
    if applied != deployed:
        raise RuntimeError("The restored program did not apply both instructions")

    comparison = Evaluate(dataset=holdout, metric=program_accuracy).compare(
        SupportProgram(model), restored
    )
    print("Selected instructions:", deployed)
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
    return comparison


if __name__ == "__main__":
    main()
