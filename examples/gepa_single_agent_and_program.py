"""Two ways to optimize prompts with msgFlux.

Set OPENAI_API_KEY and run from a checkout:
    uv run --extra optimizers python examples/gepa_single_agent_and_program.py
"""

from statistics import fmean

import msgflux as mf
from msgflux import nn
from msgflux.optim import GEPA, Evaluate


def queue_agent(model):
    return nn.Agent(
        name="router", model=model, system_prompt="Classify the support ticket."
    )


class SupportProgram(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.router = queue_agent(model)
        self.priority = nn.Agent(
            name="priority", model=model, system_prompt="Decide the ticket priority."
        )

    def forward(self, ticket):
        queue = self.router(ticket)
        priority = self.priority(f"Assigned queue: {queue}\nTicket: {ticket}")
        return {"queue": queue, "priority": priority}


def exact(predicted, expected):
    return float(predicted.strip().casefold() == expected.casefold())


def queue_score(example, prediction):
    return exact(prediction, example.labels["queue"]), {
        "expected": example.labels["queue"]
    }


def program_score(example, prediction):
    return fmean(exact(prediction[k], v) for k, v in example.labels.items()), {
        "expected": example.labels
    }


def main():
    mf.load_dotenv()
    model = mf.Model.chat_completion("openai/gpt-6-luna")
    optimizer = GEPA(reflection_model=model, max_metric_calls=18, module_selector="all")

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
    holdout = [ticket("Can I get a copy of my last receipt?", "billing", "normal")]

    # 1. Optimize a single Agent's free-text system prompt.
    single = optimizer.compile(
        lambda: queue_agent(model),
        trainset=train,
        valset=validation,
        metric=queue_score,
    )
    print("Selected router prompt:", single.search.best_text)  # noqa: T201

    # 2. Optimize both registered agents against the complete program's output.
    program = optimizer.compile(
        lambda: SupportProgram(model),
        trainset=train,
        valset=validation,
        metric=program_score,
    )
    program.save("support-program.optim.json")
    evaluation = Evaluate(dataset=holdout, metric=program_score)
    comparison = evaluation.compare(SupportProgram(model), program.build())
    print("Selected program prompts:", program.search.best_texts)  # noqa: T201
    print("Holdout score change:", comparison.mean_delta)  # noqa: T201


if __name__ == "__main__":
    main()
