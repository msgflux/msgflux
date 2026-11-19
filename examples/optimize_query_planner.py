"""Compare two QueryPlanner instructions with GPT-6 Luna."""

import runpy
from pathlib import Path

import msgflux as mf
from msgflux import nn
from msgflux.optim import GEPA, evaluate


def labeled_question(question, intents):
    return mf.Example(
        inputs={
            "question": question,
            "available_intents": (
                "search: find article IDs by keyword; lookup: retrieve the article by "
                "ID after search; analyze: compute incident metrics"
            ),
        },
        labels={"intents": intents.split()},
    )


def exact_plan(example, prediction):
    steps = prediction["plan"]
    intents = [step.get("intent") for step in steps]
    expected = example.labels["intents"]
    valid_fields = all(set(step) == {"intent", "subquery"} for step in steps)
    valid_order = all(
        action != "lookup" or "search" in intents[:index]
        for index, action in enumerate(intents)
    )
    return float(valid_fields and valid_order and intents == expected), {
        "feedback": f"Expected {expected}; got {intents}. lookup must follow search.",
        "valid_fields": valid_fields,
        "valid_order": valid_order,
    }


def main():
    # Reuse the existing example's Signature and environment setup.
    router = runpy.run_path(str(Path(__file__).with_name("intent-router.py")))
    model = mf.Model.chat_completion("openai/gpt-6-luna")

    def build():
        return nn.Agent(name="planner", model=model, signature=router["QueryPlanner"])

    train = [
        labeled_question("Find article IDs for deployment.", "search"),
        labeled_question(
            "Retrieve the complete authentication article, finding its ID first.",
            "search lookup",
        ),
        labeled_question("Compute incident metrics.", "analyze"),
    ]
    validation = [
        labeled_question(
            "Return only the titles and IDs of the performance articles.", "search"
        ),
        labeled_question(
            "Fetch the full deployment guide by first finding its article ID.",
            "search lookup",
        ),
    ]
    holdout = [
        labeled_question(
            "List the IDs of OAuth articles without retrieving their contents.",
            "search",
        ),
        labeled_question(
            "Find and then retrieve the full Docker setup document.", "search lookup"
        ),
        labeled_question("Report incident trends over the past week.", "analyze"),
    ]
    result = GEPA(model, max_metric_calls=10).compile(
        build, trainset=train, valset=validation, metric=exact_plan
    )
    print(
        f"Validation: {result.search.baseline_score:.2f} -> {result.search.best_score:.2f}"
    )
    for label, agent in (("original", build()), ("compiled", result.build())):
        report = evaluate(agent, holdout, exact_plan)
        print(f"{label} holdout: {report.mean_score:.2f}")
        for row in report.rows:
            print(
                row.example.inputs["question"], row.prediction, row.score, row.feedback
            )


if __name__ == "__main__":
    main()
