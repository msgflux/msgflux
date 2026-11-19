"""Optimize a support agent and its invoice lookup description together.

Set OPENAI_API_KEY, then run from a checkout:
    uv run --extra optimizers python examples/optimize_tool_description.py

To use an existing OpenCode Go account for the task model:
    uv run --extra optimizers python examples/optimize_tool_description.py \
        --task-model opencode-go/deepseek-v4.1-flash --account work

Add --extra mlflow before python and --mlflow after the script to track the run.
"""

import argparse
from pathlib import Path

import msgflux as mf
from msgflux import nn
from msgflux.accounts.resolver import AccountCredentialResolver
from msgflux.accounts.store import Accounts
from msgflux.optim import GEPA, Evaluate, OptimizationArtifact


def lookup(reference: str) -> str:
    """Look up a record."""
    return "rcpt-" + reference[::-1] + "-archived"


def build_agent(model, *, train_usage_guidance=False):
    agent = nn.Agent(
        name="desk",
        model=model,
        system_prompt="Help resolve the support request.",
        tools=[lookup],
    )
    agent.tool_library.description_parameter("lookup").requires_grad_(
        requires_grad=True
    )
    if train_usage_guidance:
        guidance = agent.tool_library.usage_guidance_parameter("lookup")
        guidance.copy_to_data("Use lookup to retrieve a record.")
        guidance.requires_grad_(requires_grad=True)
    return agent


def receipt(reference):
    request = f"What is the exact archived receipt code for invoice {reference}?"
    return mf.Example(request, lookup(reference))


def exact_receipt(example, prediction):
    return float(str(prediction).strip() == example.labels), {
        "feedback": (
            f"Return exactly {example.labels} from the lookup result, with no prose."
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-model", default="openai/gpt-6-luna")
    parser.add_argument("--account", help="Named msgFlux account for the task model")
    parser.add_argument("--reflection-model", default="openai/gpt-6-luna")
    parser.add_argument("--output", type=Path, default=Path("receipts.optim.json"))
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--train-usage-guidance", action="store_true")
    args = parser.parse_args()

    mf.load_dotenv()
    options = {}
    if args.account:
        provider = args.task_model.split("/", 1)[0]
        options["credential_resolver"] = AccountCredentialResolver(
            Accounts(), provider, account_name=args.account
        )
    task = mf.Model.chat_completion(args.task_model, **options)
    reflect = mf.Model.chat_completion(args.reflection_model)
    optimizer = GEPA(reflect, max_metric_calls=6, module_selector="all")

    def factory():
        return build_agent(task, train_usage_guidance=args.train_usage_guidance)

    result = optimizer.compile(
        factory,
        trainset=[receipt("ZX10"), receipt("VA27")],
        valset=[receipt("BX42")],
        metric=exact_receipt,
    )
    result.save(args.output)
    deployed = factory()
    OptimizationArtifact.load(args.output).apply(deployed)
    holdout = Evaluate(dataset=[receipt("KR17")], metric=exact_receipt)
    comparison = holdout.compare(factory(), deployed)
    print("Selected texts:", result.search.best_texts)  # noqa: T201
    print("Validation:", result.search.baseline_score, "->", result.search.best_score)  # noqa: T201
    print(  # noqa: T201
        "Holdout:",
        comparison.baseline_mean_score,
        "->",
        comparison.candidate_mean_score,
    )

    if args.mlflow:
        import mlflow  # noqa: PLC0415

        from msgflux.optim.mlflow import log_optimization  # noqa: PLC0415

        mlflow.set_experiment("joint-tool-description")
        with mlflow.start_run() as run:
            log_optimization(
                result,
                comparison,
                dataset_id="archived-receipts-v1",
                task_model=args.task_model,
                reflection_model=args.reflection_model,
            )
            print("MLflow run:", run.info.run_id)  # noqa: T201


if __name__ == "__main__":
    main()
