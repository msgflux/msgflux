"""Compare msgFlux GEPA with the saved Facility Support Analyzer tutorial in DSPy."""

import argparse
import ast
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
from typing import Literal

import msgflux as mf
from msgflux import nn
from msgflux.optim import GEPA, evaluate

DATA_SHA256 = "ef148f95bd89113982079250addfe96dc0015b6d375ea63e0450ff6751860440"
FIELDS = ("urgency", "sentiment", "categories")


class Urgency(mf.Signature):
    request: str = mf.InputField()
    reasoning: str = mf.OutputField()
    urgency: Literal["low", "medium", "high"] = mf.OutputField()


class Sentiment(mf.Signature):
    request: str = mf.InputField()
    reasoning: str = mf.OutputField()
    sentiment: Literal["positive", "neutral", "negative"] = mf.OutputField()


class Categories(mf.Signature):
    request: str = mf.InputField()
    reasoning: str = mf.OutputField()
    categories: list[
        Literal[
            "emergency_repair_services",
            "routine_maintenance_requests",
            "quality_and_safety_concerns",
            "specialized_cleaning_services",
            "general_inquiries",
            "sustainability_and_environmental_practices",
            "training_and_support_requests",
            "cleaning_services_scheduling",
            "customer_feedback_and_complaints",
            "facility_management_issues",
        ]
    ] = mf.OutputField()


SIGNATURES = dict(zip(FIELDS, (Urgency, Sentiment, Categories), strict=True))


def load_prompts(notebook):
    """Read the published instructions as data from the local notebook."""
    cells = json.loads(notebook.read_text())["cells"]
    baseline = {}
    optimized = {}
    for cell in cells:
        source = "".join(cell["source"])
        if "class FacilitySupportAnalyzerUrgency(" in source:
            for node in ast.parse(source).body:
                if isinstance(node, ast.ClassDef):
                    field = node.name.removeprefix("FacilitySupportAnalyzer").lower()
                    if field in FIELDS:
                        baseline[field] = ast.get_docstring(node)
        if "optimized_program.named_predictors()" in source:
            output = "".join("".join(item.get("text", [])) for item in cell["outputs"])
            optimized = dict(
                re.findall(
                    r"Predictor: (\w+)_module\.predict\n=+\nPrompt:\n(.*?)\n\*{5,}",
                    output,
                    re.DOTALL,
                )
            )
    if set(baseline) != set(FIELDS) or set(optimized) != set(FIELDS):
        raise ValueError(
            "Expected all three original and optimized prompts in notebook"
        )
    return baseline, optimized


def load_splits(dataset):
    raw = dataset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset differs from the documented revision")
    rows = [
        mf.Example({"request": item["fields"]["input"]}, json.loads(item["answer"]))
        for item in json.loads(raw)
    ]
    random.Random(0).shuffle(rows)
    train_end, val_end = int(0.33 * len(rows)), int(0.66 * len(rows))
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def field_metric(field):
    def score(example, prediction):
        expected = example.labels[field]
        actual = prediction[field]
        if field == "categories":
            value = fmean(
                enabled == (name in actual) for name, enabled in expected.items()
            )
        else:
            value = float(expected == actual)
        return value, {"feedback": f"Expected {expected}; predicted {actual}."}

    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--task-model", default="openai/gpt-6-luna")
    parser.add_argument("--reflection-model", default="openai/gpt-6-luna")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    mf.load_dotenv()
    task_model = mf.Model.chat_completion(args.task_model, temperature=1)
    reflector = mf.Model.chat_completion(args.reflection_model, temperature=1)
    baseline, published = load_prompts(args.notebook)
    train, validation, holdout = load_splits(args.dataset)
    results = {}
    prompts = {}
    counts = {}
    for field, signature in SIGNATURES.items():

        def build(instruction=baseline[field], signature=signature):
            return nn.Agent(
                name="facility",
                model=task_model,
                signature=signature,
                system_prompt=instruction,
            )

        metric = field_metric(field)
        fit = GEPA(reflector, max_metric_calls=18, seed=0).compile(
            build, trainset=train[:8], valset=validation[:4], metric=metric
        )
        counts[field] = fit.search.metric_calls
        prompts[field] = {
            "baseline": baseline[field],
            "published": published[field],
            "native": fit.search.best_text,
        }
        by_text = {}
        for variant, text in prompts[field].items():
            if text not in by_text:

                def evaluate_one(example, text=text):
                    return evaluate(build(text), [example], metric).rows[0].score

                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    by_text[text] = list(pool.map(evaluate_one, holdout))
            results.setdefault(variant, {})[field] = by_text[text]
            print(variant, field, round(fmean(by_text[text]), 4), flush=True)

    summary = {
        variant: {
            **{f: fmean(values[f]) for f in FIELDS},
            "overall": fmean(fmean(values[f]) for f in FIELDS),
        }
        for variant, values in results.items()
    }
    args.report.write_text(
        json.dumps(
            {
                "dataset_sha256": DATA_SHA256,
                "notebook_sha256": hashlib.sha256(
                    args.notebook.read_bytes()
                ).hexdigest(),
                "task_model": args.task_model,
                "reflection_model": args.reflection_model,
                "shuffle_seed": 0,
                "train": 8,
                "validation": 4,
                "test": len(holdout),
                "native_metric_calls": counts,
                "summary": summary,
                "scores": results,
                "instructions": prompts,
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
