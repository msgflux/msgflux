"""Run the pinned DSPy Facility Support notebook with OpenCode Go."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dspy
from reproduce_facility_support import FIELDS, load_splits, mf

from msgflux.accounts.store import Accounts

NOTEBOOK_SHA256 = "ef846b2c48c75fafd84ce23d56d3dc5341a4ffe90c351627c3d9dbc35b69b0c7"


def notebook_definitions(notebook):
    raw = notebook.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError("Expected the documented DSPy notebook revision")
    namespace = {"dspy": dspy, "json": json}
    cells = json.loads(raw)["cells"]
    # Only these three verified cells define the program, metric and feedback.
    for index in (10, 12, 17):
        code = compile(
            "".join(cells[index]["source"]), f"facility-cell-{index}", "exec"
        )
        exec(code, namespace)  # noqa: S102
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    mf.load_dotenv()
    defs = notebook_definitions(args.notebook)
    splits = [
        [
            dspy.Example(
                message=ex.inputs["request"], answer=json.dumps(ex.labels)
            ).with_inputs("message")
            for ex in split
        ]
        for split in load_splits(args.dataset)
    ]
    train, val, holdout = splits
    lm = dspy.LM(
        "openai/deepseek-v4.1-flash",
        engine="litellm",
        api_base="https://opencode.ai/inference/go/openai/v1",
        api_key=Accounts().get("opencode-go", account_name="work").value,
        extra_headers={"x-opencode-session": "facility-dspy-reference-20261118"},
        temperature=1,
        max_tokens=6000,
    )
    dspy.configure(lm=lm)
    initial = defs["program"]
    # Surface transport errors before DSPy's batch evaluator can score failures.
    defs["metric"](train[0], initial(message=train[0].message))
    saved = args.out / "optimized-program.json"
    if saved.exists():
        optimized = defs["FacilitySupportAnalyzerMM"]()
        optimized.load(str(saved))
    else:
        optimizer = dspy.GEPA(
            metric=defs["metric_with_feedback"],
            max_metric_calls=1643,
            num_threads=8,
            track_stats=True,
            use_merge=False,
            reflection_lm=dspy.LM("openai/gpt-6-luna", temperature=1, max_tokens=32000),
            log_dir=str(args.out / "gepa"),
        )
        optimized = optimizer.compile(initial, trainset=train, valset=val)
        stats = optimized.detailed_results
        (args.out / "search.json").write_text(
            json.dumps(
                {
                    "metric_calls": stats.total_metric_calls,
                    "validation_scores": stats.val_aggregate_scores,
                }
            )
        )
        optimized.save(str(saved))
    for label, model in (("baseline", initial), ("optimized", optimized)):

        def run(ex, model=model):
            prediction = model(message=ex.message)
            return {
                "score": defs["metric"](ex, prediction),
                "prediction": {k: getattr(prediction, k) for k in FIELDS},
            }

        with ThreadPoolExecutor(8) as pool:
            rows = list(pool.map(run, holdout))
        (args.out / f"{label}.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
