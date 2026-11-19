"""Compare Facility Support holdout scores and check predictions with both metrics."""

import argparse
import json
import math
import sqlite3
from contextlib import ExitStack, closing
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

from facility_full_native import prediction_key
from facility_full_reference import notebook_definitions
from reproduce_facility_support import FIELDS, field_metric, load_splits

from msgflux.optim import ComparisonResult, Evaluate, MetricSet


def cached_prediction(conn, instruction, example):
    key = prediction_key(instruction, example.inputs["request"])
    row = conn.execute("SELECT value FROM predictions WHERE key=?", (key,)).fetchone()
    if row is None:
        raise ValueError(
            "Missing cached prediction for the given instruction and request"
        )
    return json.loads(row[0])


def audit_selection(directory, field, validation, holdout, result):
    candidates = json.loads((directory / "gepa/candidates.json").read_text())
    selected = json.loads((directory / "artifact.json").read_text())["text"]
    seed = candidates[0]["current_candidate"]
    metric = field_metric(field)
    scores = {}
    with sqlite3.connect(
        (directory / "predictions.sqlite").as_uri() + "?mode=ro", uri=True
    ) as conn:
        for candidate in candidates:
            instruction = candidate["current_candidate"]
            scores[instruction] = fmean(
                metric(example, cached_prediction(conn, instruction, example))[0]
                for example in validation
            )
        for arm, instruction in (("baseline", seed), ("optimized", selected)):
            for ex, entry in zip(holdout, result["scores"][arm], strict=True):
                if (
                    cached_prediction(conn, instruction, ex)[field]
                    != entry["prediction"]
                ):
                    raise ValueError(f"{field} {arm} holdout instruction mismatch")
    for observed, expected in (
        (scores[seed], result["baseline_validation"]),
        (scores[selected], result["optimized_validation"]),
        (scores[selected], max(scores.values())),
    ):
        if not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{field} validation selection mismatch")
    return {
        "candidates": len(candidates),
        "baseline": scores[seed],
        "best": scores[selected],
    }


def audit_joint(directory, validation, holdout):
    candidates = json.loads((directory / "gepa/candidates.json").read_text())
    artifact = json.loads((directory / "artifact.json").read_text())
    selected = {u["target"]["module"]: u["text"] for u in artifact["updates"]}
    validation_scores = []
    with ExitStack() as stack:
        connections = {
            f: stack.enter_context(
                closing(
                    sqlite3.connect(
                        (directory / f / "predictions.sqlite").resolve().as_uri()
                        + "?mode=ro",
                        uri=True,
                    )
                )
            )
            for f in FIELDS
        }
        for candidate in candidates:
            validation_scores.append(
                fmean(
                    field_metric(f)(
                        ex, cached_prediction(connections[f], candidate[f], ex)
                    )[0]
                    for ex in validation
                    for f in FIELDS
                )
            )
        for arm, candidate in (("baseline", candidates[0]), ("optimized", selected)):
            rows = json.loads((directory / f"{arm}.json").read_text())
            for ex, row in zip(holdout, rows, strict=True):
                for f in FIELDS:
                    if (
                        cached_prediction(connections[f], candidate[f], ex)[f]
                        != row["prediction"][f]
                    ):
                        raise ValueError(f"Joint {arm} prediction mismatch for {f}")
    selected_score = validation_scores[candidates.index(selected)]
    if not math.isclose(
        selected_score, max(validation_scores), rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("The joint artifact differs from the best validated candidate")
    return {
        "candidates": len(candidates),
        "baseline": validation_scores[0],
        "best": selected_score,
    }


def evaluate_saved(evaluator, rows):
    # The saved predictions follow the same holdout order as the evaluator.
    if len(rows) != len(evaluator.dataset):
        raise ValueError("Expected one saved prediction per holdout example")
    predictions = iter(row["prediction"] for row in rows)
    result = evaluator(lambda **_: next(predictions))
    for entry, stored in zip(result.rows, rows, strict=True):
        component_mean = fmean(entry.metric_values[f][0] for f in FIELDS)
        if not (
            math.isclose(component_mean, entry.score, rel_tol=0, abs_tol=1e-12)
            and math.isclose(component_mean, stored["score"], rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("Holdout prediction and recorded metrics disagree")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--reference-dir", type=Path, help="Reuse an existing DSPy control"
    )
    parser.add_argument(
        "--joint-dir", type=Path, help="Also compare a joint msgFlux search"
    )
    args = parser.parse_args()
    reference = args.reference_dir or args.out / "reference"
    defs = notebook_definitions(args.notebook)
    _, validation, examples = load_splits(args.dataset)
    native = {
        field: json.loads((args.out / field / "holdout.json").read_text())
        for field in FIELDS
    }

    def notebook_metric(example, prediction):
        return defs["metric"](
            {"answer": json.dumps(example.labels)}, SimpleNamespace(**prediction)
        )

    metrics = MetricSet(
        {"overall": notebook_metric, **{f: field_metric(f) for f in FIELDS}},
        primary="overall",
    )
    evaluator = Evaluate(dataset=examples, metric=metrics)
    paired_metrics = {}
    summary = {}
    checks = 0
    paired = {}
    backends = ["msgflux", "dspy"]
    if args.joint_dir:
        backends.append("msgflux_joint")
    for backend in backends:
        arms = {}
        paired[backend] = {}
        for arm in ("baseline", "optimized"):
            if backend in {"dspy", "msgflux_joint"}:
                directory = reference if backend == "dspy" else args.joint_dir
                rows = json.loads((directory / f"{arm}.json").read_text())
            else:
                by_field = [native[field]["scores"][arm] for field in FIELDS]
                rows = [
                    {
                        "prediction": {
                            field: entry["prediction"]
                            for field, entry in zip(FIELDS, entries, strict=True)
                        },
                        "score": fmean(entry["score"] for entry in entries),
                    }
                    for entries in zip(*by_field, strict=True)
                ]
            result = evaluate_saved(evaluator, rows)
            arms[arm] = result
            checks += len(rows)
            summary[f"{backend}_{arm}"] = result.mean_scores
            paired[backend][arm] = result.scores
        comparison = ComparisonResult.pair(arms["baseline"], arms["optimized"])
        paired_metrics[backend] = {}
        for name in metrics.metrics:
            view = comparison.for_metric(name)
            paired_metrics[backend][name] = {
                "mean_delta": view.mean_delta,
                "improved": len(view.improved),
                "unchanged": len(view.unchanged),
                "regressed": len(view.regressed),
                "deltas": view.deltas,
            }
    report = {
        "metrics_verified": checks,
        "dspy_search": json.loads((reference / "search.json").read_text()),
        "validation_selection": {
            field: audit_selection(
                (args.out / field).resolve(), field, validation, examples, native[field]
            )
            for field in FIELDS
        },
        "holdout": len(examples),
        "task_model": "opencode-go/deepseek-v4.1-flash",
        "reflection_model": "openai/gpt-6-luna",
        "native_engine_evaluations": {f: native[f]["metric_calls"] for f in FIELDS},
        "test_means": summary,
        "paired_scores": paired,
        "paired_metrics": paired_metrics,
    }
    if args.joint_dir:
        report["joint_selection"] = audit_joint(args.joint_dir, validation, examples)
        report["joint_search"] = json.loads(
            (args.joint_dir / "search.json").read_text()
        )
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
