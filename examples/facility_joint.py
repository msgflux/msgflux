"""Run one GEPA search on the complete three-agent Facility Support program."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean

from facility_full_native import (
    build_facility_agent,
    prepare_cache,
    task_and_reflection_models,
)
from reproduce_facility_support import (
    FIELDS,
    field_metric,
    load_prompts,
    load_splits,
    mf,
    nn,
)

from msgflux.optim import GEPA, TextTarget


def program_metric(example, prediction):
    scores = []
    feedback = {}
    for field in FIELDS:
        score, detail = field_metric(field)(example, prediction)
        scores.append(score)
        feedback[f"{field}_specific_info"] = detail
    return fmean(scores), feedback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-metric-calls", type=int, default=1643)
    args = parser.parse_args()
    mf.load_dotenv()
    seed, _ = load_prompts(args.notebook)
    train, val, holdout = load_splits(args.dataset)
    task_model, reflection_model = task_and_reflection_models()
    databases = {field: prepare_cache(args.out / field) for field in FIELDS}

    class FacilityProgram(nn.Module):
        def __init__(self):
            super().__init__()
            for field in FIELDS:
                setattr(
                    self,
                    field,
                    build_facility_agent(
                        field, task_model, seed[field], databases[field]
                    ),
                )

        def forward(self, request):
            return {
                field: getattr(self, field)(request=request)[field] for field in FIELDS
            }

    fit = GEPA(
        reflection_model,
        max_metric_calls=args.max_metric_calls,
        workers=args.workers,
        run_dir=args.out / "gepa",
    ).compile(
        FacilityProgram,
        targets={field: TextTarget(module=field) for field in FIELDS},
        trainset=train,
        valset=val,
        metric=program_metric,
    )
    fit.save(args.out / "artifact.json")
    (args.out / "search.json").write_text(
        json.dumps(
            {
                "metric_calls": fit.search.metric_calls,
                "baseline_validation": fit.search.baseline_score,
                "optimized_validation": fit.search.best_score,
            }
        )
    )
    for arm, factory in (("baseline", FacilityProgram), ("optimized", fit.build)):

        def score_one(example, factory=factory):
            prediction = factory()(**example.inputs)
            return {
                "score": program_metric(example, prediction)[0],
                "prediction": prediction,
            }

        with ThreadPoolExecutor(args.workers) as executor:
            rows = list(executor.map(score_one, holdout))
        (args.out / f"{arm}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
