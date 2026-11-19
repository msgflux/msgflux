"""Run one full-budget msgFlux Facility Support component. Start once per field."""

import argparse
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from reproduce_facility_support import (
    FIELDS,
    SIGNATURES,
    field_metric,
    load_prompts,
    load_splits,
    mf,
    nn,
)

from msgflux.accounts.resolver import AccountCredentialResolver
from msgflux.accounts.store import Accounts
from msgflux.optim import GEPA, TextTarget, evaluate


def prediction_key(instruction, request):
    return hashlib.sha256(json.dumps([instruction, request]).encode()).hexdigest()


def task_and_reflection_models():
    task_model = mf.Model.chat_completion(
        "opencode-go/deepseek-v4.1-flash",
        credential_resolver=AccountCredentialResolver(
            Accounts(), "opencode-go", account_name="work"
        ),
    )
    return task_model, mf.Model.chat_completion("openai/gpt-6-luna")


def prepare_cache(directory):
    directory.mkdir(exist_ok=True, parents=True)
    db = directory / "predictions.sqlite"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS predictions "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
    return db


class FacilityAgent(nn.Agent):
    """Memoize each inference at the agent boundary while preserving module hooks."""

    def forward(self, *, request):
        key = prediction_key(self.get_task_instruction(), request)
        with closing(sqlite3.connect(self._prediction_db, timeout=60)) as conn, conn:
            query = "SELECT value FROM predictions WHERE key=?"
            row = conn.execute(query, (key,)).fetchone()
            if row:
                return json.loads(row[0])
            value = dict(super().forward(request=request))
            conn.execute(
                "INSERT OR IGNORE INTO predictions VALUES (?, ?)",
                (key, json.dumps(value)),
            )
            return json.loads(conn.execute(query, (key,)).fetchone()[0])


def build_facility_agent(field, task_model, instruction, db):
    agent = FacilityAgent(
        name="facility",
        model=task_model,
        signature=SIGNATURES[field],
        system_prompt=instruction,
    )
    agent._prediction_db = db
    return agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field", choices=FIELDS)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--feedback-only",
        action="store_true",
        help="Reproduce the original metric-only reflection condition",
    )
    args = parser.parse_args()
    field = args.field
    out = args.out / field
    out.mkdir(exist_ok=True, parents=True)
    mf.load_dotenv()
    task_model, reflector = task_and_reflection_models()
    seed, _ = load_prompts(args.notebook)
    train, val, test = load_splits(args.dataset)
    db = prepare_cache(out)

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.agent = build_facility_agent(field, task_model, seed[field], db)

        def forward(self, request):
            return self.agent(request=request)

    optimizer = GEPA(
        reflector, max_metric_calls=1643, workers=args.workers, run_dir=out / "gepa"
    )
    fit = optimizer.compile(
        Predictor,
        target=TextTarget(module="agent"),
        trainset=train,
        valset=val,
        metric=field_metric(field),
        include_example_context=not args.feedback_only,
    )
    fit.save(out / "artifact.json")
    results = {}
    for variant, factory in (("baseline", Predictor), ("optimized", fit.build)):

        def predict_one(example, factory=factory):
            row = evaluate(factory(), [example], field_metric(field)).rows[0]
            return {"prediction": row.prediction[field], "score": row.score}

        with ThreadPoolExecutor(args.workers) as pool:
            results[variant] = list(pool.map(predict_one, test))
    (out / "holdout.json").write_text(
        json.dumps(
            {
                "metric_calls": fit.search.metric_calls,
                "baseline_validation": fit.search.baseline_score,
                "optimized_validation": fit.search.best_score,
                "scores": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
