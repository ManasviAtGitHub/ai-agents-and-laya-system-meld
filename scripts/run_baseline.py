"""Score a baseline on the MELD test set for emotion (7 classes) and sentiment (3 classes).

    uv run scripts/run_baseline.py majority
    uv run scripts/run_baseline.py tfidf
    uv run scripts/run_baseline.py laya [--limit N] [--backend openvino]

Writes results/<method>.json (metrics + run metadata). The laya method also streams
predictions to results/predictions/laya_zero_shot_v1.jsonl and resumes from it, so an
interrupted run continues where it stopped.
"""
import argparse
import json
import time
from importlib.metadata import version

from laya_meld import data, evaluation, metrics
from laya_meld.evaluation import TASKS
from laya_meld.paths import RESULTS_DIR


def majority(train, test, args):
    most_common = {task: max(labels, key=[u[task] for u in train].count) for task, labels in TASKS.items()}
    return {u["id"]: dict(most_common) for u in test}, {"majority_label": most_common}


def tfidf(train, test, args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union

    dev = data.load("dev")
    predictions = {u["id"]: {} for u in test}
    chosen = {}
    for task, labels in TASKS.items():
        best = None
        for c in (0.3, 1.0, 3.0, 10.0):  # pick regularisation on dev, never on test
            model = make_pipeline(
                make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                           TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
                LogisticRegression(C=c, max_iter=2000),
            )
            model.fit([u["text"] for u in train], [u[task] for u in train])
            dev_f1 = metrics.evaluate([u[task] for u in dev], list(model.predict([u["text"] for u in dev])),
                                      labels)["weighted_f1"]
            if best is None or dev_f1 > best[0]:
                best = (dev_f1, c, model)
        chosen[task] = {"C": best[1], "dev_weighted_f1": best[0]}
        for u, label in zip(test, best[2].predict([u["text"] for u in test])):
            predictions[u["id"]][task] = label
    return predictions, {"features": "word 1-2gram + char_wb 2-5gram TF-IDF", "model": "LogisticRegression",
                         "selected_on_dev": chosen}


def laya(train, test, args):
    from laya_meld import backends
    from laya_meld.questions import ZERO_SHOT_V1, state_for

    out = RESULTS_DIR / "predictions" / "laya_zero_shot_v1.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        with open(out, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done[row["id"]] = row
    todo = [u for u in test if u["id"] not in done]
    print(f"{len(done)} already scored, {len(todo)} to go", flush=True)

    agent = backends.load(args.backend)
    t0 = time.perf_counter()
    with open(out, "a", encoding="utf-8") as f:
        for i, u in enumerate(todo, 1):
            answers = agent.predict(state_for(u), ZERO_SHOT_V1)["answers"]
            row = evaluation.laya_prediction(u["id"], answers)
            f.write(json.dumps(row) + "\n")
            f.flush()
            done[u["id"]] = row
            if i % 50 == 0 or i == len(todo):
                rate = (time.perf_counter() - t0) / i
                print(f"  {len(done)}/{len(test)}  {rate * 1000:.0f} ms each, "
                      f"~{rate * (len(todo) - i) / 60:.0f} min left", flush=True)
    return done, {"questions": "ZERO_SHOT_V1", "backend": args.backend, "laya_version": version("laya"),
                  "input": "previous line + speaker + line"}


METHODS = {"majority": majority, "tfidf": tfidf, "laya": laya}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--limit", type=int, help="score only the first N test utterances (smoke runs)")
    parser.add_argument("--backend", default="openvino", help="laya only: torch | onnx | openvino")
    args = parser.parse_args()

    train, test = data.load("train"), data.load("test")
    if args.limit:
        test = test[:args.limit]
    predictions, details = METHODS[args.method](train, test, args)
    results = evaluation.score(test, predictions)

    name = args.method if not args.limit else f"{args.method}_limit{args.limit}"
    evaluation.write_result(name, args.method, details, results)
    print(evaluation.summary(results))


if __name__ == "__main__":
    main()
