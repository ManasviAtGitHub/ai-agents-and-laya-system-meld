"""Score predictions on MELD and write result files, identically for every method."""
import datetime
import json
import platform

from . import data, metrics
from .paths import RESULTS_DIR

TASKS = {"emotion": data.EMOTIONS, "sentiment": data.SENTIMENTS}


def score(utterances, predictions):
    """predictions: {utterance id: {"emotion": label, "sentiment": label, ...}}"""
    return {task: metrics.evaluate([u[task] for u in utterances],
                                   [predictions[u["id"]][task] for u in utterances], labels)
            for task, labels in TASKS.items()}


def laya_prediction(utterance_id, answers):
    row = {"id": utterance_id}
    for task in TASKS:
        row[task] = answers[task]["choice"]
        row[f"{task}_probabilities"] = answers[task]["probabilities"]
    return row


def write_result(name, method, details, results, split="test", results_dir=RESULTS_DIR):
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"method": method, "split": split, "details": details,
                   "meld_commit": data.MELD_COMMIT, "python": platform.python_version(),
                   "date": datetime.date.today().isoformat(), "results": results}, f, indent=2)
    return path


def summary(results):
    return "\n".join(f"{task:<10} n={r['n']}  accuracy {r['accuracy']:.3f}  weighted F1 {r['weighted_f1']:.3f}  "
                     f"macro F1 {r['macro_f1']:.3f}" for task, r in results.items())
