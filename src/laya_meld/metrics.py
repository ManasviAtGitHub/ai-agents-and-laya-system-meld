"""Accuracy and F1 as MELD papers report them (weighted F1 is the headline number)."""


def per_class_f1(gold, pred, labels):
    scores = {}
    for label in labels:
        tp = sum(g == p == label for g, p in zip(gold, pred))
        fp = sum(p == label and g != label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        scores[label] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return scores


def evaluate(gold, pred, labels):
    f1 = per_class_f1(gold, pred, labels)
    support = {label: gold.count(label) for label in labels}
    return {
        "n": len(gold),
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold),
        "weighted_f1": sum(f1[l] * support[l] for l in labels) / len(gold),
        "macro_f1": sum(f1.values()) / len(labels),
        "per_class_f1": f1,
        "support": support,
        "confusion": {g: {p: sum(a == g and b == p for a, b in zip(gold, pred)) for p in labels}
                      for g in labels},
    }
