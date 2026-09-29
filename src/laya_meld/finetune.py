"""MELD as Laya training data.

One record per utterance, in the same shape as LocalLLaMA/typed-decisions (the dataset the
upstream fine-tuning notebook uses): a state, the questions, and a gold answer per question.
Targets are one-hot on the MELD label.

`tokenize` turns records into the items the RLCD trainer consumes. It calls Laya's own
`build_sequence` with the same max_len/head_max_len the fine-tuned checkpoint's config will
carry, so training inputs are token-for-token what inference builds.
"""
from . import data
from .questions import MELD_V1, state_for

# The trainer writes these into the fine-tuned checkpoint's rl_agent_config.json.
MAX_LEN = 1024
HEAD_MAX_LEN = 256


def gold_for(question, label):
    keys = list(question["criteria"])
    if label not in keys:
        raise ValueError(f"label {label!r} is not an option of {keys}")
    return {"label": label, "probabilities": {k: float(k == label) for k in keys}}


def records(split, questions=MELD_V1):
    for u in data.load(split):
        yield {
            "id": u["id"],
            "state": state_for(u),
            "questions": questions,
            "gold": {qid: gold_for(q, u[qid]) for qid, q in questions.items()},
        }


def tokenize(recs, tok):
    """Records -> trainer items: {ids, markers, qtype, target, label, id, qid}."""
    from laya.common import QTYPES, build_sequence, render_options

    items, skipped = [], 0
    for rec in recs:
        for qid, q in rec["questions"].items():
            internal = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
            seq, markers = build_sequence(tok, rec["state"], internal, MAX_LEN, HEAD_MAX_LEN)
            if len(markers) != len(render_options(internal)):
                skipped += 1
                continue
            probs = rec["gold"][qid]["probabilities"]
            target = [probs[k] for k in q["criteria"]]
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]],
                          "target": target, "label": target.index(max(target)),
                          "id": rec["id"], "qid": qid})
    return items, skipped
