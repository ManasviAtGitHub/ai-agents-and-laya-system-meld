"""Write MELD train/dev as Laya training records and check they tokenize cleanly.

    uv run scripts/build_finetune_data.py

Outputs data/finetune/meld_{train,dev}.jsonl (not committed: derived from MELD). The Kaggle
notebook rebuilds the same records from the same code, so this script is for inspection and
for catching problems before spending GPU time.
"""
import json
import statistics

from transformers import AutoTokenizer

from laya_meld import finetune
from laya_meld.paths import DATA_DIR


def main():
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config

    model_dir = snapshot_download("convaiinnovations/laya", allow_patterns=["tokenizer/*", "rl_agent_config.json"])
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")

    out_dir = DATA_DIR.parent / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev"):
        recs = list(finetune.records(split))
        with open(out_dir / f"meld_{split}.jsonl", "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        items, skipped = finetune.tokenize(recs, tok)
        lengths = [len(it["ids"]) for it in items]
        print(f"{split:<5} {len(recs):>6} utterances -> {len(items):>6} items, {skipped} skipped, "
              f"tokens median {statistics.median(lengths):.0f} / max {max(lengths)}")


if __name__ == "__main__":
    main()
