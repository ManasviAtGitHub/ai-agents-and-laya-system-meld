"""MELD transcripts: download and load.

The CSVs come from the official repository, pinned to one commit so every run scores
against the same labels. Audio and video are not needed for the text baselines.
"""
import collections
import csv
import urllib.request

from .paths import DATA_DIR

MELD_COMMIT = "e8cedf27b5d2877e198332c957127e16eb214afe"
MELD_URL = "https://raw.githubusercontent.com/declare-lab/MELD/{commit}/data/MELD/{split}_sent_emo.csv"
SPLITS = ("train", "dev", "test")

EMOTIONS = ("neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear")
SENTIMENTS = ("neutral", "positive", "negative")


def csv_path(split):
    return DATA_DIR / f"{split}_sent_emo.csv"


def download(splits=SPLITS):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in splits:
        path = csv_path(split)
        if not path.exists():
            urllib.request.urlretrieve(MELD_URL.format(commit=MELD_COMMIT, split=split), path)
        yield split, path


def load(split):
    """Utterances in dialogue order, each with the line spoken just before it (or "")."""
    path = csv_path(split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run scripts/download_meld.py")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    dialogues = collections.defaultdict(list)
    for row in rows:
        dialogues[int(row["Dialogue_ID"])].append(row)

    utterances = []
    for dialogue_id in sorted(dialogues):
        turns = sorted(dialogues[dialogue_id], key=lambda r: int(r["Utterance_ID"]))
        previous = ""
        for row in turns:
            utterances.append({
                "id": f"{split}/{dialogue_id}/{row['Utterance_ID']}",
                "dialogue_id": dialogue_id,
                "utterance_id": int(row["Utterance_ID"]),
                "speaker": row["Speaker"],
                "text": row["Utterance"],
                "previous": previous,
                "emotion": row["Emotion"],
                "sentiment": row["Sentiment"],
            })
            previous = f"{row['Speaker']}: {row['Utterance']}"
    return utterances
