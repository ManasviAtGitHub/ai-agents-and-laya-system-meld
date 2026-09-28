"""Download the MELD transcript CSVs (train/dev/test) into data/meld/.

    uv run scripts/download_meld.py
"""
from laya_meld import data

if __name__ == "__main__":
    for split, path in data.download():
        print(f"{split:<5} {len(data.load(split)):>6} utterances  {path}")
