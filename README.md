# ai-agents-and-laya-system-meld

A multi-agent group conversation system built on the MELD dataset (multi-party scenes from
*Friends*). [Laya](https://github.com/NandhaKishorM/laya), a small decision model that
returns typed answers with probabilities, handles perception and direction: what each line
expresses, and who should speak next. Small language models do the speaking.

```
you ─► Listener ─► Perceiver (Laya) ─► Director (Laya + rules) ─► Character agents (SLMs)
```

Everything runs locally on CPU.

## Status

Baselines established on the full MELD test set. Nothing is fine-tuned yet. Next step:
fine-tune Laya on MELD's training split and compare against the table below.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync                           # CPU-only PyTorch is pinned in pyproject.toml
uv run scripts/download_meld.py   # MELD transcripts, pinned to one upstream commit
uv run scripts/export_onnx.py     # Laya -> models/laya.onnx (~1.6 GB, a few minutes)
```

If you already have an export, set `LAYA_MODELS_DIR` to its folder instead of re-exporting.

## Baselines

```bash
uv run scripts/run_baseline.py majority
uv run scripts/run_baseline.py tfidf
uv run scripts/run_baseline.py laya
```

Results are written to `results/`: one JSON per method with metrics, the confusion matrix
and run details, plus Laya's per-utterance predictions and probabilities in
`results/predictions/`.

### Results: MELD test set (2,610 utterances, text only)

| Method | Emotion acc | Emotion weighted F1 | Emotion macro F1 | Sentiment acc | Sentiment weighted F1 | Sentiment macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Majority class (`neutral`) | 0.481 | 0.313 | 0.093 | 0.481 | 0.313 | 0.217 |
| TF-IDF + logistic regression (trained on MELD train) | **0.596** | **0.559** | **0.354** | **0.652** | **0.640** | **0.606** |
| Laya zero-shot (`ZERO_SHOT_V1`, no training) | 0.466 | 0.318 | 0.104 | 0.282 | 0.243 | 0.266 |

Weighted F1 is the number MELD papers usually report. Emotion has 7 classes, sentiment 3.

**What the zero-shot run shows.** The base Laya checkpoint doesn't transfer to this task:

- **Emotion:** it predicts `neutral` for 95% of utterances, so it scores about the same as the
  majority class. Only 5 of 345 `anger` lines and none of 402 `joy` lines were recognised.
- **Sentiment:** it errs the other way. It predicts `neutral` for only 8% of lines, although
  48% are neutral, and scores below the majority class.

A plain TF-IDF model trained on MELD beats zero-shot Laya on every metric. That sets the bar
for the next step: Laya fine-tuned on MELD's 9,989 training utterances should clearly beat
0.559 emotion weighted F1 to be worth using as the Perceiver.

**Setup details.** Laya input = previous line + speaker + current line, both questions asked in
one call, OpenVINO FP32 backend (identical answers to PyTorch in our backend check). TF-IDF
regularisation was chosen on the dev split, never on test. MELD labels are pinned to upstream
commit `e8cedf2`. On an i5-13420H CPU, Laya took about 1.1–1.6 s per utterance.

## Fine-tuning Laya on MELD

Training needs a GPU, so it runs on Kaggle:
[`notebooks/finetune_meld_2xT4_kaggle.ipynb`](notebooks/finetune_meld_2xT4_kaggle.ipynb)
(Accelerator **GPU T4 x2**, Internet **On**). The notebook clones this repo, so training data,
questions and metrics come from the same code as the baselines.

- **Data:** `laya_meld.finetune` turns each MELD utterance into two decisions (emotion and
  sentiment), using exactly the baseline questions (`MELD_V1` = `ZERO_SHOT_V1`). Only the
  weights change between the baseline and the fine-tuned run.
- **Training:** Laya's RLCD method (policy gradient with proper scoring rules, plus soft
  cross-entropy), adapted from the
  [upstream notebook](https://github.com/NandhaKishorM/laya/tree/9d955671415fc19f069b9cc998928075c1f255ec/notebooks).
  3 epochs, DDP on two T4s.
- **Model selection:** the epoch with the best **dev** emotion weighted F1 is kept, and
  temperatures are calibrated on dev. The test set is scored once, at the end.
- **Outputs:** the checkpoint (`laya-meld-v1.zip`) and `results/laya_finetuned_v1.json`,
  in the same format as the baselines.

To inspect the training records locally before spending GPU time:

```bash
uv run scripts/build_finetune_data.py   # data/finetune/meld_{train,dev}.jsonl, tokenization check
```

## Data and licences

- Code: Apache-2.0.
- MELD is **not** redistributed here. `scripts/download_meld.py` fetches the transcripts from
  [declare-lab/MELD](https://github.com/declare-lab/MELD) (GPL-3.0). The audio and video clips
  come from *Friends* and remain under their original copyright.
- Laya weights: Apache-2.0, downloaded from Hugging Face at first use.
