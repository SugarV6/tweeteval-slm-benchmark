# TweetEval SLM Benchmark

My CS465 machine learning project. I compare a classical TF-IDF + linear
model baseline against a fine-tuned Phi-3-mini (3.8B) small language model
across all 11 TweetEval sub-tasks on a single 12 GB RTX 3060.

## Tasks

Sentiment, emotion, hate, offensive, irony, emoji, and five stance targets
(abortion, atheism, climate, feminism, Hillary). Everything is cleaned and
split 80/10/10 from one raw `tweets.csv` (200,785 rows).

## Results

Mean macro-F1 across the 11 tasks:

| Model | Macro-F1 |
|---|---|
| Logistic Regression (TF-IDF word 1-2 grams + char 3-5 grams) | 0.6050 |
| Linear SVM (same features) | 0.5944 |
| Phi-3-mini + QLoRA (r=16, 4-bit NF4) | **0.6536** |

Phi-3 wins on 9 of 11 tasks. Big gains on stance (+13 to +17 F1) and
irony (+7). It loses on `emoji` and `offensive` because I had to cap
training at 1500 rows per task to fit in my GPU budget, and those two
tasks have much more data available.

Per-task numbers are in `global_comparison.csv`. The full report (6-page
IEEE draft) is `report.docx`.

## Files

```
clean_tweets.py             cleaner + 80/10/10 stratified splits
baseline_tfidf.py           LogReg + LinearSVM per task
finetune_local.py           Phi-3-mini QLoRA fine-tune
tweets.csv                  raw 200,785-row dump
task_metadata.json          per-task label maps and counts
full_baseline_results.json  LR + SVM metrics
slm_results.json            Phi-3 per-task metrics
global_comparison.csv       merged table (baselines + SLM + delta)
misclassified.csv           sample of wrong predictions
report.docx                 written report
```

## How to run

```bash
python clean_tweets.py      # ~30 sec
python baseline_tfidf.py    # ~3 min CPU
python finetune_local.py    # ~2 h on RTX 3060
```

Needs Python 3.10+, `transformers`, `peft`, `bitsandbytes`, `scikit-learn`,
`pandas`, `datasets`.

## Notes

- `slm_adapters/` (about 1.1 GB of LoRA weights) is not in the repo.
- `data/splits/` is also ignored — regenerate with `clean_tweets.py`.
- Peak VRAM during fine-tuning was around 4.3 GB.
- Single seed (42). The results would need more seeds to be sure of the
  gap on the small tasks.
