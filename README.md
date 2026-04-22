# A Multi-Task Benchmark of Shallow and Parameter-Efficient Small Language Models on TweetEval

CS465 — Machine Learning Research Project.

We unify the TweetEval corpus into one reproducible 11-task benchmark
(sentiment, emotion, hate, offensive, irony, emoji, and five stance targets)
and compare:

1. A TF-IDF classical baseline (Logistic Regression and Linear SVM) with word
   1–2 grams + char-wb 3–5 grams.
2. A fine-tuned Phi-3-mini-4k-instruct (3.8 B parameters) trained with **QLoRA**
   (4-bit NF4 + LoRA r=16) on a single RTX 3060 (12 GB).

## Headline results

| Model | Mean macro-F1 (11 tasks) |
|---|---|
| Logistic Regression + TF-IDF | 0.6050 |
| Linear SVM + TF-IDF | 0.5944 |
| **Phi-3-mini + QLoRA** | **0.6536** |

Phi-3-mini wins on 9 / 11 sub-tasks. Biggest gains: low-resource stance
(+13 to +17 F1) and irony (+7). It loses on `emoji` and `offensive`, where
the 1,500-row subsample bottlenecks the SLM.

Full per-task numbers are in [`global_comparison.csv`](global_comparison.csv)
and [`slm_results.json`](slm_results.json) / [`full_baseline_results.json`](full_baseline_results.json).
The report (IEEE format) is in [`report.docx`](report.docx).

## Repository layout

```
clean_tweets.py            # global cleaner + 80/10/10 stratified splitter
baseline_tfidf.py          # TF-IDF + LogReg / Linear SVM per task
finetune_local.py          # Phi-3-mini QLoRA fine-tune (single 3060)
tweets.csv                 # raw 200,785-row TweetEval dump
task_metadata.json         # per-task label maps + class counts
full_baseline_results.json # LR + SVM results
slm_results.json           # Phi-3-mini per-task metrics
global_comparison.csv      # merged table (baselines + SLM + Δ)
misclassified.csv          # error log (predicted vs gold)
report.docx                # written report (6-page IEEE draft)
```

## Reproducing

```bash
# 1. Clean + split
python clean_tweets.py

# 2. Classical baselines (CPU; ~3 min total)
python baseline_tfidf.py

# 3. Phi-3-mini QLoRA (single RTX 3060, ~2 h for all 11 tasks)
python finetune_local.py
```

Requirements: Python 3.10+, `transformers`, `peft`, `bitsandbytes`,
`scikit-learn`, `pandas`. The `slm_adapters/` directory (~1.1 GB of LoRA
weights) is gitignored; ask for the Hugging Face / Drive link if needed.

## Hardware

All experiments ran on a single RTX 3060 (12 GB). Peak VRAM during
QLoRA fine-tuning: ~4.3 GB.

## License

MIT (code) — see paper for dataset citations (TweetEval, SemEval tasks).
