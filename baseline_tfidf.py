"""
Task 2 — Multi-Task Baseline Benchmarking.

Trains and evaluates TF-IDF (word 1-2grams + char 3-5grams) followed by
(a) Logistic Regression and (b) Linear SVM across ALL 11 TweetEval sub-tasks,
reporting Accuracy, Macro-F1, and Macro-Precision on each task's held-out
test split. Results are written to `full_baseline_results.json`.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "data" / "splits"
META_PATH = ROOT / "task_metadata.json"
RESULTS_PATH = ROOT / "full_baseline_results.json"


def load_split(task: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(SPLIT_DIR / task / f"{name}.csv")
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    return df


def build_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        transformer_list=[
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=200_000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=200_000,
                ),
            ),
        ]
    )


def build_models() -> dict[str, object]:
    return {
        "logreg": OneVsRestClassifier(
            LogisticRegression(
                solver="liblinear",
                C=4.0,
                class_weight="balanced",
                max_iter=2000,
            ),
            n_jobs=-1,
        ),
        "linear_svm": LinearSVC(C=1.0, class_weight="balanced"),
    }


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def main() -> None:
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    tasks = sorted(metadata["tasks"].keys())
    results: dict = {"tasks": {}, "summary": {}}

    for task in tasks:
        train_df = load_split(task, "train")
        val_df = load_split(task, "val")
        test_df = load_split(task, "test")

        X_train = pd.concat([train_df["clean_text"], val_df["clean_text"]]).tolist()
        y_train = pd.concat([train_df["label"], val_df["label"]]).tolist()
        X_test = test_df["clean_text"].tolist()
        y_test = test_df["label"].tolist()

        results["tasks"][task] = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_classes": int(metadata["tasks"][task]["n_classes"]),
            "models": {},
        }

        vectorizer = build_vectorizer()
        t0 = time.time()
        vectorizer.fit(X_train)
        Xtr = vectorizer.transform(X_train)
        Xte = vectorizer.transform(X_test)
        vec_time = time.time() - t0

        for name, clf in build_models().items():
            t0 = time.time()
            clf.fit(Xtr, y_train)
            fit_time = time.time() - t0
            preds = clf.predict(Xte)
            metrics = evaluate(y_test, preds)
            metrics["fit_time_sec"] = round(fit_time, 2)
            results["tasks"][task]["models"][name] = metrics
            print(
                f"[baseline] {task:<18} {name:<11} "
                f"acc={metrics['accuracy']:.4f} f1={metrics['macro_f1']:.4f} "
                f"prec={metrics['macro_precision']:.4f} (fit {fit_time:.1f}s)"
            )

        results["tasks"][task]["vectorize_time_sec"] = round(vec_time, 2)

    # Per-model summary (averaged across tasks).
    for model in ["logreg", "linear_svm"]:
        macro_f1s = [
            results["tasks"][t]["models"][model]["macro_f1"] for t in tasks
        ]
        accs = [results["tasks"][t]["models"][model]["accuracy"] for t in tasks]
        precs = [
            results["tasks"][t]["models"][model]["macro_precision"] for t in tasks
        ]
        results["summary"][model] = {
            "mean_macro_f1": float(sum(macro_f1s) / len(macro_f1s)),
            "mean_accuracy": float(sum(accs) / len(accs)),
            "mean_macro_precision": float(sum(precs) / len(precs)),
            "n_tasks": len(tasks),
        }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[baseline] wrote {RESULTS_PATH}")
    for m, s in results["summary"].items():
        print(
            f"[baseline] SUMMARY {m}: mean_macro_f1={s['mean_macro_f1']:.4f} "
            f"mean_acc={s['mean_accuracy']:.4f} "
            f"mean_macro_prec={s['mean_macro_precision']:.4f}"
        )


if __name__ == "__main__":
    main()
