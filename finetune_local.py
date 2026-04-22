"""
Task 3 — Local SLM Multi-Task Fine-Tuning.

Fine-tunes `microsoft/Phi-3-mini-4k-instruct` independently on every one of
the 11 TweetEval sub-tasks from Task 1, using 4-bit NF4 quantization +
LoRA (r=16, alpha=32) so that the whole pipeline fits on a 12 GB RTX 3060.
For each task the script:

    1. Loads the stratified train/val/test splits produced by clean_tweets.py.
    2. Builds a Phi-3 Sequence-Classification head on top of the frozen,
       4-bit quantized backbone.
    3. Trains LoRA adapters with `Trainer` using paged-AdamW-8bit.
    4. Evaluates on the held-out test split (macro-F1, accuracy,
       macro-precision).
    5. Saves per-task LoRA adapters, a global comparison CSV vs the TF-IDF
       baselines, and a combined `misclassified.csv` sample.

The script is idempotent — if `slm_results.json` already contains a task's
metrics it is skipped, so training can be resumed after a crash.
"""

from __future__ import annotations

import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Phi3ForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# ------------------------- config --------------------------------------

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "data" / "splits"
META_PATH = ROOT / "task_metadata.json"
BASELINE_PATH = ROOT / "full_baseline_results.json"
SLM_RESULTS_PATH = ROOT / "slm_results.json"
COMPARISON_CSV = ROOT / "global_comparison.csv"
MISCLASSIFIED_CSV = ROOT / "misclassified.csv"
ADAPTER_DIR = ROOT / "slm_adapters"

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MAX_LEN = 128
MAX_TRAIN_PER_TASK = 1500       # cap to keep wall-clock tractable on a 3060
MAX_TEST_PER_TASK = 600
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ------------------------- data ----------------------------------------

def _read(task: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(SPLIT_DIR / task / f"{split}.csv")
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    return df


def _subsample(df: pd.DataFrame, cap: int, seed: int = SEED) -> pd.DataFrame:
    if len(df) <= cap:
        return df
    # stratified subsample to preserve label balance
    per_class = cap // df["label"].nunique()
    parts = []
    for lbl, g in df.groupby("label"):
        parts.append(g.sample(min(len(g), max(1, per_class)), random_state=seed))
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(out) > cap:
        out = out.head(cap)
    return out


def build_hf_dataset(df: pd.DataFrame, tokenizer) -> Dataset:
    ds = Dataset.from_pandas(df[["clean_text", "label"]].rename(
        columns={"clean_text": "text"}))
    def tok(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=MAX_LEN, padding=False
        )
    ds = ds.map(tok, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    return ds


# ------------------------- modeling ------------------------------------

def make_model(n_labels: int, tokenizer):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Phi3ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=n_labels,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="eager",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        modules_to_save=["score"],   # keep the classification head trainable
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "macro_precision": precision_score(
            labels, preds, average="macro", zero_division=0
        ),
    }


# ------------------------- training loop -------------------------------

def train_one_task(task: str, tokenizer, metadata) -> dict:
    print(f"\n================= TASK: {task} =================")
    n_labels = int(metadata["tasks"][task]["n_classes"])
    label_names = metadata["tasks"][task]["label_names"]

    train_df = _read(task, "train")
    val_df = _read(task, "val")
    test_df = _read(task, "test")

    train_df = _subsample(train_df, MAX_TRAIN_PER_TASK)
    val_df = _subsample(val_df, min(400, MAX_TEST_PER_TASK))
    test_df = _subsample(test_df, MAX_TEST_PER_TASK)

    print(f"[{task}] train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"n_labels={n_labels}")

    train_ds = build_hf_dataset(train_df, tokenizer)
    val_ds = build_hf_dataset(val_df, tokenizer)
    test_ds = build_hf_dataset(test_df, tokenizer)

    model = make_model(n_labels, tokenizer)

    # Epochs: more epochs for small tasks, fewer for large
    n_train = len(train_df)
    if n_train <= 800:
        epochs = 4
    elif n_train <= 1500:
        epochs = 3
    else:
        epochs = 2

    out_dir = ADAPTER_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0,
        logging_steps=25,
        save_strategy="no",
        eval_strategy="no",
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        seed=SEED,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0

    # -------- test evaluation --------
    model.eval()
    eval_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=16, collate_fn=collator
    )
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**batch).logits.float()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro",
                                   zero_division=0)),
        "macro_precision": float(precision_score(all_labels, all_preds,
                                                 average="macro",
                                                 zero_division=0)),
        "n_train": n_train,
        "n_test": len(test_df),
        "epochs": epochs,
        "train_time_sec": round(train_sec, 1),
    }
    print(f"[{task}] TEST acc={metrics['accuracy']:.4f} "
          f"f1={metrics['macro_f1']:.4f} prec={metrics['macro_precision']:.4f} "
          f"(train {train_sec:.0f}s)")

    # Save LoRA adapter
    try:
        model.save_pretrained(str(out_dir / "lora"))
    except Exception as e:
        print(f"[{task}] adapter save warning: {e}")

    # -------- misclassifications (up to 20) --------
    texts = test_df["clean_text"].tolist()
    wrong_rows = []
    for i, (yt, yp) in enumerate(zip(all_labels.tolist(), all_preds.tolist())):
        if yt != yp:
            wrong_rows.append({
                "task": task,
                "text": texts[i],
                "true_label": label_names[yt] if yt < len(label_names) else yt,
                "pred_label": label_names[yp] if yp < len(label_names) else yp,
            })
    random.shuffle(wrong_rows)
    wrong_rows = wrong_rows[:20]

    # cleanup GPU
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    return {"metrics": metrics, "misclassified": wrong_rows}


# ------------------------- main ----------------------------------------

def main() -> None:
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    tasks = sorted(metadata["tasks"].keys())
    print(f"[finetune] {len(tasks)} tasks: {tasks}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Resume support
    slm_results: dict = {"model": MODEL_NAME, "tasks": {}}
    if SLM_RESULTS_PATH.exists():
        slm_results = json.loads(SLM_RESULTS_PATH.read_text())
        print(f"[finetune] resumed; already have: "
              f"{list(slm_results.get('tasks', {}).keys())}")

    all_misclassified: list[dict] = []
    mis_path = MISCLASSIFIED_CSV
    if mis_path.exists():
        try:
            all_misclassified = pd.read_csv(mis_path).to_dict(orient="records")
        except Exception:
            all_misclassified = []

    # small → large so we fail fast on the easy tasks
    order = sorted(tasks, key=lambda t: metadata["tasks"][t]["n_train"])

    for task in order:
        if task in slm_results.get("tasks", {}):
            print(f"[finetune] {task}: already done, skip")
            continue
        try:
            result = train_one_task(task, tokenizer, metadata)
        except torch.cuda.OutOfMemoryError as e:
            print(f"[finetune] OOM on {task}: {e}")
            gc.collect(); torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[finetune] ERROR on {task}: {type(e).__name__}: {e}")
            gc.collect(); torch.cuda.empty_cache()
            continue

        slm_results["tasks"][task] = result["metrics"]
        SLM_RESULTS_PATH.write_text(json.dumps(slm_results, indent=2))

        all_misclassified.extend(result["misclassified"])
        if all_misclassified:
            pd.DataFrame(all_misclassified).to_csv(mis_path, index=False)

    # ----- global comparison CSV -----
    rows = []
    for task in tasks:
        base_lr = (baseline["tasks"].get(task, {}).get("models", {})
                   .get("logreg", {}))
        base_sv = (baseline["tasks"].get(task, {}).get("models", {})
                   .get("linear_svm", {}))
        slm = slm_results["tasks"].get(task, {})
        row = {
            "task": task,
            "n_classes": metadata["tasks"][task]["n_classes"],
            "n_train_total": metadata["tasks"][task]["n_train"],
            "n_test_total": metadata["tasks"][task]["n_test"],
            "baseline_logreg_f1": base_lr.get("macro_f1"),
            "baseline_logreg_acc": base_lr.get("accuracy"),
            "baseline_svm_f1": base_sv.get("macro_f1"),
            "baseline_svm_acc": base_sv.get("accuracy"),
            "slm_f1": slm.get("macro_f1"),
            "slm_acc": slm.get("accuracy"),
            "slm_macro_precision": slm.get("macro_precision"),
            "slm_train_samples": slm.get("n_train"),
            "slm_test_samples": slm.get("n_test"),
            "slm_epochs": slm.get("epochs"),
            "slm_train_time_sec": slm.get("train_time_sec"),
        }
        if row["slm_f1"] is not None and row["baseline_logreg_f1"] is not None:
            row["delta_vs_logreg_f1"] = row["slm_f1"] - row["baseline_logreg_f1"]
        else:
            row["delta_vs_logreg_f1"] = None
        if row["slm_f1"] is not None and row["baseline_svm_f1"] is not None:
            row["delta_vs_svm_f1"] = row["slm_f1"] - row["baseline_svm_f1"]
        else:
            row["delta_vs_svm_f1"] = None
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(COMPARISON_CSV, index=False)
    print(f"[finetune] wrote {COMPARISON_CSV}")
    print(comp_df[["task", "baseline_logreg_f1", "slm_f1",
                   "delta_vs_logreg_f1"]].to_string(index=False))

    if all_misclassified:
        pd.DataFrame(all_misclassified).to_csv(mis_path, index=False)
        print(f"[finetune] wrote {mis_path} ({len(all_misclassified)} rows)")


if __name__ == "__main__":
    main()
