from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "tweets.csv"
SPLIT_DIR = ROOT / "data" / "splits"
META_PATH = ROOT / "task_metadata.json"

LABEL_NAMES: dict[str, list[str]] = {
    "emoji": [
        "red_heart", "smiling_face_with_hearteyes", "face_with_tears_of_joy",
        "two_hearts", "fire", "smiling_face_with_smiling_eyes",
        "smiling_face_with_sunglasses", "sparkles", "blue_heart", "face_blowing_a_kiss",
        "camera", "United_States", "sun", "purple_heart", "winking_face",
        "hundred_points", "beaming_face_with_smiling_eyes", "Christmas_tree",
        "camera_with_flash", "winking_face_with_tongue",
    ],
    "emotion": ["anger", "joy", "optimism", "sadness"],
    "hate": ["non-hate", "hate"],
    "irony": ["non_irony", "irony"],
    "offensive": ["non-offensive", "offensive"],
    "sentiment": ["negative", "neutral", "positive"],
    "stance_abortion": ["none", "against", "favor"],
    "stance_atheism": ["none", "against", "favor"],
    "stance_climate": ["none", "against", "favor"],
    "stance_feminist": ["none", "against", "favor"],
    "stance_hillary": ["none", "against", "favor"],
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
EMOJI_RE = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)
WS_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    t = raw.replace("\n", " ").replace("\r", " ")
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub("@user", t)
    t = HASHTAG_RE.sub(r"\1", t)
    t = EMOJI_RE.sub(" ", t)
    t = re.sub(r"&amp;", "&", t)
    t = re.sub(r"&lt;", "<", t)
    t = re.sub(r"&gt;", ">", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def stratified_split(df: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    counts = df["label"].value_counts()
    strat = df["label"] if counts.min() >= 2 else None
    train, rest = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=strat, shuffle=True
    )
    rest_counts = rest["label"].value_counts()
    strat_rest = rest["label"] if rest_counts.min() >= 2 else None
    val, test = train_test_split(
        rest, test_size=0.50, random_state=seed, stratify=strat_rest, shuffle=True
    )
    return {"train": train, "val": val, "test": test}


def main() -> None:
    print(f"[clean_tweets] Loading {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    n_raw = len(df)
    print(f"[clean_tweets] raw rows = {n_raw:,}")

    df = df.dropna(subset=["text"]).copy()
    df["label"] = df["label"].astype(int)

    df["clean_text"] = df["text"].astype(str).map(clean_text)

    before = len(df)
    df = df[df["clean_text"].str.len() > 0].copy()
    print(f"[clean_tweets] dropped {before - len(df):,} empty rows after cleaning")
    print(f"[clean_tweets] usable rows = {len(df):,}")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    metadata: dict = {
        "source_file": str(RAW_CSV.name),
        "raw_row_count": n_raw,
        "usable_row_count": int(len(df)),
        "tasks": {},
        "cleaning": {
            "url_removed": True,
            "mention_normalized_to_@user": True,
            "hashtag_hash_stripped": True,
            "emoji_removed": True,
            "html_entities_decoded": True,
            "whitespace_collapsed": True,
        },
        "split_strategy": "stratified 80/10/10 (seed=42)",
    }

    for task in sorted(df["about"].unique()):
        sub = df[df["about"] == task].reset_index(drop=True)
        splits = stratified_split(sub)

        task_dir = SPLIT_DIR / task
        task_dir.mkdir(parents=True, exist_ok=True)
        for name, part in splits.items():
            part[["clean_text", "label"]].to_csv(
                task_dir / f"{name}.csv", index=False
            )

        label_counts_total = sub["label"].value_counts().sort_index().to_dict()
        label_counts_total = {int(k): int(v) for k, v in label_counts_total.items()}
        n_classes = len(label_counts_total)
        names = LABEL_NAMES.get(task, [str(i) for i in range(n_classes)])

        metadata["tasks"][task] = {
            "n_total": int(len(sub)),
            "n_train": int(len(splits["train"])),
            "n_val": int(len(splits["val"])),
            "n_test": int(len(splits["test"])),
            "n_classes": int(n_classes),
            "label_names": names[:n_classes],
            "label_counts_total": label_counts_total,
            "label_counts_train": {
                int(k): int(v)
                for k, v in splits["train"]["label"].value_counts().sort_index().items()
            },
            "label_counts_val": {
                int(k): int(v)
                for k, v in splits["val"]["label"].value_counts().sort_index().items()
            },
            "label_counts_test": {
                int(k): int(v)
                for k, v in splits["test"]["label"].value_counts().sort_index().items()
            },
            "split_dir": str(task_dir.relative_to(ROOT)).replace("\\", "/"),
        }
        print(
            f"[clean_tweets] {task:<18} n={len(sub):>6}  "
            f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} "
            f"classes={n_classes}"
        )

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[clean_tweets] wrote {META_PATH}")
    print(f"[clean_tweets] wrote splits under {SPLIT_DIR}")


if __name__ == "__main__":
    main()
