"""Convert the labeled TSV into JSON sentences and create train/val/test splits.

Reads   data/labeled/labeling_template.tsv
Writes  data/splits/train.json / val.json / test.json
        each item: {"tokens": [...], "labels": [...]}

NOTE: with <200 sentences this split is a pipeline smoke test only.
Re-split ONCE when labeling reaches 200-300, then freeze the test set.
"""

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELED_PATH = ROOT / "data" / "labeled" / "labeling_template.tsv"
SPLITS_DIR = ROOT / "data" / "splits"
SEED = 42
VALID_LABELS = {"B-SKILL", "I-SKILL", "O"}


def load_sentences(path):
    sentences = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = int(row["sentence_id"])
            token = (row["token"] or "").strip()
            label = (row["label"] or "").strip()
            if label not in VALID_LABELS:
                sys.exit(f"Invalid label '{label}' in sentence {sid} - run scripts/validate_labels.py first")
            item = sentences.setdefault(sid, {"tokens": [], "labels": []})
            item["tokens"].append(token)
            item["labels"].append(label)
    return [sentences[sid] for sid in sorted(sentences)]


def main():
    sentences = load_sentences(LABELED_PATH)
    rng = random.Random(SEED)
    order = list(range(len(sentences)))
    rng.shuffle(order)

    n = len(sentences)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    splits = {
        "train": [sentences[i] for i in order[:n_train]],
        "val": [sentences[i] for i in order[n_train : n_train + n_val]],
        "test": [sentences[i] for i in order[n_train + n_val :]],
    }

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        out = SPLITS_DIR / f"{name}.json"
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tokens = sum(len(s["tokens"]) for s in data)
        skills = sum(1 for s in data for label in s["labels"] if label == "B-SKILL")
        print(f"{name:5}: {len(data):3d} sentences | {tokens:5d} tokens | {skills:3d} skill mentions -> {out.name}")

    print(f"\nSeed: {SEED} (do not change - keeps splits reproducible)")
    print("Reminder: smoke-test split. Re-split once at 200-300 sentences, then freeze.")


if __name__ == "__main__":
    main()
