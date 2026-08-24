"""Validate the labeled TSV before converting it to a training dataset.

Usage:
    python scripts/validate_labels.py [path/to/file.tsv]

Checks:
    - header is sentence_id, token_id, token, label
    - labels are only B-SKILL / I-SKILL / O
    - no empty labels in the middle of the file
    - I-SKILL never starts an entity (must follow B-SKILL or I-SKILL)

Prints dataset stats and every skill extracted so far.
"""

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "labeled" / "labeling_template.tsv"
VALID_LABELS = {"B-SKILL", "I-SKILL", "O"}


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        sys.exit(f"File not found: {path}")

    errors = []
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames or []
        if [h.strip() for h in header[:4]] != ["sentence_id", "token_id", "token", "label"]:
            sys.exit(f"Unexpected header: {header}")
        for line_no, row in enumerate(reader, start=2):
            label = (row["label"] or "").strip()
            token = (row["token"] or "").strip()
            sid = row["sentence_id"].strip()
            if label and label not in VALID_LABELS:
                errors.append(f"line {line_no}: invalid label '{label}' (token '{token}')")
            if not token:
                errors.append(f"line {line_no}: empty token")
            rows.append({"sid": sid, "token": token, "label": label})

    prev_label = None
    prev_sid = None
    for line_no, row in enumerate(rows, start=2):
        if row["label"] == "I-SKILL":
            if row["sid"] != prev_sid or prev_label not in {"B-SKILL", "I-SKILL"}:
                errors.append(
                    f"line {line_no}: I-SKILL starts a new entity "
                    f"(sentence {row['sid']}, token '{row['token']}') - use B-SKILL instead"
                )
        if row["label"]:
            prev_label = row["label"]
            prev_sid = row["sid"]

    labeled = [r for r in rows if r["label"]]
    unlabeled = len(rows) - len(labeled)
    sids = sorted({r["sid"] for r in labeled}, key=int)
    partial = [s for s in sids if any(r["sid"] == s and not r["label"] for r in rows)]

    skills = []
    current = []
    for row in rows:
        if row["label"] == "B-SKILL":
            if current:
                skills.append(" ".join(current))
            current = [row["token"]]
        elif row["label"] == "I-SKILL" and current:
            current.append(row["token"])
        else:
            if current:
                skills.append(" ".join(current))
            current = []
    if current:
        skills.append(" ".join(current))

    print(f"Rows total        : {len(rows)}")
    print(f"Rows labeled      : {len(labeled)}")
    print(f"Rows unlabeled    : {unlabeled}")
    print(f"Sentences labeled : {len(sids)} (ids {sids[0]}..{sids[-1]})" if sids else "Sentences labeled: 0")
    if partial:
        print(f"Partially labeled sentences (some tokens missing a label): {', '.join(partial)}")
    print(f"Skill mentions    : {len(skills)}")
    print(f"Unique skills     : {len(set(skills))}")
    for skill, count in Counter(skills).most_common():
        print(f"  {count:>3}x  {skill}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("\nOK - BIO format is valid.")


if __name__ == "__main__":
    main()
