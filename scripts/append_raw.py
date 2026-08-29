"""Append newly collected raw sentences to the existing labeling template.

Never touches existing rows/labels. New sentences continue the sentence_id
sequence with empty labels, ready for labeling. Deduplicates against
sentences already indexed (by lowercase text).
"""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "jds.txt"
TEMPLATE_PATH = ROOT / "data" / "labeled" / "labeling_template.tsv"
INDEX_PATH = ROOT / "data" / "labeled" / "sentences_index.tsv"
DELIM = "\t"

TOKEN_RE = re.compile(r"[^\W_]+(?:[._#+][^\W_]+)*|_+|[^\w\s]+", re.UNICODE)


def tokenize(sentence):
    return TOKEN_RE.findall(sentence)


def parse_sentences(raw_text):
    sentences = []
    source = ""
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("==="):
            continue
        if line.startswith("["):
            source = line
            continue
        sentences.append((source, line))
    return sentences


def ensure_trailing_newline(path):
    text = path.read_text(encoding="utf-8-sig")
    if text and not text.endswith("\n"):
        with path.open("a", encoding="utf-8", newline="") as f:
            f.write("\r\n")


def main():
    ensure_trailing_newline(TEMPLATE_PATH)
    ensure_trailing_newline(INDEX_PATH)
    with INDEX_PATH.open(encoding="utf-8-sig", newline="") as f:
        index_rows = list(csv.reader(f, delimiter=DELIM))
    header, existing = index_rows[0], index_rows[1:]
    known_texts = {row[2].strip().lower() for row in existing}
    max_sid = max(int(row[0]) for row in existing)

    new_sentences = []
    seen_now = set()
    skipped_dup_total = 0
    for source, text in parse_sentences(RAW_PATH.read_text(encoding="utf-8")):
        key = text.lower()
        if key in known_texts or key in seen_now:
            skipped_dup_total += 1
            continue
        seen_now.add(key)
        new_sentences.append((source, text))

    if not new_sentences:
        print("Nothing new to append.")
        return

    with TEMPLATE_PATH.open("a", encoding="utf-8-sig", newline="") as f, INDEX_PATH.open(
        "a", encoding="utf-8-sig", newline=""
    ) as g:
        writer = csv.writer(f, delimiter=DELIM)
        index_writer = csv.writer(g, delimiter=DELIM)
        appended_tokens = 0
        for offset, (source, text) in enumerate(new_sentences, start=1):
            sid = max_sid + offset
            index_writer.writerow([sid, source, text])
            for tid, token in enumerate(tokenize(text), start=1):
                writer.writerow([sid, tid, token, ""])
                appended_tokens += 1

    print(f"New sentences appended : {len(new_sentences)} (ids {max_sid + 1}..{max_sid + len(new_sentences)})")
    print(f"New tokens to label    : {appended_tokens}")
    print(f"Duplicates skipped     : {skipped_dup_total}")


if __name__ == "__main__":
    main()
