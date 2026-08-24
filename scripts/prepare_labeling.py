"""Build the BIO labeling template from raw JD sentences.

Reads   data/raw/jds.txt
Writes  data/labeled/labeling_template.tsv  -> fill the `label` column
        data/labeled/sentences_index.tsv    -> sentence_id -> source/text lookup

TSV (tab-separated) so Excel/Google Sheets split columns automatically
regardless of OS locale. Labels: B-SKILL / I-SKILL / O.
Exact-duplicate sentences are removed automatically.
"""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "jds.txt"
OUT_DIR = ROOT / "data" / "labeled"
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


def dedupe(sentences):
    seen = set()
    unique = []
    removed = 0
    for source, text in sentences:
        key = text.lower()
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append((source, text))
    return unique, removed


def main():
    raw = RAW_PATH.read_text(encoding="utf-8")
    sentences, removed = dedupe(parse_sentences(raw))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    template_path = OUT_DIR / "labeling_template.tsv"
    index_path = OUT_DIR / "sentences_index.tsv"

    total_tokens = 0
    with template_path.open("w", newline="", encoding="utf-8-sig") as f, index_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as g:
        writer = csv.writer(f, delimiter=DELIM)
        index_writer = csv.writer(g, delimiter=DELIM)
        writer.writerow(["sentence_id", "token_id", "token", "label"])
        index_writer.writerow(["sentence_id", "source", "text"])
        for sid, (source, text) in enumerate(sentences, start=1):
            index_writer.writerow([sid, source, text])
            for tid, token in enumerate(tokenize(text), start=1):
                writer.writerow([sid, tid, token, ""])
                total_tokens += 1

    print(f"Unique sentences : {len(sentences)} ({removed} duplicates removed)")
    print(f"Tokens to label  : {total_tokens}")
    print(f"Template         : {template_path}")
    print(f"Index            : {index_path}")


if __name__ == "__main__":
    main()
