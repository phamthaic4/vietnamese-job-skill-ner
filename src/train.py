"""Fine-tune PhoBERT cho Skill NER (BIO tagging).

Cách dùng:
  Smoke test (máy local, CPU, chỉ kiểm tra alignment):
      python src/train.py --smoke
  Train thật (trên Colab GPU):
      python src/train.py

TODO: điền code vào các chỗ [TODO]. Đọc docs/week3_training_guide.md trước khi bắt đầu.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Colab: đổi thành Path(".") hoặc path bạn upload — xem guide phần B
DATA_DIR = ROOT / "data" / "splits"

LABEL_LIST = ["O", "B-SKILL", "I-SKILL"]
MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 128


# [TODO 1] label map
# Viết 2 dict: label2id ({nhãn: số}) và id2label ({số: nhãn}) từ LABEL_LIST.
# Gợi ý: enumerate(LABEL_LIST)
label2id = None
id2label = None


# [TODO 2] load tokenizer + model
# Load AutoTokenizer (use_fast=True!) và AutoModelForTokenClassification
#   - tokenizer: from MODEL_NAME, use_fast=True
#   - model: from MODEL_NAME, truyền num_labels=len(LABEL_LIST),
#     id2label=..., label2id=... (để checkpoint lưu được tên nhãn)
# Gợi ý: from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer = None
model = None


# [TODO 3] tokenize + align labels
# Viết hàm tokenize_and_align(examples):
#   - gọi tokenizer(examples["tokens"], is_split_into_words=True,
#     truncation=True, max_length=MAX_LENGTH)
#   - với mỗi câu i trong batch: lấy word_ids = tokenized.word_ids(i)
#     với mỗi subword: nếu word_id là None -> -100; nếu là subword ĐẦU của từ
#     (word_id != word_id_trước đó) -> label2id[examples["labels"][i][word_id]];
#     ngược lại -> -100
#   - trả về dict chứa tokenized + "labels"
def tokenize_and_align(examples):
    pass


# [TODO 4] PyTorch Dataset
# Class SkillNERDataset(torch.utils.data.Dataset):
#   __init__(self, data, tokenizer): lưu data (list {"tokens","labels"}) + tokenizer
#   __len__: số câu
#   __getitem__(self, idx):
#       tokenized = tokenize_and_align({"tokens": [tokens_câu], "labels": [labels_câu]})
#       trả về dict các tensor (hoặc list -> collator sẽ lo)
# Gợi ý: lấy item đầu của từng key trong tokenized: {k: v[0] for k, v in tokenized.items()}
class SkillNERDataset:
    pass


# [TODO 5] compute_metrics với seqeval
# Viết hàm compute_metrics(eval_pred):
#   - eval_pred.predictions shape (n, seq, num_labels) -> argmax axis=-1
#   - LỌC bỏ chỗ labels == -100 (cả predictions lẫn labels), trả về list các list nhãn CHỮ
#     qua id2label
#   - seqeval.metrics.f1_score / precision_score / recall_score trên 2 list đó
# Gợi ý: from seqeval.metrics import f1_score, precision_score, recall_score
def compute_metrics(eval_pred):
    pass


def load_split(name):
    import json

    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def run_smoke():
    """Kiểm tra alignment bằng mắt trước khi đốt tiền GPU — xem guide phần A."""
    data = load_split("train")[:3]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    out = tokenize_and_align({"tokens": [s["tokens"] for s in data], "labels": [s["labels"] for s in data]})
    for i, s in enumerate(data):
        print(f"\n=== Câu {i} ===")
        for w, l in zip(s["tokens"], s["labels"]):
            print(f"  word: {w!r:20} label: {l}")
        print("  --- subwords ---")
        for tid, wid in enumerate(out["word_ids"](0) if hasattr(out, "word_ids") else out["word_ids"][i]):
            pass  # [TODO 6] in từng subtoken + word_id + nhãn số + nhãn chữ
    print("\nKiểm tra bằng mắt: subword ĐẦU của từ giữ nhãn, còn lại -100?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="chỉ kiểm tra alignment rồi thoát")
    args = parser.parse_args()

    if label2id is None or tokenizer is None or model is None:
        sys.exit("Chưa điền TODO 1-2!")

    if args.smoke:
        run_smoke()
        return

    # [TODO 7] train loop
    # 1. load train/val: load_split("train"), load_split("val") -> SkillNERDataset
    # 2. from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
    #    collator = DataCollatorForTokenClassification(tokenizer)
    # 3. TrainingArguments với hyperparams đã chốt trong guide:
    #       output_dir="runs/exp_001", learning_rate=3e-5, num_train_epochs=5,
    #       per_device_train_batch_size=16, weight_decay=0.01,
    #       eval_strategy="epoch", save_strategy="epoch",
    #       load_best_model_at_end=True, metric_for_best_model="f1",
    #       report_to="none"
    # 4. Trainer(model=..., args=..., train_dataset=..., eval_dataset=...,
    #             data_collator=collator, compute_metrics=compute_metrics)
    # 5. trainer.train()
    # 6. Đánh giá TRÊN TEST: load test -> SkillNERDataset -> trainer.predict(test_ds)
    #    in ra metrics (đây là con số ghi vào results/experiments.csv)
    # 7. Save: trainer.save_model("model/best") + tokenizer.save_pretrained("model/best")
    pass


if __name__ == "__main__":
    main()
