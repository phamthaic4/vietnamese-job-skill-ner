"""Fine-tune PhoBERT cho Skill NER (BIO tagging).

Cách dùng:
  Smoke test (máy local, CPU, chỉ kiểm tra alignment):
      python src/train.py --smoke
  Train thật (trên Colab GPU):
      python src/train.py
  Train + log lên Weights & Biases:
      python src/train.py --wandb

Config đã chốt (exp_001): LR 3e-5, batch 16, 5 epochs, max_length 128.
Kết quả kỳ vọng trên test set: P 0.6538 / R 0.6892 / F1 0.6711.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

# Đảm bảo in tiếng Việt không lỗi trên Windows CMD (cp1252)
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Chạy trên Jupyter / Colab (không có biến __file__)
    ROOT = Path(".")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data" / "splits"

LABEL_LIST = ["O", "B-SKILL", "I-SKILL"]
MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 128


# Label map
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for i, label in enumerate(LABEL_LIST)}


# Tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_LIST), id2label=id2label, label2id=label2id)


# Tokenize + align labels
# Subword đầu tiên của mỗi từ giữ nhãn thật, các subword sau gán -100 (bị ignore trong loss).
def tokenize_and_align(examples):
    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    batch_word_ids = []
    
    # Lấy id của các special token
    cls_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    sep_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id
    
    for tokens, labels in zip(examples["tokens"], examples["labels"]):
        input_ids = [cls_token_id]
        label_ids = [-100]
        word_ids = [None]
        
        for w_idx, (word, label) in enumerate(zip(tokens, labels)):
            # Tokenize từng từ (bỏ qua special token vì ta tự thêm vào)
            subwords = tokenizer.encode(word, add_special_tokens=False)
            if not subwords:
                continue
            
            input_ids.extend(subwords)
            # Mảnh đầu tiên giữ nhãn thật, các mảnh sau gán -100
            label_ids.append(label2id[label])
            label_ids.extend([-100] * (len(subwords) - 1))
            # Tất cả các subwords của từ này đều có cùng word_id = w_idx
            word_ids.extend([w_idx] * len(subwords))
            
        # Thêm token kết thúc câu
        input_ids.append(sep_token_id)
        label_ids.append(-100)
        word_ids.append(None)
        
        # Cắt ngắn nếu dài hơn MAX_LENGTH
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH - 1] + [sep_token_id]
            label_ids = label_ids[:MAX_LENGTH - 1] + [-100]
            word_ids = word_ids[:MAX_LENGTH - 1] + [None]
            
        attention_mask = [1] * len(input_ids)
        
        batch_input_ids.append(input_ids)
        batch_labels.append(label_ids)
        batch_attention_mask.append(attention_mask)
        batch_word_ids.append(word_ids)
        
    return {
        "input_ids": batch_input_ids,
        "labels": batch_labels,
        "attention_mask": batch_attention_mask,
        "word_ids": batch_word_ids,
    }


# PyTorch Dataset
class SkillNERDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokenized = tokenize_and_align({
            "tokens": [item["tokens"]],
            "labels": [item["labels"]]
        })
        
        return {
            "input_ids": tokenized["input_ids"][0],
            "labels": tokenized["labels"][0],
            "attention_mask": tokenized["attention_mask"][0],
        }


# compute_metrics với seqeval — đánh giá theo entity, không phải theo token
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    preds = np.argmax(predictions, axis=-1)

    true_predictions = []
    true_labels = []

    for pred_sentence, label_sentence in zip(preds, labels):
        cur_preds = []
        cur_labels = []

        for p, lbl in zip(pred_sentence, label_sentence):
            if lbl != -100:
                cur_preds.append(id2label[p])
                cur_labels.append(id2label[lbl])

        true_predictions.append(cur_preds)
        true_labels.append(cur_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions)
    }


def load_split(name):
    import json

    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def run_smoke():
    """Kiểm tra alignment bằng mắt trước khi đốt tiền GPU — xem guide phần A."""
    data = load_split("train")[:3]
    out = tokenize_and_align({"tokens": [s["tokens"] for s in data], "labels": [s["labels"] for s in data]})
    
    for i, s in enumerate(data):
        print(f"\n=== Câu {i} ===")
        for w, lbl in zip(s["tokens"], s["labels"]):
            print(f"  word: {w!r:20} label: {lbl}")
        print("  --- subwords alignment ---")
        inp_ids = out["input_ids"][i]
        lbl_ids = out["labels"][i]
        w_ids = out["word_ids"][i]
        for tid, lid, wid in zip(inp_ids, lbl_ids, w_ids):
            subtoken = tokenizer.decode([tid])
            lbl_name = id2label[lid] if lid != -100 else "-100"
            wid_str = str(wid) if wid is not None else "None"
            print(f"  subtoken: {subtoken!r:15} | word_id: {wid_str:5s} | label: {lid:4d} ({lbl_name})")
            
    print("\nKiểm tra bằng mắt: subword ĐẦU của từ giữ nhãn, còn lại -100?")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="chỉ kiểm tra alignment rồi thoát")
    parser.add_argument("--wandb", action="store_true", help="log metrics lên Weights & Biases")
    args = parser.parse_args()

    if args.smoke:
        run_smoke()
        return

    print(">>> Đang nạp dữ liệu train / val / test...")
    raw_train = load_split("train")
    raw_val = load_split("val")
    raw_test = load_split("test")
    train_dataset = SkillNERDataset(raw_train)
    val_dataset = SkillNERDataset(raw_val)
    test_dataset = SkillNERDataset(raw_test)
    # Data collator tự động pad các câu trong cùng batch về cùng độ dài, pad label bằng -100
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    # Cấu hình tham số huấn luyện (Hyperparameters)
    training_args = TrainingArguments(
        output_dir="runs/exp_001",
        learning_rate=3e-5,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        report_to="wandb" if args.wandb else "none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print(">>> Bắt đầu huấn luyện mô hình...")
    trainer.train()
    print("\n>>> Đang đánh giá trên tập TEST...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print("\n" + "="*40)
    print("KẾT QUẢ TRÊN TẬP TEST:")
    print(f" - Precision: {test_results.get('eval_precision', 0):.4f}")
    print(f" - Recall:    {test_results.get('eval_recall', 0):.4f}")
    print(f" - F1-Score:  {test_results.get('eval_f1', 0):.4f}")
    print("="*40)
    # Lưu lại model và tokenizer tốt nhất
    save_path = ROOT / "model" / "best"
    print(f"\n>>> Đang lưu model và tokenizer vào {save_path}...")
    trainer.save_model(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(">>> Hoàn thành Nấc 1!")



if __name__ == "__main__":
    main()
