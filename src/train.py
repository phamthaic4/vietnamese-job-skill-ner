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
import torch
from torch.utils.data import Dataset
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score 
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)

# Đảm bảo in tiếng Việt không lỗi trên Windows CMD (cp1252)
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

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
label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for i, label in enumerate(LABEL_LIST)}


# [TODO 2] load tokenizer + model
# Load AutoTokenizer (use_fast=True!) và AutoModelForTokenClassification
#   - tokenizer: from MODEL_NAME, use_fast=True
#   - model: from MODEL_NAME, truyền num_labels=len(LABEL_LIST),
#     id2label=..., label2id=... (để checkpoint lưu được tên nhãn)
# Gợi ý: from transformers import AutoTokenizer, AutoModelForTokenClassification
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_LIST), id2label=id2label, label2id=label2id)


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


# [TODO 4] PyTorch Dataset
# Class SkillNERDataset(torch.utils.data.Dataset):
#   __init__(self, data, tokenizer): lưu data (list {"tokens","labels"}) + tokenizer
#   __len__: số câu
#   __getitem__(self, idx):
#       tokenized = tokenize_and_align({"tokens": [tokens_câu], "labels": [labels_câu]})
#       trả về dict các tensor (hoặc list -> collator sẽ lo)
# Gợi ý: lấy item đầu của từng key trong tokenized: {k: v[0] for k, v in tokenized.items()}
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


# [TODO 5] compute_metrics với seqeval
# Viết hàm compute_metrics(eval_pred):
#   - eval_pred.predictions shape (n, seq, num_labels) -> argmax axis=-1
#   - LỌC bỏ chỗ labels == -100 (cả predictions lẫn labels), trả về list các list nhãn CHỮ
#     qua id2label
#   - seqeval.metrics.f1_score / precision_score / recall_score trên 2 list đó
# Gợi ý: from seqeval.metrics import f1_score, precision_score, recall_score
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    preds = np.argmax(predictions, axis=-1)

    true_predictions = []
    true_labels = []

    for pred_sentence, label_sentence in zip(preds, labels):
        cur_preds = []
        cur_labels = []

        for p, l in zip(pred_sentence, label_sentence):
            if l != -100:
                cur_preds.append(id2label[p])
                cur_labels.append(id2label[l])

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
        for w, l in zip(s["tokens"], s["labels"]):
            print(f"  word: {w!r:20} label: {l}")
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
        report_to="none",  # Tắt wandb tạm thời ở local, khi lên Colab có thể bật
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
    print(f"KẾT QUẢ TRÊN TẬP TEST:")
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
