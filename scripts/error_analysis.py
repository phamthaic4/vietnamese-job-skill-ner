"""Error Analysis Script: Đánh giá chi tiết lỗi dự đoán trên tập Test.

Phân loại lỗi thành 3 nhóm chuẩn mực:
1. False Negative (FN): Kỹ năng thực tế bị mô hình bỏ sót hoàn toàn (đặc biệt theo dõi OOV).
2. False Positive (FP): Mô hình đoán nhầm từ thường thành kỹ năng.
3. Boundary Mismatch: Bắt đúng một phần nhưng sai ranh giới (ví dụ: 'Spring Boot' -> 'Spring').

Xuất kết quả ra file docs/error_analysis.md.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "best"
TEST_PATH = ROOT / "data" / "splits" / "test.json"
TRAIN_PATH = ROOT / "data" / "splits" / "train.json"


def extract_spans_from_bio(tokens, labels):
    """Trích xuất danh sách các span kỹ năng từ cặp (tokens, labels) theo chuẩn BIO."""
    spans = []
    current = []
    for token, label in zip(tokens, labels):
        if label == "B-SKILL":
            if current:
                spans.append(" ".join(current))
            current = [token]
        elif label == "I-SKILL":
            if current:
                current.append(token)
            else:
                current = [token]
        else:  # label == "O"
            if current:
                spans.append(" ".join(current))
                current = []
    if current:
        spans.append(" ".join(current))
    return spans


def main():
    print(">>> Đang nạp dữ liệu train để xác định danh sách từ vựng (In-vocab vs OOV)...")
    with open(TRAIN_PATH, encoding="utf-8") as f:
        train_data = json.load(f)
    train_vocab = set()
    for s in train_data:
        for span in extract_spans_from_bio(s["tokens"], s["labels"]):
            train_vocab.add(span.lower())

    with open(TEST_PATH, encoding="utf-8") as f:
        test_data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Đang tải mô hình từ {MODEL_DIR} lên thiết bị {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()

    id2label = model.config.id2label
    cls_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    sep_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    total_gold_entities = 0
    total_pred_entities = 0
    total_tp = 0

    # Đếm riêng In-vocab (đã học) vs OOV (chưa thấy ở train) để stress-test
    iv_total = 0
    iv_tp = 0
    oov_total = 0
    oov_tp = 0

    fn_list = []  # Bỏ sót
    fp_list = []  # Đoán nhầm
    boundary_list = []  # Sai ranh giới
    sample_error_cases = []

    print(f">>> Đang suy luận trên {len(test_data)} câu của tập Test...")
    for idx, sample in enumerate(test_data):
        tokens = sample["tokens"]
        gold_labels = sample["labels"]
        gold_spans = extract_spans_from_bio(tokens, gold_labels)
        total_gold_entities += len(gold_spans)
        for g_span in gold_spans:
            if g_span.lower() in train_vocab:
                iv_total += 1
            else:
                oov_total += 1

        # Tokenize từng từ
        input_ids = [cls_id]
        word_ids = [None]
        for w_idx, word in enumerate(tokens):
            subwords = tokenizer.encode(word, add_special_tokens=False)
            if not subwords:
                continue
            input_ids.extend(subwords)
            word_ids.extend([w_idx] * len(subwords))
        input_ids.append(sep_id)
        word_ids.append(None)

        # Dự đoán
        inp_tensor = torch.tensor([input_ids], device=device)
        with torch.no_grad():
            outputs = model(inp_tensor)
            preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()

        # Word-level decoding: lấy subword đầu
        word_preds = {}
        seen_words = set()
        for wid, pid in zip(word_ids, preds):
            if wid is not None and wid not in seen_words:
                word_preds[wid] = id2label.get(pid, "O")
                seen_words.add(wid)

        pred_labels = [word_preds.get(w_idx, "O") for w_idx in range(len(tokens))]
        pred_spans = extract_spans_from_bio(tokens, pred_labels)
        total_pred_entities += len(pred_spans)

        # Đối chiếu Span giữa Gold và Pred
        gold_matched = set()
        pred_matched = set()

        # 1. Khớp chính xác (Exact match)
        for g_idx, g_span in enumerate(gold_spans):
            for p_idx, p_span in enumerate(pred_spans):
                if p_idx not in pred_matched and g_span.lower() == p_span.lower():
                    gold_matched.add(g_idx)
                    pred_matched.add(p_idx)
                    total_tp += 1
                    if g_span.lower() in train_vocab:
                        iv_tp += 1
                    else:
                        oov_tp += 1
                    break

        # 2. Khớp một phần / Sai ranh giới (Boundary mismatch)
        for g_idx, g_span in enumerate(gold_spans):
            if g_idx in gold_matched:
                continue
            for p_idx, p_span in enumerate(pred_spans):
                if p_idx in pred_matched:
                    continue
                # Kiểm tra chồng lấn chuỗi
                g_low, p_low = g_span.lower(), p_span.lower()
                if g_low in p_low or p_low in g_low:
                    boundary_list.append((g_span, p_span))
                    gold_matched.add(g_idx)
                    pred_matched.add(p_idx)
                    break

        # 3. Còn lại ở Gold -> False Negative (Bỏ sót)
        for g_idx, g_span in enumerate(gold_spans):
            if g_idx not in gold_matched:
                is_oov = g_span.lower() not in train_vocab
                fn_list.append((g_span, is_oov))

        # 4. Còn lại ở Pred -> False Positive (Đoán thừa)
        for p_idx, p_span in enumerate(pred_spans):
            if p_idx not in pred_matched:
                fp_list.append(p_span)

        # Lưu lại mẫu câu bị lỗi để đưa vào tài liệu
        has_error = (
            len(gold_matched) < len(gold_spans)
            or len(pred_matched) < len(pred_spans)
            or any(g_idx not in gold_matched for g_idx in range(len(gold_spans)))
        )
        if has_error and len(sample_error_cases) < 15:
            sentence_text = " ".join(tokens)
            sample_error_cases.append(
                {
                    "id": idx + 1,
                    "text": sentence_text,
                    "gold": gold_spans,
                    "pred": pred_spans,
                }
            )

    # Thống kê phân loại lỗi
    fn_oov_count = sum(1 for _, is_oov in fn_list if is_oov)
    fn_invocab_count = len(fn_list) - fn_oov_count

    print("\n" + "=" * 60)
    print("BÁO CÁO PHÂN TÍCH LỖI (ERROR ANALYSIS REPORT):")
    print(f" - Tổng thực thể thực tế (Gold Entities): {total_gold_entities}")
    print(f" - Tổng thực thể dự đoán (Pred Entities): {total_pred_entities}")
    print(f" - Khớp chính xác (True Positives):       {total_tp}")
    print("-" * 60)
    print(f"1. False Negatives (Bỏ sót):               {len(fn_list)}")
    print(f"   + Do từ mới lạ (OOV):                   {fn_oov_count} ({fn_oov_count/len(fn_list):.1%})")
    print(f"   + Bỏ sót từ đã học (In-vocab):          {fn_invocab_count}")
    print(f"2. False Positives (Báo động giả):         {len(fp_list)}")
    print(f"3. Boundary Mismatch (Sai ranh giới):      {len(boundary_list)}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("OOV STRESS-TEST: Recall In-vocab vs OOV (kỹ năng chưa từng thấy ở train):")
    iv_recall = iv_tp / iv_total if iv_total else 0
    oov_recall = oov_tp / oov_total if oov_total else 0
    print(f" - In-vocab entities: {iv_tp}/{iv_total} = {iv_recall:.1%}")
    print(f" - OOV entities:      {oov_tp}/{oov_total} = {oov_recall:.1%}")
    print(f" - Chênh lệch Recall (OOV - In-vocab): {oov_recall - iv_recall:+.1%}")
    print("=" * 60)

    # Note: docs/error_analysis.md được duy trì thủ công (tổng hợp từ output in bên trên
    # + phân tích sâu + kiểm chứng end-to-end). Không ghi đè file md tự động.
    print("\n>>> Đã in báo cáo thống kê lên console. Chi tiết phân tích và fix: docs/error_analysis.md")


if __name__ == "__main__":
    main()
