"""Inference script: Trích xuất kỹ năng (Job Skills) từ văn bản tiếng Việt/Anh.

Sử dụng mô hình PhoBERT-base đã fine-tune tại model/best.
Có tích hợp tầng Post-processing: làm sạch ký tự, gộp B-I span, loại bỏ trùng lặp.

Cách dùng:
    # Chạy thử câu mẫu mặc định:
    python src/predict.py
    
    # Chạy với câu văn bản tùy ý:
    python src/predict.py --text "Yêu cầu 2 năm kinh nghiệm với Python, Docker và Kubernetes."
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Union
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Thiết lập encoding UTF-8 cho console Windows
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "best"

# Regex tách từ: giữ nguyên ký tự kỹ thuật dính vào từ (C++, C/C++, CI/CD, C#, 2.5)
TOKEN_RE = re.compile(r"(?:[^\W_]+[./+#_]*)+|_+|[^\w\s]+", re.UNICODE)


class SkillExtractor:
    def __init__(self, model_dir: Union[Path, str] = MODEL_DIR, device: str = None):
        """Nạp model và tokenizer đã lưu từ thư mục model_dir."""
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục mô hình tại: {model_dir}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f">>> Đang tải mô hình từ {model_dir} lên thiết bị: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label

    def tokenize_words(self, text: str):
        """Tách câu thô thành danh sách các từ/ký hiệu theo regex chuẩn."""
        return TOKEN_RE.findall(text)

    def extract_skills(self, text: str):
        """Trích xuất danh sách các kỹ năng từ chuỗi văn bản (kèm post-processing)."""
        words = self.tokenize_words(text)
        if not words:
            return []

        # BPE tokenize từng từ và lưu mapping subwords -> word index
        cls_id = self.tokenizer.bos_token_id or self.tokenizer.cls_token_id
        sep_id = self.tokenizer.eos_token_id or self.tokenizer.sep_token_id

        input_ids = [cls_id]
        word_ids = [None]

        for w_idx, word in enumerate(words):
            subwords = self.tokenizer.encode(word, add_special_tokens=False)
            if not subwords:
                continue
            input_ids.extend(subwords)
            word_ids.extend([w_idx] * len(subwords))

        input_ids.append(sep_id)
        word_ids.append(None)

        # Dự đoán nhãn qua mô hình
        attention_mask = [1] * len(input_ids)
        input_tensor = {
            "input_ids": torch.tensor([input_ids], device=self.device),
            "attention_mask": torch.tensor([attention_mask], device=self.device),
        }
        with torch.no_grad():
            outputs = self.model(**input_tensor)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()

        # Mỗi từ lấy label của subword đầu tiên thuộc về nó
        word_labels = {}
        seen_words = set()
        for wid, pid in zip(word_ids, preds):
            if wid is not None and wid not in seen_words:
                word_labels[wid] = self.id2label.get(pid, "O")
                seen_words.add(wid)

        # Gom B/I-SKILL liên tiếp thành span; "(" / ")" chen giữa không phá span
        raw_spans = []
        current_span = []
        span_last_idx = None

        for w_idx, word in enumerate(words):
            label = word_labels.get(w_idx, "O")

            if word == "(" or word == ")":
                continue

            if label == "B-SKILL":
                if current_span and span_last_idx == w_idx - 1:
                    # B-SKILL liền kề B-SKILL: skill nhiều từ như "Spring Boot"
                    current_span.append(word)
                    span_last_idx = w_idx
                else:
                    if current_span:
                        raw_spans.append(" ".join(current_span))
                    current_span = [word]
                    span_last_idx = w_idx
            elif label == "I-SKILL":
                if current_span:
                    current_span.append(word)
                    span_last_idx = w_idx
                else:
                    # I- đứng đầu không có B- đi trước -> coi như bắt đầu span
                    current_span = [word]
                    span_last_idx = w_idx
            else:
                if current_span:
                    raw_spans.append(" ".join(current_span))
                current_span = []
                span_last_idx = None

        if current_span:
            raw_spans.append(" ".join(current_span))

        return self.post_process(raw_spans)

    @staticmethod
    def post_process(spans):
        """Làm sạch các span kỹ năng:
        - Xóa dấu phẩy, chấm, gạch ngang dính ở đầu/cuối từ: 'Python,' -> 'Python'
        - Loại bỏ các từ rỗng hoặc quá ngắn không có nghĩa
        - Loại bỏ trùng lặp (giữ nguyên thứ tự xuất hiện)
        """
        cleaned = []
        seen = set()

        for span in spans:
            # Xóa các dấu câu ở đầu/cuối (giữ "." đầu cho .NET, xóa "." cuối câu)
            token = span.strip(" \t\n\r,;:()[]{}'\"")
            if token.endswith(".") and not token.startswith("."):
                token = token[:-1]
            # Chuẩn hóa khoảng trắng kép
            token = re.sub(r"\s+", " ", token)

            if not token or len(token) <= 1 and token.lower() not in {"c", "r"}:
                continue

            # Kiểm tra trùng lặp theo dạng lowercase
            token_lower = token.lower()
            if token_lower not in seen:
                seen.add(token_lower)
                cleaned.append(token)

        return cleaned


def main():
    parser = argparse.ArgumentParser(description="Trích xuất kỹ năng nghề nghiệp từ văn bản.")
    parser.add_argument(
        "--text",
        type=str,
        default=(
            "Thành thạo ít nhất 1 ngôn ngữ Backend: Ưu tiên Python, Go, C/C++ hoặc Java (Spring Boot). "
            "Có kinh nghiệm làm việc với hệ quản trị cơ sở dữ liệu PostgreSQL, MySQL, Redis và triển khai Docker, Kubernetes, CI/CD."
        ),
        help="Chuỗi văn bản JD hoặc CV cần trích xuất skill.",
    )
    args = parser.parse_args()

    extractor = SkillExtractor()
    print("\n" + "=" * 60)
    print("VĂN BẢN ĐẦU VÀO:")
    print(args.text)
    print("-" * 60)

    skills = extractor.extract_skills(args.text)
    print(f"KỸ NĂNG TRÍCH XUẤT ĐƯỢC ({len(skills)} skills):")
    for idx, skill in enumerate(skills, 1):
        print(f"  {idx}. {skill}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
