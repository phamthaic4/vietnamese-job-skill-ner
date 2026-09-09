# Week 3 Guide — Huấn luyện PhoBERT NER (tự làm từng bước)

> Mục tiêu cuối tuần: `src/train.py` chạy trên Colab → con số **entity F1 đầu tiên** của dự án + model được lưu.
> Cách làm: đọc guide này → điền code vào các TODO trong `src/train.py` → smoke test trên máy (CPU) → train thật trên Colab.

---

## Bức tranh tổng thể — pipeline 6 bước

```
data/splits/*.json
      │
      ▼
[1] Label map:  O=0, B-SKILL=1, I-SKILL=2   (máy chỉ hiểu số)
      │
      ▼
[2] Tokenize + SUBWORD ALIGNMENT   ← phần khó nhất, dành 60% thời gian ở đây
      │
      ▼
[3] PyTorch Dataset + DataCollator (padding)
      │
      ▼
[4] Model: PhoBERT + đầu token-classification
      │
      ▼
[5] Trainer + TrainingArguments + compute_metrics (seqeval)
      │
      ▼
[6] Train → Evaluate trên test → Save model + tokenizer
```

---

## Khái niệm phải hiểu trước khi viết code

### 1. Vì sao cần subword alignment?

PhoBERT dùng **BPE tokenization**: 1 từ có thể bị cắt thành nhiều mảnh con (subword).
Ví dụ: `Docker` → `["Do", "cker"]`.

Nhãn của chúng ta gắn theo **từ**, nhưng model dự đoán theo **subword**. Vậy mỗi mảnh con
nhận nhãn gì? Quy ước chuẩn (HF tutorial):

- **Subword đầu tiên** của từ → nhận nhãn của từ đó
- **Các subword tiếp theo** → nhận `-100` (hằng số đặc biệt mà `CrossEntropyLoss` **bỏ qua**)
- Special tokens (`<s>`, `</s>`, `<pad>`) → `-100`

Ví dụ cụ thể (bạn sẽ tự in ra để kiểm chứng trong smoke test):

```
words:    ["Thành", "thạo", "Docker"]     ← dữ liệu của bạn
labels:   ["O",    "O",    "B-SKILL"]

subtokens: <s>  Thành  thạo  Do  cker  </s>
word_ids:  None    0      1    2     2  None   ← word_ids() cho biết mảnh nào thuộc từ nào
labels:   -100     0      0    1  -100  -100   ← chỉ mảnh ĐẦU giữ nhãn
```

### 2. `word_ids()` và `is_split_into_words=True`

- Gọi tokenizer với `is_split_into_words=True` và input là **list các từ** → tokenizer biết
  ranh giới từ, và `tokenized.word_ids(batch_index=i)` trả về mapping "mảnh con → index từ".
- ⚠️ Cần `use_fast=True` khi load tokenizer (chỉ fast tokenizer mới có `word_ids()`).
  Nếu phobert không có bản fast → xem mục Troubleshooting.

### 3. Vì sao label_all_tokens=False (chỉ label subword đầu)?

Cách chuẩn của HF tutorial: chỉ 1 subword/đại diện mỗi từ mang nhãn → số nhãn = số từ.
Lựa chọn kia (`label_all_tokens=True`: mọi subword đều mang nhãn) cũng chạy được, nhưng
lúc decode phải gom subword lại phức tạp hơn. Giữ cách chuẩn.

### 4. seqeval: entity-level ≠ token accuracy

- Token accuracy ~95% là vô nghĩa khi 93% token là O (đoán O hết cũng được 93%).
- `seqeval` ghép các B-/I- liên tiếp thành **entity** rồi tính Precision/Recall/F1 **cấp entity**
  (một entity được tính đúng chỉ khi cả span khớp). Đây là số sẽ ghi vào README.

### 5. DataCollatorForTokenClassification làm gì?

Padding batch: input_ids pad bằng `pad_token_id`, **labels pad bằng -100** (không phải 0!)
→ loss không bị "dối" bởi các ô padding. Dùng sẵn, đừng tự viết.

---

## Phần A — Smoke test trên máy (CPU, không cần GPU, ~10 phút)

Làm ngay trên máy, mục đích: **chắc chắn bước 1–3 đúng trước khi đốt thời gian trên Colab.**

1. Điền TODO 1–3 trong `src/train.py`.
2. Chạy: `python src/train.py --smoke`
   (chế độ smoke chỉ tokenize + in bảng alignment của 3 câu đầu + thoát)
3. Tự kiểm chứng bằng mắt:
   - Số `-100` xuất hiện ở đúng chỗ? (subword tiếp theo + special tokens)
   - Mảnh subword ĐẦU của từ có nhãn của từ đó?
   - Khi decode ngược `id2label` vẫn ra `O / B-SKILL / I-SKILL` đúng nghĩa?

Nếu bảng alignment đúng 100% → 80% rủi ro của Week 3 đã được gỡ.

## Phần B — Train trên Colab (~15–20 phút tổng)

1. Mở Colab → Runtime → Change runtime type → **T4 GPU**.
2. Cell 1: `!pip install -q transformers datasets seqeval accelerate`
3. Đưa dữ liệu vào — chọn 1 trong 2:
   - **Cách nhanh:** sidebar Files → upload `train.json`, `val.json`, `test.json` + file `train.py`
   - **Cách sạch:** nếu repo public → `!git clone <repo-url>` rồi `cd` vào
4. Sửa `DATA_DIR` trong train.py cho khớp path trên Colab.
5. Chạy train. Dataset 244 câu là rất nhỏ: **~1–2 phút/epoch trên T4**, tổng < 15 phút.
6. Sau khi xong:
   - Ghi kết quả vào `results/experiments.csv` (row `exp_001` — baseline)
   - Tải model về: zip thư mục output → download (hoặc lưu lên Drive)
   - Commit: kết quả CSV + (nếu nhỏ) không commit weights — weights để release sau

### Hyperparameters baseline (đã chốt trong kế hoạch — đừng đổi trước exp_001)

```python
learning_rate = 3e-5
num_train_epochs = 5
per_device_train_batch_size = 16   # OOM thì giảm 8
weight_decay = 0.01
eval_strategy = "epoch"            # mỗi epoch đánh giá val 1 lần
load_best_model_at_end = True      # giữ checkpoint tốt nhất theo F1
```

### Điều gì là "thành công" của exp_001?

- Entity F1 **bất kỳ** được in ra từ test set (plan ghi: > 0.6 là ổn cho v1)
- Model + tokenizer save thành công
- Một row trong `results/experiments.csv`

Đừng so val/test F1 với ai cả — dataset 350 câu, số này là **baseline để vượt** ở Week 4, không phải thành tích.

---

## Troubleshooting (gặp lỗi → tra đây trước khi hỏi)

| Lỗi / hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `'RobertaTokenizer' object has no attribute 'word_ids'` | Đang dùng slow tokenizer | `AutoTokenizer.from_pretrained(..., use_fast=True)`; nếu vẫn lỗi → fallback: tự tokenize từng từ bằng `tokenizer(word)` rồi ghép (hỏi mình sẽ có snippet) |
| `CUDA out of memory` | Batch quá lớn cho T4 với max_length 128 | `per_device_train_batch_size=8`, hoặc fp16=True |
| Loss không giảm / F1 = 0 sau 1 epoch | Label map sai thứ tự, hoặc quên -100 masking khi decode | In lại bảng alignment phần smoke test |
| F1 cao lạ thường (0.9+) ngay epoch 1 | Nghi leakage: kiểm tra split có dính câu trùng không (đã dedupe từ Week 0 nên khả năng thấp — nhưng đừng tin niềm vui miễn phí) | Đọc vài prediction thực tế bằng tay |
| seqeval báo label mix-up | List nhãn decode còn lẫn -100 | Trong compute_metrics phải lọc `label != -100` trước |
| Model tải chậm lần đầu trên Colab | HF download ~500MB | Bình thường; lần sau sẽ có cache trong session |

---

## Checklist hoàn thành Week 3 (khớp plan)

- [ ] `src/train.py` hoàn chỉnh, smoke test pass
- [ ] Chạy train trên Colab, không lỗi
- [ ] Entity F1 (test) in ra → điền vào `results/experiments.csv` (exp_001)
- [ ] Model + tokenizer save được
- [ ] Ghi 2–3 dòng học được vào `PROGRESS.md` (mục Notes)
