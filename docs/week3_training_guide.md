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

### 2. `word_ids()` và thực tế với PhoBERT (quan trọng!)

- Lý thuyết chuẩn: gọi tokenizer với `is_split_into_words=True`, rồi `tokenized.word_ids(i)`
  cho biết "mảnh con thuộc về từ thứ mấy". Nhưng hàm này **chỉ tồn tại ở fast tokenizer**.
- **Thực tế đã kiểm chứng (Aug 2026): `vinai/phobert-base` KHÔNG có bản fast**
  (`tokenizer.is_fast == False`) → `word_ids()` không dùng được.
- Cách đúng cho PhoBERT: **vòng lặp thủ công từng từ** như TODO 3 trong `src/train.py`
  (`tokenizer.encode(word, add_special_tokens=False)` cho từng từ, miếng đầu giữ nhãn,
  miếng sau `-100`, tự track `word_id` bằng index của từ trong vòng lặp).
- Vì sao cách thủ công vẫn đúng? PhoBERT dùng BPE kiểu `@@` (đánh dấu tiếp diễn ở cuối
  miếng) — điểm cắt chỉ phụ thuộc bên trong từ, nên encode riêng lẻ cho ID y hệt
  encode cả câu (đã kiểm chứng: `Docker` → `[4448, 18095]` ở cả hai cách).
  Với tokenizer kiểu `Ġ`-prefix (RoBERTa tiếng Anh) thì cách này mới sai.

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

### Experiment tracking (bổ sung theo plan Week 3 — WandB)

Plan yêu cầu log Loss & F1 curves chuyên nghiệp. Cách gọn nhẹ nhất:

1. Tạo tài khoản free tại [wandb.ai](https://wandb.ai) → lấy API key
2. Trên Colab:
   ```python
   !pip install -q wandb
   import wandb; wandb.login()
   ```
3. Trong `TrainingArguments`: đổi `report_to="none"` → `report_to="wandb"`,
   và truyền `run_name="exp_001-baseline"` để runs có tên trên dashboard
4. Sau khi train xong: `wandb.finish()`
5. Screenshot dashboard (loss curve + F1 curve) → dùng cho README sau này

> Không muốn xài dịch vụ ngoài? Giữ `report_to="none"` và copy số liệu từ log
> console vào `results/experiments.csv` — vẫn thỏa checkpoint, chỉ là ít "pro" hơn.

### Bonus track — XLM-RoBERTa baseline (theo plan Week 3)

Plan gợi ý chuẩn bị 1 baseline `xlm-roberta-base` để trả lời câu phỏng vấn
*"Tại sao chọn PhoBERT thay vì multilingual model?"*. Cách làm rẻ nhất:

- Code đã viết sẵn cho phép đổi model vì mọi thứ dùng biến `MODEL_NAME`
- Chỉ cần chạy lại toàn bộ pipeline với `MODEL_NAME = "xlm-roberta-base"`
  (output_dir đổi thành `runs/exp_xlmr_001`) → có bảng so sánh PhoBERT vs XLM-R
- ⚠️ Làm SAU khi exp_001 (PhoBERT) xong và lưu model — không chạy song song
  hai model cùng lúc trên 1 Colab session (RAM/GPU sẽ nổ)
- Lưu ý nhỏ: XLM-R không dùng `use_fast=True` thì không có `word_ids()` —
  giữ nguyên `use_fast=True`; nếu tokenizer slow thì tra bảng Troubleshooting

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
- [ ] **(Bonus theo plan)** WandB dashboard log Loss/F1 curves
- [ ] Entity F1 (test) in ra → điền vào `results/experiments.csv` (exp_001)
- [ ] Model + tokenizer save được
- [ ] **(Bonus theo plan)** Bảng so sánh exp_001 (PhoBERT) vs XLM-RoBERTa baseline — lưu cho Week 4
- [ ] Ghi 2–3 dòng học được vào `PROGRESS.md` (mục Notes)

## Nhìn trước sang Week 4 (đã update trong plan — để định hướng)

Các việc này **chưa làm bây giờ**, nhưng để bạn biết data đã chuẩn bị sẵn cho chúng:

- **OOV Stress-testing:** EDA đã chỉ ra 31.8% skill ở test là OOV → Week 4 sẽ tách
  In-vocab vs OOV Recall riêng. Guide này chưa cần làm gì thêm.
- **3 dạng lỗi kinh điển** (FP / FN / Boundary mismatch) khi đọc error analysis —
  khi decode prediction ở TODO 7, chú ý lưu vài dự đoán sai làm mẫu cho Week 4.
- **ONNX export + benchmark** là của Week 5 — cần `predict.py` (Week 4) trước.
