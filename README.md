# Vietnamese Job Skill NER

Fine-tune [vinai/phobert-base](https://huggingface.co/vinai/phobert-base) to extract **SKILL** entities from Vietnamese job description / CV text using BIO tagging.

> **Status:** Work in progress — Week 0 (setup & data collection)

## Problem

Recruiter tools need to know which skills a job post or CV mentions. This project trains a NER model to automatically extract IT skills (e.g. Python, SQL, Docker) from Vietnamese text.

## Data

| Item | Value |
|------|-------|
| Source | IT job posts (TopCV, ITviec, VietnamWorks) |
| Size v1 | 200–300 labeled sentences |
| Format | JSON: `{ "tokens": [...], "labels": [...] }` |
| Labels | `B-SKILL`, `I-SKILL`, `O` |
| Split | 70% train / 15% val / 15% test |

Example:

```
Token:   Tuyển   Python  developer  có  kinh nghiệm  SQL
Label:   O       B-SKILL O          O   O           B-SKILL
```

## Method

- Model: `vinai/phobert-base`
- Task: token classification with BIO tagging
- Subword label alignment for PhoBERT tokenizer
- Metrics: entity-level Precision / Recall / F1 via `seqeval`

## Results & Hyperparameter Experiments

Toàn bộ quá trình huấn luyện được theo dõi trực tiếp qua [Weights & Biases (WandB)](https://wandb.ai). Dưới đây là bảng so sánh thực nghiệm điều chỉnh Learning Rate (LR sweep) và so sánh với mô hình đa ngôn ngữ (Multilingual baseline):

| Experiment | Model | Learning Rate | Epochs | Batch Size | Precision (Test) | Recall (Test) | **F1-Score (Test)** | Ghi chú / Nhận xét |
|---|---|---|---|---|---|---|---|---|
| **exp_001** | `vinai/phobert-base` | **3e-5** | **5** | **16** | **0.6538** | **0.6892** | **`0.6711`** | **Tối ưu nhất (Best checkpoint)**; F1 val đạt 0.6173 ở epoch 4 |
| **exp_002** | `xlm-roberta-base` | 3e-5 | 5 | 8 | 0.6329 | 0.6757 | `0.6536` | Multilingual baseline; PhoBERT vượt trội **+1.75% F1** dù nhẹ hơn một nửa |
| **exp_003** | `vinai/phobert-base` | 2e-5 | 5 | 16 | 0.5000 | 0.6892 | `0.5795` | LR thấp: Recall giữ nguyên nhưng Precision sụt giảm mạnh (dự đoán thừa nhiều) |
| **exp_004** | `vinai/phobert-base` | 4e-5 | 5 | 16 | 0.5556 | 0.6757 | `0.6098` | LR cao: Bước nhảy gradient quá lớn, F1 thấp hơn mốc 3e-5 |

> 📌 **Kết luận kỹ thuật (Key Takeaways):**
> 1. **Đường cong học tập theo LR:** $2\times 10^{-5} (0.5795) < 4\times 10^{-5} (0.6098) < \mathbf{3\times 10^{-5} (0.6711)}$. Tốc độ học $\text{LR} = 3\times 10^{-5}$ là điểm hội tụ tối ưu cho PhoBERT trên tập dữ liệu này.
> 2. **Monolingual vs Multilingual:** Mô hình đơn ngữ chuyên biệt cho tiếng Việt (`PhoBERT-base` ~540MB) đạt hiệu năng cao hơn mô hình đa ngôn ngữ (`XLM-RoBERTa-base` ~1.12GB) tới $+1.75\%$ F1, đồng thời huấn luyện và suy luận nhanh hơn gấp $2.5\times$.

## Error Analysis
Xem chi tiết báo cáo mổ xẻ lỗi và các trường hợp sai ranh giới (Boundary), bỏ sót OOV tại [docs/error_analysis.md](docs/error_analysis.md).

## Repository Structure

```
vietnamese-job-skill-ner/
├── data/            # raw, labeled, splits
├── docs/            # skill vocab, error analysis report
├── notebooks/       # EDA (label distribution, length, OOV analysis)
├── scripts/         # data conversion, error analysis utilities
├── src/             # train.py, predict.py (with post-processing)
├── model/           # saved checkpoints (model/best)
├── results/         # experiment logs (experiments.csv)
└── tests/
```

## How to Run

```bash
pip install -r requirements.txt

# Huấn luyện mô hình (Fine-tune):
python src/train.py

# Thử nghiệm suy luận trích xuất kỹ năng:
python src/predict.py --text "Tuyển Backend Engineer thành thạo Python, Docker, PostgreSQL và Redis."
```

## Roadmap

- [x] Week 1: setup + collect 100+ JD sentences + label first 50
- [x] Week 2: label to 200–300 + data pipeline + EDA (O 92.8%, OOV 31.8%)
- [x] Week 3: first training run + seqeval baseline F1 (0.6711) + WandB tracking
- [x] Week 4: hyperparameter experiments (LR sweep) + Multilingual comparison + error analysis + predict.py
- [ ] Week 5: polish README, ONNX export benchmark, tag release v1.0.0
