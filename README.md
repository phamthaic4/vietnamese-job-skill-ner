# Vietnamese Job Skill NER

Fine-tune [vinai/phobert-base](https://huggingface.co/vinai/phobert-base) to extract **SKILL** entities from Vietnamese job-description and CV text (BIO tagging).

> **Status:** v1.0.0 — PhoBERT test F1 **0.6711**. CPU inference: PyTorch **35.39 ms** → ONNX Runtime **12.55 ms** (**2.82×**).
> **Weights:** [v1.0.0 release](https://github.com/phamthaic4/vietnamese-job-skill-ner/releases/tag/v1.0.0) — `model.onnx` · `model.safetensors` · `tokenizer.zip`.

## Problem

Recruiter tools need the skills a job post or CV actually mentions. This project trains a NER model to extract IT skills (Python, SQL, Docker, …) from mixed Vietnamese/English text.

## Data

| Item | Value |
|------|-------|
| Source | IT job posts (TopCV, ITviec, VietnamWorks) + 5 CVs |
| Size v1 | 350 labeled sentences, 372 skill mentions (186 unique) |
| Format | JSON: `{ "tokens": [...], "labels": [...] }` |
| Labels | `B-SKILL`, `I-SKILL`, `O` |
| Split | 244 train / 52 val / 54 test (seed 42) |

```
Token:   Tuyển   Python  developer  có  kinh nghiệm  SQL
Label:   O       B-SKILL O          O   O           B-SKILL
```

## Method

- Model: `vinai/phobert-base` (locked after ablation vs `xlm-roberta-base`)
- Task: token classification, BIO
- Subword label alignment for PhoBERT’s slow tokenizer
- Metrics: entity-level Precision / Recall / F1 (`seqeval`)
- Inference: `src/predict.py` with post-processing (glue `C/C++`, `CI/CD`; merge adjacent `B-SKILL` spans)

## Results

Training runs are logged on [Weights & Biases](https://wandb.ai). Decision metric is **test F1** (52-sentence val set is noisy).

| Experiment | Model | LR | Epochs | Batch | P | R | **Test F1** | Notes |
|---|---|---|---|---|---|---|---|---|
| **exp_001** | `vinai/phobert-base` | **3e-5** | **5** | **16** | **0.6538** | **0.6892** | **0.6711** | Best checkpoint; val F1 0.6173 @ epoch 4 |
| exp_002 | `xlm-roberta-base` | 3e-5 | 5 | 8 | 0.6329 | 0.6757 | 0.6536 | Multilingual baseline; PhoBERT **+1.75% F1**, ~2.5× faster |
| exp_003 | `vinai/phobert-base` | 2e-5 | 5 | 16 | 0.5000 | 0.6892 | 0.5795 | Too-low LR: recall holds, precision collapses |
| exp_004 | `vinai/phobert-base` | 4e-5 | 5 | 16 | 0.5556 | 0.6757 | 0.6098 | Too-high LR: worse than 3e-5 |

**Locked config:** PhoBERT-base, LR `3e-5`, batch 16, 5 epochs.

### CPU latency (batch 1, local)

| Runtime | Mean latency | Notes |
|---|---|---|
| PyTorch `model/best` | 35.39 ms | torch CPU |
| ONNX Runtime `model/best/model.onnx` | 12.55 ms | opset 14 |

Same weights, **2.82×** faster. Re-run: `python scripts/export_onnx.py`.

### OOV robustness (test set)

| Group | Exact-match recall |
|---|---|
| In-vocab (52 / 74) | 71.2% |
| OOV (22 / 74) | 54.5% (−16.6%) |

Half of false negatives (5/10) are OOV skills — matches EDA (31.8% OOV unique skills). Full breakdown: [docs/error_analysis.md](docs/error_analysis.md).

## Quick start

```bash
pip install -r requirements.txt
```

### Model weights

Weights are **not** in git (the checkpoint is ~513 MB). Download them from the [v1.0.0 release](https://github.com/phamthaic4/vietnamese-job-skill-ner/releases/tag/v1.0.0):

| File | Size | Use |
|---|---|---|
| [`model.onnx`](https://github.com/phamthaic4/vietnamese-job-skill-ner/releases/download/v1.0.0/model.onnx) | 513 MB | ONNX Runtime inference — 12.55 ms CPU |
| [`model.safetensors`](https://github.com/phamthaic4/vietnamese-job-skill-ner/releases/download/v1.0.0/model.safetensors) | 513 MB | `transformers` / `from_pretrained()` |
| [`tokenizer.zip`](https://github.com/phamthaic4/vietnamese-job-skill-ner/releases/download/v1.0.0/tokenizer.zip) | 0.8 MB | `config.json` + `vocab.txt` + `bpe.codes` + `tokenizer_config.json` + `added_tokens.json` |

Put all three in `model/best/` (extract `tokenizer.zip` in place):

```bash
mkdir -p model/best
# move model.onnx, model.safetensors, tokenizer.zip into model/best/
unzip tokenizer.zip -d model/best/
```

Then run:

```bash
python src/predict.py --text "Tuyển Backend Engineer thành thạo Python, Docker, PostgreSQL và Redis."
```

Full guide: [docs/reproduce.md](docs/reproduce.md).

Train from the frozen splits (Colab GPU recommended):

```bash
python src/train.py --smoke          # CPU smoke: alignment only
python src/train.py                  # full fine-tune → model/best
```

Export ONNX + benchmark:

```bash
python scripts/export_onnx.py
```

## Reproduce

| Step | Command / location |
|---|---|
| Labeled data | `data/splits/{train,val,test}.json` (frozen, seed 42) |
| Train | `python src/train.py` — PhoBERT, LR 3e-5, batch 16, 5 epochs |
| Weights | [v1.0.0 release](https://github.com/phamthaic4/vietnamese-job-skill-ner/releases/tag/v1.0.0) → extract into `model/best/` |
| Predict | `python src/predict.py --text "..."` |
| Error analysis | `python scripts/error_analysis.py` |
| ONNX | `python scripts/export_onnx.py` |

Details: [docs/reproduce.md](docs/reproduce.md).

## Repository

```
vietnamese-job-skill-ner/
├── data/splits/          # frozen train / val / test JSON
├── docs/                 # error analysis, skill vocab, reproduce
├── notebooks/            # 01_eda, 02_error_analysis
├── scripts/              # labeling, convert, error_analysis, export_onnx
├── src/train.py          # fine-tune
├── src/predict.py        # inference + post-processing
├── model/best/           # checkpoint + model.onnx (local, gitignored)
└── results/experiments.csv
```

## Roadmap

- [x] Week 1: setup + collect 100+ JD sentences + label first 50
- [x] Week 2: label to 200–300 + data pipeline + EDA (O 92.8%, OOV 31.8%)
- [x] Week 3: first training run + seqeval baseline F1 (0.6711) + WandB
- [x] Week 4: LR sweep + XLM-R comparison + error analysis + `predict.py`
- [x] Week 5: README v1, ONNX CPU benchmark (2.82×), reproduce docs
