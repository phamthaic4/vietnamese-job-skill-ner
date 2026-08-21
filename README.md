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

## Results

| Experiment | LR | Epochs | Batch | F1 (test) | Notes |
|------------|-----|--------|-------|-----------|-------|
| exp_001 | 3e-5 | 5 | 16 | TBD | baseline |

## Repository Structure

```
vietnamese-job-skill-ner/
├── data/            # raw, labeled, splits
├── docs/            # skill vocab, error analysis
├── notebooks/       # EDA
├── scripts/         # data conversion utilities
├── src/             # train, evaluate, predict
├── model/           # saved checkpoints
├── results/         # experiment logs
└── tests/
```

## How to Run

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py --text "Tuyển Python developer có kinh nghiệm SQL"
```

## Roadmap

- [ ] Week 1: setup + collect 100+ JD sentences + label first 50
- [ ] Week 2: label to 200–300 + data pipeline + EDA
- [ ] Week 3: first training run + seqeval baseline F1
- [ ] Week 4: hyperparameter experiments + error analysis + predict.py
- [ ] Week 5: polish README, tag release v1.0.0
