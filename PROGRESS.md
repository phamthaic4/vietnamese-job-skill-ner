# Progress Log

## Week 0 — 2026-08-21

**Hours spent:** 0

### Done
- [x] Create repo structure + README skeleton
- [x] Read HuggingFace token classification tutorial
- [x] Collect 30 JD sentences from ITviec/TopCV (57 unique after dedupe)
- [x] Label 20 sentences in spreadsheet (BIO) — overdelivered: all 57 sentences, 109 skill mentions, 68 unique skills, BIO valid
- [x] Colab: load vinai/phobert-base successfully

### Blockers
- None

### Next week
- [x] Collect raw toward 100+ — overdelivered: 757 unique (frontend/mobile/qa/devops/rpa/payment + 5 CVs)
- [x] Label toward ~100 total — overdelivered: 350 labeled, 372 skill mentions, 186 unique skills
- [x] Audit-fix 97 label cells (generic nouns to O, split Java Spring Boot / .NET spans, X-API rule)
- [x] Final split frozen: 244 train / 52 val / 54 test (seed 42)
- [x] notebooks/01_eda.ipynb (O 92.8%, max len 92, OOV 31.8%)
- [x] Week 3: train.py + first PhoBERT fine-tuning run on Colab → TEST F1 = 0.6711 (P 0.6538 / R 0.6892)
- [x] Week 3 Bonus: XLM-RoBERTa baseline run on Colab → TEST F1 = 0.6536 (P 0.6329 / R 0.6757)

### Notes / learnings
- Week 0 done ahead of plan: 57 labeled sentences (target was 20)
- Tokenizer splits "+" and "-" (e.g. "C ++", "Multi - agent") — PhoBERT subword alignment handled manually (no fast tokenizer for PhoBERT, @@-style BPE is word-independent)
- AutoModel load drops lm_head keys (UNEXPECTED warning is expected)
- exp_001: epoch 1 F1=0 (cold start, fresh classifier head predicts all-O) → normal; best val epoch 4 (0.6173); test 0.6711 > plan target 0.6
- Recall (0.689) > Precision (0.654): model captures skills well, boundary errors on compound skills are common
- Experiment tracked live with Weights & Biases (WandB runs: exp_001-phobert-baseline, exp_002-xlm-roberta)
- Model Comparison (Ablation): PhoBERT-base (F1=0.6711) outperforms XLM-RoBERTa-base (F1=0.6536) by +1.75% F1 on Vietnamese IT JDs, despite XLM-R having 2x larger parameter size (~1.1GB vs ~540MB) and training ~2.5x slower. Confirms monolingual pre-trained tokenizer/representations benefit Vietnamese NER.
- Lesson on val noise: XLM-R had HIGHER best-val F1 (0.6966) yet LOWER test F1 (0.6536) than PhoBERT (val 0.6173 / test 0.6711) — with only 52 val sentences, val F1 is noisy and model selection by val does not guarantee test ranking. Report test F1 as the decision metric.
