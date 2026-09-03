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
- [ ] notebooks/01_eda.ipynb
- [ ] Week 3: train.py + first PhoBERT fine-tuning run on Colab

### Notes / learnings
- Week 0 done ahead of plan: 57 labeled sentences (target was 20)
- Tokenizer splits "+" and "-" (e.g. "C ++", "Multi - agent") — PhoBERT subword alignment will handle in Week 2
- AutoModel load drops lm_head keys (UNEXPECTED warning is expected)
