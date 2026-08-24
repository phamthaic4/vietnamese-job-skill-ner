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
- Collect more raw sentences toward 100+ (have 57)
- Write docs/skill_vocab.md (~50 common IT skills as labeling reference)
- Label toward 200-300 total
- Write scripts/convert_to_hf_dataset.py + train/val/test split

### Notes / learnings
- Week 0 done ahead of plan: 57 labeled sentences (target was 20)
- Tokenizer splits "+" and "-" (e.g. "C ++", "Multi - agent") — PhoBERT subword alignment will handle in Week 2
- AutoModel load drops lm_head keys (UNEXPECTED warning is expected)
