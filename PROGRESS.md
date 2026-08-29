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
- [ ] Collect raw toward 100+ (diversify: frontend/mobile/QA/DevOps + ~15 CV lines)
- [ ] Label toward ~100 total (vocab rules now frozen in docs/skill_vocab.md)
- [x] docs/skill_vocab.md + audit of 57 sentences (API/RAG canonical rules applied)
- [x] scripts/convert_to_hf_dataset.py + 70/15/15 split (seed 42, smoke-test only)

### Notes / learnings
- Week 0 done ahead of plan: 57 labeled sentences (target was 20)
- Tokenizer splits "+" and "-" (e.g. "C ++", "Multi - agent") — PhoBERT subword alignment will handle in Week 2
- AutoModel load drops lm_head keys (UNEXPECTED warning is expected)
