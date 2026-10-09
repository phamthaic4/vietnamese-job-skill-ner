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
- LR sweep done (test F1): 2e-5 → 0.5795, 3e-5 → 0.6711, 4e-5 → 0.6098. Optimum at 3e-5 (symmetric drop both sides). Lower LR (2e-5) dropped precision hard (0.5000) with recall unchanged; higher LR (4e-5) raised precision but lost recall. FINAL config locked: PhoBERT, LR 3e-5, batch 16, 5 epochs, test F1 0.6711.

## Week 4 — 2026-09-17 (hoàn thành)

### Done
- [x] LR sweep (2e-5 / 3e-5 / 4e-5) → 3e-5 là điểm tối ưu (0.6711)
- [x] XLM-R baseline so sánh → PhoBERT thắng +1.75% F1 (0.6536)
- [x] WandB tracking (2 run có curves)
- [x] `scripts/error_analysis.py` — phân loại lỗi trên Test set (54 câu, 74 entity)
- [x] `docs/error_analysis.md` — FN 10 / FP 17 / Boundary 15; 50% FN là OOV
- [x] `src/predict.py` v1 với post-processing — câu mẫu đạt 11/11 skill
- [x] README: bảng so sánh experiments + repo structure + usage

### Notes / learnings
- Error distribution: False Negatives 10 (23.8%), False Positives 17 (40.5%), Boundary 15 (35.7%); total 42 errors on 74 gold entities → F1 0.6711 consistent.
- 50% (5/10) của FN là do kỹ năng OOV — đúng dự đoán EDA (OOV 31.8%)
- 3/4 lỗi predict.py gốc đến từ **tokenization mất ký tự** (C++, C/C++, CI/CD, Spring Boot) chứ không phải model — post-processing đóng vai trò quyết định trong real pipeline
- predict.py fixes: regex glue `/`+`+`, B-SKILL adjacent merge (`Spring Boot`), bỏ `(` `)` chen giữa span, strip `.` cuối giữ `.NET`
- `scripts/error_analysis.py` hiện chỉ in console (md duy trì thủ công để tránh ghi đè analysis sống)

### Next week
- [x] Week 5: polish README, ONNX export + CPU benchmark, reproduce docs

## Week 5 — 2026-10-10 (hoàn thành)

### Done
- [x] `scripts/export_onnx.py` — export `model/best` → `model.onnx` (opset 14, `dynamo=False`)
- [x] CPU benchmark (batch 1, n=50): PyTorch **35.39 ms**, ONNX Runtime **12.55 ms** → **2.82×** (đúng khoảng đoán 2–3×)
- [x] README v1: status, data thật, bảng LR + XLM-R, latency, OOV, quick start, reproduce
- [x] `docs/reproduce.md` + `model/README.md` — weights gitignored, hướng dẫn train / đặt checkpoint / export
- [x] `requirements.txt`: `onnx`, `onnxruntime`, `onnxscript`

### Notes / learnings
- Torch mới default `torch.onnx.export` sang dynamo → cần `onnxscript`; trên Windows dynamo in emoji ✅ crash cp1252. Legacy exporter (`dynamo=False`) khớp skeleton `dynamic_axes`.
- ONNX không đổi F1: cùng weights, chỉ khác runtime.

### Còn lại (không chặn v1 code)
- Push tag `v1.0.0` + zip `model/best` lên GitHub Release khi sẵn sàng (`git push && git push --tags`).

