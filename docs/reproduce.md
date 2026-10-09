# Reproduce v1.0.0

How to get the same checkpoint, metrics, and ONNX latency numbers.

## 1. Environment

```bash
pip install -r requirements.txt
```

Python 3.12+ on Windows/Linux. Training used Colab GPU; inference and ONNX export were measured on **CPU**.

## 2. Data (already in the repo)

Frozen split, seed 42:

| File | Sentences |
|---|---|
| `data/splits/train.json` | 244 |
| `data/splits/val.json` | 52 |
| `data/splits/test.json` | 54 |

Do not reshuffle. Relabeling would invalidate the reported F1.

## 3. Weights (`model/best`)

`model/` is gitignored (PhoBERT checkpoint is ~500MB). Place files here:

```
model/best/
  config.json
  model.safetensors   # or pytorch_model.bin
  tokenizer files (vocab.txt, merges, tokenizer_config.json, …)
  model.onnx          # optional; created by export_onnx.py
```

Get the folder by either:

1. **Train** (below), or
2. Copy a previously trained `model/best` from the machine / Colab that ran exp_001.

## 4. Train (locked config)

```bash
python src/train.py --smoke    # CPU: tokenizer / alignment only
python src/train.py            # GPU recommended
```

Locked hyperparameters:

| Item | Value |
|---|---|
| Base | `vinai/phobert-base` |
| LR | `3e-5` |
| Batch | 16 |
| Epochs | 5 |
| Max length | 128 |

Expected **test** numbers for exp_001 (do not pick the checkpoint by val F1 alone):

| Precision | Recall | F1 |
|---|---|---|
| 0.6538 | 0.6892 | **0.6711** |

Val F1 was 0.6173 at epoch 4. With 52 val sentences, val ranking can disagree with test (see XLM-R: higher val, lower test).

## 5. Inference

```bash
python src/predict.py --text "Yêu cầu 2 năm kinh nghiệm với Python, Docker và Kubernetes."
```

Post-processing is part of the v1 pipeline (glue `/` `+`, merge adjacent `B-SKILL`, strip trailing `.` but keep `.NET`).

## 6. ONNX export + CPU benchmark

Requires `onnx`, `onnxruntime`, `onnxscript` (listed in `requirements.txt`).

```bash
python scripts/export_onnx.py
```

Writes `model/best/model.onnx` and prints mean latency over 50 runs (5 warmup). Measured locally:

| Runtime | Mean latency |
|---|---|
| PyTorch CPU | 35.39 ms |
| ONNX Runtime CPU | 12.55 ms |
| Speedup | **2.82×** |

Exporter uses the TorchScript path (`dynamo=False`, opset 14) so `dynamic_axes` matches the Week 5 skeleton on Windows.

## 7. Error analysis

```bash
python scripts/error_analysis.py
```

Narrative report (maintained by hand): [error_analysis.md](error_analysis.md).
