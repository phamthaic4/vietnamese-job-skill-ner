# `model/best`

This directory is gitignored. Drop the fine-tuned PhoBERT checkpoint here:

```
model/best/config.json
model/best/*.safetensors   # or pytorch_model.bin
model/best/               # tokenizer files from AutoTokenizer.save_pretrained
model/best/model.onnx     # optional, from scripts/export_onnx.py
```

See [docs/reproduce.md](../docs/reproduce.md).
