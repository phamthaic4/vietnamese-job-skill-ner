# `model/best`

This directory is gitignored. Drop the fine-tuned PhoBERT checkpoint here:

```
model/best/config.json
model/best/*.safetensors   # or pytorch_model.bin
model/best/                 # tokenizer files from AutoTokenizer.save_pretrained
model/best/model.onnx       # optional, from scripts/export_onnx.py
```

## Download

Get the files from the [v1.0.0 release](https://github.com/phamthaic4/vietnamese-job-skill-ner/releases/tag/v1.0.0):

| Asset | Size | Contents |
|---|---|---|
| [`model.onnx`](https://github.com/phamthaic4/vietnamese-job-skill-ner/releases/download/v1.0.0/model.onnx) | 513 MB | ONNX graph (opset 14) |
| [`model.safetensors`](https://github.com/phamthaic4/vietnamese-job-skill-ner/releases/download/v1.0.0/model.safetensors) | 513 MB | Fine-tuned weights |
| [`tokenizer.zip`](https://github.com/phamthaic4/vietnamese-job-skill-ner/releases/download/v1.0.0/tokenizer.zip) | 0.8 MB | `config.json`, `tokenizer_config.json`, `vocab.txt`, `bpe.codes`, `added_tokens.json` |

```bash
mkdir -p model/best
# move the three files into model/best/
unzip tokenizer.zip -d model/best/
```

See [docs/reproduce.md](../docs/reproduce.md).
