"""Export model/best to ONNX and benchmark torch vs onnxruntime latency."""

import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "best"
ONNX_PATH = ROOT / "model" / "best" / "model.onnx"
TEXT = "Yêu cầu 2 năm kinh nghiệm với Python, Docker và Kubernetes."
N_RUNS = 50


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    encoded = tokenizer(TEXT, return_tensors="pt")
    input_ids = encoded["input_ids"].long()
    attention_mask = encoded["attention_mask"].long()
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1
    assert attention_mask.shape == input_ids.shape

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(ONNX_PATH),
        opset_version=14,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        dynamo=False,  # legacy exporter: matches skeleton API; avoids Windows utf-8 crash in new path
    )
    print(f"Exported: {ONNX_PATH}")

    # Warmup + torch latency
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)

        torch_times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            torch_times.append(time.perf_counter() - t0)

    import onnxruntime as ort

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }

    for _ in range(5):
        _ = session.run(None, ort_inputs)

    ort_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        _ = session.run(None, ort_inputs)
        ort_times.append(time.perf_counter() - t0)

    torch_ms = float(np.mean(torch_times) * 1000)
    ort_ms = float(np.mean(ort_times) * 1000)
    speedup = torch_ms / ort_ms if ort_ms > 0 else float("inf")

    print(f"Torch mean latency : {torch_ms:.2f} ms  (n={N_RUNS})")
    print(f"ORT   mean latency : {ort_ms:.2f} ms  (n={N_RUNS})")
    print(f"Speedup (torch/ort): {speedup:.2f}x")


if __name__ == "__main__":
    main()
