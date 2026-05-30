"""Re-quantize FP32 ONNX models to Android-compatible INT8.

Quantizes only MatMul/Gemm ops (not Conv), avoiding ConvInteger(opset 10)
which is unsupported by onnxruntime-android.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

TARGETS = [
    ("fashion_siglip_fp32.onnx", "fashion_siglip_int8.onnx", 224),
    ("siglip2_fp32.onnx",        "siglip2_int8.onnx",        224),
    ("dinov2s_fp32.onnx",        "dinov2s_int8.onnx",        224),
]


def requantize(fp32_name: str, int8_name: str) -> Path:
    fp32 = MODELS_DIR / fp32_name
    int8 = MODELS_DIR / int8_name
    assert fp32.exists(), f"Missing FP32 file: {fp32}"
    print(f"[{fp32_name}] {fp32.stat().st_size / 1e6:.1f} MB → INT8 (MatMul/Gemm only)…")
    quantize_dynamic(
        model_input=fp32.as_posix(),
        model_output=int8.as_posix(),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"  → {int8.name}: {int8.stat().st_size / 1e6:.1f} MB")
    return int8


def validate(fp32_path: Path, int8_path: Path, size: int, label: str):
    # Verify that the Android-compatible INT8 model gives cosine ≥ 0.99 vs FP32
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 3, size, size), dtype=np.float32)

    s_fp = ort.InferenceSession(fp32_path.as_posix(), providers=["CPUExecutionProvider"])
    s_q = ort.InferenceSession(int8_path.as_posix(), providers=["CPUExecutionProvider"])

    name_fp = s_fp.get_inputs()[0].name
    name_q = s_q.get_inputs()[0].name
    y_fp = s_fp.run(None, {name_fp: x})[0].reshape(-1)
    y_q = s_q.run(None, {name_q: x})[0].reshape(-1)

    cos = float(y_fp @ y_q / (np.linalg.norm(y_fp) * np.linalg.norm(y_q) + 1e-8))
    status = "✅" if cos >= 0.99 else "⚠️"
    print(f"  {status} cosine(FP32, INT8) = {cos:.5f} [{label}]")


def main():
    for fp32_name, int8_name, size in TARGETS:
        int8_path = requantize(fp32_name, int8_name)
        validate(MODELS_DIR / fp32_name, int8_path, size, fp32_name.replace("_fp32.onnx", ""))
    print("\nDone. Copy the updated int8 models into Android assets:")
    print("  cp models/{fashion_siglip_int8,siglip2_int8,dinov2s_int8}.onnx \\")
    print("     /Users/i.kostiunina/StudioProjects/ImageSearchApp/app/src/main/assets/models/")


if __name__ == "__main__":
    main()
