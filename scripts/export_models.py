#!/usr/bin/env python3
"""Export all on-device models to ONNX FP32 + INT8.

Exports:
  1) FashionSigLIP image-tower  -> models/fashion_siglip_{fp32,int8}.onnx
  2) SigLIP2 image-tower        -> models/siglip2_{fp32,int8}.onnx
  3) DINOv2-small (texture)     -> models/dinov2s_{fp32,int8}.onnx
  4) U2Netp (segmentation)      -> models/u2netp.onnx  (no quantization)

All encoders: NCHW float32 [1, 3, 224, 224], mean=0.5, std=0.5
INT8 quantizes MatMul/Gemm only (Conv stays FP32 for Android ORT compatibility).

Usage:
  python3 scripts/export_models.py [--out models] [--skip-int8] [--only fashion|siglip2|dino|u2net]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_OUT = ROOT / "models"

FASHION_MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"
SIGLIP2_MODEL = "google/siglip2-base-patch16-224"
DINO_MODEL = "facebook/dinov2-small"

# Test images for sanity check
TEST_IMAGES = [
    DATA_DIR / "images_sam2" / "4984.jpg",
    DATA_DIR / "images_sam2" / "4931.jpg",
]


def _pick_test_image() -> Path:
    for p in TEST_IMAGES:
        if p.exists():
            return p
    candidates = list((DATA_DIR / "images_sam2").glob("*.jpg"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No test image found. Place a .jpg in data/images_sam2/ or data/images/")


def _to_tensor_norm(img: Image.Image, size: int, mean, std) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)[None, ...]  # NCHW
    return torch.from_numpy(arr)


def _torch_onnx_export(wrapper: nn.Module, dummy: torch.Tensor, fp32_path: Path) -> None:
    """Export via legacy TorchScript path (dynamo=False) for HF transformers compatibility."""
    torch.onnx.export(
        wrapper, dummy, fp32_path.as_posix(),
        input_names=["image"], output_names=["embedding"],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        dynamo=False,
    )


# ───────────────────────────────────────────────────────────────────
# 1. FashionSigLIP (open_clip)
# ───────────────────────────────────────────────────────────────────

class FashionImageTower(nn.Module):
    """open_clip image tower with L2 normalization."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model.encode_image(x)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        return feat


def export_fashion_siglip(out_dir: Path, do_int8: bool) -> tuple[Path, Path | None]:
    import open_clip

    print("\n[1/4] FashionSigLIP (open_clip)…")
    model, _, _ = open_clip.create_model_and_transforms(FASHION_MODEL)
    model.eval()
    wrapper = FashionImageTower(model).eval()

    # Preprocessing for marqo-fashionSigLIP via open_clip (224×224, OpenAI mean/std)
    dummy = torch.randn(1, 3, 224, 224)

    fp32_path = out_dir / "fashion_siglip_fp32.onnx"
    _torch_onnx_export(wrapper, dummy, fp32_path)
    print(f"  FP32 → {fp32_path} ({fp32_path.stat().st_size / 1e6:.1f} MB)")

    int8_path = None
    if do_int8:
        int8_path = _quantize_int8(fp32_path)
        _validate_cosine(fp32_path, int8_path, input_size=224, label="FashionSigLIP")

    return fp32_path, int8_path


# ───────────────────────────────────────────────────────────────────
# 2. SigLIP2 image tower (transformers)
# ───────────────────────────────────────────────────────────────────

class SigLIP2ImageTower(nn.Module):
    """SigLIP2 vision-only tower -> 768-dim L2-normalized embedding.
    
    Uses model.vision_model directly to avoid requiring input_ids from the full model.
    """

    def __init__(self, model):
        super().__init__()
        self.vision = model.vision_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.vision(pixel_values=x).pooler_output
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
        return out


def export_siglip2(out_dir: Path, do_int8: bool) -> tuple[Path, Path | None]:
    from transformers import AutoModel

    print("\n[2/4] SigLIP2 image-tower…")
    model = AutoModel.from_pretrained(SIGLIP2_MODEL).eval()
    wrapper = SigLIP2ImageTower(model).eval()

    dummy = torch.randn(1, 3, 224, 224)
    fp32_path = out_dir / "siglip2_fp32.onnx"
    _torch_onnx_export(wrapper, dummy, fp32_path)
    print(f"  FP32 → {fp32_path} ({fp32_path.stat().st_size / 1e6:.1f} MB)")

    int8_path = None
    if do_int8:
        int8_path = _quantize_int8(fp32_path)
        _validate_cosine(fp32_path, int8_path, input_size=224, label="SigLIP2")

    return fp32_path, int8_path


# ───────────────────────────────────────────────────────────────────
# 3. DINOv2-small (transformers) — patch tokens mean
# ───────────────────────────────────────────────────────────────────

class Dinov2TextureTower(nn.Module):
    """DINOv2 bag-of-patches mean descriptor, L2-normalized (384-dim)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        patches = out.last_hidden_state[:, 1:, :]  # skip CLS token
        feat = patches.mean(dim=1)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        return feat


def export_dinov2(out_dir: Path, do_int8: bool) -> tuple[Path, Path | None]:
    from transformers import AutoModel

    print("\n[3/4] DINOv2-small (texture)…")
    model = AutoModel.from_pretrained(DINO_MODEL).eval()
    wrapper = Dinov2TextureTower(model).eval()

    dummy = torch.randn(1, 3, 224, 224)
    fp32_path = out_dir / "dinov2s_fp32.onnx"
    _torch_onnx_export(wrapper, dummy, fp32_path)
    print(f"  FP32 → {fp32_path} ({fp32_path.stat().st_size / 1e6:.1f} MB)")

    int8_path = None
    if do_int8:
        int8_path = _quantize_int8(fp32_path)
        _validate_cosine(fp32_path, int8_path, input_size=224, label="DINOv2")

    return fp32_path, int8_path


# ───────────────────────────────────────────────────────────────────
# 4. U²-Netp (rembg cached)
# ───────────────────────────────────────────────────────────────────

def export_u2netp(out_dir: Path) -> Path:
    """Copy U2Netp ONNX from rembg cache to out_dir."""
    print("\n[4/4] U2Netp (query segmentation)…")
    try:
        from rembg.sessions import U2netpSession
        session = U2netpSession.download_models()
        src = Path(session)
    except Exception:
        src = Path.home() / ".u2net" / "u2netp.onnx"
        if not src.exists():
            print("  ⚠ rembg model not found. Run: pip install rembg && python -c \"from rembg import remove; remove(b'')\"")
            raise FileNotFoundError(src)

    dst = out_dir / "u2netp.onnx"
    shutil.copyfile(src, dst)
    print(f"  ONNX → {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


# ───────────────────────────────────────────────────────────────────
# INT8 quantization + validation
# ───────────────────────────────────────────────────────────────────

def _quantize_int8(fp32_path: Path) -> Path:
    """Dynamic INT8 quantization (MatMul/Gemm only, Conv stays FP32 for Android ORT compatibility)."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    int8_path = fp32_path.with_name(fp32_path.stem.replace("_fp32", "_int8") + ".onnx")
    print(f"  INT8 quantization (MatMul/Gemm only) → {int8_path.name}…")
    quantize_dynamic(
        model_input=fp32_path.as_posix(),
        model_output=int8_path.as_posix(),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"  INT8 → {int8_path} ({int8_path.stat().st_size / 1e6:.1f} MB)")
    return int8_path


def _validate_cosine(fp32_path: Path, int8_path: Path, input_size: int, label: str):
    """Verify cosine similarity between FP32 and INT8 outputs (target >= 0.99)."""
    import onnxruntime as ort

    img_path = _pick_test_image()
    x = _to_tensor_norm(Image.open(img_path), input_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).numpy()

    sess_fp = ort.InferenceSession(fp32_path.as_posix(), providers=["CPUExecutionProvider"])
    sess_q = ort.InferenceSession(int8_path.as_posix(), providers=["CPUExecutionProvider"])
    v_fp = sess_fp.run(None, {sess_fp.get_inputs()[0].name: x})[0][0]
    v_q = sess_q.run(None, {sess_q.get_inputs()[0].name: x})[0][0]

    cos = float(np.dot(v_fp, v_q) / (np.linalg.norm(v_fp) * np.linalg.norm(v_q) + 1e-8))
    status = "✅" if cos >= 0.99 else "⚠"
    print(f"  {status} {label}: cosine(FP32, INT8) = {cos:.5f}  (target ≥ 0.99)")


# ───────────────────────────────────────────────────────────────────
# main
# ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--only", choices=["fashion", "siglip2", "dino", "u2net"],
                    default=None)
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    do_int8 = not args.skip_int8

    if args.only in (None, "fashion"):
        export_fashion_siglip(out, do_int8)
    if args.only in (None, "siglip2"):
        export_siglip2(out, do_int8)
    if args.only in (None, "dino"):
        export_dinov2(out, do_int8)
    if args.only in (None, "u2net"):
        export_u2netp(out)

    print("\nDone. Next → python3 scripts/export_catalog_for_android.py")


if __name__ == "__main__":
    main()

