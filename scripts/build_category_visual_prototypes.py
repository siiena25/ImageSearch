#!/usr/bin/env python3
"""Build visual category prototypes (mean per-category embeddings) for the on-device router.

Instead of text prompts, the router compares query embeddings against visual centroids
computed from actual catalog images. This is more robust for edge cases like children's
fashion, dark dresses, and denim shades where text-only routing fails.

Tries both SigLIP2 and FashionSigLIP, picks the one with the larger inter-category margin.

Output:
  android_bundle/category_visual_prototypes.bin   (4 × 768 float32)
  android_bundle/category_visual_prototypes.json

Usage:
  python3 scripts/build_category_visual_prototypes.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BUNDLE_DIR = ROOT / "android_bundle"
CATALOG_PATH = DATA_DIR / "catalog_multicat.jsonl"

# Both encoders are evaluated; the one with higher inter-category margin is used
ENCODERS = {
    "siglip2":        ROOT / "models" / "siglip2_int8.onnx",
    "fashion_siglip": ROOT / "models" / "fashion_siglip_int8.onnx",
}

MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr.astype(np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def compute_prototypes(encoder_name: str, model_path: Path, by_cat: dict[str, list]) -> dict[str, np.ndarray]:
    print(f"\n{'='*60}\n[{encoder_name}] {model_path.name}\n{'='*60}")
    sess = ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    prototypes: dict[str, np.ndarray] = {}
    for cat, items in by_cat.items():
        vecs = []
        for it in tqdm(items, desc=f"{encoder_name}/{cat}"):
            img_path = DATA_DIR / it["image"]
            if not img_path.exists():
                continue
            x = preprocess(Image.open(img_path))
            v = sess.run(None, {in_name: x})[0][0]
            vecs.append(normalize(v))
        if not vecs:
            continue
        proto = normalize(np.mean(np.stack(vecs), axis=0))
        prototypes[cat] = proto.astype(np.float32)
    return prototypes


def margin_stats(prototypes: dict[str, np.ndarray]) -> float:
    """Return mean margin between top-1 and top-2 cosine scores (higher = clearer routing)."""
    cats = list(prototypes.keys())
    cross = np.zeros((len(cats), len(cats)), dtype=np.float32)
    for i, c1 in enumerate(cats):
        for j, c2 in enumerate(cats):
            cross[i, j] = float(np.dot(prototypes[c1], prototypes[c2]))
    print("\n       " + "  ".join(f"{c:>10}" for c in cats))
    for i, c1 in enumerate(cats):
        row = [f"{cross[i,j]:.3f}" for j in range(len(cats))]
        print(f"{c1:>10}  " + "  ".join(f"{v:>10}" for v in row))

    diag = np.diag(cross).copy()
    np.fill_diagonal(cross, -np.inf)
    second_best = cross.max(axis=1)
    margins = diag - second_best
    print(f"\n  margins: {dict(zip(cats, margins.round(3).tolist()))}")
    print(f"  mean margin = {float(margins.mean()):.3f}")
    return float(margins.mean())


def main():
    rows = []
    with CATALOG_PATH.open() as f:
        for line in f:
            rows.append(json.loads(line))
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    results: dict[str, tuple[float, dict[str, np.ndarray]]] = {}
    for name, path in ENCODERS.items():
        protos = compute_prototypes(name, path, by_cat)
        m = margin_stats(protos)
        results[name] = (m, protos)

    best_name = max(results, key=lambda k: results[k][0])
    print(f"\n🏆 Best encoder for routing: {best_name} "
          f"(margin {results[best_name][0]:.3f})")
    prototypes = results[best_name][1]

    # Save in manifest.router_categories order - critical for index alignment!
    manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text())
    ordered_cats = manifest["router_categories"]
    arr = np.zeros((len(ordered_cats), 768), dtype=np.float32)
    for i, c in enumerate(ordered_cats):
        if c not in prototypes:
            raise RuntimeError(f"category '{c}' from manifest missing in prototypes")
        arr[i] = prototypes[c]

    out_bin = BUNDLE_DIR / "category_visual_prototypes.bin"
    out_json = BUNDLE_DIR / "category_visual_prototypes.json"
    arr.tofile(out_bin)
    out_json.write_text(json.dumps({
        "categories": ordered_cats,
        "dim": int(arr.shape[1]),
        "encoder": best_name,
        "computed_on": "cropped_catalog_images",
        "mean_margin": round(results[best_name][0], 3),
    }, indent=2))
    print(f"\n✅ {out_bin}  ({out_bin.stat().st_size / 1024:.1f} KB)")
    print(f"✅ {out_json}")


if __name__ == "__main__":
    main()

