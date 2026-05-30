#!/usr/bin/env python3
"""Benchmark the on-device ONNX pipeline against the offline retrieval pipeline.

Runs the full ORT pipeline (router → encode → texture → color rerank → top-10)
and measures recall@10 vs retrieve_topk_multicat.py ground truth.
Target: mean recall@10 >= 0.9

Usage:
  python3 bench_onnx_desktop.py
  python3 bench_onnx_desktop.py --full-rerank
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from color_v10 import (
    color_gate,
    color_similarity,
    extract_color_signature,
    signature_from_vector,
    SIGNATURE_VECTOR_LEN,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

ALPHA, BETA, GAMMA = 0.45, 0.15, 0.40

# Preprocessing (mean=0.5/std=0.5 for open_clip & SigLIP2 family)
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DINO_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(img: Image.Image, size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, ...]  # NCHW
    return arr.astype(np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def _cosine_matrix(q: np.ndarray, m: np.ndarray) -> np.ndarray:
    return m @ q


def _load_bundle(bundle: Path):
    manifest = json.loads((bundle / "manifest.json").read_text())
    n = manifest["n_items"]
    d_emb = manifest["embedding_dim"]
    d_tex = manifest["texture_dim"]

    embs = np.fromfile(bundle / "embeddings.bin", dtype=np.float32).reshape(n, d_emb)
    tex = (
        np.fromfile(bundle / "texture.bin", dtype=np.float32).reshape(n, d_tex)
        if d_tex > 0 and (bundle / "texture.bin").exists() else None
    )
    color = np.fromfile(bundle / "color_signatures.bin",
                        dtype=np.float32).reshape(n, SIGNATURE_VECTOR_LEN)
    cat_text = np.fromfile(bundle / "category_text_embeddings.bin",
                           dtype=np.float32).reshape(-1, d_emb)
    cats = json.loads((bundle / "category_text_embeddings.json").read_text())["categories"]
    catalog = json.loads((bundle / "catalog.json").read_text())
    return manifest, embs, tex, color, cat_text, cats, catalog


def _route(siglip2_sess, x: np.ndarray, cat_text: np.ndarray, cats: list[str]) -> tuple[str, np.ndarray]:
    name = siglip2_sess.get_inputs()[0].name
    v = siglip2_sess.run(None, {name: x})[0][0]
    v = _normalize(v)
    sims = cat_text @ v
    return cats[int(np.argmax(sims))], v


def _encode(sess, x: np.ndarray) -> np.ndarray:
    name = sess.get_inputs()[0].name
    v = sess.run(None, {name: x})[0][0]
    return _normalize(v)


def _retrieve_ort(query_path: Path, sessions, bundle_data, top_k: int = 10,
                  full_rerank: bool = False):
    """
    Emulates the on-device pipeline.

    Default (full_rerank=False) - semantic-only cosine, comparable 1:1
    with `retrieve_topk_multicat.py` (offline ground truth is also semantic-only).
    This isolates ONNX export validation: checks that INT8-query embeddings
    are compatible with FP32 catalog embeddings.

    full_rerank=True - adds texture (DINOv2) + color (v10) + triple rerank
    + color_gate. This is the full on-device pipeline, but it diverges from
    the offline result because `retrieve_topk_multicat.py` does not rerank.
    """
    fashion_sess, siglip2_sess, dino_sess = sessions
    manifest, embs, tex, color, cat_text, cats, catalog = bundle_data

    img = Image.open(query_path).convert("RGB")
    x_clip = _preprocess(img, 224, MEAN, STD)

    # 1) Category router
    cat, sig_vec = _route(siglip2_sess, x_clip, cat_text, cats)

    # 2) Encoder branch
    if cat == "watches":
        q_emb = sig_vec # SigLIP2 already computed
    else:
        q_emb = _encode(fashion_sess, x_clip)

    # 3) Filter catalog by category
    idx = np.array([i for i, c in enumerate(catalog) if c["category"] == cat],
                   dtype=np.int64)
    if len(idx) == 0:
        return cat, []

    sub_emb = embs[idx]
    sem = sub_emb @ q_emb

    if not full_rerank:
        order = np.argsort(-sem)[:top_k]
        return cat, [catalog[idx[k]]["product_id"] for k in order]

    # Full pipeline (for end-to-end debug, not for recall vs offline)
    x_dino = _preprocess(img, 224, DINO_MEAN, DINO_STD)
    if tex is not None and dino_sess is not None:
        q_tex = _encode(dino_sess, x_dino)
        sub_tex = tex[idx]
        norms = np.linalg.norm(sub_tex, axis=1) + 1e-8
        tex_sim = (sub_tex @ q_tex) / norms
        tex_sim = np.clip(tex_sim, 0.0, 1.0)
    else:
        tex_sim = np.zeros_like(sem)

    q_sig = extract_color_signature(img)
    col_sim = np.zeros(len(idx), dtype=np.float32)
    for k, i in enumerate(idx):
        c_sig = signature_from_vector(color[i])
        col_sim[k] = color_similarity(q_sig, c_sig)

    base = ALPHA * sem + BETA * tex_sim + GAMMA * col_sim
    gates = np.array([color_gate(c) for c in col_sim], dtype=np.float32)
    final = base * gates

    order = np.argsort(-final)[:top_k]
    return cat, [catalog[idx[k]]["product_id"] for k in order]


def _retrieve_offline(query_path: Path, top_k: int = 10) -> list[str]:
    # run retrieve_topk_multicat.py and parse the JSON result
    out_json = ROOT / "data" / "_bench_offline_result.json"
    tmp_grid = ROOT / "data" / "_bench_offline_grid.jpg"
    cmd = [
        sys.executable, "retrieve_topk_multicat.py",
        "--query", str(query_path),
        "--topk", str(top_k),
        "--save-json", str(out_json),
        "--save-grid", str(tmp_grid),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("offline pipeline failed:\n" + res.stderr)
        return []
    data = json.loads(out_json.read_text())
    return [r["product_id"] for r in data.get("results", [])][:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--bundle", type=Path, default=ROOT / "android_bundle")
    ap.add_argument("--queries", nargs="+", type=Path, default=[
        DATA_DIR / "multicat_demo_dress.jpg",
        DATA_DIR / "multicat_demo_tshirt.jpg",
        DATA_DIR / "multicat_demo_jeans.jpg",
        DATA_DIR / "multicat_demo_watch.jpg",
    ])
    ap.add_argument("--variant", choices=["int8", "fp32"], default="int8")
    ap.add_argument("--full-rerank", action="store_true",
                    help="Enable texture+color+triple rerank in the ORT pipeline. "
                         "Defaults to semantic-only — like-with-like with offline.")
    args = ap.parse_args()

    suffix = "int8" if args.variant == "int8" else "fp32"
    fashion_path = args.models / f"fashion_siglip_{suffix}.onnx"
    siglip2_path = args.models / f"siglip2_{suffix}.onnx"
    dino_path = args.models / f"dinov2s_{suffix}.onnx"

    providers = ["CPUExecutionProvider"]
    fashion_sess = ort.InferenceSession(fashion_path.as_posix(), providers=providers)
    siglip2_sess = ort.InferenceSession(siglip2_path.as_posix(), providers=providers)
    dino_sess = (
        ort.InferenceSession(dino_path.as_posix(), providers=providers)
        if dino_path.exists() else None
    )
    bundle_data = _load_bundle(args.bundle)

    print(f"\nBench ORT pipeline ({suffix}, "
          f"{'full rerank' if args.full_rerank else 'semantic-only'}) "
          f"on {len(args.queries)} queries\n")
    print(f"{'query':<32} {'cat':<10} {'recall@10':>10}")
    print("-" * 56)

    recalls = []
    for q in args.queries:
        cat, ort_top = _retrieve_ort(q, (fashion_sess, siglip2_sess, dino_sess),
                                     bundle_data, top_k=10,
                                     full_rerank=args.full_rerank)
        offline_top = _retrieve_offline(q, top_k=10)
        if not offline_top:
            r = float("nan")
        else:
            r = len(set(ort_top) & set(offline_top)) / 10.0
        recalls.append(r)
        print(f"{q.name:<32} {cat:<10} {r * 10:>9.0f}/10")

    valid = [r for r in recalls if not np.isnan(r)]
    if valid:
        avg = sum(valid) / len(valid)
        target = 0.9
        status = "✅" if avg >= target else "⚠"
        print(f"\n{status} mean recall@10 = {avg:.2f}  (target ≥ {target})")


if __name__ == "__main__":
    main()

