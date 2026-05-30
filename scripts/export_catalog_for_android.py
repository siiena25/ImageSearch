#!/usr/bin/env python3
"""Build Android asset bundle from the multi-category catalog.

Output structure (default: android_bundle/):
  embeddings.bin               (N × 768 float32)
  color_signatures.bin         (N × 26  float32)
  texture.bin                  (N × 384 float32, optional)
  category_text_embeddings.bin (4 × 768 float32, category router)
  category_visual_prototypes.bin (4 × 768 float32, visual router)
  catalog.json                 [{product_id, category, encoder, title, thumb}]
  thumbnails/<product_id>.jpg  (256×256, q80)
  manifest.json

Usage:
  python3 scripts/export_catalog_for_android.py [--out android_bundle]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from color_v10 import (
    SIGNATURE_VECTOR_LEN,
    extract_color_signature,
    signature_to_vector,
)

DATA_DIR = ROOT / "data"

CATALOG_PATH = DATA_DIR / "catalog_multicat.jsonl"
METADATA_PATH = DATA_DIR / "metadata_multicat.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings_multicat.npy"
TEXTURE_PATH = DATA_DIR / "patch_features_dino.npy"
TEXTURE_IDS_PATH = DATA_DIR / "product_ids_dino.json"

THUMB_SIZE = 256
THUMB_QUALITY = 80

DEFAULT_OUT = ROOT / "android_bundle"


def _load_catalog() -> list[dict]:
    rows = []
    with CATALOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_metadata() -> dict[str, dict]:
    with METADATA_PATH.open(encoding="utf-8") as f:
        items = json.load(f)
    return {it["product_id"]: it for it in items}


def _resolve_thumbnail_source(row: dict) -> Path:
    """Prefer original (non-cropped) images for thumbnails shown to users.
    
    Falls back to SAM2-cropped image if original is not available.
    Note: embeddings/color_signatures still use cropped images for distribution
    match with the on-device query pipeline (which also segments via U2Netp).
    """
    pid = row["product_id"]
    cat = row.get("category", "")
    candidates = [
        DATA_DIR / "images" / f"{pid}.jpg",
        DATA_DIR / "images_multicat_hires" / cat / f"{pid}.jpg",
        DATA_DIR / "images_multicat_raw" / cat / f"{pid}.jpg",
        DATA_DIR / row["image"],  # cropped fallback
    ]
    for p in candidates:
        if p.exists():
            return p
    return DATA_DIR / row["image"]


def _build_thumbnails(rows: list[dict], out_dir: Path, force: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nThumbnails → {out_dir}")
    n_orig = 0
    n_cropped_fallback = 0
    for r in tqdm(rows, desc="thumbs"):
        dst = out_dir / f"{r['product_id']}.jpg"
        if dst.exists() and not force:
            continue
        src = _resolve_thumbnail_source(r)
        if "images_sam2" in src.as_posix():
            n_cropped_fallback += 1
        else:
            n_orig += 1
        img = Image.open(src).convert("RGB")
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        canvas = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (255, 255, 255))
        ox = (THUMB_SIZE - img.width) // 2
        oy = (THUMB_SIZE - img.height) // 2
        canvas.paste(img, (ox, oy))
        canvas.save(dst, "JPEG", quality=THUMB_QUALITY, optimize=True)
    print(f"  thumbnails: {n_orig} originals + {n_cropped_fallback} cropped fallback")


def _build_color_signatures(rows: list[dict]) -> np.ndarray:
    print(f"\nColor signatures (v10) for {len(rows)} items")
    out = np.zeros((len(rows), SIGNATURE_VECTOR_LEN), dtype=np.float32)
    for i, r in enumerate(tqdm(rows, desc="color")):
        img = Image.open(DATA_DIR / r["image"]).convert("RGB")
        sig = extract_color_signature(img)
        out[i] = signature_to_vector(sig)
    return out


def _build_category_text_embeddings() -> np.ndarray | None:
    """Build category text embeddings (4 × 768) for the on-device router.
    
    Pre-computed so the device doesn't need a tokenizer or text encoder.
    """
    print("\nCategory text embeddings (SigLIP2)…")
    try:
        import torch
        from transformers import AutoModel, AutoProcessor

        from category_router import CATEGORY_PROMPTS, ROUTER_MODEL_NAME
    except Exception as e:
        print(f"  ⚠ Skip (failed to import router/transformers): {e}")
        return None

    processor = AutoProcessor.from_pretrained(ROUTER_MODEL_NAME)
    model = AutoModel.from_pretrained(ROUTER_MODEL_NAME).eval()
    # Use text_model directly to avoid requiring pixel_values from the full model
    text_model = model.text_model

    cats = list(CATEGORY_PROMPTS.keys())
    vectors = []
    with torch.no_grad():
        for cat in cats:
            prompts = CATEGORY_PROMPTS[cat]
            inp = processor(text=prompts, return_tensors="pt", padding="max_length")
            text_inputs = {k: v for k, v in inp.items()
                           if k in ("input_ids", "attention_mask")}
            feats = text_model(**text_inputs).pooler_output  # [N, 768]
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            v = feats.mean(dim=0)
            v = v / (v.norm() + 1e-8)
            vectors.append(v.cpu().numpy().astype(np.float32))
    arr = np.stack(vectors, axis=0)  # (4, 768)
    print(f"  shape={arr.shape}, categories={cats}")
    return arr, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-thumbs", action="store_true")
    ap.add_argument("--skip-texture", action="store_true")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    rows = _load_catalog()
    meta = _load_metadata()
    print(f"Catalog: {len(rows)} items")

    # 1) embeddings.bin  (N × 768 float32)
    embs = np.load(EMBEDDINGS_PATH).astype(np.float32)
    assert embs.shape[0] == len(rows), (
        f"embeddings_multicat.npy size ({embs.shape[0]}) != catalog ({len(rows)})"
    )
    embs.tofile(out / "embeddings.bin")
    print(f"\nembeddings.bin: shape={embs.shape}, {(out/'embeddings.bin').stat().st_size / 1e6:.2f} MB")

    # 2) texture.bin  (N × 384 float32, re-ordered to match catalog row order)
    if not args.skip_texture and TEXTURE_PATH.exists() and TEXTURE_IDS_PATH.exists():
        tex = np.load(TEXTURE_PATH).astype(np.float32)
        with TEXTURE_IDS_PATH.open() as f:
            tex_ids = json.load(f)
        id_to_idx = {pid: i for i, pid in enumerate(tex_ids)}
        ordered = np.zeros((len(rows), tex.shape[1]), dtype=np.float32)
        missing = 0
        for i, r in enumerate(rows):
            j = id_to_idx.get(r["product_id"])
            if j is None:
                missing += 1
            else:
                ordered[i] = tex[j]
        ordered.tofile(out / "texture.bin")
        print(f"texture.bin: shape={ordered.shape}, missing={missing}, {(out/'texture.bin').stat().st_size / 1e6:.2f} MB")
    else:
        print("texture.bin: skipped")

    # 3) color_signatures.bin  (N × 26 float32)
    color = _build_color_signatures(rows)
    color.tofile(out / "color_signatures.bin")
    print(f"color_signatures.bin: shape={color.shape}, {(out/'color_signatures.bin').stat().st_size / 1e3:.1f} KB")

    # 4) category_text_embeddings.bin  (4 × 768 float32)
    res = _build_category_text_embeddings()
    if res is not None:
        text_emb, cats = res
        text_emb.tofile(out / "category_text_embeddings.bin")
        with (out / "category_text_embeddings.json").open("w") as f:
            json.dump({"categories": cats, "dim": int(text_emb.shape[1])}, f, indent=2)
        print(f"category_text_embeddings.bin: shape={text_emb.shape}")

    # 5) catalog.json
    enriched = []
    for r in rows:
        m = meta.get(r["product_id"], {})
        enriched.append({
            "product_id": r["product_id"],
            "category": r.get("category", m.get("category", "unknown")),
            "encoder": m.get("encoder", "fashion"),
            "title": r.get("title", ""),
            "thumb": f"thumbnails/{r['product_id']}.jpg",
        })
    with (out / "catalog.json").open("w") as f:
        json.dump(enriched, f, ensure_ascii=False)
    print(f"\ncatalog.json: {len(enriched)} entries")

    # 6) thumbnails/
    if not args.skip_thumbs:
        _build_thumbnails(rows, out / "thumbnails")

    # 7) manifest.json
    has_visual_proto = (out / "category_visual_prototypes.bin").exists()
    manifest = {
        "version": 2 if has_visual_proto else 1,
        "n_items": len(rows),
        "embedding_dim": int(embs.shape[1]),
        "texture_dim": 384 if (out / "texture.bin").exists() else 0,
        "color_signature_dim": SIGNATURE_VECTOR_LEN,
        "router_categories": (
            json.loads((out / "category_text_embeddings.json").read_text())["categories"]
            if (out / "category_text_embeddings.json").exists() else []
        ),
        "router_method": "visual_prototypes" if has_visual_proto else "text_embeddings",
        "thumb_size": THUMB_SIZE,
    }
    with (out / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest.json:\n{json.dumps(manifest, indent=2)}")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\n✅ Bundle ready: {out}  ({total / 1e6:.1f} MB)")
    print("Copy to Android assets:")
    print(f"  cp -r {out}/* /Users/i.kostiunina/StudioProjects/ImageSearchApp/app/src/main/assets/catalog/")


if __name__ == "__main__":
    main()
