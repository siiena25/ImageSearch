#!/usr/bin/env python3
"""Build DINOv2 patch features for texture re-ranking.

Computes bag-of-patches mean embeddings (384-dim) for each catalog image.
Output: data/patch_features_dino.npy, data/product_ids_dino.json
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CATALOG_PATH = DATA_DIR / "catalog_multicat.jsonl"
CATALOG_FALLBACK_PATH = DATA_DIR / "catalog_sam2.jsonl"  # fallback for dresses-only

OUTPUT_PATCH_FEATURES = DATA_DIR / "patch_features_dino.npy"
OUTPUT_PRODUCT_IDS_DINO = DATA_DIR / "product_ids_dino.json"

MODEL_NAME = "facebook/dinov2-small"

BATCH_SIZE = 32


def load_catalog(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
    return rows


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def compute_patch_features_batch(
    images: list[Image.Image],
    processor: AutoImageProcessor,
    model: AutoModel,
    device: str,
) -> np.ndarray:
    """Compute L2-normalized bag-of-patches mean for a batch of images."""
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=False)
        hidden = outputs.last_hidden_state  # [B, 1+N_patches, D]

    patch_mean = hidden[:, 1:, :].mean(dim=1)  # skip CLS token, mean over patches
    patch_mean = patch_mean / patch_mean.norm(dim=-1, keepdim=True)
    return patch_mean.cpu().numpy().astype(np.float32)


def main() -> int:
    if CATALOG_PATH.exists():
        catalog_path = CATALOG_PATH
        print(f"Catalog: {catalog_path}")
    elif CATALOG_FALLBACK_PATH.exists():
        catalog_path = CATALOG_FALLBACK_PATH
        print(f"⚠️  Using fallback catalog: {catalog_path}")
    else:
        raise FileNotFoundError("No catalog found.")

    catalog = load_catalog(catalog_path)
    print(f"Catalog size: {len(catalog)} items")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"\n❌ Failed to load {MODEL_NAME}: {e}")
        return 1

    model.eval().to(device)

    valid_rows = [r for r in catalog if (DATA_DIR / r["image"]).exists()]
    missing = len(catalog) - len(valid_rows)
    if missing > 0:
        print(f"⚠️  Missing images: {missing}")

    product_ids = []
    all_features = []
    failed = 0

    for i in tqdm(range(0, len(valid_rows), BATCH_SIZE), desc="DINOv2 patch features"):
        batch = valid_rows[i : i + BATCH_SIZE]
        images = []
        ids = []

        for row in batch:
            try:
                img = Image.open(DATA_DIR / row["image"])
                images.append(ensure_rgb(img))
                ids.append(str(row["product_id"]))
            except Exception as e:
                print(f"\n[WARN] {row['product_id']}: {e}")
                failed += 1

        if not images:
            continue

        try:
            features = compute_patch_features_batch(images, processor, model, device)
            product_ids.extend(ids)
            all_features.extend(features)
        except Exception as e:
            print(f"\n[WARN] Batch {i}: {e}")
            failed += len(images)

    if not all_features:
        print("❌ No features computed.")
        return 1

    features_array = np.stack(all_features, axis=0)

    np.save(OUTPUT_PATCH_FEATURES, features_array)
    with OUTPUT_PRODUCT_IDS_DINO.open("w", encoding="utf-8") as f:
        json.dump(product_ids, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"✅ Done!")
    print(f"   Patch features shape: {features_array.shape}")
    print(f"     → {features_array.shape[0]} items × {features_array.shape[1]}-dim")
    print(f"   Product IDs: {len(product_ids)}")
    print(f"   Failed: {failed}")
    print(f"\nSaved: {OUTPUT_PATCH_FEATURES}")
    print(f"Saved: {OUTPUT_PRODUCT_IDS_DINO}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

