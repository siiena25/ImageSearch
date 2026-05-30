#!/usr/bin/env python3
"""Build multi-category catalog (catalog_multicat.jsonl).

Categories:
  dresses  -> existing catalog_sam2.jsonl
  tshirts, jeans, watches -> ashraq/fashion-product-images-small

Output: data/catalog_multicat.jsonl, data/images_multicat_raw/{category}/*.jpg
"""

import json
import os
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Existing dress catalog (already segmented by SAM2)
EXISTING_DRESS_CATALOG = DATA_DIR / "catalog_sam2.jsonl"

# Output
OUTPUT_CATALOG = DATA_DIR / "catalog_multicat.jsonl"
RAW_IMAGES_DIR = DATA_DIR / "images_multicat_raw"

# Number of new items per category
NEW_CATEGORY_LIMITS = {
    "tshirts": 100,
    "jeans":   80,
    "watches": 100,
}

# Filter by articleType in the ashraq dataset
ARTICLE_TYPE_MAP = {
    "tshirts": "Tshirts",
    "jeans":   "Jeans",
    "watches": "Watches",
}

# Encoder to use per category
CATEGORY_ENCODER = {
    "dresses": "fashion",   # Marqo FashionSigLIP
    "tshirts": "fashion",
    "jeans":   "fashion",
    "watches": "universal", # google/siglip2-base
}

DATASET_NAME = "ashraq/fashion-product-images-small"


def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def load_existing_dresses() -> list[dict]:
    # Load the existing 399 dresses from catalog_sam2.jsonl and add the category field.
    if not EXISTING_DRESS_CATALOG.exists():
        raise FileNotFoundError(
            f"{EXISTING_DRESS_CATALOG} not found. Build the dress catalog first."
        )

    rows = []
    with EXISTING_DRESS_CATALOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["category"] = "dresses"
            # image already points to images_sam2/{id}.jpg - keep as-is
            rows.append(row)
    return rows


def fetch_category_items(category: str, limit: int) -> list[dict]:
    """
    Download up to `limit` items from ashraq for the given category (articleType).
    Saves raw images to images_multicat_raw/{category}/{id}.jpg.
    Returns a list of catalog records (without seg_method — segmentation is a separate step).
    """
    from datasets import load_dataset

    article_type = ARTICLE_TYPE_MAP[category]
    out_dir = RAW_IMAGES_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n→ Loading {category} (articleType={article_type}, limit={limit})...")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    rows = []
    saved = 0
    scanned = 0
    for item in ds:
        scanned += 1
        if str(item.get("articleType", "")).strip() != article_type:
            continue

        pid = str(item["id"]).strip()
        # Category prefix so product_ids don't collide with dresses
        unique_id = f"{category[:3]}_{pid}"

        img_filename = f"{unique_id}.jpg"
        img_path = out_dir / img_filename

        if not img_path.exists():
            try:
                pil = ensure_rgb(item["image"])
                pil.save(img_path, format="JPEG", quality=95)
            except Exception as e:
                print(f"  [WARN] failed to save {unique_id}: {e}")
                continue

        rel_path = f"images_multicat_raw/{category}/{img_filename}"

        # Title: "{ArticleType} {id}" - predictable for UI
        title = f"{article_type.rstrip('s')} {pid}"

        rows.append({
            "product_id": unique_id,
            "title": title,
            "image": rel_path,        # raw for now; segment_multicat.py will update to sam2 path
            "category": category,
        })
        saved += 1

        if saved >= limit:
            break

    print(f"  saved: {saved} / scanned: {scanned}")
    return rows


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Existing dresses
    print(f"Loading existing dresses from {EXISTING_DRESS_CATALOG.name}...")
    all_rows = load_existing_dresses()
    print(f"  loaded: {len(all_rows)} dresses")

    # 2) New categories
    for category, limit in NEW_CATEGORY_LIMITS.items():
        new_rows = fetch_category_items(category, limit)
        all_rows.extend(new_rows)

    # 3) Save
    with OUTPUT_CATALOG.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    stats = Counter(r["category"] for r in all_rows)

    print(f"\n{'='*54}")
    print(f"✅ Multi-category catalog saved: {OUTPUT_CATALOG}")
    print(f"   Total items: {len(all_rows)}")
    for cat, cnt in stats.most_common():
        enc = CATEGORY_ENCODER[cat]
        print(f"     {cat:10s} : {cnt:4d} items  [encoder={enc}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
