#!/usr/bin/env python3
"""Re-fetch hi-res images for tshirts/jeans/watches from ceyda/fashion-products-small.

ashraq dataset provides low-res thumbnails (~60x80). This script fetches the
hi-res versions (~384x512) using matching product IDs from the ceyda dataset.

Updates data/catalog_multicat.jsonl image paths to the hi-res versions.
"""

import json
from pathlib import Path
from collections import Counter

from datasets import load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CATALOG = DATA_DIR / "catalog_multicat.jsonl"
HIRES_DIR = DATA_DIR / "images_multicat_hires"

ASHRAQ = "ashraq/fashion-product-images-small" # source of articleType metadata
CEYDA = "ceyda/fashion-products-small" # source of hi-res images

ARTICLE_TYPE_MAP = {
    "tshirts": "Tshirts",
    "jeans":   "Jeans",
    "watches": "Watches",
}


def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def load_catalog(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def save_catalog(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    catalog = load_catalog(CATALOG)
    # Collect required ids per category (without prefix)
    # product_id format: "tsh_10065" → raw_id = "10065"
    need: dict[str, set[str]] = {c: set() for c in ARTICLE_TYPE_MAP}
    for r in catalog:
        cat = r["category"]
        if cat in need:
            raw_id = r["product_id"].split("_", 1)[1]
            need[cat].add(raw_id)

    print("Need hi-res images for:")
    for cat, s in need.items():
        print(f"  {cat:8s}: {len(s)}")

    all_need_ids = set()
    id_to_cat = {}
    for cat, ids in need.items():
        for i in ids:
            all_need_ids.add(i)
            id_to_cat[i] = cat

    HIRES_DIR.mkdir(parents=True, exist_ok=True)
    for cat in need:
        (HIRES_DIR / cat).mkdir(parents=True, exist_ok=True)

    # Stream ceyda dataset — save images matching the required IDs
    print(f"\nStreaming {CEYDA} (will take 1-2 min)...")
    ds = load_dataset(CEYDA, split="train", streaming=True)

    saved_per_cat = Counter()
    sample_sizes = {}
    n_scanned = 0
    remaining = set(all_need_ids)

    for item in ds:
        n_scanned += 1
        if n_scanned % 5000 == 0:
            print(f"  scanned {n_scanned} | remaining {len(remaining)}")

        iid = str(item.get("id", "")).strip()
        if iid not in remaining:
            continue
        cat = id_to_cat[iid]
        try:
            pil = ensure_rgb(item["image"])
            if cat not in sample_sizes:
                sample_sizes[cat] = pil.size
            out_path = HIRES_DIR / cat / f"{cat[:3]}_{iid}.jpg"
            pil.save(out_path, format="JPEG", quality=95)
            saved_per_cat[cat] += 1
            remaining.discard(iid)
        except Exception as e:
            print(f"  [WARN] id={iid}: {e}")

        if not remaining:
            print("  all found ✅")
            break

    print(f"\nHi-res samples by category:")
    for cat, sz in sample_sizes.items():
        print(f"  {cat:8s}: size={sz}  saved={saved_per_cat[cat]}/{len(need[cat])}")

    # Update catalog: image → hi-res path
    # Also clear seg_method since re-segmentation is required
    missing_rows = []
    for r in catalog:
        cat = r["category"]
        if cat not in need:
            continue
        raw_id = r["product_id"].split("_", 1)[1]
        hires_path = HIRES_DIR / cat / f"{cat[:3]}_{raw_id}.jpg"
        if hires_path.exists():
            r["image"] = f"images_multicat_hires/{cat}/{cat[:3]}_{raw_id}.jpg"
            r.pop("seg_method", None)   # needs re-segmentation
        else:
            missing_rows.append(r["product_id"])

    if missing_rows:
        print(f"\n⚠️  {len(missing_rows)} products not found in ceyda (left unchanged):")
        for m in missing_rows[:20]:
            print(f"    {m}")

    save_catalog(CATALOG, catalog)
    print(f"\n✅ Catalog updated: {CATALOG}")
    print(f"   Images: {HIRES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
