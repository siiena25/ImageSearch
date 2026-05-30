#!/usr/bin/env python3
"""Segment new catalog categories using Grounded-SAM2.

Reads catalog_multicat.jsonl, applies GroundingDINO + SAM2 per category,
saves results to images_sam2_multicat/{category}/ and updates catalog.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

INPUT_CATALOG = DATA_DIR / "catalog_multicat.jsonl"
OUTPUT_CATALOG = DATA_DIR / "catalog_multicat.jsonl"   # updated in-place
OUTPUT_IMAGES_DIR = DATA_DIR / "images_sam2_multicat"

GDINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM2_MODEL = "facebook/sam2.1-hiera-small"

# Text prompts per category.
# v2: narrow prompts. Adding "clothing"/"garment"/"top" captures EVERYTHING worn
# by the model (tshirt + jeans + hands), causing SAM2 to segment the whole body.
CATEGORY_PROMPTS = {
    "tshirts": "t-shirt. tee.",
    "jeans":   "jeans.",
    "watches": "watch. wristwatch.",
}

# Bbox post-processing heuristic — for tshirts/jeans on models, Grounding DINO
# may return a bbox covering the entire torso (tshirt + jeans together).
# If the aspect ratio (h/w) indicates the bbox is too tall —
# for tshirts keep the TOP `TSHIRT_KEEP_TOP` fraction of the height;
# for jeans keep the BOTTOM `JEANS_KEEP_BOTTOM` fraction.
# This trims the extra garment from the bbox.
TSHIRT_TALL_RATIO = 1.3       # if h/w > 1.3 → bbox includes the lower half too
TSHIRT_KEEP_TOP = 0.55
JEANS_SHORT_RATIO = 1.4       # if h/w < 1.4 → bbox includes the upper body
JEANS_KEEP_BOTTOM = 0.60

GDINO_BOX_THRESHOLD = 0.22   # watches are smaller than dresses → slightly lower threshold
GDINO_TEXT_THRESHOLD = 0.18
MIN_BBOX_AREA_RATIO = 0.03
BACKGROUND_COLOR = (255, 255, 255)


def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def apply_mask(img: Image.Image, mask: np.ndarray) -> Image.Image:
    img_arr = np.array(img, dtype=np.uint8)
    bg = np.full_like(img_arr, BACKGROUND_COLOR)
    return Image.fromarray(np.where(mask[:, :, np.newaxis], img_arr, bg).astype(np.uint8))


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_catalog(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_catalog(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def detect_bbox(img: Image.Image, prompt: str, processor, model, device: str):
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"], target_sizes=[img.size[::-1]]
    )[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    keep = scores >= GDINO_BOX_THRESHOLD
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return None
    w, h = img.size
    img_area = w * h
    valid = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        ratio = (x2 - x1) * (y2 - y1) / img_area
        if ratio >= MIN_BBOX_AREA_RATIO:
            valid.append((box, ratio))
    if not valid:
        return None
    best = max(valid, key=lambda x: x[1])[0]
    x1, y1, x2, y2 = best
    pad_x, pad_y = int((x2 - x1) * 0.03), int((y2 - y1) * 0.03)
    return (max(0, int(x1) - pad_x), max(0, int(y1) - pad_y),
            min(w, int(x2) + pad_x), min(h, int(y2) + pad_y))


def refine_bbox_for_category(bbox, category: str):
    """
    Crop the bbox height when it captures more than the target garment.
    - tshirts: bbox too tall → keep top (~55%)
    - jeans:   bbox too square → keep bottom (~60%)
    """
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return bbox
    ar = h / w

    if category == "tshirts" and ar > TSHIRT_TALL_RATIO:
        new_h = int(h * TSHIRT_KEEP_TOP)
        y2 = y1 + new_h
    elif category == "jeans" and ar < JEANS_SHORT_RATIO:
        new_h = int(h * JEANS_KEEP_BOTTOM)
        y1 = y2 - new_h
    return x1, y1, x2, y2


def segment_with_bbox(img: Image.Image, predictor, bbox):
    predictor.set_image(np.array(ensure_rgb(img)))
    input_box = np.array(bbox, dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=None, point_labels=None,
        box=input_box[np.newaxis, :], multimask_output=True,
    )
    mask = masks[int(np.argmax(scores))].astype(bool)
    # Strict clip: mask cannot extend beyond the bbox
    x1, y1, x2, y2 = bbox
    clip = np.zeros_like(mask, dtype=bool)
    clip[y1:y2, x1:x2] = True
    return mask & clip


def segment_center(img: Image.Image, predictor):
    predictor.set_image(np.array(ensure_rgb(img)))
    w, h = img.size
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[w // 2, h // 2]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )
    img_area = w * h
    best_mask, best_score = None, -1.0
    for mask, score in zip(masks, scores):
        ratio = mask.sum() / img_area
        if 0.02 <= ratio <= 0.95 and score > best_score:
            best_score, best_mask = float(score), mask
    return best_mask


def main() -> int:
    catalog = load_catalog(INPUT_CATALOG)

    # Select items not yet segmented (image path does not start with images_sam2*)
    # They may live in images_multicat_raw/ OR images_multicat_hires/
    to_process = [r for r in catalog if not r["image"].startswith("images_sam2")]
    already_done = len(catalog) - len(to_process)
    print(f"Catalog: {len(catalog)} items | already segmented: {already_done} | to process: {len(to_process)}")

    if not to_process:
        print("Nothing to segment. ✅")
        return 0

    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    print(f"\nLoading Grounding DINO: {GDINO_MODEL}")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL)
    gdino_model.eval().to(device)

    print(f"Loading SAM2: {SAM2_MODEL}")
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_predictor = SAM2ImagePredictor.from_pretrained(SAM2_MODEL, device=device)

    # Build product_id → row lookup for in-place updates
    pid_to_row = {r["product_id"]: r for r in catalog}

    stats = {"grounded_sam2": 0, "sam2_center": 0, "original": 0, "failed": 0}

    for row in tqdm(to_process, desc="Segmenting new items"):
        pid = row["product_id"]
        rel_in = row["image"]
        category = row["category"]
        input_path = DATA_DIR / rel_in

        if not input_path.exists():
            stats["failed"] += 1
            continue

        # Output path
        out_dir = OUTPUT_IMAGES_DIR / category
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = Path(rel_in).name
        out_path = out_dir / out_name
        rel_out = f"images_sam2_multicat/{category}/{out_name}"

        # Idempotency
        if out_path.exists():
            pid_to_row[pid]["image"] = rel_out
            pid_to_row[pid].setdefault("seg_method", "cached")
            continue

        try:
            img = ensure_rgb(Image.open(input_path))
            prompt = CATEGORY_PROMPTS[category]

            bbox = detect_bbox(img, prompt, gdino_processor, gdino_model, device)
            if bbox is not None:
                bbox = refine_bbox_for_category(bbox, category)
            method = "original"
            result = img

            if bbox is not None:
                mask = segment_with_bbox(img, sam2_predictor, bbox)
                result = apply_mask(img, mask)
                method = "grounded_sam2"
                stats["grounded_sam2"] += 1
            else:
                mask = segment_center(img, sam2_predictor)
                if mask is not None:
                    result = apply_mask(img, mask)
                    method = "sam2_center"
                    stats["sam2_center"] += 1
                else:
                    stats["original"] += 1

            result.save(out_path, format="JPEG", quality=95)
            pid_to_row[pid]["image"] = rel_out
            pid_to_row[pid]["seg_method"] = method
        except Exception as e:
            tqdm.write(f"[WARN] {pid}: {e}")
            stats["failed"] += 1

    # Write updated catalog
    save_catalog(OUTPUT_CATALOG, list(pid_to_row.values()))

    print(f"\n{'='*54}")
    print(f"✅ Done. Stats: {stats}")
    print(f"Images: {OUTPUT_IMAGES_DIR}")
    print(f"Catalog updated: {OUTPUT_CATALOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
