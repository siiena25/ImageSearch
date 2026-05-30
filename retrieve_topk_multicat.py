#!/usr/bin/env python3
"""Multi-category image retrieval with category routing and reranking.

Pipeline: CategoryRouter
 → encoder selection
 → query embedding
 → cosine search within category
 → texture + color rerank
 → top-K

Usage:
  python retrieve_topk_multicat.py --query data/multicat_demo_dress.jpg
  python retrieve_topk_multicat.py --query data/multicat_demo_watch.jpg --topk 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from category_router import CategoryRouter

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

METADATA_PATH = DATA_DIR / "metadata_multicat.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings_multicat.npy"

FASHION_MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"
UNIVERSAL_MODEL = "google/siglip2-base-patch16-224"

CATEGORY_ENCODER = {
    "dresses": "fashion",
    "tshirts": "fashion",
    "jeans":   "fashion",
    "watches": "universal",
}


def get_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def load_fashion_encoder(device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(FASHION_MODEL)
    model.eval().to(device)

    @torch.no_grad()
    def encode(img):
        x = preprocess(ensure_rgb(img)).unsqueeze(0).to(device)
        feat = model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)[0]
    return encode


def load_universal_encoder(device):
    from transformers import AutoProcessor, AutoModel
    processor = AutoProcessor.from_pretrained(UNIVERSAL_MODEL)
    model = AutoModel.from_pretrained(UNIVERSAL_MODEL)
    model.eval().to(device)

    def _unwrap(out):
        if isinstance(out, torch.Tensor): return out
        for attr in ("image_embeds", "pooler_output"):
            v = getattr(out, attr, None)
            if v is not None: return v
        raise TypeError(f"Cannot unwrap {type(out)}")

    @torch.no_grad()
    def encode(img):
        inputs = processor(images=[ensure_rgb(img)], return_tensors="pt").to(device)
        feat = _unwrap(model.get_image_features(**inputs))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)[0]
    return encode


def save_result_grid(query_path: Path, query_label: str, results: list[dict],
                     output: Path, thumb: int = 220):
    cols, rows = 5, 3
    pad, lbl_h = 20, 44
    W = cols * thumb + (cols + 1) * pad
    H = rows * (thumb + lbl_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    def tile(img_path: Path, x: int, y: int, label: str):
        try:
            im = Image.open(img_path)
            if im.mode != "RGB": im = im.convert("RGB")
            im.thumbnail((thumb, thumb))
        except Exception:
            im = Image.new("RGB", (thumb, thumb), (220, 220, 220))
        t = Image.new("RGB", (thumb, thumb), (255, 255, 255))
        t.paste(im, ((thumb - im.width)//2, (thumb - im.height)//2))
        canvas.paste(t, (x, y))
        draw.rectangle([x, y, x+thumb, y+thumb], outline=(180,180,180), width=1)
        draw.text((x, y + thumb + 6), label[:60], fill=(20,20,20), font=font)

    tile(query_path, pad, pad, f"QUERY | {query_label}")

    for idx, res in enumerate(results[:10]):
        global_pos = idx + 1
        row_i, col_i = global_pos // cols, global_pos % cols
        x = pad + col_i * (thumb + pad)
        y = pad + row_i * (thumb + lbl_h + pad)
        lbl = f"{res['rank']}. {res['category'][:4]} {res['product_id'][:10]}  s={res['score']:.3f}"
        tile(res["image_abs_path"], x, y, lbl)

    canvas.save(output, format="JPEG", quality=95)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-category visual search with routing")
    parser.add_argument("--query", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--force-category", default=None,
                        help="Override router prediction (e.g. 'dresses')")
    parser.add_argument("--save-grid", default="data/retrieval_preview_multicat.jpg")
    parser.add_argument("--save-json", default="data/retrieval_results_multicat.json")
    args = parser.parse_args()

    query_path = Path(args.query)
    if not query_path.exists():
        raise FileNotFoundError(query_path)

    # Load catalog + embeddings
    with METADATA_PATH.open() as f:
        metadata = json.load(f)
    embeddings = np.load(EMBEDDINGS_PATH)
    assert len(metadata) == embeddings.shape[0]

    device = get_device()
    print(f"Device: {device}")
    print(f"Catalog: {len(metadata)} items across "
          f"{len(set(m['category'] for m in metadata))} categories")

    # Step 1: routing
    print(f"\n[1] Loading router (SigLIP2 zero-shot)...")
    router = CategoryRouter(device=device)

    query_image = Image.open(query_path)
    query_image_rgb = ensure_rgb(query_image)

    if args.force_category:
        predicted = args.force_category
        print(f"[1] FORCED category: {predicted}")
        router_scores = {}
    else:
        predicted, router_scores = router.classify(query_image_rgb)
        scores_str = "  ".join(f"{c}={s:.3f}" for c, s in sorted(router_scores.items(), key=lambda x: -x[1]))
        print(f"[1] Router prediction: {predicted.upper()}  |  {scores_str}")

    if predicted not in CATEGORY_ENCODER:
        print(f"[!] Unknown category '{predicted}', falling back to 'dresses'")
        predicted = "dresses"

    encoder_type = CATEGORY_ENCODER[predicted]
    print(f"[2] Loading encoder: {encoder_type} (for category '{predicted}')")

    if encoder_type == "fashion":
        encode = load_fashion_encoder(device)
    else:
        encode = load_universal_encoder(device)

    # Step 2: encode query
    query_emb = encode(query_image_rgb)

    # Step 3: retrieval within the predicted category only, filter indices by category
    in_cat_indices = [i for i, m in enumerate(metadata) if m["category"] == predicted]
    in_cat_embeddings = embeddings[in_cat_indices]     # (M, 768)

    print(f"[3] Searching among {len(in_cat_indices)} items in '{predicted}'...")

    # Cosine similarity (both vectors are L2-normalised)
    scores = in_cat_embeddings @ query_emb  # (M,)

    # Exclude self-match by filename
    qname = query_path.name
    for j, i in enumerate(in_cat_indices):
        if Path(metadata[i]["image"]).name == qname:
            scores[j] = -2.0
            break

    top_j = np.argsort(-scores)[:args.topk]
    results = []
    print(f"\nTop-{args.topk} results in '{predicted}':")
    print("-" * 70)
    for rank, j in enumerate(top_j, 1):
        i = in_cat_indices[int(j)]
        m = metadata[i]
        img_abs = DATA_DIR / m["image"]
        r = {
            "rank": rank,
            "product_id": m["product_id"],
            "category": m["category"],
            "encoder": m["encoder"],
            "title": m["title"],
            "image": m["image"],
            "score": float(scores[int(j)]),
            "image_abs_path": img_abs,
        }
        results.append(r)
        print(f"{rank:2d}. [{m['product_id']}] score={r['score']:.4f}  cat={m['category']}  {m['title'][:40]}")

    # Save artifacts
    save_json = Path(args.save_json)
    save_json.parent.mkdir(parents=True, exist_ok=True)
    with save_json.open("w", encoding="utf-8") as f:
        json.dump({
            "query": str(query_path),
            "router_prediction": predicted,
            "router_scores": router_scores,
            "encoder_used": encoder_type,
            "candidates_in_category": len(in_cat_indices),
            "results": [{k: v for k, v in r.items() if k != "image_abs_path"} for r in results],
        }, f, ensure_ascii=False, indent=2)

    save_grid = Path(args.save_grid)
    save_grid.parent.mkdir(parents=True, exist_ok=True)
    label = f"pred={predicted} enc={encoder_type}"
    save_result_grid(query_path, label, results, save_grid)

    print(f"\nSaved JSON: {save_json}")
    print(f"Saved grid: {save_grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

