#!/usr/bin/env python3
"""Build catalog embeddings with category-routing.

  dresses/tshirts/jeans -> FashionSigLIP (768-dim)
  watches               -> SigLIP2 universal (768-dim)

Output: data/embeddings_multicat.npy, data/metadata_multicat.json
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CATALOG_PATH = DATA_DIR / "catalog_multicat.jsonl"
OUT_EMBEDDINGS = DATA_DIR / "embeddings_multicat.npy"
OUT_METADATA = DATA_DIR / "metadata_multicat.json"

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


def load_catalog(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


# Fashion encoder via open_clip
class FashionEncoder:
    def __init__(self, device: str):
        import open_clip
        print(f"Loading fashion encoder: {FASHION_MODEL}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(FASHION_MODEL)
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def encode(self, img: Image.Image) -> np.ndarray:
        x = self.preprocess(ensure_rgb(img)).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)[0]


# ── Universal SigLIP2 ────────────────────────────────────────────
class UniversalEncoder:
    def __init__(self, device: str):
        from transformers import AutoProcessor, AutoModel
        print(f"Loading universal encoder: {UNIVERSAL_MODEL}")
        self.processor = AutoProcessor.from_pretrained(UNIVERSAL_MODEL)
        self.model = AutoModel.from_pretrained(UNIVERSAL_MODEL)
        self.model.eval().to(device)
        self.device = device

    @staticmethod
    def _unwrap(out):
        if isinstance(out, torch.Tensor): return out
        for attr in ("image_embeds", "pooler_output"):
            v = getattr(out, attr, None)
            if v is not None: return v
        raise TypeError(f"Cannot unwrap {type(out)}")

    @torch.no_grad()
    def encode(self, img: Image.Image) -> np.ndarray:
        inputs = self.processor(images=[ensure_rgb(img)], return_tensors="pt").to(self.device)
        feat = self._unwrap(self.model.get_image_features(**inputs))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)[0]


def main() -> int:
    catalog = load_catalog(CATALOG_PATH)
    print(f"Catalog: {len(catalog)} items")

    device = get_device()
    print(f"Device: {device}")

    # Group by encoder type - load each model only once
    by_encoder_type = {"fashion": [], "universal": []}
    for row in catalog:
        enc = CATEGORY_ENCODER[row["category"]]
        by_encoder_type[enc].append(row)

    print(f"  fashion   encoder → {len(by_encoder_type['fashion'])} items")
    print(f"  universal encoder → {len(by_encoder_type['universal'])} items")

    # Collect results in catalog order (important for reproducibility)
    results_by_pid = {}

    # Fashion encoder
    if by_encoder_type["fashion"]:
        enc = FashionEncoder(device)
        for row in tqdm(by_encoder_type["fashion"], desc="Fashion encoder"):
            img_path = DATA_DIR / row["image"]
            try:
                img = Image.open(img_path)
                emb = enc.encode(img)
                results_by_pid[row["product_id"]] = (emb, "fashion")
            except Exception as e:
                tqdm.write(f"[WARN] {row['product_id']}: {e}")
        del enc
        torch.cuda.empty_cache() if device == "cuda" else None

    # Universal encoder
    if by_encoder_type["universal"]:
        enc = UniversalEncoder(device)
        for row in tqdm(by_encoder_type["universal"], desc="Universal encoder"):
            img_path = DATA_DIR / row["image"]
            try:
                img = Image.open(img_path)
                emb = enc.encode(img)
                results_by_pid[row["product_id"]] = (emb, "universal")
            except Exception as e:
                tqdm.write(f"[WARN] {row['product_id']}: {e}")
        del enc

    # Assemble in catalog order
    embeddings = []
    metadata = []
    for row in catalog:
        pid = row["product_id"]
        if pid not in results_by_pid:
            continue
        emb, encoder_used = results_by_pid[pid]
        embeddings.append(emb)
        metadata.append({
            "product_id": pid,
            "title": row["title"],
            "image": row["image"],
            "category": row["category"],
            "encoder": encoder_used,
        })

    embeddings = np.vstack(embeddings).astype(np.float32)

    np.save(OUT_EMBEDDINGS, embeddings)
    with OUT_METADATA.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Stats
    from collections import Counter
    per_cat = Counter(m["category"] for m in metadata)
    per_enc = Counter(m["encoder"] for m in metadata)

    print(f"\n{'='*54}")
    print(f"✅ Saved embeddings: {embeddings.shape} → {OUT_EMBEDDINGS}")
    print(f"✅ Saved metadata:   {len(metadata)} entries → {OUT_METADATA}")
    print(f"  by category: {dict(per_cat)}")
    print(f"  by encoder:  {dict(per_enc)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

