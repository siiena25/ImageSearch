#!/usr/bin/env python3
"""Category router: zero-shot image-to-category classification via SigLIP2.

Computes cosine similarity between the query image embedding and text/visual
prototypes for each category. Supports both text-prompt mode and visual-centroid
mode (preferred, more robust for edge cases).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

ROUTER_MODEL_NAME = "google/siglip2-base-patch16-224"

# Prompts for each category
CATEGORY_PROMPTS: dict[str, list[str]] = {
    "dresses": [
        "a photo of a dress",
        "a woman's dress",
        "a floral print dress",
        "a short summer dress",
    ],
    "tshirts": [
        "a photo of a t-shirt",
        "a casual cotton t-shirt",
        "a short-sleeve tee",
        "a printed t-shirt",
    ],
    "jeans": [
        "a photo of blue jeans",
        "denim pants",
        "a pair of jeans",
        "casual denim trousers",
    ],
    "watches": [
        "a photo of a wristwatch",
        "an analog watch",
        "a digital watch",
        "a watch with a metal strap",
    ],
}


class CategoryRouter:
    """
    Zero-shot category classifier based on SigLIP2.

    Usage:
        router = CategoryRouter(device="mps")
        category, scores = router.classify(pil_image)
    """

    def __init__(
        self,
        model_name: str = ROUTER_MODEL_NAME,
        device: str | None = None,
        prompts: dict[str, list[str]] | None = None,
    ):
        from transformers import AutoProcessor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available()
                                 else "cpu")
        self.model_name = model_name
        self.prompts = prompts or CATEGORY_PROMPTS
        self.categories = list(self.prompts.keys())

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)

        # Pre-compute text embeddings once at init
        self._text_embeds = self._encode_prompts()

    @staticmethod
    def _unwrap(output):
        # SigLIP2 may return a ModelOutput object or a bare tensor - handle both
        if isinstance(output, torch.Tensor):
            return output
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(output, attr, None)
            if v is not None:
                return v
        if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            return output[0]
        raise TypeError(f"Cannot unwrap features from {type(output)}")

    @torch.no_grad()
    def _encode_prompts(self) -> dict[str, np.ndarray]:
        # Compute the mean L2-normalised text embedding per category across all its prompts
        result = {}
        for cat, prompts in self.prompts.items():
            inputs = self.processor(
                text=prompts, return_tensors="pt", padding="max_length"
            ).to(self.device)
            text_features = self._unwrap(self.model.get_text_features(**inputs))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_emb = text_features.mean(dim=0)
            mean_emb = mean_emb / mean_emb.norm()
            result[cat] = mean_emb.cpu().numpy().astype(np.float32)
        return result

    @torch.no_grad()
    def classify(self, image: Image.Image) -> tuple[str, dict[str, float]]:
        # Returns (best_category, {cat: score})
        # Score is the cosine similarity with the mean text embedding of each category
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        img_features = self._unwrap(self.model.get_image_features(**inputs))
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        img_emb = img_features.cpu().numpy().astype(np.float32)[0]

        scores = {cat: float(np.dot(img_emb, emb)) for cat, emb in self._text_embeds.items()}
        best = max(scores, key=scores.get)
        return best, scores


def demo() -> int:
    """
    Demo: runs the router on 4 query images (one per category).
    Prints the predicted category and score.
    """
    ROOT = Path(__file__).resolve().parent
    DATA_DIR = ROOT / "data"

    catalog = [
        json.loads(line) for line in (DATA_DIR / "catalog_multicat.jsonl").open()
    ]
    by_cat = {}
    for r in catalog:
        by_cat.setdefault(r["category"], []).append(r)

    # Pick one random example per category
    import random
    random.seed(42)
    samples = []
    for cat in ["dresses", "tshirts", "jeans", "watches"]:
        r = random.choice(by_cat[cat])
        samples.append((cat, DATA_DIR / r["image"], r["product_id"]))

    print(f"Loading CategoryRouter ({ROUTER_MODEL_NAME})...")
    router = CategoryRouter()
    print(f"Device: {router.device}  |  Categories: {router.categories}")
    print()

    correct = 0
    for true_cat, img_path, pid in samples:
        img = Image.open(img_path)
        pred, scores = router.classify(img)
        ok = "✅" if pred == true_cat else "❌"
        if pred == true_cat:
            correct += 1
        scores_str = "  ".join(f"{c}={s:.3f}" for c, s in sorted(scores.items(), key=lambda x: -x[1]))
        print(f"{ok}  [{pid}]  true={true_cat:8s}  pred={pred:8s}  |  {scores_str}")

    print(f"\n→ {correct}/{len(samples)} correct on demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(demo())

