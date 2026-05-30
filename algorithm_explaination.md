# Visual Product Search — algorithm explanation

## Overall pipeline

```text
User takes a photo
        ↓
QUERY PREPROCESSING   ← U2Netp removes background / model / clutter
        ↓
CATEGORY ROUTER       ← SigLIP2 text embeddings + cosine to category prototypes
        ↓
SEMANTIC EMBEDDING    ← FashionSigLIP (fashion) or SigLIP2 (other)
        ↓
COSINE RETRIEVAL      ← top-50 candidates from catalog
        ↓
TEXTURE + COLOR RERANK ← DINOv2 patch features + HSV color signature
        ↓
Top-10 results shown to user
```

---

## QUERY PREPROCESSING

```text
[photo of a dress on a model / gray studio background]
        ↓
U2Netp segmentation
        ↓
[only the product foreground — transparent PNG]
```

### What happens here

At this stage we remove everything that is not part of the product itself:

- studio background,
- mannequin or human model,
- shadows and visual clutter,
- other objects near the product.

The model generates a **saliency map** — a grayscale probability mask the same size as the input image.
Each pixel value in the mask is a number between 0 and 1:

- 1 (white) = belongs to the foreground (the product),
- 0 (black) = background.

We threshold this mask, apply it to the original image, and get a PNG with transparent background containing only the product.

### What is a saliency map?

A saliency map is **not** a depth map.  
A depth map encodes how far each pixel is from the camera (3D geometry).  
A saliency map encodes **how visually important** each pixel is — essentially, what the eye (or model) considers the main subject.

U2Netp is trained to produce a binary-like probability mask that separates the most salient (visually dominant) object from the rest of the image.

### What is encoder-decoder architecture?

U2Netp uses an **encoder-decoder** architecture:

- **Encoder** — a series of convolution layers that repeatedly downsample the image, compressing spatial detail into abstract features.  
  Think of it as "zooming out" step by step: `224×224 → 112×112 → 56×56 → 28×28 → 14×14`.  
  At each step the network learns higher-level patterns (edges → textures → shapes → objects).

- **Decoder** — the reverse: it gradually upsamples the compressed features back to the original resolution, combining them with skip connections from the encoder.  
  Think of it as "zooming back in" to reconstruct a pixel-level mask.

The result is a full-resolution probability mask where each pixel has a score.

### Why U2Netp (not SAM 2)?

| | U2Netp | SAM 2 |
|---|---|---|
| Model size | ~4 MB | ~180 MB |
| Inference on phone | ~300 ms | too slow |
| Prompt required | No — fully automatic | Optional |
| Output | Saliency mask | Precise object mask |

For on-device inference on a mobile phone, U2Netp is the right trade-off.  
SAM 2 was used during **catalog preprocessing** on a desktop (it is more accurate, but too large for the phone).

---

## CATEGORY ROUTER

```text
[cropped product image]
        ↓
SigLIP2 image embedding  →  768-dim vector
        ↓
cosine similarity with category text prototypes
("a photo of a dress", "a photo of jeans", "a photo of a watch", ...)
        ↓
top-1 or top-2 categories with score > threshold
```

### What happens here

We need to know **which expert model** to use for retrieval, and **which sub-catalog** to search in.

The router works by comparing the query image embedding against pre-computed **text embeddings** for each category label (e.g. `"a photo of a dress"`, `"a photo of jeans"`, etc.).

The category whose text embedding is closest (highest cosine similarity) to the query image embedding wins.

### Why text-image similarity works here

SigLIP2 is a **vision-language model** trained to align images and text in the same embedding space.  
This means the vector for an image of a dress is naturally close to the vector for the sentence `"a photo of a dress"`.

### Multi-category queries

If the second-best category is also above the confidence threshold, we search in **both** sub-catalogs and merge the results.  
This handles ambiguous queries (e.g. a cropped image of jeans that could also match dresses).

---

## SEMANTIC EMBEDDING

```text
[cropped product image — 224×224 px]
        ↓
FashionSigLIP   (if category is fashion)
   OR
SigLIP2         (if category is watches/electronics/etc.)
        ↓
768-dim normalized embedding vector
```

### What happens here

The image is encoded into a **dense vector** (768 numbers) that captures the visual meaning of the product.

This vector encodes:
- type and silhouette of the garment,
- general color distribution,
- style, neckline, sleeve length,
- category-level semantics.

### Why two models?

**FashionSigLIP** was fine-tuned specifically on fashion e-commerce images. It produces better embeddings for clothing because it has seen millions of product shots, and its representation space is structured around fashion attributes.

**SigLIP2** is a general-purpose vision-language model. For non-fashion categories (watches, electronics) it outperforms FashionSigLIP because it was not specialized to clothing.

### INT8 quantization

Both models are exported as ONNX INT8 models to reduce their size from ~370 MB (FP32) to ~90 MB each, while losing less than 0.5% cosine similarity vs. the full-precision version. This makes them usable on an Android device.

---

## COSINE RETRIEVAL

```text
query embedding q  (768-dim, unit norm)
        ↓
dot product with all catalog embeddings E  (N × 768 matrix)
        ↓
cosine similarity scores  s_i = q · e_i
        ↓
top-50 candidates by score
```

### What happens here

We compare the query vector against every catalog embedding using **cosine similarity**.  
Cosine similarity measures the angle between two vectors — if the angle is small (vectors point in the same direction), the products are semantically similar.

Because all vectors are L2-normalized to length 1, cosine similarity = dot product, which is a fast operation even on mobile hardware (658 items × 768 dims ≈ microseconds).

The output of this step is a shortlist of 50 candidates, ranked by semantic similarity.

### Why 50 candidates?

We retrieve 50 (not 10) to give the reranking step enough material to work with.  
Reranking is more expensive per item, so we limit its input to a manageable shortlist.

---

## TEXTURE + COLOR RERANK

```text
top-50 candidates
        ↓
┌──────────────────────────┐   ┌──────────────────────────┐
│   DINOv2 texture score   │   │   HSV color score        │
│                          │   │                          │
│   query DINOv2 patch     │   │   query HSV histogram    │
│   features (384-dim)     │   │   (26-bin compact sig.)  │
│   vs catalog features    │   │   vs catalog signatures  │
│   → cosine sim t_i       │   │   → weighted dist c_i    │
└──────────────────────────┘   └──────────────────────────┘
                    ↓
        final_score_i = α · semantic_i + β · texture_i + γ · color_i
        (α=0.50, β=0.30, γ=0.20)
                    ↓
        re-sorted top-10
```

### DINOv2 texture features

**DINOv2** is a self-supervised vision transformer trained by Meta on 142M images.  
Unlike SigLIP which is trained to match text, DINOv2 is trained to produce **patch-level features** — local descriptors for small regions of the image.

We use the 384-dim [CLS] token output as a compact texture descriptor.  
DINOv2 is especially good at capturing **texture, pattern, and fabric detail** — exactly what matters for print-aware retrieval (floral vs. polka-dot vs. plain).

### HSV color signature

We convert the cropped product image to **HSV color space** (Hue, Saturation, Value) and build a 26-dimensional compact signature:

- **18 hue bins** (dominant colors),
- **4 saturation bins** (vivid vs. muted),
- **4 value bins** (bright vs. dark).

HSV is chosen over RGB because hue is a perceptually meaningful color axis — it directly corresponds to "red", "blue", "green" etc., independent of lighting conditions.

### Why reranking is necessary

Semantic embedding captures **what** the product is, but not **exactly how it looks**.  
Two dresses may have identical SigLIP vectors but completely different prints and colors.

Reranking adds the "fine-grained visual similarity" layer on top of coarse semantic retrieval.

---

## CATALOG PREPROCESSING (offline, on desktop)

```text
For each catalog product image (done once, on desktop):

[raw marketplace image]
        ↓
SAM 2 segmentation          ← precise mask, no prompt
        ↓
[cropped product — transparent PNG]
        ↓
FashionSigLIP / SigLIP2     ← semantic embedding (768-dim)
DINOv2                      ← texture embedding (384-dim)
U2Netp color extraction     ← HSV color signature (26-dim)
        ↓
embeddings.bin, texture.bin, color_signatures.bin
```

### Why SAM 2 for catalog but U2Netp for query?

| | SAM 2 | U2Netp |
|---|---|---|
| Quality | Very high | Good |
| Model size | ~180 MB | ~4 MB |
| Speed | ~1 s/image on GPU | ~300 ms on CPU |
| Where used | Desktop, offline | Phone, online |

For the catalog, we have time and compute — we run SAM 2 once and store the results.  
For the query at inference time, we need a model that fits in ~4 MB and runs in real time on the phone.

---

## ANDROID ON-DEVICE INFERENCE

```text
All models bundled in app assets:
  fashion_siglip_int8.onnx   (~91 MB)
  siglip2_int8.onnx          (~96 MB)
  dinov2s_int8.onnx          (~23 MB)
  u2netp.onnx                (~4 MB)

Catalog data:
  embeddings.bin             (658 items × 768 dims)
  texture.bin                (658 items × 384 dims)
  color_signatures.bin       (658 items × 26 dims)
  catalog.json               (product metadata + thumbnail paths)

Runtime: ONNX Runtime Mobile (ORT)
```

### Inference pipeline on device

1. User takes photo or picks from gallery.
2. U2Netp removes background → cropped RGBA bitmap.
3. SigLIP2 image tower → category decision.
4. FashionSigLIP or SigLIP2 → query embedding.
5. Dot product scan over `embeddings.bin` → top-50.
6. DINOv2 → texture embedding for query.
7. Score blend → top-10 final ranking.
8. Results displayed as product thumbnails.

Total latency on Pixel 10: **~700–900 ms** end to end.
