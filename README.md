# ImageSearch

End-to-end on-device visual product search for marketplaces.  
Combines SAM 2 segmentation · FashionSigLIP / SigLIP 2 embeddings · DINOv2 texture · HSV colour reranking · category router · Android demo (Pixel 10, Android 16).

---

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Script execution order

Run every script from the **project root** (`ImageSearch/`).

### Step 1 — Download & build the raw catalog

```bash
python3 scripts/build_catalog_from_asos.py # dresses from ASOS CSV
python3 scripts/build_multicat_catalog.py # tshirts, jeans, watches from HF datasets
python3 scripts/refetch_multicat_hires.py # re-download hi-res images for all categories
```

### Step 2 — Segment catalog images with Grounded-SAM 2

```bash
python3 scripts/segment_multicat.py # background + model removal (offline, GPU recommended)
```

Outputs: `data/images_sam2_multicat/{category}/*.jpg`

### Step 3 — Build embeddings & feature files

```bash
python3 scripts/build_embeddings_multicat.py # FashionSigLIP / SigLIP2 embeddings → data/
python3 scripts/build_patch_features_dino.py # DINOv2 texture vectors   → data/
```

### Step 4 — Export ONNX models

```bash
python3 scripts/export_models.py # fp32 + int8 for all 4 models → models/
# If you already have fp32 and only need to re-quantize:
python3 scripts/requantize_int8_for_android.py
```

### Step 5 — Build visual category prototypes (router)

```bash
python3 scripts/build_category_visual_prototypes.py # → android_bundle/category_visual_prototypes.bin
```

### Step 6 — Export Android asset bundle

```bash
python3 scripts/export_catalog_for_android.py # → android_bundle/{embeddings,texture,color_signatures,...}.bin
```

### Step 7 — (Optional) Quick desktop retrieval test

```bash
python3 retrieve_topk_multicat.py --query data/multicat_demo_dress.jpg
python3 retrieve_topk_multicat.py --query data/multicat_demo_jeans.jpg
```

### Step 8 — (Optional) Recall benchmark

```bash
python3 bench_onnx_desktop.py
# target: mean recall@10 ≥ 0.90
```

---

## Android app

1. Copy `android_bundle/` into `ImageSearchApp/app/src/main/assets/bundle/`  
   (or adjust `BundleLoader.kt` to point to the right path).
2. Open `ImageSearchApp` in Android Studio.
3. Build & run on a Pixel 10 (API 36, Android 16).

---

## Project structure

```
ImageSearch/
├── category_router.py          # zero-shot category router
├── color_v10.py                # HSV colour signature (mirrored to Kotlin)
├── retrieve_topk_multicat.py   # CLI retrieval pipeline
├── bench_onnx_desktop.py       # recall@10 benchmark
├── requirements.txt
├── data/                       # catalogs, embeddings, demo images
├── models/                     # ONNX models (fp32 + int8)
├── android_bundle/             # assets exported for the Android app
└── scripts/                    # preprocessing & build scripts (run in order above)
    ├── build_catalog_from_asos.py
    ├── build_multicat_catalog.py
    ├── refetch_multicat_hires.py
    ├── segment_multicat.py
    ├── build_embeddings_multicat.py
    ├── build_patch_features_dino.py
    ├── build_category_visual_prototypes.py
    ├── export_models.py
    ├── requantize_int8_for_android.py
    ├── export_catalog_for_android.py
    └── setup_project.sh
```
