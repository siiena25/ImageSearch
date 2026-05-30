#!/usr/bin/env bash
# =============================================================================
#  setup_project.sh — Full end-to-end build for Visual Product Search
# =============================================================================
#
#  What this script does (step by step):
#  ─────────────────────────────────────────────────────────────────────────────
#  0.  Check Python / pip
#  1.  Create virtual environment and install dependencies
#  2.  Download products_asos.csv (if missing)
#  3.  Build dresses catalog from ASOS CSV → data/catalog_sam2.jsonl
#  4.  Build multi-category catalog (dresses + tshirts + jeans + watches)
#      → data/catalog_multicat.jsonl + download raw images
#  5.  Re-fetch hi-res images for tshirts/jeans/watches
#  6.  Grounded-SAM2 segmentation → data/images_sam2_multicat/
#  7.  Build embeddings with category routing
#      → data/embeddings_multicat.npy + data/metadata_multicat.json
#  8.  DINOv2 patch features for texture/pattern re-ranking
#      → data/patch_features_dino.npy
#  9.  Category visual prototypes for the router
#      → android_bundle/category_visual_prototypes.bin
#  10. Export ONNX models: FashionSigLIP + SigLIP2 + DINOv2 + U2Netp
#      → models/*.onnx (FP32 + INT8)
#  11. Build Android bundle → android_bundle/
#  12. Quick benchmark (recall@10 ≥ 0.9)
#
#  Requirements:
#  ─────────────────────────────────────────────────────────────────────────────
#  • Python 3.10+ (3.12 recommended)
#  • ~10 GB free disk space
#  • Internet access (for downloading HuggingFace models and datasets)
#  • GPU is optional - everything runs on CPU too, just slower
#
#  Usage:
#      chmod +x setup_project.sh
#      ./setup_project.sh
#
#  To skip already-completed steps run individual scripts directly,
#  or use flags:
#      ./setup_project.sh --skip-segmentation   # skip step 6 (slow!)
#      ./setup_project.sh --skip-export         # skip ONNX export
#      ./setup_project.sh --only-android        # run steps 9-12 only
# =============================================================================

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

step() {
    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  STEP $1: $2${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════${NC}"
}

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_SEGMENTATION=false
SKIP_EXPORT=false
ONLY_ANDROID=false

for arg in "$@"; do
    case $arg in
        --skip-segmentation) SKIP_SEGMENTATION=true ;;
        --skip-export)       SKIP_EXPORT=true ;;
        --only-android)      ONLY_ANDROID=true ;;
        --help|-h)
            sed -n '2,50p' "$0" | grep -E "^\s*#" | sed 's/^.*#//'
            exit 0
            ;;
    esac
done

# ── Locate project root (script's own directory) ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "Project root: $SCRIPT_DIR"

# =============================================================================
# STEP 0 — Python check
# =============================================================================
step 0 "Checking Python"

if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install Python 3.10+ and try again."
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Found Python $PY_VER"

PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
if [[ $PY_MAJOR -lt 3 || ($PY_MAJOR -eq 3 && $PY_MINOR -lt 10) ]]; then
    error "Python 3.10+ required (got $PY_VER)"
fi

if $ONLY_ANDROID; then
    info "--only-android: jumping to step 9"
fi

# =============================================================================
# STEP 1 — Virtual environment + dependencies
# =============================================================================
if ! $ONLY_ANDROID; then

step 1 "Setting up virtual environment"

if [[ ! -d ".venv" ]]; then
    info "Creating .venv ..."
    python3 -m venv .venv
else
    info ".venv already exists, skipping creation"
fi

# Activate
# shellcheck source=/dev/null
source .venv/bin/activate
info "Activated .venv"

info "Upgrading pip ..."
pip install --quiet --upgrade pip

info "Installing requirements.txt ..."
pip install --quiet -r requirements.txt

# SAM2 sometimes needs a manual install from GitHub
if ! python3 -c "import sam2" 2>/dev/null; then
    warn "sam2 not importable via pip — trying GitHub install ..."
    pip install --quiet "git+https://github.com/facebookresearch/sam2.git" || \
        warn "SAM2 install failed. Segmentation step will be skipped."
fi

fi  # !ONLY_ANDROID

# Ensure venv is active for remaining steps
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

# =============================================================================
# STEP 2 — Download products_asos.csv (if missing)
# =============================================================================
if ! $ONLY_ANDROID; then

step 2 "Checking products_asos.csv"

if [[ ! -f "products_asos.csv" ]]; then
    warn "products_asos.csv not found."
    info "Attempting to download from HuggingFace datasets ..."
    python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
import shutil, pathlib
path = hf_hub_download(
    repo_id="ashraq/fashion-product-images-small",
    filename="products.csv",
    repo_type="dataset",
    local_dir=".",
)
shutil.copy(path, "products_asos.csv")
print(f"  Saved to products_asos.csv")
PYEOF
else
    info "products_asos.csv already present ($(du -sh products_asos.csv | cut -f1))"
fi

fi  # !ONLY_ANDROID

# =============================================================================
# STEP 3 — Build dresses catalog
# =============================================================================
if ! $ONLY_ANDROID; then

step 3 "Building dresses catalog → data/catalog_sam2.jsonl"

if [[ -f "data/catalog_sam2.jsonl" ]]; then
    COUNT=$(wc -l < data/catalog_sam2.jsonl | tr -d ' ')
    info "data/catalog_sam2.jsonl already exists ($COUNT items) — skipping"
else
    mkdir -p data/images
    python3 build_catalog_from_asos.py \
        --input products_asos.csv \
        --output data/catalog_sam2.jsonl \
        --limit 400
    info "Done: $(wc -l < data/catalog_sam2.jsonl | tr -d ' ') dresses"
fi

fi  # !ONLY_ANDROID

# =============================================================================
# STEP 4 — Build multicat catalog + download raw images
# =============================================================================
if ! $ONLY_ANDROID; then

step 4 "Building multicat catalog (dresses+tshirts+jeans+watches)"

if [[ -f "data/catalog_multicat.jsonl" ]]; then
    COUNT=$(wc -l < data/catalog_multicat.jsonl | tr -d ' ')
    info "data/catalog_multicat.jsonl already exists ($COUNT items) — skipping"
else
    python3 build_multicat_catalog.py
    info "Done: $(wc -l < data/catalog_multicat.jsonl | tr -d ' ') items"
fi

fi  # !ONLY_ANDROID

# =============================================================================
# STEP 5 — Refetch hi-res images for tshirts/jeans/watches
# =============================================================================
if ! $ONLY_ANDROID; then

step 5 "Refetching hi-res images for tshirts/jeans/watches"

HIRES_COUNT=$(find data/images_multicat_hires -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
if [[ $HIRES_COUNT -gt 200 ]]; then
    info "Hi-res images already present ($HIRES_COUNT files) — skipping"
else
    python3 refetch_multicat_hires.py
    info "Done: $(find data/images_multicat_hires -name '*.jpg' | wc -l | tr -d ' ') hi-res images"
fi

fi  # !ONLY_ANDROID

# =============================================================================
# STEP 6 — Grounded-SAM2 segmentation
# =============================================================================
if ! $ONLY_ANDROID && ! $SKIP_SEGMENTATION; then

step 6 "Grounded-SAM2 segmentation → data/images_sam2_multicat/"

SAM_COUNT=$(find data/images_sam2_multicat -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
CATALOG_COUNT=$(wc -l < data/catalog_multicat.jsonl | tr -d ' ')

if [[ $SAM_COUNT -ge $((CATALOG_COUNT - 10)) ]]; then
    info "Segmentation already complete ($SAM_COUNT/$CATALOG_COUNT) — skipping"
else
    info "Segmenting $CATALOG_COUNT items (may take 30-60 min on CPU) ..."
    python3 segment_multicat.py
    info "Done: $(find data/images_sam2_multicat -name '*.jpg' | wc -l | tr -d ' ') segmented images"
fi

elif $SKIP_SEGMENTATION; then
    warn "STEP 6 skipped (--skip-segmentation). Using existing images."
fi

# =============================================================================
# STEP 7 — Build embeddings with category routing
# =============================================================================
if ! $ONLY_ANDROID; then

step 7 "Building multicat embeddings → data/embeddings_multicat.npy"

if [[ -f "data/embeddings_multicat.npy" && -f "data/metadata_multicat.json" ]]; then
    info "Embeddings already exist — skipping"
else
    python3 build_embeddings_multicat.py
    info "Done"
fi

fi  # !ONLY_ANDROID

# =============================================================================
# STEP 8 — DINOv2 patch features for texture/pattern re-ranking
# =============================================================================
if ! $ONLY_ANDROID; then

step 8 "Building DINOv2 patch features → data/patch_features_dino.npy"

if [[ -f "data/patch_features_dino.npy" && -f "data/product_ids_dino.json" ]]; then
    info "Patch features already exist — skipping"
else
    python3 build_patch_features_dino.py
    info "Done"
fi

fi  # !ONLY_ANDROID

# =============================================================================
# STEP 9 — Category visual prototypes (router)
# =============================================================================
step 9 "Building category visual prototypes → android_bundle/"

if [[ -f "android_bundle/category_visual_prototypes.bin" ]]; then
    info "Prototypes already exist — skipping"
else
    mkdir -p android_bundle
    python3 build_category_visual_prototypes.py
    info "Done"
fi

# =============================================================================
# STEP 10 — Export ONNX models
# =============================================================================
if ! $SKIP_EXPORT; then

step 10 "Exporting ONNX models → models/"

mkdir -p models

ONNX_COUNT=$(find models -name "*.onnx" 2>/dev/null | wc -l | tr -d ' ')
if [[ $ONNX_COUNT -ge 8 ]]; then
    info "ONNX models already exported ($ONNX_COUNT files) — skipping"
    info "  To re-export: python3 export_models.py"
else
    info "Exporting (takes ~5-10 min) ..."
    python3 export_models.py
    info "Done: $(find models -name '*.onnx' | wc -l | tr -d ' ') ONNX files"
fi

else
    warn "STEP 10 skipped (--skip-export)"
fi

# =============================================================================
# STEP 11 — Build Android bundle
# =============================================================================
step 11 "Building Android asset bundle → android_bundle/"

python3 export_catalog_for_android.py
info "Bundle ready:"
du -sh android_bundle/embeddings.bin android_bundle/catalog.json android_bundle/thumbnails/ 2>/dev/null | \
    awk '{printf "    %-45s %s\n", $2, $1}'

# =============================================================================
# STEP 12 — Quick benchmark
# =============================================================================
step 12 "Running recall@10 benchmark"

python3 bench_onnx_desktop.py
BENCH_EXIT=$?
if [[ $BENCH_EXIT -ne 0 ]]; then
    warn "Benchmark returned non-zero exit code ($BENCH_EXIT). Check output above."
fi

# =============================================================================
# Done
# =============================================================================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   ✅  BUILD COMPLETE                                 ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Next steps:"
echo "  • Copy android_bundle/ to your Android project's assets/"
echo "  • Copy models/*.onnx to Android project's assets/models/"
echo "  • Run bench:       python3 bench_onnx_desktop.py"
echo "  • Run retrieval:   python3 retrieve_topk_multicat.py --query data/multicat_demo_dress.jpg"
echo ""
