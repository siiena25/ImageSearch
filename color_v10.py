#!/usr/bin/env python3

# HSV color signature v10 — shared module for retrieval and catalog export.

# API (mirrored 1:1 in Kotlin ColorSignatureV10.kt):
#   extract_color_signature(pil_image) -> dict (hue_hist[24], mono_ratio, is_monochrome)
#   color_similarity(sig_a, sig_b) -> float in [0, 1]
#   signature_to_vector(sig) -> np.ndarray[26]
#   signature_from_vector(vec) -> dict

from __future__ import annotations

import numpy as np
from PIL import Image

# Constants (bit-for-bit identical to retrieve_topk_v2.py v10)
HUE_BINS = 24
PRINT_SAT_THRESHOLD = 50
PRINT_V_MIN = 50
PRINT_V_MAX = 240
WHITE_THRESHOLD = 245
MIN_PRINT_PIXELS = 200

# Hard kill / jaccard gate
PEAK_DIST_KILL = 2
JACCARD_KILL_THRESHOLD = 0.30
JACCARD_FULL_THRESHOLD = 0.40
PEAK_SIM_DIVISOR = 2.5  # exp(-d²/2.5)

# Color gate (applied after triple rerank)
COLOR_GATE_FLOOR = 0.05
COLOR_GATE_FULL = 0.30


def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def extract_color_signature(image: Image.Image) -> dict:
    # Compute HSV colour signature for the given image
    image = _ensure_rgb(image)
    rgb = np.asarray(image).astype(np.uint8)
    hsv = np.asarray(image.convert("HSV")).astype(np.uint8)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    not_white = ~(
        (rgb[:, :, 0] >= WHITE_THRESHOLD)
        & (rgb[:, :, 1] >= WHITE_THRESHOLD)
        & (rgb[:, :, 2] >= WHITE_THRESHOLD)
    )
    print_mask = (
        not_white
        & (s >= PRINT_SAT_THRESHOLD)
        & (v >= PRINT_V_MIN)
        & (v <= PRINT_V_MAX)
    )

    n_fg = int(not_white.sum())
    n_print = int(print_mask.sum())
    mono_ratio = 1.0 - (n_print / n_fg) if n_fg > 0 else 1.0
    is_monochrome = n_print < MIN_PRINT_PIXELS

    if is_monochrome:
        hue_hist = np.zeros(HUE_BINS, dtype=np.float32)
    else:
        hist, _ = np.histogram(h[print_mask], bins=HUE_BINS, range=(0, 256))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        hue_hist = hist

    return {
        "hue_hist": hue_hist,
        "mono_ratio": float(mono_ratio),
        "is_monochrome": bool(is_monochrome),
    }


def _circular_hue_distance(a: int, b: int, n: int = HUE_BINS) -> int:
    d = abs(a - b)
    return min(d, n - d)


def _weighted_peak_hue(hue_hist: np.ndarray) -> int:
    smoothed = (
        0.25 * np.roll(hue_hist, 1)
        + 0.5 * hue_hist
        + 0.25 * np.roll(hue_hist, -1)
    )
    return int(np.argmax(smoothed))


def _dominant_hue_bins(hue_hist: np.ndarray, coverage: float = 0.70,
                       max_bins: int = 6) -> set[int]:
    if hue_hist.sum() <= 0:
        return set()
    order = np.argsort(-hue_hist)
    cumulative = np.cumsum(hue_hist[order])
    n = int(np.searchsorted(cumulative, coverage)) + 1
    n = min(max(n, 1), max_bins)
    out: set[int] = set()
    total = len(hue_hist)
    for b in order[:n]:
        b = int(b)
        out.add(b)
        out.add((b - 1) % total)
        out.add((b + 1) % total)
    return out


def color_similarity(sig_a: dict, sig_b: dict) -> float:
    both_mono = sig_a["is_monochrome"] and sig_b["is_monochrome"]
    one_mono = sig_a["is_monochrome"] != sig_b["is_monochrome"]

    if both_mono:
        return 0.7
    if one_mono:
        return 0.0

    hue_inter = float(np.minimum(sig_a["hue_hist"], sig_b["hue_hist"]).sum())
    peak_a = _weighted_peak_hue(sig_a["hue_hist"])
    peak_b = _weighted_peak_hue(sig_b["hue_hist"])
    peak_dist = _circular_hue_distance(peak_a, peak_b)
    peak_sim = float(np.exp(-(peak_dist ** 2) / PEAK_SIM_DIVISOR))

    dom_a = _dominant_hue_bins(sig_a["hue_hist"])
    dom_b = _dominant_hue_bins(sig_b["hue_hist"])
    if dom_a and dom_b:
        union_size = len(dom_a | dom_b)
        jaccard = len(dom_a & dom_b) / union_size if union_size > 0 else 0.0
    else:
        jaccard = 0.0

    if peak_dist > PEAK_DIST_KILL and jaccard < JACCARD_KILL_THRESHOLD:
        return 0.0

    hue_match = 0.2 * hue_inter + 0.3 * peak_sim + 0.5 * jaccard
    mono_diff = abs(sig_a["mono_ratio"] - sig_b["mono_ratio"])
    mono_multiplier = max(float(np.exp(-(mono_diff * 2.0) ** 2)), 0.1)
    jaccard_gate = min(jaccard / JACCARD_FULL_THRESHOLD, 1.0)

    score = hue_match * mono_multiplier * jaccard_gate
    return float(np.clip(score, 0.0, 1.0))


def color_gate(col_score: float) -> float:
    # Final multiplicative gate applied to the base score after triple rerank.
    return COLOR_GATE_FLOOR + (1.0 - COLOR_GATE_FLOOR) * min(
        col_score, COLOR_GATE_FULL
    ) / COLOR_GATE_FULL


def signature_to_vector(sig: dict) -> np.ndarray:
    """
    Serialize a signature into a float32 vector of length 26:
      [0..23]  hue_hist
      [24] mono_ratio
      [25] is_monochrome (0.0 / 1.0)
    Used for the color_signatures.bin bundle file on Android
    """
    out = np.zeros(26, dtype=np.float32)
    out[:HUE_BINS] = sig["hue_hist"]
    out[HUE_BINS] = sig["mono_ratio"]
    out[HUE_BINS + 1] = 1.0 if sig["is_monochrome"] else 0.0
    return out


def signature_from_vector(vec: np.ndarray) -> dict:
    return {
        "hue_hist": vec[:HUE_BINS].astype(np.float32),
        "mono_ratio": float(vec[HUE_BINS]),
        "is_monochrome": bool(vec[HUE_BINS + 1] > 0.5),
    }


SIGNATURE_VECTOR_LEN = 26
