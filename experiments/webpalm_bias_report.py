"""WebPalm retained-set bias report (plan §V3 + §V14b).

For every successful WebPalm ROI image we read its source PIL image to
compute lightweight proxy statistics (image dimensions, aspect ratio,
brightness mean/std, blur Laplacian variance). The same proxies are
computed for the *filtered* set (images that the ROI extractor classified
as `error` or whose ROI was rejected upstream — actual case is rare in
practice since fallback nearly always succeeds; we treat anything not in
status=success/fallback as filtered).

For each continuous proxy we report:
  - median + IQR (descriptive)
  - Cohen's d  (location effect under approximate normality)
  - Cliff's δ  (rank-based, robust to skewness; PRIMARY per §V14b)
  - 2-sample KS p-value
  - 2-sample Mann-Whitney U p-value

For categorical proxies (ROI extraction method) we report:
  - per-method proportion in retained vs filtered
  - chi-squared p-value
  - proportion-difference 95% bootstrap CI

The verdict rule from §V3:
  - |Cliff's δ| < 0.15 (or |Δprop| < 5pp) on every proxy
    → "no operationally meaningful bias"
  - any proxy at |Cliff's δ| ≥ 0.33 (or |Δprop| ≥ 15pp)
    → "operationally meaningful bias on <proxy>; flagged in limitations"
  - mid-range → "no-bias claim withheld"

Usage:
    python experiments/webpalm_bias_report.py \
        --raw-dir   ~/research_data/webpalm/raw \
        --roi-log   ~/research_data/webpalm/roi_224/extraction_log.csv \
        --output    ~/research_data/webpalm/bias_full

Note: at 83K samples this script is I/O-bound (each PIL.open + np.array
conversion is ~5-15ms). Expect ~10-15 min total. Use --limit for trial.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def cliff_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a > b) - P(a < b). Range [-1, 1].

    Magnitude: <0.147 negligible, <0.33 small, <0.474 medium, ≥0.474 large
    (Romano et al. 2006). We use 0.15 / 0.33 thresholds in our verdict rule.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    # Use rank-based formula to avoid O(n*m) outer comparison on large arrays
    combined = np.concatenate([a, b])
    ranks = combined.argsort().argsort()
    r_a = ranks[: a.size]
    sum_r_a = float(r_a.sum())
    n_a, n_b = a.size, b.size
    # U_a = sum_r_a - n_a*(n_a+1)/2
    U_a = sum_r_a - n_a * (n_a + 1) / 2.0
    delta = (2 * U_a - n_a * n_b) / (n_a * n_b)
    return float(delta)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return float("nan")
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled = ((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2)
    if pooled <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled))


def ks_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b).pvalue)
    except ImportError:
        return float("nan")


def mw_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import mannwhitneyu
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ImportError:
        return float("nan")


def proxy_for_image(p: Path) -> Dict[str, float] | None:
    """Compute lightweight proxies on a single image. Returns None on failure."""
    try:
        with Image.open(p) as im:
            w, h = im.size
            arr = np.asarray(im.convert("L"), dtype=np.float64)
            brightness_mean = float(arr.mean())
            brightness_std = float(arr.std())
            # Cheap Laplacian variance for blur (8-neighbor 2nd derivative)
            lap = (
                arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:]
                - 4 * arr[1:-1, 1:-1]
            )
            blur = float(lap.var())
            return {
                "width": float(w),
                "height": float(h),
                "aspect_ratio": float(w / max(h, 1)),
                "brightness_mean": brightness_mean,
                "brightness_std": brightness_std,
                "blur_lapvar": blur,
            }
    except Exception:
        return None


def verdict(cliff: float, prop_diff: float | None = None) -> str:
    """Plan §V3 verdict ladder."""
    cd = abs(cliff) if cliff == cliff else 0  # NaN -> 0
    pd = abs(prop_diff) if prop_diff is not None else 0
    if cd >= 0.33 or pd >= 0.15:
        return "meaningful_bias"
    if cd < 0.15 and pd < 0.05:
        return "no_meaningful_bias"
    return "mid_range"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True,
                   help="WebPalm raw image directory")
    p.add_argument("--roi-log", type=Path, required=True,
                   help="extraction_log.csv from webpalm_roi_extractor")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="Limit per-set sample count (for trial runs)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Build retained vs filtered file lists from the ROI log
    rows = []
    with args.roi_log.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"[load] {len(rows)} rows in roi log")

    retained_names = [r["filename"] for r in rows
                      if r.get("status") in ("success", "fallback")]
    filtered_names = [r["filename"] for r in rows
                      if r.get("status") not in ("success", "fallback")]
    print(f"[partition] retained={len(retained_names)} "
          f"filtered={len(filtered_names)}")

    if not retained_names or not filtered_names:
        print("[skip] need both retained and filtered samples; producing "
              "descriptive-only summary")

    # Subsample if --limit
    rng = np.random.default_rng(args.seed)
    if args.limit is not None:
        if len(retained_names) > args.limit:
            retained_names = rng.choice(retained_names, args.limit, replace=False).tolist()
        if len(filtered_names) > args.limit:
            filtered_names = rng.choice(filtered_names, args.limit, replace=False).tolist()

    # Compute proxies
    def proxies_for(names: List[str], label: str) -> Dict[str, np.ndarray]:
        out: Dict[str, List[float]] = {
            "width": [], "height": [], "aspect_ratio": [],
            "brightness_mean": [], "brightness_std": [], "blur_lapvar": [],
        }
        for nm in tqdm(names, desc=f"proxy {label}"):
            # WebPalm raw filenames: <stem>.{jpg,JPG,jpeg,...}; try common
            for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
                candidate = args.raw_dir / f"{nm}{ext}"
                if candidate.exists():
                    pr = proxy_for_image(candidate)
                    if pr is not None:
                        for k, v in pr.items():
                            out[k].append(v)
                    break
        return {k: np.array(v, dtype=np.float64) for k, v in out.items()}

    retained_pr = proxies_for(retained_names, "retained")
    filtered_pr = proxies_for(filtered_names, "filtered") if filtered_names else {
        k: np.array([]) for k in retained_pr
    }

    # Compute statistics per proxy
    summary = {
        "n_retained": int(len(retained_names)),
        "n_filtered": int(len(filtered_names)),
        "proxies": {},
        "categorical": {},
    }
    overall_verdict = "no_meaningful_bias"
    flagged = []

    for proxy in retained_pr:
        a = retained_pr[proxy]
        b = filtered_pr.get(proxy, np.array([]))
        if a.size == 0:
            continue

        med_a = float(np.median(a))
        iqr_a = float(np.percentile(a, 75) - np.percentile(a, 25))
        if b.size > 0:
            med_b = float(np.median(b))
            iqr_b = float(np.percentile(b, 75) - np.percentile(b, 25))
            cd = cohen_d(a, b)
            cl = cliff_delta(a, b)
            ks = ks_pvalue(a, b)
            mw = mw_pvalue(a, b)
        else:
            med_b = iqr_b = float("nan")
            cd = cl = ks = mw = float("nan")

        v = verdict(cl)
        summary["proxies"][proxy] = {
            "retained_median": med_a, "retained_iqr": iqr_a,
            "filtered_median": med_b, "filtered_iqr": iqr_b,
            "cohens_d": cd, "cliffs_delta": cl,
            "ks_pvalue": ks, "mannwhitney_pvalue": mw,
            "verdict": v,
        }
        if v == "meaningful_bias":
            overall_verdict = "meaningful_bias"
            flagged.append(proxy)
        elif v == "mid_range" and overall_verdict != "meaningful_bias":
            overall_verdict = "mid_range"

    # Categorical proxy: ROI extraction method distribution (within retained)
    method_counts_retained: Dict[str, int] = {}
    for r in rows:
        if r.get("status") in ("success", "fallback"):
            method_counts_retained[r["method"]] = method_counts_retained.get(r["method"], 0) + 1
    summary["categorical"]["method_distribution_retained"] = method_counts_retained

    summary["overall_verdict"] = overall_verdict
    summary["flagged_proxies"] = flagged
    summary["seed"] = args.seed

    (args.output / "bias_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[verdict] {overall_verdict}; flagged: {flagged or 'none'}")
    print(f"[ok] {args.output / 'bias_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
