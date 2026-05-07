"""WebPalm ROI Extractor (specialized for web-collected palmprint images).

WebPalm characteristics (from RegPalm 2025, TIFS):
  - 83,145 palm images collected from public web sources
  - Diverse devices, lighting, poses, skin tones
  - Single image per identity
  - Variable resolution; palms typically occupy 20-40% of image short side
    (vs BMPD's 50%+ for smartphone close-ups)
  - Some images have EXIF orientation tags (iPhone, etc.)

Why a separate extractor:
  The default `roi_extractor.py` is tuned for BMPD smartphone close-ups
  (SIZE_SCALE=1.4, lower=0.27, upper=0.42 of short_side). On WebPalm trial
  runs, this caps 34% of ROIs at the lower bound (~100-150px), resulting in
  too-small palm crops. WebPalm needs:

  - Larger SIZE_SCALE so V1V2 distance maps to a larger ROI
  - Higher LOWER bound to ensure minimum useful ROI even for small palms
  - Higher UPPER bound to allow large palms when V1V2 is large
  - Slightly less inward shift (INWARD_DIST_RATIO 0.65 → 0.55) to keep
    more of the palm-line region in frame
  - EXIF auto-rotation via PIL before passing to MediaPipe / cropping
  - Output 224×224 by default (RegPalm-compatible; matches their backbone
    input size)
  - skin_ratio kept as diagnostic only (YCrCb thresholds calibrated for
    BMPD give false negatives on diverse skin tones)

Reuses helpers from `roi_extractor.py`:
  - `detect_landmarks` / `_detect_landmarks_with_handedness`
  - `compute_landmark_diagnostics`
  - `compute_roi_skin_ratio`
  - `crop_roi`
  - `save_roi`
  - `save_debug_vis`

Overrides:
  - `compute_roi_from_landmarks` → `_compute_roi_from_landmarks_webpalm`
  - `compute_roi_fallback` → `_compute_roi_fallback_webpalm`
  - Filename parsing → trivial: user_id = stem

Usage:
  python ROI/webpalm_roi_extractor.py \\
      --input  ~/palm_images \\
      --output ~/webpalm_roi \\
      --sizes  224 \\
      --vis    # optional, expensive but useful for first runs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# Reuse the helpers from the BMPD extractor
sys.path.insert(0, str(Path(__file__).resolve().parent))
import roi_extractor as base  # noqa: E402


# ---------------------------------------------------------------------------
# WebPalm-specific tuning constants (re-derived from trial-100 analysis)
# ---------------------------------------------------------------------------

# V15 (post-trial-100 finger-contamination fix): inward is now SIZE-relative,
# not V1V2-relative. This guarantees the ROI top edge sits below the V1V2
# line by 10% of the ROI side, regardless of whether `size` came from
# `SIZE_SCALE × V1V2` or from a clamp at the lower/upper bound. The previous
# v1 tuning (SIZE_SCALE=1.9, INWARD_DIST_RATIO=0.55) put the top edge at
# `M − 0.4 × V1V2`, which let finger pixels into the crop on small/medium
# palms. See plan §V15 for the geometric derivation.
WP_SIZE_SCALE = 1.7             # was 1.9; modest reduction
WP_SIZE_LOWER_BOUND_RATIO = 0.35  # unchanged
WP_SIZE_UPPER_BOUND_RATIO = 0.55  # unchanged
WP_INWARD_RATIO_OF_SIZE = 0.60    # NEW: inward_dist = 0.60 × size
                                   # (replaces V1V2-relative INWARD_DIST_RATIO)
# Top edge of ROI relative to V1V2 line:
#   (inward - size/2) = (0.60 - 0.50) × size = +0.10 × size below V1V2
# i.e., the ROI is fully below V1V2 with a 10% margin proportional to ROI side.

WP_FALLBACK_DT_SIZE_SCALE = 3.0   # ↑ from 2.6; match the larger landmark scale
WP_FALLBACK_LOWER_BOUND_RATIO = 0.35
WP_FALLBACK_UPPER_BOUND_RATIO = 0.55


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

def _compute_roi_from_landmarks_webpalm(lm, img_h: int, img_w: int):
    """WebPalm-tuned variant of compute_roi_from_landmarks.

    Identical structure to base.compute_roi_from_landmarks but uses the
    WebPalm constants for SIZE_SCALE / size bounds / INWARD_DIST_RATIO.
    """
    pt = lambda i: np.array([lm[i].x * img_w, lm[i].y * img_h], dtype=np.float64)
    p0 = pt(0)
    V1 = (pt(5) + pt(9)) / 2.0
    V2 = (pt(13) + pt(17)) / 2.0
    M = (V1 + V2) / 2.0

    palm_vec = V2 - V1
    v1v2_dist = float(np.linalg.norm(palm_vec))
    angle_deg = float(np.degrees(np.arctan2(palm_vec[1], palm_vec[0])))

    # Fingers-up canonicalization (verbatim from base)
    angle_rad = np.radians(angle_deg)
    s_a = float(np.sin(angle_rad))
    c_a = float(np.cos(angle_rad))
    to_wrist = p0 - M
    to_wrist_rot_y = to_wrist[0] * s_a + to_wrist[1] * c_a
    if to_wrist_rot_y < 0.0:
        angle_deg += 180.0
        if angle_deg > 180.0:
            angle_deg -= 360.0

    short_side = float(min(img_h, img_w))
    size = v1v2_dist * WP_SIZE_SCALE
    size = max(size, short_side * WP_SIZE_LOWER_BOUND_RATIO)
    size = min(size, short_side * WP_SIZE_UPPER_BOUND_RATIO)
    size = max(size, 100.0)

    perp = np.array([-palm_vec[1], palm_vec[0]], dtype=np.float64)
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm < 1e-6:
        inward_unit = np.array([0.0, 1.0])
    else:
        inward_unit = perp / perp_norm
        if float(np.dot(inward_unit, p0 - M)) < 0.0:
            inward_unit = -inward_unit

    # V15 fix: size-relative inward (was v1v2_dist * WP_INWARD_DIST_RATIO).
    # Guarantees ROI top edge sits below V1V2 line by 10% of `size`.
    inward_dist = size * WP_INWARD_RATIO_OF_SIZE
    center = M + inward_unit * inward_dist
    return float(center[0]), float(center[1]), size, angle_deg


def _compute_roi_fallback_webpalm(image_bgr: np.ndarray):
    """WebPalm-tuned skin_fallback (larger DT_SIZE_SCALE, looser bounds)."""
    h, w = image_bgr.shape[:2]
    short_side = float(min(h, w))

    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cx, cy = w / 2.0, h / 2.0
        size = short_side * WP_SIZE_LOWER_BOUND_RATIO
        return cx, cy, size, 0.0, "center_crop"

    largest = max(contours, key=cv2.contourArea)
    hand_mask = np.zeros_like(mask)
    cv2.drawContours(hand_mask, [largest], -1, 255, thickness=cv2.FILLED)

    dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
    _, max_dist_val, _, max_loc = cv2.minMaxLoc(dist)
    cx, cy = float(max_loc[0]), float(max_loc[1])

    rect = cv2.minAreaRect(largest)
    (_, _), (rw, rh), rect_angle = rect
    if rw < rh:
        angle_deg = rect_angle
    else:
        angle_deg = rect_angle + 90.0
    while angle_deg > 90.0:
        angle_deg -= 180.0
    while angle_deg <= -90.0:
        angle_deg += 180.0

    moments = cv2.moments(largest)
    if moments['m00'] > 0:
        cent_x = moments['m10'] / moments['m00']
        cent_y = moments['m01'] / moments['m00']
        to_finger_x = cent_x - cx
        to_finger_y = cent_y - cy
        ar = np.radians(angle_deg)
        s_f, c_f = float(np.sin(ar)), float(np.cos(ar))
        to_finger_rot_y = to_finger_x * s_f + to_finger_y * c_f
        if to_finger_rot_y > 0.0:
            angle_deg += 180.0
            while angle_deg > 180.0:
                angle_deg -= 360.0

    size = float(max_dist_val) * WP_FALLBACK_DT_SIZE_SCALE
    size = max(size, short_side * WP_FALLBACK_LOWER_BOUND_RATIO)
    size = min(size, short_side * WP_FALLBACK_UPPER_BOUND_RATIO)
    size = max(size, 100.0)
    return cx, cy, size, float(angle_deg), "skin_fallback"


# ---------------------------------------------------------------------------
# EXIF-aware loader
# ---------------------------------------------------------------------------

def load_image_with_exif(path: Path) -> np.ndarray | None:
    """Load image as BGR after applying EXIF orientation if any.

    cv2.imread ignores EXIF tags, which leaves iPhone (and many web JPEG)
    images rotated incorrectly. We use PIL.ImageOps.exif_transpose first.
    """
    try:
        with Image.open(str(path)) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                im = im.convert("RGB")
            arr = np.asarray(im)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CSV schema (extends base with a `quality_flag` summary column)
# ---------------------------------------------------------------------------

WP_CSV_FIELDS = list(base._CSV_FIELDS) + ["quality_flag"]


def _quality_flag(row: dict) -> str:
    """Compose a coarse quality label from diagnostics. Diagnostic only."""
    flags = []
    try:
        if int(row.get("roi_size_px", 0)) < 130:
            flags.append("small_roi")
    except (TypeError, ValueError):
        pass
    method = row.get("method", "")
    # finger_extension only meaningful when landmarks exist; skip for fallback
    if method.startswith("landmark"):
        try:
            ext = row.get("finger_extension")
            if ext not in (None, ""):
                ext_v = float(ext)
                if ext_v < 1.0 or ext_v > 3.0:
                    flags.append("weird_finger")
        except (TypeError, ValueError):
            pass
        if str(row.get("landmarks_in_bounds")).lower() == "false":
            flags.append("out_of_bounds")
    if method == "skin_fallback":
        flags.append("fallback")
    if method == "center_crop":
        flags.append("center_crop")
    return "|".join(flags) if flags else "clean"


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="WebPalm ROI Extractor")
    p.add_argument("--input", type=str, required=True, help="WebPalm 이미지 폴더")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--sizes", type=int, nargs="+", default=[224],
                   help="출력 해상도; default 224 (RegPalm-compatible)")
    p.add_argument("--vis", action="store_true",
                   help="debug 시각화 저장 (느림; 검수용)")
    p.add_argument("--limit", type=int, default=None,
                   help="처리할 이미지 수 제한")
    args = p.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    patterns = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG", "*.bmp", "*.BMP")
    images = set()
    for pat in patterns:
        images.update(in_root.rglob(pat))
    images = sorted(images)
    if args.limit:
        images = images[: int(args.limit)]
    print(f"WebPalm: 총 {len(images)} 이미지")

    log_rows: list[dict] = []
    for img_path in tqdm(images, desc="WebPalm ROI"):
        meta = {"user_id": img_path.stem, "type": "G", "hand": "U",
                "index": "1", "stem": img_path.stem}
        try:
            image = load_image_with_exif(img_path)
            if image is None:
                log_rows.append(base._err_row(img_path, "load_failed", meta))
                continue
            img_h, img_w = image.shape[:2]
            note = ""

            lm, mp_handed = base._detect_landmarks_with_handedness(image, "high")
            if lm is None:
                lm, mp_handed = base._detect_landmarks_with_handedness(image, "low")
                if lm is not None:
                    note = "low_confidence"
            if mp_handed:
                meta["hand"] = mp_handed

            diag = {}
            if lm is not None:
                diag = base.compute_landmark_diagnostics(lm, img_h, img_w)
                cx, cy, size, angle = _compute_roi_from_landmarks_webpalm(
                    lm, img_h, img_w
                )
                method = "landmark_low_conf" if note == "low_confidence" else "landmark"
                status = "success"
            else:
                cx, cy, size, angle, method = _compute_roi_fallback_webpalm(image)
                status = "fallback"

            roi = base.crop_roi(image, cx, cy, size, angle)
            if roi is None or roi.size == 0:
                log_rows.append(base._err_row(img_path, "crop_failed", meta))
                continue

            skin_ratio = base.compute_roi_skin_ratio(roi)
            base.save_roi(roi, out_root, meta["user_id"], meta["stem"], args.sizes)

            if args.vis:
                debug_path = out_root / "debug" / meta["user_id"] / f"{meta['stem']}_debug.jpg"
                base.save_debug_vis(image, lm, cx, cy, size, angle, method,
                                     debug_path, diag, skin_ratio)

            row = {
                "filename": meta["stem"],
                "user_id": meta["user_id"],
                "type": meta["type"],
                "hand": meta["hand"],
                "index": meta["index"],
                "method": method,
                "roi_size_px": int(size),
                "rotation_deg": round(float(angle), 2),
                "status": status,
                "note": note,
                "mp_handedness": mp_handed if mp_handed else "",
                "roi_skin_ratio": skin_ratio,
                "img_w": img_w,
                "img_h": img_h,
            }
            row.update(diag)
            row["quality_flag"] = _quality_flag(row)
            log_rows.append(row)
        except Exception as e:
            log_rows.append(base._err_row(
                img_path, f"exception:{type(e).__name__}:{e}", meta
            ))

    # Write CSV with extended schema
    csv_path = out_root / "extraction_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=WP_CSV_FIELDS)
        writer.writeheader()
        for r in log_rows:
            # ensure quality_flag exists for error rows
            r.setdefault("quality_flag", "error")
            writer.writerow({k: r.get(k, "") for k in WP_CSV_FIELDS})

    # Summary
    from collections import Counter
    statuses = Counter(r["status"] for r in log_rows)
    methods = Counter(r["method"] for r in log_rows)
    flags = Counter(r.get("quality_flag", "error") for r in log_rows)
    print(f"\nWebPalm 완료: 총 {len(log_rows)}")
    print(f"  status: {dict(statuses)}")
    print(f"  method: {dict(methods)}")
    print(f"  quality_flag: {dict(flags)}")
    print(f"로그: {csv_path}")


if __name__ == "__main__":
    main()
