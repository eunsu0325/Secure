"""BMPD Palmprint ROI Extractor.

V1V2 골짜기 기반 Zhang 표준 방식으로 BMPD 손바닥 이미지에서 정사각형 ROI를 추출한다.
F타입(전체손)은 MediaPipe Hands로 랜드마크 검출 → V1V2 ROI.
S타입(클로즈업)은 MediaPipe 우회하고 중앙 정사각형 크롭.

사용 예시 (Colab):
    !unzip -q /content/drive/MyDrive/데이터셋/archive.zip -d /content
    !pip install -q mediapipe tqdm
    !python ROI/roi_extractor.py \\
        --input  /content/BMPD \\
        --output /content/output \\
        --vis
"""

from __future__ import annotations

import argparse
import csv
import re
import urllib.request
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision


_FILENAME_RE = re.compile(r"^(\d{3})_([FS])_([LR])_(\d+)$")

SIZE_SCALE = 1.4
FALLBACK_SIZE_SCALE = 0.6  # legacy (no longer used after Phase 4)

# Phase 3: landmark ROI tuning (data-driven from extraction_log analysis)
SIZE_LOWER_BOUND_RATIO = 0.27   # min ROI size = short_side × 0.27 (살짝 키움)
SIZE_UPPER_BOUND_RATIO = 0.42   # max ROI size = short_side × 0.42 (살짝 키움)
INWARD_DIST_RATIO = 0.65        # inward offset = V1V2_dist × 0.65 (size/2 수준으로 환원)

# Phase 4: skin_fallback redesign
FALLBACK_DT_SIZE_SCALE = 2.6    # ROI size = palm_inscribed_radius × 2.6 (≈ 손바닥 너비)
FALLBACK_LOWER_BOUND_RATIO = 0.27  # landmark과 동일한 lower clamp
FALLBACK_UPPER_BOUND_RATIO = 0.42  # landmark과 동일한 upper clamp (살짝 키움)

_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_TASK_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"


def _ensure_task_model() -> Path:
    if not _TASK_PATH.exists():
        print(f"[Tasks API] Downloading {_TASK_URL}")
        urllib.request.urlretrieve(_TASK_URL, _TASK_PATH)
    return _TASK_PATH


def _make_detector(min_conf: float):
    base = _mp_tasks.BaseOptions(model_asset_path=str(_ensure_task_model()))
    options = _mp_vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=1,
        min_hand_detection_confidence=min_conf,
        running_mode=_mp_vision.RunningMode.IMAGE,
    )
    return _mp_vision.HandLandmarker.create_from_options(options)


_HANDS_HIGH = None
_HANDS_LOW = None


def _get_detector(level: Literal["high", "low"]):
    global _HANDS_HIGH, _HANDS_LOW
    if _HANDS_HIGH is None:
        _HANDS_HIGH = _make_detector(0.5)
        _HANDS_LOW = _make_detector(0.3)
    return _HANDS_HIGH if level == "high" else _HANDS_LOW

_CSV_FIELDS = [
    "filename", "user_id", "type", "hand", "index",
    "method", "roi_size_px", "rotation_deg", "status", "note",
    "v1v2_dist_px", "v1v2_norm_ratio", "p5_p17_dist_px",
    "landmarks_in_bounds", "wrist_perp_score",
    "collinearity_score", "thumb_palm_alignment",
    "finger_extension", "inward_sign_value",
    "roi_skin_ratio",
    "img_w", "img_h",
]


def parse_filename(path: Path) -> dict | None:
    m = _FILENAME_RE.match(path.stem)
    if m is None:
        return None
    return {
        "user_id": m.group(1),
        "type": m.group(2),
        "hand": m.group(3),
        "index": m.group(4),
        "stem": path.stem,
    }


def detect_landmarks(image_bgr: np.ndarray, level: Literal["high", "low"]):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detector = _get_detector(level)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    return result.hand_landmarks[0]


def compute_roi_from_landmarks(lm, img_h: int, img_w: int) -> tuple[float, float, float, float]:
    pt = lambda i: np.array([lm[i].x * img_w, lm[i].y * img_h], dtype=np.float64)

    p0 = pt(0)
    V1 = (pt(5) + pt(9)) / 2.0
    V2 = (pt(13) + pt(17)) / 2.0
    M = (V1 + V2) / 2.0

    palm_vec = V2 - V1
    v1v2_dist = float(np.linalg.norm(palm_vec))
    angle_deg = float(np.degrees(np.arctan2(palm_vec[1], palm_vec[0])))

    # Fingers-up canonicalization: after rotating image by -angle_deg, the wrist (p0) must
    # end up BELOW M (positive y in cv2 convention). Rotated y of (p0 - M) is
    # dx * sin(angle) + dy * cos(angle). If negative, add 180° to flip orientation.
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
    size = v1v2_dist * SIZE_SCALE
    size = max(size, short_side * SIZE_LOWER_BOUND_RATIO)
    size = min(size, short_side * SIZE_UPPER_BOUND_RATIO)
    size = max(size, 100.0)

    perp = np.array([-palm_vec[1], palm_vec[0]], dtype=np.float64)
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm < 1e-6:
        inward_unit = np.array([0.0, 1.0])
    else:
        inward_unit = perp / perp_norm
        if float(np.dot(inward_unit, p0 - M)) < 0.0:
            inward_unit = -inward_unit

    inward_dist = v1v2_dist * INWARD_DIST_RATIO
    center = M + inward_unit * inward_dist

    return float(center[0]), float(center[1]), size, angle_deg


def compute_landmark_diagnostics(lm, img_h: int, img_w: int) -> dict:
    pt = lambda i: np.array([lm[i].x * img_w, lm[i].y * img_h], dtype=np.float64)

    p0 = pt(0)
    p1 = pt(1); p2 = pt(2)
    p5 = pt(5); p8 = pt(8)
    p9 = pt(9); p13 = pt(13); p17 = pt(17)

    V1 = (p5 + p9) / 2.0
    V2 = (p13 + p17) / 2.0
    M = (V1 + V2) / 2.0
    palm_vec = V2 - V1
    inward_vec = p0 - M

    v1v2_dist = float(np.linalg.norm(palm_vec))
    p5_p17_dist = float(np.linalg.norm(p17 - p5))
    short_side = float(min(img_h, img_w))
    v1v2_ratio = v1v2_dist / short_side if short_side > 0 else 0.0

    margin = -0.05
    in_bounds = all(
        margin <= lm[i].x <= 1.0 - margin and margin <= lm[i].y <= 1.0 - margin
        for i in range(21)
    )

    inward_norm = float(np.linalg.norm(inward_vec))
    palm_norm = float(np.linalg.norm(palm_vec))
    if inward_norm > 1e-6 and palm_norm > 1e-6:
        cos_a = abs(float(np.dot(inward_vec, palm_vec)) / (inward_norm * palm_norm))
        wrist_perp_score = round(1.0 - cos_a, 4)
    else:
        wrist_perp_score = 0.0

    # collinearity: 5,9,13,17이 일직선이어야 함. 9, 13의 (5↔17 직선)에 대한 perp distance RMS / line length.
    if p5_p17_dist > 1e-6:
        line_unit = (p17 - p5) / p5_p17_dist

        def _perp(p):
            v = p - p5
            return abs(line_unit[0] * v[1] - line_unit[1] * v[0])

        rms = float(np.sqrt((_perp(p9) ** 2 + _perp(p13) ** 2) / 2.0))
        collinearity_score = round(rms / p5_p17_dist, 4)
    else:
        collinearity_score = 1.0

    # thumb 방향 vs palm_vec 평행도. 0=수직(정상), 1=평행(이상).
    thumb_dir = p2 - p1
    thumb_norm = float(np.linalg.norm(thumb_dir))
    if thumb_norm > 1e-6 and palm_norm > 1e-6:
        cos_t = abs(float(np.dot(thumb_dir, palm_vec)) / (thumb_norm * palm_norm))
        thumb_palm_alignment = round(cos_t, 4)
    else:
        thumb_palm_alignment = 0.0

    # 검지 베이스(5)→끝(8) 거리 / V1V2 거리. 정상 1.5~2.5, 잘림 시 작음.
    finger_5_8 = float(np.linalg.norm(p8 - p5))
    finger_extension = round(finger_5_8 / v1v2_dist, 4) if v1v2_dist > 1e-6 else 0.0

    # cross product 부호 (정규화). 좌/우손 + 뒤집힘 검출.
    if palm_norm > 1e-6 and inward_norm > 1e-6:
        cross = float(palm_vec[0] * inward_vec[1] - palm_vec[1] * inward_vec[0])
        inward_sign_value = round(cross / (palm_norm * inward_norm), 4)
    else:
        inward_sign_value = 0.0

    return {
        "v1v2_dist_px": round(v1v2_dist, 1),
        "v1v2_norm_ratio": round(v1v2_ratio, 4),
        "p5_p17_dist_px": round(p5_p17_dist, 1),
        "landmarks_in_bounds": in_bounds,
        "wrist_perp_score": wrist_perp_score,
        "collinearity_score": collinearity_score,
        "thumb_palm_alignment": thumb_palm_alignment,
        "finger_extension": finger_extension,
        "inward_sign_value": inward_sign_value,
    }


def compute_roi_skin_ratio(roi_bgr: np.ndarray) -> float:
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )
    return round(float((mask > 0).mean()), 4)


def compute_roi_fallback(image_bgr: np.ndarray) -> tuple[float, float, float, float, str]:
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
        size = short_side * 0.30
        return cx, cy, size, 0.0, "center_crop"

    largest = max(contours, key=cv2.contourArea)

    hand_mask = np.zeros_like(mask)
    cv2.drawContours(hand_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # 1) palm core 추정: 거리변환 최댓값 위치 (손가락 분기점이 아닌 손바닥 본체의 가장 두꺼운 지점)
    dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
    _, max_dist_val, _, max_loc = cv2.minMaxLoc(dist)
    cx, cy = float(max_loc[0]), float(max_loc[1])

    # 2) 회전 각도 추정: minAreaRect의 짧은 축(=손바닥 너비, V1V2와 같은 방향)을 수평으로
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

    # 3) Fingers-up canonicalization: contour centroid is shifted toward fingers from
    # the inscribed-circle center (palm core). Vector palm_core → centroid points TOWARD
    # fingers. After rotation, this vector must have NEGATIVE y (point upward) so fingers
    # are at the top of the ROI.
    moments = cv2.moments(largest)
    if moments['m00'] > 0:
        cent_x = moments['m10'] / moments['m00']
        cent_y = moments['m01'] / moments['m00']
        to_finger_x = cent_x - cx
        to_finger_y = cent_y - cy
        angle_rad_f = np.radians(angle_deg)
        s_f = float(np.sin(angle_rad_f))
        c_f = float(np.cos(angle_rad_f))
        to_finger_rot_y = to_finger_x * s_f + to_finger_y * c_f
        if to_finger_rot_y > 0.0:
            angle_deg += 180.0
            while angle_deg > 180.0:
                angle_deg -= 360.0

    # 4) size: 손바닥 내접원 지름 × 2.6 ≈ 손바닥 너비, 그 후 clamp
    size = float(max_dist_val) * FALLBACK_DT_SIZE_SCALE
    size = max(size, short_side * FALLBACK_LOWER_BOUND_RATIO)
    size = min(size, short_side * FALLBACK_UPPER_BOUND_RATIO)
    size = max(size, 100.0)

    return cx, cy, size, float(angle_deg), "skin_fallback"


def crop_roi(image_bgr: np.ndarray, cx: float, cy: float, size: float, angle_deg: float) -> np.ndarray | None:
    img_h, img_w = image_bgr.shape[:2]

    M = cv2.getRotationMatrix2D((img_w / 2.0, img_h / 2.0), -angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image_bgr, M, (img_w, img_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )

    src = np.array([cx, cy, 1.0], dtype=np.float64)
    cx_r, cy_r = M @ src
    cx_r = int(round(cx_r))
    cy_r = int(round(cy_r))

    half = int(size // 2)
    x1, y1 = cx_r - half, cy_r - half
    x2, y2 = cx_r + half, cy_r + half

    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - img_h)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - img_w)

    if pad_top or pad_bottom or pad_left or pad_right:
        rotated = cv2.copyMakeBorder(
            rotated, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_REFLECT_101,
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    roi = rotated[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    h_roi, w_roi = roi.shape[:2]
    if h_roi != w_roi:
        s = min(h_roi, w_roi)
        roi = roi[(h_roi - s) // 2:(h_roi - s) // 2 + s,
                  (w_roi - s) // 2:(w_roi - s) // 2 + s]
    return roi


def save_roi(roi_bgr: np.ndarray, output_root: Path, user_id: str, stem: str, sizes: list[int]) -> None:
    for size in sizes:
        out_dir = output_root / str(size) / user_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_roi.jpg"
        resized = cv2.resize(roi_bgr, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])


def save_debug_vis(
    image_bgr: np.ndarray, lm, cx: float, cy: float, size: float, angle_deg: float,
    method: str, out_path: Path, diag: dict | None = None,
    skin_ratio: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = image_bgr.copy()
    img_h, img_w = vis.shape[:2]

    if lm is not None:
        for i in range(21):
            x = int(lm[i].x * img_w)
            y = int(lm[i].y * img_h)
            cv2.circle(vis, (x, y), 3, (0, 200, 0), -1)
        for idx in (0, 5, 9, 13, 17):
            x = int(lm[idx].x * img_w)
            y = int(lm[idx].y * img_h)
            cv2.circle(vis, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(vis, str(idx), (x + 9, y - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, str(idx), (x + 9, y - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        V1 = ((lm[5].x + lm[9].x) / 2.0 * img_w, (lm[5].y + lm[9].y) / 2.0 * img_h)
        V2 = ((lm[13].x + lm[17].x) / 2.0 * img_w, (lm[13].y + lm[17].y) / 2.0 * img_h)
        M_pt = ((V1[0] + V2[0]) / 2.0, (V1[1] + V2[1]) / 2.0)
        cv2.circle(vis, (int(V1[0]), int(V1[1])), 9, (255, 100, 0), 2)
        cv2.circle(vis, (int(V2[0]), int(V2[1])), 9, (255, 100, 0), 2)
        cv2.circle(vis, (int(M_pt[0]), int(M_pt[1])), 6, (255, 100, 255), -1)
        cv2.arrowedLine(vis, (int(V1[0]), int(V1[1])), (int(V2[0]), int(V2[1])),
                        (255, 0, 255), 2, tipLength=0.05)
        cv2.arrowedLine(vis, (int(M_pt[0]), int(M_pt[1])), (int(cx), int(cy)),
                        (0, 200, 255), 2, tipLength=0.1)

    cv2.drawMarker(vis, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)

    box = cv2.boxPoints(((float(cx), float(cy)), (float(size), float(size)), -float(angle_deg)))
    box = box.astype(np.int32)
    cv2.polylines(vis, [box], isClosed=True, color=(0, 255, 255), thickness=4)

    skin_str = f"{skin_ratio:.3f}" if skin_ratio is not None else "-"
    lines = [f"{method} | size={int(size)}px | angle={angle_deg:.1f}deg | skin={skin_str}"]
    if diag:
        lines.append(
            f"V1V2={diag.get('v1v2_dist_px', '-')}px"
            f" | ratio={diag.get('v1v2_norm_ratio', '-')}"
            f" | p5_p17={diag.get('p5_p17_dist_px', '-')}px"
        )
        lines.append(
            f"in_bounds={diag.get('landmarks_in_bounds', '-')}"
            f" | wrist_perp={diag.get('wrist_perp_score', '-')}"
            f" | colin={diag.get('collinearity_score', '-')}"
        )
        lines.append(
            f"thumb_align={diag.get('thumb_palm_alignment', '-')}"
            f" | finger_ext={diag.get('finger_extension', '-')}"
            f" | inward_sign={diag.get('inward_sign_value', '-')}"
        )

    y0 = 30
    for line in lines:
        cv2.putText(vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 26

    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _err_row(img_path: Path, note: str, meta: dict | None = None) -> dict:
    base = {f: "" for f in _CSV_FIELDS}
    base["filename"] = img_path.stem
    base["status"] = "error"
    base["note"] = note
    if meta:
        base["user_id"] = meta["user_id"]
        base["type"] = meta["type"]
        base["hand"] = meta["hand"]
        try:
            base["index"] = int(meta["index"])
        except (ValueError, TypeError):
            base["index"] = meta["index"]
    return base


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict]) -> None:
    total = len(rows)
    success = sum(1 for r in rows if r["status"] == "success")
    fallback = sum(1 for r in rows if r["status"] == "fallback")
    errors = sum(1 for r in rows if r["status"] == "error")
    print(f"\n완료: 총 {total}개 | 성공 {success} | fallback {fallback} | 오류 {errors}")
    methods: dict[str, int] = {}
    for r in rows:
        methods[r["method"]] = methods.get(r["method"], 0) + 1
    print("method 분포:")
    for k, v in sorted(methods.items(), key=lambda kv: -kv[1]):
        print(f"  {k or '(empty)'}: {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BMPD Palmprint ROI Extractor")
    p.add_argument("--input", type=str, required=True, help="BMPD 루트 폴더 경로")
    p.add_argument("--output", type=str, required=True, help="출력 루트 폴더 경로")
    p.add_argument("--sizes", type=int, nargs="+", default=[128, 256], help="출력 해상도 리스트")
    p.add_argument("--vis", action="store_true", help="디버그 시각화 저장")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    images = sorted({*input_root.rglob("*.JPG"), *input_root.rglob("*.jpg")})
    print(f"총 {len(images)}개 이미지 발견")

    log_rows: list[dict] = []

    for img_path in tqdm(images, desc="ROI 추출"):
        meta: dict | None = None
        try:
            meta = parse_filename(img_path)
            if meta is None:
                log_rows.append(_err_row(img_path, "invalid_filename"))
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                log_rows.append(_err_row(img_path, "load_failed", meta))
                continue

            img_h, img_w = image.shape[:2]
            note = ""

            lm = detect_landmarks(image, "high")
            if lm is None:
                lm = detect_landmarks(image, "low")
                if lm is not None:
                    note = "low_confidence"

            diag: dict = {}
            if lm is not None:
                diag = compute_landmark_diagnostics(lm, img_h, img_w)
                cx, cy, size, angle = compute_roi_from_landmarks(lm, img_h, img_w)
                method = "landmark_low_conf" if note == "low_confidence" else "landmark"
                status = "success"
            else:
                cx, cy, size, angle, method = compute_roi_fallback(image)
                status = "fallback"

            roi = crop_roi(image, cx, cy, size, angle)
            if roi is None or roi.size == 0:
                log_rows.append(_err_row(img_path, "crop_failed", meta))
                continue

            skin_ratio = compute_roi_skin_ratio(roi)

            save_roi(roi, output_root, meta["user_id"], meta["stem"], args.sizes)

            if args.vis:
                debug_path = output_root / "debug" / meta["user_id"] / f"{meta['stem']}_debug.jpg"
                save_debug_vis(image, lm, cx, cy, size, angle, method, debug_path, diag, skin_ratio)

            row = {
                "filename": meta["stem"],
                "user_id": meta["user_id"],
                "type": meta["type"],
                "hand": meta["hand"],
                "index": int(meta["index"]),
                "method": method,
                "roi_size_px": int(size),
                "rotation_deg": round(float(angle), 2),
                "status": status,
                "note": note,
                "roi_skin_ratio": skin_ratio,
                "img_w": img_w,
                "img_h": img_h,
            }
            row.update(diag)
            log_rows.append(row)
        except Exception as e:
            log_rows.append(_err_row(img_path, f"exception:{type(e).__name__}:{e}", meta))

    _write_csv(output_root / "extraction_log.csv", log_rows)
    _print_summary(log_rows)
    print(f"로그 저장: {output_root / 'extraction_log.csv'}")


if __name__ == "__main__":
    main()
