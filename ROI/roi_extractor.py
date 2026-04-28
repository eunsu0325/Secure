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
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm

try:
    import mediapipe as mp
    _mp_hands = mp.solutions.hands
except (AttributeError, ImportError):
    from mediapipe.python.solutions import hands as _mp_hands


_FILENAME_RE = re.compile(r"^(\d{3})_([FS])_([LR])_(\d+)$")

SIZE_SCALE = 1.3
FALLBACK_SIZE_SCALE = 0.6

_HANDS_HIGH = _mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
)
_HANDS_LOW = _mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3
)

_CSV_FIELDS = [
    "filename", "user_id", "type", "hand", "index",
    "method", "roi_size_px", "rotation_deg", "status", "note",
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
    detector = _HANDS_HIGH if level == "high" else _HANDS_LOW
    result = detector.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    return result.multi_hand_landmarks[0].landmark


def compute_roi_from_landmarks(lm, img_h: int, img_w: int) -> tuple[float, float, float, float]:
    pt = lambda i: np.array([lm[i].x * img_w, lm[i].y * img_h], dtype=np.float64)

    V1 = (pt(5) + pt(9)) / 2.0
    V2 = (pt(13) + pt(17)) / 2.0

    palm_vec = V2 - V1
    angle_deg = float(np.degrees(np.arctan2(palm_vec[1], palm_vec[0])))

    size = float(np.linalg.norm(palm_vec)) * SIZE_SCALE
    size = max(size, 100.0)

    M = (V1 + V2) / 2.0
    wrist = pt(0)
    inward = wrist - M
    norm = float(np.linalg.norm(inward))
    if norm < 1e-6:
        inward = np.array([0.0, 1.0])
    else:
        inward = inward / norm
    center = M + inward * (size / 2.0)

    return float(center[0]), float(center[1]), size, angle_deg


def compute_roi_fallback(image_bgr: np.ndarray) -> tuple[float, float, float, float, str]:
    h, w = image_bgr.shape[:2]
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
        size = float(min(h, w)) * FALLBACK_SIZE_SCALE
        return cx, cy, size, 0.0, "center_crop"

    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    size = float(min(bw, bh)) * FALLBACK_SIZE_SCALE
    return cx, cy, size, 0.0, "skin_fallback"


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
    method: str, out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis = image_bgr.copy()
    img_h, img_w = vis.shape[:2]

    if lm is not None:
        for idx in (0, 5, 9, 13, 17):
            x = int(lm[idx].x * img_w)
            y = int(lm[idx].y * img_h)
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
        V1 = ((lm[5].x + lm[9].x) / 2.0 * img_w, (lm[5].y + lm[9].y) / 2.0 * img_h)
        V2 = ((lm[13].x + lm[17].x) / 2.0 * img_w, (lm[13].y + lm[17].y) / 2.0 * img_h)
        cv2.circle(vis, (int(V1[0]), int(V1[1])), 8, (255, 100, 0), 2)
        cv2.circle(vis, (int(V2[0]), int(V2[1])), 8, (255, 100, 0), 2)

    cv2.drawMarker(vis, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    box = cv2.boxPoints(((float(cx), float(cy)), (float(size), float(size)), -float(angle_deg)))
    box = box.astype(np.int32)
    cv2.polylines(vis, [box], isClosed=True, color=(0, 255, 255), thickness=4)

    text = f"{method} | size={int(size)}px | angle={angle_deg:.1f}deg"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

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

            if lm is not None:
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

            save_roi(roi, output_root, meta["user_id"], meta["stem"], args.sizes)

            if args.vis:
                debug_path = output_root / "debug" / meta["user_id"] / f"{meta['stem']}_debug.jpg"
                save_debug_vis(image, lm, cx, cy, size, angle, method, debug_path)

            log_rows.append({
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
            })
        except Exception as e:
            log_rows.append(_err_row(img_path, f"exception:{type(e).__name__}:{e}", meta))

    _write_csv(output_root / "extraction_log.csv", log_rows)
    _print_summary(log_rows)
    print(f"로그 저장: {output_root / 'extraction_log.csv'}")


if __name__ == "__main__":
    main()
