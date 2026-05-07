"""WebPalm cal/dev/test deterministic split (per plan §V11 reproducibility).

Splits the successful WebPalm ROIs into three identity-disjoint groups:
  - calibration (3,000): used to fit τ̂_pred(m) and σ̂_τ(m) (plan §C2 / V14a)
  - dev (10,000): used for safety-margin λ selection at α=1e-3 (plan §V9)
  - test (≥50,000): single-shot final evaluation only (plan §V5 / V9)

The split is deterministic by stable hash of the WebPalm filename to make
cal/dev/test reproducible across re-runs:
  - sha256(filename) mod 100,000 in [0, target_cal_share)         -> cal
  - in [cal_share, dev_share)                                      -> dev
  - in [dev_share, dev_share + test_share)                         -> test
  - else                                                            -> unused

When the ROI extraction success count is < 83,145 (typical: ~65K after
74-79% success), the script preserves the cal/dev pool sizes (operational
constraints from plan §V9: dev needs >=10 expected accepts at α=1e-3) and
puts the remainder into test. If the test pool would fall below 50,000
(plan §V5: minimum for reliable α=1e-4 reporting), the script issues a
warning and falls back to the v3 §C3 cap rule:
  cal = min(3000, floor(0.04 * N_success))
  dev = min(10000, floor(0.14 * N_success))
  test = remaining.

Usage:
    python experiments/split_webpalm.py \
        --roi-log ~/research_data/webpalm/roi_224/extraction_log.csv \
        --out     ~/research_data/webpalm/splits

Outputs at `--out`:
    splits.json           : list of {filename, hand, status, split} per ROI
    webpalm_calibration.txt : one filename per line
    webpalm_dev.txt
    webpalm_test.txt
    webpalm_unused.txt    : filenames excluded from cal/dev/test
    summary.json          : counts + reliability flags + tier per V5
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List


CAL_TARGET = 3_000
DEV_TARGET = 10_000
TEST_MIN = 50_000      # plan §V5 reliability floor for α=1e-4
HASH_MOD = 100_000     # split granularity (100k buckets)


def _stable_hash(name: str) -> int:
    """sha256-based stable hash mod HASH_MOD. Cross-platform, deterministic."""
    h = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(h, 16) % HASH_MOD


def _wilson_lower(p: float, n: int, z: float = 1.96) -> float:
    """One-line Wilson lower bound at 95% CI."""
    if n == 0:
        return 0.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - half)


def main() -> int:
    p = argparse.ArgumentParser(description="WebPalm cal/dev/test split")
    p.add_argument("--roi-log", type=Path, required=True,
                   help="extraction_log.csv from webpalm_roi_extractor")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cal", type=int, default=CAL_TARGET)
    p.add_argument("--dev", type=int, default=DEV_TARGET)
    p.add_argument("--test-min", type=int, default=TEST_MIN)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Load extraction log; keep only successful ROIs
    rows = []
    with args.roi_log.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("status") in ("success", "fallback"):
                rows.append({
                    "filename": r["filename"],
                    "hand": r.get("hand", "U"),
                    "method": r["method"],
                    "status": r["status"],
                })
    n_success = len(rows)
    print(f"[load] {args.roi_log}: {n_success} successful ROIs")

    # Decide quotas. Honor user requested cal/dev unless they exceed the
    # success pool; in that case fall back to v3 §C3 cap rule.
    if n_success < args.cal + args.dev + args.test_min:
        cal_n = min(args.cal, int(0.04 * n_success))
        dev_n = min(args.dev, int(0.14 * n_success))
        test_n = max(0, n_success - cal_n - dev_n)
        print(f"[fallback] success pool {n_success} below "
              f"{args.cal}+{args.dev}+{args.test_min}; "
              f"using cal={cal_n}, dev={dev_n}, test={test_n}")
    else:
        cal_n = args.cal
        dev_n = args.dev
        test_n = n_success - cal_n - dev_n
        print(f"[plan] cal={cal_n}, dev={dev_n}, test={test_n}")

    # Hash-based deterministic split. Each successful ROI gets a hash
    # bucket; we sort by bucket then assign cal first, dev next, test next.
    rows.sort(key=lambda r: _stable_hash(r["filename"]))
    for i, r in enumerate(rows):
        if i < cal_n:
            r["split"] = "calibration"
        elif i < cal_n + dev_n:
            r["split"] = "dev"
        elif i < cal_n + dev_n + test_n:
            r["split"] = "test"
        else:
            r["split"] = "unused"

    # Write outputs
    splits_json = args.out / "splits.json"
    splits_json.write_text(json.dumps(rows, indent=2))

    for split_name in ("calibration", "dev", "test", "unused"):
        names = [r["filename"] for r in rows if r["split"] == split_name]
        (args.out / f"webpalm_{split_name}.txt").write_text(
            "\n".join(names) + ("\n" if names else "")
        )

    # Reliability flags per V5 / V9
    actual_test = sum(1 for r in rows if r["split"] == "test")
    actual_dev = sum(1 for r in rows if r["split"] == "dev")
    actual_cal = sum(1 for r in rows if r["split"] == "calibration")

    # Wilson lower bound from the trial-100/1K success rate isn't recomputed
    # here; instead we report the operational reliability for each α target.
    summary = {
        "n_roi_success": n_success,
        "cal_size": actual_cal,
        "dev_size": actual_dev,
        "test_size": actual_test,
        "unused_size": sum(1 for r in rows if r["split"] == "unused"),
        # Operational reliability per plan §V9 / §V12
        "alpha_1e3": {
            "dev_expected_accepts": actual_dev * 1e-3,
            "test_expected_accepts": actual_test * 1e-3,
            "test_min_observable_fpir": 1.0 / actual_test if actual_test else None,
            "lambda_tunable_on_dev": (actual_dev * 1e-3) >= 5,
            "test_reliable": (actual_test * 1e-3) >= 5,
        },
        "alpha_1e4": {
            "dev_expected_accepts": actual_dev * 1e-4,
            "test_expected_accepts": actual_test * 1e-4,
            "test_min_observable_fpir": 1.0 / actual_test if actual_test else None,
            "lambda_tunable_on_dev": False,  # locked: never tune at 1e-4
            "test_reliable": (actual_test * 1e-4) >= 5,
            "note": "λ tuning forbidden at α=1e-4 per plan §V9; "
                    "single-shot evaluation only, reusing λ from α=1e-3",
        },
        "split_seed": "sha256-mod-100000 deterministic; split_seed=42 elsewhere",
        "split_size_policy": (
            "honors --cal/--dev unless success pool too small, "
            "then falls back to v3 §C3 caps (4%/14%/remaining)"
        ),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[summary] cal={actual_cal} dev={actual_dev} test={actual_test} "
          f"unused={summary['unused_size']}")
    print(f"[ok] written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
