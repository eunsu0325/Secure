"""Build Tongji ROI manifest for exp1_baselines (MFN/IR50 + ArcFace).

Tongji ROI layout::

    Tongji/
    ├── session1/
    │   ├── 00001.bmp ... 06000.bmp
    └── session2/
        ├── 00001.bmp ... 06000.bmp

Identity convention:
    palm_id = (image_number - 1) // 10
    sample_id = (image_number - 1) % 10

So images 00001..00010 belong to palm_id=0, images 00011..00020 to palm_id=1, etc.
600 palm_ids total (0..599), 10 images per palm per session.

4-way identity split (seed=42 shuffle of all 600 palm_ids):
    base_ids:           320  (backbone ArcFace training only)
    threshold_val_ids:   80  (fixed tau calibration only)
    future_ids:         150  (sequential enrollment evaluation)
    external_test_ids:   50  (held-out never-enrolled evaluation)

Sample split per identity (legacy enroll/dev/test format for tooling compatibility):
    enroll = session1 all 10 images (used for evaluation gallery/prototypes)
    test   = session2 all 10 images (used for evaluation queries)
    dev    = unused (kept for legacy validator compatibility)

Note for backbone training:
    base_ids use BOTH session1 AND session2 (combined enroll+test paths) for ArcFace
    training. The enroll/test split semantics here are for evaluation, not training.

Usage::

    python exp1_baselines/build_tongji_for_baselines.py \\
        --tongji_root /path/to/Tongji \\
        --output_dir exp1_baselines/manifests \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp1_common.dataset_manifest import is_image_file


_FILENAME_RE = re.compile(r"^(\d+)$")


def _palm_id_from_n(n: int) -> int:
    return (n - 1) // 10


def _sample_id_from_n(n: int) -> int:
    return (n - 1) % 10


def _walk_session(session_dir: Path) -> Dict[int, List[Tuple[int, Path]]]:
    """Return palm_id -> list of (sample_id, image_path)."""
    out: Dict[int, List[Tuple[int, Path]]] = defaultdict(list)
    if not session_dir.is_dir():
        return out
    for p in sorted(session_dir.rglob("*")):
        if not p.is_file() or not is_image_file(p):
            continue
        m = _FILENAME_RE.match(p.stem)
        if not m:
            continue
        n = int(m.group(1))
        if n < 1 or n > 6000:
            continue
        out[_palm_id_from_n(n)].append((_sample_id_from_n(n), p))
    for pid in out:
        out[pid].sort(key=lambda x: x[0])
    return out


def _rel_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Tongji manifest for exp1_baselines"
    )
    parser.add_argument("--tongji_root", required=True, type=str,
                        help="Path containing session1/ and session2/")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--n_base", default=320, type=int)
    parser.add_argument("--n_threshold_val", default=80, type=int)
    parser.add_argument("--n_future", default=150, type=int)
    parser.add_argument("--n_external_test", default=50, type=int)
    parser.add_argument("--n_threshold_gallery", default=40, type=int,
                        help="threshold_val_ids[:n] used as calibration gallery")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.tongji_root)
    s1 = _walk_session(root / "session1")
    s2 = _walk_session(root / "session2")
    if not s1 or not s2:
        raise FileNotFoundError(
            f"Tongji root must contain session1/ and session2/: {root}"
        )

    n_s1 = sum(len(v) for v in s1.values())
    n_s2 = sum(len(v) for v in s2.values())
    print(f"[walk] session1: {n_s1} images across {len(s1)} palm_ids")
    print(f"[walk] session2: {n_s2} images across {len(s2)} palm_ids")

    assert n_s1 == 6000, f"session1 must have 6000 images, got {n_s1}"
    assert n_s2 == 6000, f"session2 must have 6000 images, got {n_s2}"
    assert len(s1) == 600, f"session1 must have 600 palm_ids, got {len(s1)}"
    assert len(s2) == 600, f"session2 must have 600 palm_ids, got {len(s2)}"
    assert set(s1) == set(s2), "session1 and session2 palm_id sets must match"

    for pid in sorted(s1):
        assert len(s1[pid]) == 10, f"palm {pid} session1: {len(s1[pid])} != 10"
        assert len(s2[pid]) == 10, f"palm {pid} session2: {len(s2[pid])} != 10"

    all_palm_ids = sorted(s1.keys())
    total_needed = (
        args.n_base + args.n_threshold_val + args.n_future + args.n_external_test
    )
    assert total_needed == 600, (
        f"split sums to {total_needed}, must equal 600"
    )
    assert args.n_threshold_gallery <= args.n_threshold_val, (
        "threshold_gallery size must be <= threshold_val size"
    )

    rng = np.random.RandomState(args.seed)
    shuffled = list(rng.permutation(all_palm_ids))

    cursor = 0
    base_palms = sorted(int(x) for x in shuffled[cursor:cursor + args.n_base])
    cursor += args.n_base
    threshold_val_palms = [int(x) for x in shuffled[cursor:cursor + args.n_threshold_val]]
    cursor += args.n_threshold_val
    future_palms = [int(x) for x in shuffled[cursor:cursor + args.n_future]]
    cursor += args.n_future
    external_test_palms = sorted(
        int(x) for x in shuffled[cursor:cursor + args.n_external_test]
    )

    threshold_gallery_palms = threshold_val_palms[:args.n_threshold_gallery]
    threshold_unknown_palms = threshold_val_palms[args.n_threshold_gallery:]

    # No overlap check
    overlap_sets = [
        ("base", set(base_palms)),
        ("threshold_val", set(threshold_val_palms)),
        ("future", set(future_palms)),
        ("external_test", set(external_test_palms)),
    ]
    for i, (name_a, set_a) in enumerate(overlap_sets):
        for name_b, set_b in overlap_sets[i + 1:]:
            inter = set_a & set_b
            assert not inter, f"{name_a} ∩ {name_b} = {inter}"

    union_size = sum(len(s) for _, s in overlap_sets)
    assert union_size == 600, f"union size {union_size} != 600"

    # Canonical label assignment: base sorted, threshold_val shuffle, future shuffle,
    # external_test sorted.
    canonical_palm_order = (
        list(base_palms)
        + list(threshold_val_palms)
        + list(future_palms)
        + list(external_test_palms)
    )

    # Build sample_split, all.txt records, and split bookkeeping.
    base_set = set(base_palms)
    threshold_val_set = set(threshold_val_palms)
    threshold_gallery_set = set(threshold_gallery_palms)
    threshold_unknown_set = set(threshold_unknown_palms)
    future_set = set(future_palms)
    external_test_set = set(external_test_palms)

    sample_split: Dict[int, Dict[str, List[str]]] = {}
    label_to_palm: Dict[int, int] = {}
    base_ids: List[int] = []
    threshold_val_ids: List[int] = []
    threshold_gallery_ids: List[int] = []
    threshold_unknown_ids: List[int] = []
    future_ids: List[int] = []
    external_test_ids: List[int] = []

    txt_records: List[Tuple[str, int]] = []

    for label, pid in enumerate(canonical_palm_order):
        s1_paths = [_rel_to_project(p) for _, p in s1[pid]]
        s2_paths = [_rel_to_project(p) for _, p in s2[pid]]
        assert len(s1_paths) == 10 and len(s2_paths) == 10

        sample_split[label] = {
            "enroll": s1_paths,   # session1 all 10
            "dev": [],            # unused
            "test": s2_paths,     # session2 all 10
        }
        label_to_palm[label] = pid

        # all.txt: include session1 + session2 paths for this label
        for p in s1_paths + s2_paths:
            txt_records.append((p, label))

        if pid in base_set:
            base_ids.append(label)
        if pid in threshold_val_set:
            threshold_val_ids.append(label)
            if pid in threshold_gallery_set:
                threshold_gallery_ids.append(label)
            elif pid in threshold_unknown_set:
                threshold_unknown_ids.append(label)
        if pid in future_set:
            future_ids.append(label)
        if pid in external_test_set:
            external_test_ids.append(label)

    # Write all.txt (sorted by (label, path) for byte-stability)
    txt_records.sort(key=lambda r: (r[1], r[0]))
    with open(out_dir / "all.txt", "w", encoding="utf-8") as f:
        for path, label in txt_records:
            f.write(f"{path} {label}\n")

    # Write identity_split.json — 4-way split for baseline pipeline
    identity_split = {
        "seed": int(args.seed),
        "split_mode": "identity_disjoint_baseline_protocol_analysis",
        "total_identities": 600,
        "base_ids": sorted(int(x) for x in base_ids),
        "threshold_val_ids": [int(x) for x in threshold_val_ids],
        "threshold_gallery_ids": [int(x) for x in threshold_gallery_ids],
        "threshold_unknown_ids": [int(x) for x in threshold_unknown_ids],
        "future_ids": [int(x) for x in future_ids],
        "external_test_ids": sorted(int(x) for x in external_test_ids),
        "enrollment_order": [int(x) for x in future_ids],
    }
    with open(out_dir / "identity_split.json", "w", encoding="utf-8") as f:
        json.dump(identity_split, f, indent=2, sort_keys=False, ensure_ascii=False)
        f.write("\n")

    # Write sample_split.json — legacy enroll/dev/test format
    sample_payload: Dict[str, Dict[str, List[str]]] = {}
    for label, parts in sample_split.items():
        sample_payload[str(label)] = {
            "enroll": list(parts["enroll"]),
            "dev": list(parts["dev"]),
            "test": list(parts["test"]),
        }
    with open(out_dir / "sample_split.json", "w", encoding="utf-8") as f:
        json.dump(sample_payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    # Write metadata.json
    metadata = {
        "dataset_name": "tongji_roi_baselines",
        "identity_unit": "palm_side",
        "subject_disjoint": False,
        "tongji_root": str(root.resolve()),
        "selected_identity_count": 600,
        "base_count": len(base_ids),
        "threshold_val_count": len(threshold_val_ids),
        "threshold_gallery_count": len(threshold_gallery_ids),
        "threshold_unknown_count": len(threshold_unknown_ids),
        "future_count": len(future_ids),
        "external_test_count": len(external_test_ids),
        "label_to_palm_id": {str(l): int(p) for l, p in label_to_palm.items()},
        "sample_split_policy": (
            "enroll = session1 all 10 images (gallery/prototype). "
            "test = session2 all 10 images (probes). "
            "dev = unused. "
            "Backbone training of base_ids uses session1+session2 (combined "
            "enroll+test paths)."
        ),
        "seed": int(args.seed),
        "split_construction": (
            f"shuffle(sorted(palm_ids), seed={args.seed}) → "
            f"base[{args.n_base}], threshold_val[{args.n_threshold_val}], "
            f"future[{args.n_future}], external_test[{args.n_external_test}]"
        ),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    print(f"[manifest] wrote {out_dir}/all.txt ({len(txt_records)} records)")
    print(f"[manifest] wrote {out_dir}/identity_split.json")
    print(f"[manifest] wrote {out_dir}/sample_split.json")
    print(f"[manifest] wrote {out_dir}/metadata.json")
    print(
        f"[split] base={len(base_ids)} threshold_val={len(threshold_val_ids)} "
        f"(gallery={len(threshold_gallery_ids)}, unknown={len(threshold_unknown_ids)}) "
        f"future={len(future_ids)} external_test={len(external_test_ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
