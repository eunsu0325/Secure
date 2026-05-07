"""IITD Palmprint V1 manifest builder (D7a).

Reads IITD V1 Segmented BMPs and emits the canonical 5-way manifest.

IITD specifics vs. Tongji (plan: Identity Convention + Sample Role Policy):
  - Tier 1 identity_id: P0a ✓ (subject pairing from folder structure) +
    P0b ✓ (L/R from `Right`/`Left` folder name) → "subject_<id>-l" / "-r".
  - Non-sessioned: every palm has a single image pool. session_id is "".
  - sample_role for base_anchor / future = "candidate"; the score-matrix
    builder picks K=3 enroll sample_ids per palm per enroll_seed (IER-7) so
    no double-use bug can occur.
  - Eligibility (applied before splitting): subject MUST have BOTH `Right`
    AND `Left` palm folders, each with ≥ MIN_IMGS_PER_PALM images.
  - palm_id = subject_id * 2 + 0 (left) or +1 (right), so
    `contralateral_xor1(palm_id)` still yields the partner palm.

Folder/filename conventions verified at builder runtime:
  IITD_Palmprint_V1/Segmented/{Right,Left}/<sub3>_<rep>.bmp
  - sub3:  3-digit zero-padded subject id (file-side is 1-based)
  - rep:   1+ digits (typically 1..5 or 1..6)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from exp1_baselines.datasets.manifest_schema import (
    MANIFEST_COLUMNS,
    HandSide,
    IdentityNamingTier,
    ManifestRow,
    SampleRole,
    SubjectSplit,
    make_identity_id,
)
from exp1_baselines.datasets.split import (
    DATASET_QUOTAS,
    SplitResult,
    subject_level_split,
)
from exp1_baselines.datasets.validate_manifest import (
    validate_canonical_columns,
    validate_final_split_counts,
    validate_no_legacy_fields,
    validate_palm_identity_consistency,
    validate_subject_split_disjointness,
    validate_train_backbone_size,
)


MIN_IMGS_PER_PALM = 5
IITD_FILENAME_RE = re.compile(r"^(\d+)_(\d+)$")
RIGHT_FOLDER_CANDIDATES = ("Right", "right", "Right Hand")
LEFT_FOLDER_CANDIDATES = ("Left", "left", "Left Hand")


def _find_folder(parent: Path, candidates: Sequence[str]) -> Path:
    for name in candidates:
        candidate = parent / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"none of {list(candidates)} exists under {parent}"
    )


def _scan_palm_images(
    folder: Path,
) -> Dict[int, List[Path]]:
    """Return {file_subject_id (1-based): sorted list of image paths}."""
    out: Dict[int, List[Path]] = defaultdict(list)
    for p in folder.iterdir():
        if p.suffix.lower() != ".bmp":
            continue
        m = IITD_FILENAME_RE.match(p.stem)
        if m is None:
            continue
        sub = int(m.group(1))
        out[sub].append(p)
    for sub in out:
        out[sub] = sorted(out[sub], key=lambda x: int(IITD_FILENAME_RE.match(x.stem).group(2)))
    return out


def _verify_iitd_root(root: Path) -> Tuple[Path, Path]:
    """Resolve and assert the Segmented/Right + Segmented/Left folders."""
    if not root.is_dir():
        raise FileNotFoundError(f"IITD root not a directory: {root}")
    seg = root / "Segmented"
    if not seg.is_dir():
        # Permit the case where the user already pointed at .../Segmented/
        if (root / "Right").is_dir() or (root / "right").is_dir():
            seg = root
        else:
            raise FileNotFoundError(
                f"could not find Segmented/ under {root}; pass --root that "
                f"contains a `Segmented/Right` and `Segmented/Left` subtree"
            )
    right = _find_folder(seg, RIGHT_FOLDER_CANDIDATES)
    left = _find_folder(seg, LEFT_FOLDER_CANDIDATES)
    return right, left


def _val_ckpt_count(n: int) -> int:
    """≥1 val image per palm; ~20% with floor(n / 5)."""
    return max(1, n // 5)


def _row_for_image(
    *,
    image_path: Path,
    subject_id: int,
    palm_id: int,
    hand_side: str,
    sample_id: int,
    subject_split: str,
    sample_role: str,
) -> ManifestRow:
    return ManifestRow(
        image_path=str(image_path),
        dataset="iitd",
        subject_id=subject_id,
        palm_id=palm_id,
        identity_id=make_identity_id(
            IdentityNamingTier.TIER1, subject_id, palm_id, hand_side
        ),
        hand_side=hand_side,
        session_id="",
        phase_id="",
        sample_id=sample_id,
        subject_split=subject_split,
        sample_role=sample_role,
        is_enroll=(sample_role == SampleRole.ENROLL.value),
        is_query=(sample_role == SampleRole.QUERY.value),
        roi_status="ok",
    )


def _emit_palm(
    *,
    image_paths: Sequence[Path],
    subject_id: int,
    palm_id: int,
    hand_side: str,
    subject_split: str,
) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    n = len(image_paths)
    if subject_split == SubjectSplit.TRAIN_BACKBONE.value:
        n_val = _val_ckpt_count(n)
        n_train = n - n_val
        if n_train <= 0:
            raise ValueError(
                f"subject {subject_id} palm {palm_id}: only {n} imgs, "
                f"all assigned to val_ckpt — eligibility check should have rejected"
            )
        for sid, p in enumerate(image_paths):
            role = (
                SampleRole.TRAIN.value if sid < n_train
                else SampleRole.VAL_CKPT.value
            )
            rows.append(_row_for_image(
                image_path=p, subject_id=subject_id, palm_id=palm_id,
                hand_side=hand_side, sample_id=sid,
                subject_split=subject_split, sample_role=role,
            ))
        return rows

    if subject_split in (SubjectSplit.BASE_ANCHOR.value, SubjectSplit.FUTURE.value):
        # Non-sessioned: all imgs marked `candidate`; score-matrix builder
        # picks K=3 enroll + complement-query per enroll_seed (IER-7).
        for sid, p in enumerate(image_paths):
            rows.append(_row_for_image(
                image_path=p, subject_id=subject_id, palm_id=palm_id,
                hand_side=hand_side, sample_id=sid,
                subject_split=subject_split,
                sample_role=SampleRole.CANDIDATE.value,
            ))
        return rows

    if subject_split in (SubjectSplit.EXTERNAL_DEV.value, SubjectSplit.EXTERNAL_TEST.value):
        # Externals: every img is a query (no K applies; never enrolled).
        for sid, p in enumerate(image_paths):
            rows.append(_row_for_image(
                image_path=p, subject_id=subject_id, palm_id=palm_id,
                hand_side=hand_side, sample_id=sid,
                subject_split=subject_split,
                sample_role=SampleRole.QUERY.value,
            ))
        return rows

    raise ValueError(f"unknown subject_split: {subject_split}")


def build_iitd_manifest(
    root: Path,
    out_dir: Path,
    seed: int = 42,
) -> Path:
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    right_dir, left_dir = _verify_iitd_root(root)
    right_imgs = _scan_palm_images(right_dir)
    left_imgs = _scan_palm_images(left_dir)

    raw_subject_ids = sorted(set(right_imgs.keys()) | set(left_imgs.keys()))
    if not raw_subject_ids:
        raise RuntimeError(f"no IITD palm images found under {root}")

    # Eligibility: BOTH hands present, each with ≥ MIN_IMGS_PER_PALM
    eligible_subjects: List[int] = []
    excluded_reasons: Dict[str, str] = {}
    for sub in raw_subject_ids:
        r = right_imgs.get(sub, [])
        l = left_imgs.get(sub, [])
        reasons = []
        if len(r) < MIN_IMGS_PER_PALM:
            reasons.append(f"right has {len(r)} imgs (<{MIN_IMGS_PER_PALM})")
        if len(l) < MIN_IMGS_PER_PALM:
            reasons.append(f"left has {len(l)} imgs (<{MIN_IMGS_PER_PALM})")
        if reasons:
            excluded_reasons[str(sub)] = "; ".join(reasons)
        else:
            eligible_subjects.append(sub)

    # Map file-side 1-based subject -> 0-based subject_id (split unit)
    file_to_subject_id = {f: i for i, f in enumerate(eligible_subjects)}

    split: SplitResult = subject_level_split(
        list(file_to_subject_id.values()),
        DATASET_QUOTAS["iitd"],
        seed=seed,
    )

    rows: List[ManifestRow] = []
    for split_name in ("train_backbone", "base_anchor", "future",
                       "external_dev", "external_test"):
        for sid in split.assignments[split_name]:
            file_sub = eligible_subjects[sid]
            for hand_side, palm_offset, imgs in (
                (HandSide.LEFT.value, 0, left_imgs[file_sub]),
                (HandSide.RIGHT.value, 1, right_imgs[file_sub]),
            ):
                palm_id = sid * 2 + palm_offset
                rows += _emit_palm(
                    image_paths=imgs,
                    subject_id=sid,
                    palm_id=palm_id,
                    hand_side=hand_side,
                    subject_split=split_name,
                )

    # Write CSV
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_csv_dict())

    final_split_counts = {
        split_name: {
            "subjects": len(split.assignments[split_name]),
            "palms": len(split.assignments[split_name]) * 2,
        }
        for split_name in (
            "train_backbone", "base_anchor", "future",
            "external_dev", "external_test",
        )
    }
    metadata = {
        "dataset": "iitd",
        "schema_version": "exp1_canonical_v1",
        "split_seed": seed,
        "raw_subject_count": len(raw_subject_ids),
        "eligible_subject_count": len(eligible_subjects),
        "excluded_subject_count": len(excluded_reasons),
        "exclusion_reason": excluded_reasons,
        "final_split_counts": final_split_counts,
        "iitd_p0a_status": "verified_from_file_structure",
        "iitd_p0a_evidence": (
            "Right/Left folders + paired subject ids enforce P0a"
        ),
        "iitd_p0b_status": "verified_from_file_structure",
        "iitd_p0b_evidence": "hand_side derived from {Right,Left} folder name",
        "iitd_identity_naming": "subject_<id>-l / subject_<id>-r (Tier 1)",
        "iitd_session": "non_sessioned (sample_role=candidate for base/future)",
        "iitd_palm_id_formula": (
            "palm_id = subject_id * 2 + (0 if left else 1); "
            "contralateral via palm_id XOR 1"
        ),
        "iitd_min_imgs_per_palm": MIN_IMGS_PER_PALM,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Validate the just-written manifest
    import pandas as pd
    df = pd.read_csv(manifest_path, dtype={"session_id": str, "phase_id": str},
                      keep_default_na=False)
    validate_canonical_columns(df)
    validate_subject_split_disjointness(df)
    validate_palm_identity_consistency(df)
    validate_no_legacy_fields(df)
    validate_final_split_counts(df, metadata)
    validate_train_backbone_size(df, "iitd")

    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build IITD canonical manifest.")
    p.add_argument("--root", type=Path, required=True,
                   help="IITD V1 root containing Segmented/{Right,Left}/")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    manifest_path = build_iitd_manifest(args.root, args.out, seed=args.seed)
    print(f"wrote {manifest_path}")
    print(f"wrote {args.out / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
