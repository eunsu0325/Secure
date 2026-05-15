"""XJTU-UP-Huawei manifest builder (V19.10 Scenario A — third main dataset).

Produces canonical 5-way subject-level manifest for the Huawei-device subset
of XJTU-UP. Folder structure verified 2026-05-15 from local archive
`데이터셋/Huawei.zip等2个文件/huawei/`.

Dataset citation + license (per XJTU-UP authors' email to dataset user,
2026-05-15):
    Original dataset paper:
      Shao, Zhong, Du, "Effective Deep Ensemble Hashing for Open-Set
      Palmprint Recognition," Journal of Electronic Imaging, 29(1):013018,
      2020.

    The dataset authors request citation of the following recent
    palmprint papers from their group as well:
      [1] Shao, Shi, Du, Zeng, Zhong, "Robust Palmprint Recognition via
          Multi-stage Noisy Label Selection and Correction," IEEE TIP,
          2025, DOI 10.1109/TIP.2025.3588040.
      [2] Shao, Li, Zhong, "Generating Stylized Features for Single-Source
          Cross-Dataset Palmprint Recognition with Unseen Target Dataset,"
          IEEE TIP, vol. 33, pp. 4911-4922, 2024.
      [3] Shao, Zou, Liu, Guo, Zhong, "Learning to Generalize Unseen
          Dataset for Cross-Dataset Palmprint Recognition," IEEE TIFS,
          vol. 19, pp. 3788-3799, 2024.
      [4] Shao, Liu, Li, Zhong, "Privacy Preserving Palmprint Recognition
          via Federated Metric Learning," IEEE TIFS, vol. 19, pp. 878-891,
          2024.

    License: academic research only (per authors' email). Download link
    (authors-provided, 2026-05-15): https://share.weiyun.com/5CafBpq
    (password e9mz4d). The XJTU-UP project releases four sub-databases
    (iPhone 6s + HUAWEI, each with Nature and Flash conditions); the
    Huawei subset is what V19 main uses.

Dataset structure (verified 2026-05-15):
    huawei/
      Flash/               # acquisition condition 1: flash photography
        L_001/, L_002/, ..., L_100/     # left palms, 100 subjects
        R_001/, R_002/, ..., R_100/     # right palms, 100 subjects
        each folder: ~10 IMG_NNNN.JPG, 280×280 JPEG, RGB
      Nature/              # acquisition condition 2: natural lighting
        L_001/..L_100/, R_001/..R_100/
        same structure as Flash, same subjects (paired by id)

Identity / split conventions (plan §V19.10):
    - 100 subjects × 2 hands = 200 palm identities (Tier 1: hand_side
      verified from folder prefix L/R → identity_id = "subject_{N}-l"/"-r").
    - palm_id assignment: palm_id = 2*(subject_id - 1) + (0 if L else 1).
      i.e., (subj 1, L) → 0, (subj 1, R) → 1, (subj 2, L) → 2, etc.
      This mirrors Tongji's even=L / odd=R convention but with VERIFIED L/R
      (Tongji is Tier 2 with -a/-b because P0b unverified).
    - subject_id naming: extracted from folder name "L_NNN" or "R_NNN" → NNN.
    - session_id ∈ {Flash, Nature} — Tongji-style sessioned policy.
    - sample_id: assigned by sorting filenames lexicographically within each
      (subject, hand, condition) folder, then enumerating 0..N-1.

Sample role policy (Tongji-style cross-condition; plan §V19.10):
    train_backbone:
      sample_role = train      for sample_id 0..N-2 in BOTH conditions
      sample_role = val_ckpt   for sample_id N-1 in BOTH conditions
    base_anchor / future (enroll session = Flash):
      sample_role = enroll     for sample_id ∈ {0,1,2} of Flash (K=3 default)
      sample_role = unused     for sample_id ∈ {3..N-1} of Flash
      sample_role = query      for ALL Nature samples
    external_dev / external_test:
      sample_role = unused     for ALL Flash samples
      sample_role = query      for ALL Nature samples

Eligibility (per-subject):
    - Must have BOTH L and R folders in BOTH Flash AND Nature.
    - Each (subject, hand, condition) must have ≥ MIN_IMGS_PER_GROUP samples
      (default 5; matches IITD/BJTU eligibility thresholds).

Usage:
    python -m exp1_baselines.datasets.xjtu_up_huawei_manifest \\
        --root '/path/to/huawei' \\
        --out experiments/generated/xjtu_up_huawei_full \\
        --seed 42 --K 3 --enroll_seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

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


XJTU_UP_HUAWEI_DATASET_NAME = "xjtu_up_huawei"
XJTU_UP_HUAWEI_NUM_SUBJECTS_EXPECTED = 100
XJTU_UP_HUAWEI_CONDITIONS = ("Flash", "Nature")
PALM_FOLDER_RE = re.compile(r"^([LR])_(\d{3})$")
MIN_IMGS_PER_GROUP_DEFAULT = 5


def discover_palms(root: Path) -> Dict[Tuple[int, str, str], List[Path]]:
    """Walk root/{Flash,Nature}/{L,R}_NNN/ and collect image paths.

    Returns a dict keyed by (subject_id, hand_side, condition) → sorted list
    of image Paths. Missing conditions/hands are simply absent from the dict
    (caller decides eligibility).
    """
    out: Dict[Tuple[int, str, str], List[Path]] = {}
    for cond in XJTU_UP_HUAWEI_CONDITIONS:
        cond_dir = root / cond
        if not cond_dir.is_dir():
            raise FileNotFoundError(
                f"expected condition directory missing: {cond_dir}"
            )
        for sub in sorted(cond_dir.iterdir()):
            if not sub.is_dir():
                continue
            m = PALM_FOLDER_RE.match(sub.name)
            if not m:
                # ignore non-conforming names (e.g., hidden files)
                continue
            hand_letter = m.group(1)  # 'L' or 'R'
            subject_id = int(m.group(2))
            hand_side = HandSide.LEFT.value if hand_letter == "L" else HandSide.RIGHT.value
            imgs = sorted(
                p for p in sub.iterdir()
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".bmp", ".png")
            )
            out[(subject_id, hand_side, cond)] = imgs
    return out


def assert_xjtu_up_huawei_file_integrity(
    root: Path, *, min_imgs_per_group: int = MIN_IMGS_PER_GROUP_DEFAULT,
) -> Dict[str, int]:
    """Verify the discovered dataset shape and collect coverage metadata.

    Returns a dict with counts per (condition, hand_side) for the
    metadata.json file integrity record.
    """
    palms = discover_palms(root)
    counts: Dict[str, int] = {}
    for cond in XJTU_UP_HUAWEI_CONDITIONS:
        for hand in (HandSide.LEFT.value, HandSide.RIGHT.value):
            keys = [k for k in palms if k[1] == hand and k[2] == cond]
            counts[f"{cond}_{hand}_palm_folders"] = len(keys)
            total_imgs = sum(len(palms[k]) for k in keys)
            counts[f"{cond}_{hand}_total_images"] = total_imgs
    counts["distinct_subject_ids"] = len(set(k[0] for k in palms))
    counts["distinct_palms"] = len(set((k[0], k[1]) for k in palms))
    counts["distinct_groups"] = len(palms)  # (subj, hand, cond) tuples
    counts["total_images"] = sum(len(v) for v in palms.values())
    return counts


def determine_eligible_subjects(
    palms: Dict[Tuple[int, str, str], List[Path]],
    *,
    min_imgs_per_group: int = MIN_IMGS_PER_GROUP_DEFAULT,
) -> Tuple[List[int], Dict[int, str]]:
    """A subject is eligible iff BOTH L and R hands present in BOTH conditions,
    each group having ≥ min_imgs_per_group images.

    Returns (sorted eligible subject ids, exclusion reasons per ineligible subject).
    """
    all_subject_ids = sorted({k[0] for k in palms})
    eligible: List[int] = []
    exclusion_reason: Dict[int, str] = {}
    for sid in all_subject_ids:
        missing: List[str] = []
        too_few: List[str] = []
        for hand in (HandSide.LEFT.value, HandSide.RIGHT.value):
            for cond in XJTU_UP_HUAWEI_CONDITIONS:
                key = (sid, hand, cond)
                if key not in palms:
                    missing.append(f"{cond}/{hand}")
                elif len(palms[key]) < min_imgs_per_group:
                    too_few.append(f"{cond}/{hand}={len(palms[key])}")
        if missing or too_few:
            reasons = []
            if missing:
                reasons.append(f"missing={missing}")
            if too_few:
                reasons.append(f"too_few(min={min_imgs_per_group})={too_few}")
            exclusion_reason[sid] = "; ".join(reasons)
        else:
            eligible.append(sid)
    return eligible, exclusion_reason


def _palm_id_from(subject_id: int, hand_side: str) -> int:
    """palm_id assignment: even = L, odd = R within each subject.

    palm_id = 2*(subject_id - 1) + (0 if L else 1)
    subject 1: (0, L) (1, R)
    subject 2: (2, L) (3, R)
    ...
    """
    is_left = hand_side == HandSide.LEFT.value
    return 2 * (subject_id - 1) + (0 if is_left else 1)


def _assign_roles_for_palm(
    palm_image_paths: Dict[str, List[Path]],
    subject_split: str,
    *,
    K: int = 3,
) -> List[Tuple[Path, str, str, int]]:
    """Decide (path, session_id, sample_role, sample_id) tuples for one palm.

    Inputs:
        palm_image_paths: {condition_name: sorted list of image Paths}.
        subject_split: target subject_split for this palm.
        K: number of enroll prototypes per palm in base_anchor/future.

    Outputs:
        list of (path, session_id, sample_role, sample_id) — one per image.

    Role assignment per plan §V19.10 (Tongji-style cross-condition):
        train_backbone:
            train     for sample_id 0..N-2 in BOTH conditions
            val_ckpt  for sample_id N-1 in BOTH conditions
        base_anchor / future:
            enroll    for Flash sample_id ∈ {0..K-1}
            unused    for Flash sample_id ∈ {K..N-1}
            query     for ALL Nature
        external_dev / external_test:
            unused    for ALL Flash
            query     for ALL Nature
    """
    out: List[Tuple[Path, str, str, int]] = []
    for cond, imgs in palm_image_paths.items():
        for sample_id, img in enumerate(imgs):
            if subject_split == SubjectSplit.TRAIN_BACKBONE.value:
                # last image in each (palm, condition) is val_ckpt; rest are train
                if sample_id == len(imgs) - 1:
                    role = SampleRole.VAL_CKPT.value
                else:
                    role = SampleRole.TRAIN.value
            elif subject_split in (
                SubjectSplit.BASE_ANCHOR.value,
                SubjectSplit.FUTURE.value,
            ):
                if cond == "Flash":
                    role = (
                        SampleRole.ENROLL.value
                        if sample_id < K
                        else SampleRole.UNUSED.value
                    )
                else:  # Nature
                    role = SampleRole.QUERY.value
            elif subject_split in (
                SubjectSplit.EXTERNAL_DEV.value,
                SubjectSplit.EXTERNAL_TEST.value,
            ):
                if cond == "Flash":
                    role = SampleRole.UNUSED.value
                else:  # Nature
                    role = SampleRole.QUERY.value
            else:
                raise ValueError(f"unknown subject_split: {subject_split!r}")
            out.append((img, cond, role, sample_id))
    return out


def build_xjtu_up_huawei_manifest(
    root: Path,
    out_dir: Path,
    *,
    seed: int = 42,
    K: int = 3,
    enroll_seed: int = 0,  # accepted for API parity; XJTU-UP uses deterministic 0..K-1
    min_imgs_per_group: int = MIN_IMGS_PER_GROUP_DEFAULT,
) -> Dict:
    """Build the XJTU-UP-Huawei manifest + metadata files.

    Args:
        root: directory containing Flash/ and Nature/ subdirs.
        out_dir: output directory (created if missing). Writes manifest.csv +
            metadata.json.
        seed: deterministic seed for subject_level_split shuffle.
        K: number of enroll prototypes per palm in base_anchor/future.
        enroll_seed: kept for API parity; XJTU-UP simply uses sorted sample_id
            0..K-1 (sample_id is itself deterministic per filename sort).
        min_imgs_per_group: minimum images required per (subject, hand, cond)
            for that subject to be eligible.

    Returns:
        metadata dict (also written as metadata.json).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    file_integrity = assert_xjtu_up_huawei_file_integrity(
        root, min_imgs_per_group=min_imgs_per_group
    )

    palms = discover_palms(root)
    eligible, excl_reasons = determine_eligible_subjects(
        palms, min_imgs_per_group=min_imgs_per_group
    )
    if not eligible:
        raise RuntimeError(
            "no eligible subjects after applying eligibility filter; "
            f"exclusion reasons: {excl_reasons}"
        )

    quotas = DATASET_QUOTAS[XJTU_UP_HUAWEI_DATASET_NAME]
    split = subject_level_split(eligible, quotas, seed=seed)

    rows: List[ManifestRow] = []
    subj_to_split = {
        sid: split_name
        for split_name, subj_list in split.assignments.items()
        for sid in subj_list
    }
    for subject_id in eligible:
        subject_split_name = subj_to_split[subject_id]
        for hand in (HandSide.LEFT.value, HandSide.RIGHT.value):
            palm_id = _palm_id_from(subject_id, hand)
            identity_id = make_identity_id(
                IdentityNamingTier.TIER1, subject_id, palm_id, hand,
            )
            per_palm_paths = {
                cond: palms[(subject_id, hand, cond)]
                for cond in XJTU_UP_HUAWEI_CONDITIONS
            }
            for img_path, session_id, sample_role, sample_id in _assign_roles_for_palm(
                per_palm_paths, subject_split_name, K=K,
            ):
                rows.append(ManifestRow(
                    image_path=str(img_path),
                    dataset=XJTU_UP_HUAWEI_DATASET_NAME,
                    subject_id=subject_id,
                    palm_id=palm_id,
                    identity_id=identity_id,
                    hand_side=hand,
                    session_id=session_id,
                    phase_id="",
                    sample_id=sample_id,
                    subject_split=subject_split_name,
                    sample_role=sample_role,
                    is_enroll=(sample_role == SampleRole.ENROLL.value),
                    is_query=(sample_role == SampleRole.QUERY.value),
                    roi_status="ok",
                ))

    # Write manifest CSV
    manifest_csv = out_dir / "manifest.csv"
    with open(manifest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_dict())

    # Counts (final_split_counts is the source of truth; nested-dict format
    # per validate_final_split_counts — each split → {subjects, palms}).
    final_split_counts = {
        split_name: {
            "subjects": len(subj_list),
            "palms": 2 * len(subj_list),
        }
        for split_name, subj_list in split.assignments.items()
    }

    metadata = {
        "dataset": XJTU_UP_HUAWEI_DATASET_NAME,
        "plan_section": "V19.10 Scenario A (added 2026-05-15)",
        "seed": seed,
        "K": K,
        "enroll_seed": enroll_seed,
        "min_imgs_per_group": min_imgs_per_group,
        "file_integrity": file_integrity,
        "raw_subject_count": len({k[0] for k in palms}),
        "eligible_subject_count": len(eligible),
        "excluded_subject_count": len(excl_reasons),
        "exclusion_reasons": excl_reasons,
        "nominal_quotas": dict(quotas),
        "final_split_counts": final_split_counts,
        # XJTU-UP-Huawei specific evidence
        "xjtu_up_huawei_acquisition": (
            "Huawei smartphone subset of XJTU-UP (Shao et al. 2020). Two "
            "acquisition conditions (Flash, Nature) treated as Tongji-style "
            "session_id; identity is Tier 1 (subject_<id>-l/-r) since "
            "hand_side is verified from folder prefix L/R."
        ),
        "xjtu_up_huawei_sample_role_policy": (
            "train_backbone: pool BOTH conditions, last sample per (palm, cond) "
            "→ val_ckpt; base_anchor/future: Flash sample_id 0..K-1 = enroll, "
            "rest of Flash = unused, ALL Nature = query; external_dev/test: "
            "ALL Flash = unused, ALL Nature = query."
        ),
        "xjtu_up_huawei_resolution_note": (
            "Source ROI is 280x280 JPEG RGB; dataset loader resizes to 112x112 "
            "per V19.2 unified recipe."
        ),
        "spec_version": "V19.10",
    }
    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Validators take a pandas DataFrame; load now that the CSV is written.
    import pandas as pd
    df = pd.read_csv(manifest_csv)
    validate_canonical_columns(df)
    validate_subject_split_disjointness(df)
    validate_palm_identity_consistency(df)
    validate_no_legacy_fields(df)
    validate_final_split_counts(df, metadata)
    validate_train_backbone_size(df, XJTU_UP_HUAWEI_DATASET_NAME)

    print(f"[xjtu_up_huawei_manifest] wrote {manifest_csv} ({len(rows)} rows)")
    print(f"[xjtu_up_huawei_manifest] wrote {metadata_path}")
    print(f"[xjtu_up_huawei_manifest] final_split_counts:")
    for split_name, counts in final_split_counts.items():
        print(f"    {split_name}: {counts['subjects']} subjects, {counts['palms']} palms")
    return metadata


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="path to huawei/ root (contains Flash/ and Nature/)")
    parser.add_argument("--out", required=True, type=Path,
                        help="output dir for manifest.csv + metadata.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=3,
                        help="enroll prototypes per palm")
    parser.add_argument("--enroll_seed", type=int, default=0,
                        help="enroll sample seed (kept for API parity)")
    parser.add_argument("--min_imgs_per_group", type=int,
                        default=MIN_IMGS_PER_GROUP_DEFAULT,
                        help="minimum imgs per (subj, hand, cond) group")
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"[err] root not a directory: {args.root}", file=sys.stderr)
        return 2

    build_xjtu_up_huawei_manifest(
        args.root,
        args.out,
        seed=args.seed,
        K=args.K,
        enroll_seed=args.enroll_seed,
        min_imgs_per_group=args.min_imgs_per_group,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
