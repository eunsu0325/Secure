"""BJTU-V2 (ROI) manifest builder (D7b).

Sessioned dataset with F/S phase tokens in the filename. Per the plan
"Dataset Structure Verification (D1c reference)":

  BJTU_PalmV2(ROI)/
  ├── train/  (F-phase images)
  │   └── <subjectID>_F_<hand><rep>C<class>.JPG
  ├── test/   (S-phase images)
  │   └── <subjectID>_S_<hand><rep>C<class>.JPG
  ├── train_left.txt / train_right.txt / test_*.txt  (ignored — see plan)
  └── unknown_dev_file.txt / unknown_test_file.txt   (ignored — see plan)

Conventions (plan: Identity Convention + Sample Role Policy):
  - Tier 1 identity_id (hand_side from filename L/R token)
  - F = enrollment phase, S = query phase
  - palm_id = subject_id * 2 + (0 if L else 1) (matches contralateral_xor1)
  - sample_id = rep - 1 (zero-based within phase)
  - Eligibility: BOTH L AND R palms must each have ≥5 F-phase AND ≥5 S-phase imgs
  - train_backbone: F + S all used as training images; image-level val_ckpt holdout
    (last 20% of (phase, sample_id)-sorted images per palm)
  - base_anchor / future: F enroll for K=3 (enroll_seed=0 → sample_ids [0..K-1]),
    F unused for the rest, S all → query
  - external_dev / external_test: F unused, S all → query (sessioned externals
    use only the query phase, plan: Sample Role Policy)
  - The dataset-author split files (`unknown_dev_file.txt` etc.) are detected
    if present and recorded in metadata as INTENTIONALLY NOT USED.
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


MIN_IMGS_PER_PHASE_PALM = 5
BJTU_FILENAME_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<phase>[FS])_(?P<hand>[LR])(?P<rep>\d+)C(?P<cls>\d+)$"
)
BJTU_PREDEFINED_SPLIT_FILES = (
    "train_left.txt", "train_right.txt",
    "test_left.txt", "test_right.txt",
    "unknown_dev_file.txt", "unknown_test_file.txt",
)


def _scan_bjtu_folder(
    folder: Path,
) -> Dict[Tuple[int, str, str], List[Tuple[int, Path]]]:
    """Return {(file_subject, hand, phase): sorted [(rep, path)]}."""
    out: Dict[Tuple[int, str, str], List[Tuple[int, Path]]] = defaultdict(list)
    for p in folder.iterdir():
        if p.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        m = BJTU_FILENAME_RE.match(p.stem)
        if m is None:
            continue
        sub = int(m.group("subject"))
        hand = m.group("hand")
        phase = m.group("phase")
        rep = int(m.group("rep"))
        out[(sub, hand, phase)].append((rep, p))
    for key in out:
        out[key] = sorted(out[key], key=lambda x: x[0])
    return out


def _verify_bjtu_root(root: Path) -> Tuple[Path, Path, List[str]]:
    """Resolve train/, test/ folders and detect (but do not use) author splits."""
    if not root.is_dir():
        raise FileNotFoundError(f"BJTU root not a directory: {root}")
    train = root / "train"
    test = root / "test"
    if not train.is_dir() or not test.is_dir():
        raise FileNotFoundError(
            f"expected `train/` and `test/` under {root}; found "
            f"train={train.is_dir()}, test={test.is_dir()}"
        )
    detected = [name for name in BJTU_PREDEFINED_SPLIT_FILES
                if (root / name).exists()]
    return train, test, detected


def _val_ckpt_count(n: int) -> int:
    return max(1, n // 5)


def _row_for_image(
    *,
    image_path: Path,
    subject_id: int,
    palm_id: int,
    hand_side: str,
    phase_id: str,
    sample_id: int,
    subject_split: str,
    sample_role: str,
) -> ManifestRow:
    return ManifestRow(
        image_path=str(image_path),
        dataset="bjtu_v2",
        subject_id=subject_id,
        palm_id=palm_id,
        identity_id=make_identity_id(
            IdentityNamingTier.TIER1, subject_id, palm_id, hand_side
        ),
        hand_side=hand_side,
        session_id="",
        phase_id=phase_id,
        sample_id=sample_id,
        subject_split=subject_split,
        sample_role=sample_role,
        is_enroll=(sample_role == SampleRole.ENROLL.value),
        is_query=(sample_role == SampleRole.QUERY.value),
        roi_status="ok",
    )


def _emit_train_backbone_palm(
    *,
    f_imgs: Sequence[Tuple[int, Path]],
    s_imgs: Sequence[Tuple[int, Path]],
    subject_id: int,
    palm_id: int,
    hand_side: str,
) -> List[ManifestRow]:
    """train_backbone: all F+S imgs used; last 20% (phase-sorted) → val_ckpt."""
    flat: List[Tuple[str, int, Path]] = []  # (phase, sample_id, path)
    for rep_one_based, p in f_imgs:
        flat.append(("F", rep_one_based - 1, p))
    for rep_one_based, p in s_imgs:
        flat.append(("S", rep_one_based - 1, p))
    flat.sort(key=lambda x: (x[0], x[1]))
    n = len(flat)
    n_val = _val_ckpt_count(n)
    n_train = n - n_val
    rows: List[ManifestRow] = []
    for i, (phase, sid, p) in enumerate(flat):
        role = (
            SampleRole.TRAIN.value if i < n_train
            else SampleRole.VAL_CKPT.value
        )
        rows.append(_row_for_image(
            image_path=p, subject_id=subject_id, palm_id=palm_id,
            hand_side=hand_side, phase_id=phase, sample_id=sid,
            subject_split=SubjectSplit.TRAIN_BACKBONE.value,
            sample_role=role,
        ))
    return rows


def _emit_eval_sessioned_palm(
    *,
    f_imgs: Sequence[Tuple[int, Path]],
    s_imgs: Sequence[Tuple[int, Path]],
    subject_id: int,
    palm_id: int,
    hand_side: str,
    subject_split: str,
    enroll_sample_ids: Sequence[int],
    is_external: bool,
) -> List[ManifestRow]:
    """base_anchor / future / externals → F enroll/unused, S query."""
    rows: List[ManifestRow] = []
    enroll_set = set(int(x) for x in enroll_sample_ids)
    for rep_one_based, p in f_imgs:
        sid = rep_one_based - 1
        if is_external:
            role = SampleRole.UNUSED.value  # externals never enroll
        else:
            role = (
                SampleRole.ENROLL.value if sid in enroll_set
                else SampleRole.UNUSED.value
            )
        rows.append(_row_for_image(
            image_path=p, subject_id=subject_id, palm_id=palm_id,
            hand_side=hand_side, phase_id="F", sample_id=sid,
            subject_split=subject_split, sample_role=role,
        ))
    for rep_one_based, p in s_imgs:
        sid = rep_one_based - 1
        rows.append(_row_for_image(
            image_path=p, subject_id=subject_id, palm_id=palm_id,
            hand_side=hand_side, phase_id="S", sample_id=sid,
            subject_split=subject_split, sample_role=SampleRole.QUERY.value,
        ))
    return rows


def build_bjtu_manifest(
    root: Path,
    out_dir: Path,
    seed: int = 42,
    K: int = 3,
    enroll_seed: int = 0,
) -> Path:
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir, test_dir, predefined = _verify_bjtu_root(root)
    train_imgs = _scan_bjtu_folder(train_dir)
    test_imgs = _scan_bjtu_folder(test_dir)

    # Build a per-(subject, hand) view: F-imgs from train/, S-imgs from test/.
    # If a phase token appears in the wrong folder, surface it (we still
    # group by phase token from filename — folder is informational).
    from collections import defaultdict as _dd
    palm_index: Dict[Tuple[int, str], Dict[str, List[Tuple[int, Path]]]] = _dd(
        lambda: {"F": [], "S": []}
    )
    for (sub, hand, phase), entries in train_imgs.items():
        palm_index[(sub, hand)][phase].extend(entries)
    for (sub, hand, phase), entries in test_imgs.items():
        palm_index[(sub, hand)][phase].extend(entries)
    # Re-sort after merge
    for key in palm_index:
        for ph in ("F", "S"):
            palm_index[key][ph] = sorted(
                palm_index[key][ph], key=lambda x: x[0]
            )

    raw_subjects = sorted(set(s for (s, _) in palm_index.keys()))
    if not raw_subjects:
        raise RuntimeError(f"no BJTU images found under {root}")

    # Eligibility: BOTH L and R, each with ≥MIN F AND ≥MIN S
    eligible_subjects: List[int] = []
    excluded: Dict[str, str] = {}
    for sub in raw_subjects:
        reasons = []
        for hand in ("L", "R"):
            f = palm_index.get((sub, hand), {"F": [], "S": []})["F"]
            s = palm_index.get((sub, hand), {"F": [], "S": []})["S"]
            if len(f) < MIN_IMGS_PER_PHASE_PALM:
                reasons.append(f"{hand} has {len(f)} F imgs")
            if len(s) < MIN_IMGS_PER_PHASE_PALM:
                reasons.append(f"{hand} has {len(s)} S imgs")
        if reasons:
            excluded[str(sub)] = "; ".join(reasons)
        else:
            eligible_subjects.append(sub)

    # 0-based subject_ids in stable file-subject order
    file_to_subject_id = {f: i for i, f in enumerate(eligible_subjects)}

    split: SplitResult = subject_level_split(
        list(file_to_subject_id.values()),
        DATASET_QUOTAS["bjtu_v2"],
        seed=seed,
    )

    # Baseline enroll-sample selection (enroll_seed=0 → [0..K-1])
    if enroll_seed != 0:
        import numpy as np
        rng = np.random.default_rng(enroll_seed)
        # We can't use a global pool because BJTU palms have variable F-counts;
        # for the manifest baseline we record the global enroll_seed but
        # apply per-palm clipping to whatever F-pool that palm has.
        # The score-matrix builder reapplies this at runtime per IER-7.
        baseline_indices_global = list(range(K))
    baseline_enroll_sample_ids = list(range(K))

    rows: List[ManifestRow] = []
    for split_name in ("train_backbone", "base_anchor", "future",
                       "external_dev", "external_test"):
        for sid in split.assignments[split_name]:
            file_sub = eligible_subjects[sid]
            for hand_side, palm_offset, hand_token in (
                (HandSide.LEFT.value, 0, "L"),
                (HandSide.RIGHT.value, 1, "R"),
            ):
                palm_id = sid * 2 + palm_offset
                phase_imgs = palm_index[(file_sub, hand_token)]
                if split_name == SubjectSplit.TRAIN_BACKBONE.value:
                    rows += _emit_train_backbone_palm(
                        f_imgs=phase_imgs["F"], s_imgs=phase_imgs["S"],
                        subject_id=sid, palm_id=palm_id, hand_side=hand_side,
                    )
                else:
                    is_external = split_name in (
                        SubjectSplit.EXTERNAL_DEV.value,
                        SubjectSplit.EXTERNAL_TEST.value,
                    )
                    # Clip baseline to palm's actual F-pool size.
                    f_pool = len(phase_imgs["F"])
                    palm_enroll_ids = [
                        sid_ for sid_ in baseline_enroll_sample_ids
                        if sid_ < f_pool
                    ]
                    rows += _emit_eval_sessioned_palm(
                        f_imgs=phase_imgs["F"], s_imgs=phase_imgs["S"],
                        subject_id=sid, palm_id=palm_id, hand_side=hand_side,
                        subject_split=split_name,
                        enroll_sample_ids=palm_enroll_ids,
                        is_external=is_external,
                    )

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
        "dataset": "bjtu_v2",
        "schema_version": "exp1_canonical_v1",
        "split_seed": seed,
        "K": K,
        "enroll_seed_baseline": enroll_seed,
        "enroll_sample_ids_baseline_global": baseline_enroll_sample_ids,
        "raw_subject_count": len(raw_subjects),
        "eligible_subject_count": len(eligible_subjects),
        "excluded_subject_count": len(excluded),
        "exclusion_reason": excluded,
        "final_split_counts": final_split_counts,
        "bjtu_p0a_status": "verified_from_file_structure",
        "bjtu_p0b_status": "verified_from_file_structure",
        "bjtu_p0_evidence": (
            "L/R hand_side parsed from filename token; subject pairing "
            "from common <subjectID> across L and R filenames"
        ),
        "bjtu_identity_naming": "subject_<id>-l / subject_<id>-r (Tier 1)",
        "bjtu_phase_semantics": "F = enrollment, S = query (sessioned)",
        "bjtu_palm_id_formula": (
            "palm_id = subject_id * 2 + (0 if L else 1); "
            "contralateral via palm_id XOR 1"
        ),
        "bjtu_min_imgs_per_phase_palm": MIN_IMGS_PER_PHASE_PALM,
        "bjtu_predefined_split_used": False,
        "bjtu_predefined_split_files_present": predefined,
        "bjtu_split_policy_reason": (
            "canonical subject-level seed=42 split applied uniformly across "
            "Tongji/IITD/BJTU for protocol-diagnostic comparability"
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    import pandas as pd
    df = pd.read_csv(manifest_path, dtype={"session_id": str, "phase_id": str},
                      keep_default_na=False)
    validate_canonical_columns(df)
    validate_subject_split_disjointness(df)
    validate_palm_identity_consistency(df)
    validate_no_legacy_fields(df)
    validate_final_split_counts(df, metadata)
    validate_train_backbone_size(df, "bjtu_v2")

    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build BJTU-V2 canonical manifest.")
    p.add_argument("--root", type=Path, required=True,
                   help="BJTU_PalmV2(ROI) root containing train/ test/")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--enroll-seed", type=int, default=0)
    args = p.parse_args(argv)

    manifest_path = build_bjtu_manifest(
        root=args.root, out_dir=args.out, seed=args.seed,
        K=args.K, enroll_seed=args.enroll_seed,
    )
    print(f"wrote {manifest_path}")
    print(f"wrote {args.out / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
