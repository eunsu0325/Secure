"""Tongji manifest builder (D1d) — replaces legacy build_tongji_for_baselines.py.

Produces canonical 5-way subject-level manifest per the Exp1 spec:

  out_dir/
    manifest.csv     : 12000 rows in MANIFEST_COLUMNS order
    metadata.json    : P0 evidence (IER-1), file integrity (IER-5),
                       final_split_counts (IER-2 source-of-truth)

Tongji conventions (plan: Identity Convention + IER-5):
  - 600 palms, 300 subjects (P0a VERIFIED-INFERRED: subject_id = palm_id // 2)
  - palm_id  = (local_image_number - 1) // 10   in 0..599
  - sample_id = (local_image_number - 1) % 10    in 0..9
  - sessions: session1 = enroll session, session2 = query session
  - identity_id (Tier 2): subject_<id>-a (palm_id even) / -b (palm_id odd)
  - hand_side = 'unknown' (P0b not verified from public docs)

Sample role policy (sessioned, plan: Sample Role Policy):
  train_backbone:
    sample_role = train     for sample_id 0..8 in BOTH sessions (18 imgs/palm)
    sample_role = val_ckpt  for sample_id 9   in BOTH sessions (2 imgs/palm)
  base_anchor / future (enroll session = session1):
    sample_role = enroll    for sample_id ∈ {0,1,2}  (K=3 default; enroll_seed=0)
    sample_role = unused    for sample_id ∈ {3..9}
    sample_role = query     for ALL session2 imgs (10/palm)
  external_dev / external_test:
    sample_role = unused    for ALL session1 imgs
    sample_role = query     for ALL session2 imgs

is_enroll / is_query are derived directly from sample_role at write-time,
so they always agree with the role and need no separate logic.

Usage:
    python -m exp1_baselines.datasets.tongji_manifest \
        --root /path/to/Tongji_ROI --out experiments/generated/tongji \
        --seed 42 --K 3 --enroll-seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    validate_tongji_file_integrity_metadata,
    validate_tongji_p0_metadata,
    validate_train_backbone_size,
)


# Tongji constants (verified from public documentation + IER-5 file integrity)
TONGJI_NUM_PALMS = 600
TONGJI_NUM_SUBJECTS = 300
TONGJI_IMGS_PER_SESSION_PER_PALM = 10
TONGJI_TOTAL_IMGS_PER_SESSION = 6000  # 600 palms × 10 imgs
TONGJI_FILENAME_FMT = "{:05d}.bmp"

TONGJI_P0A_EVIDENCE: List[str] = [
    "official Tongji ContactlessPalm page reports 300 subjects + 600 palms",
    "official page states 10 sequential image numbers per palm "
    "('00001~00010 first palm, 00011~00020 second palm')",
    "2 palms per subject implies adjacent palm_id pairing as standard inference; "
    "README inspection at data acquisition can elevate to plain VERIFIED",
]


def assert_tongji_file_integrity(root: Path) -> Dict[str, int]:
    """Per IER-5: verify both sessions contain all 6000 expected files."""
    counts: Dict[str, int] = {}
    for sess in ("session1", "session2"):
        sess_dir = root / sess
        if not sess_dir.is_dir():
            raise FileNotFoundError(
                f"Tongji session directory missing: {sess_dir}"
            )
        bmps = sorted(p for p in sess_dir.iterdir() if p.suffix == ".bmp")
        counts[sess] = len(bmps)
        if len(bmps) != TONGJI_TOTAL_IMGS_PER_SESSION:
            raise ValueError(
                f"{sess_dir}: expected {TONGJI_TOTAL_IMGS_PER_SESSION} .bmp files, "
                f"found {len(bmps)} (IER-5 file count assertion failed)"
            )
        # Filename existence: every 1..6000 must exist
        for n in range(1, TONGJI_TOTAL_IMGS_PER_SESSION + 1):
            expected = sess_dir / TONGJI_FILENAME_FMT.format(n)
            if not expected.exists():
                raise FileNotFoundError(
                    f"missing Tongji file {expected} (IER-5 filename existence)"
                )
    return counts


def palm_id_from_filename_number(local_n: int) -> int:
    return (local_n - 1) // TONGJI_IMGS_PER_SESSION_PER_PALM


def sample_id_from_filename_number(local_n: int) -> int:
    return (local_n - 1) % TONGJI_IMGS_PER_SESSION_PER_PALM


def palm_ids_for_subject(subject_id: int) -> Tuple[int, int]:
    """P0a VERIFIED-INFERRED: subject owns adjacent palm_ids (2k, 2k+1)."""
    return (subject_id * 2, subject_id * 2 + 1)


def select_enroll_sample_ids(K: int, enroll_seed: int) -> List[int]:
    """Pick K enroll sample_ids out of session1's [0..9] (sessioned policy).

    enroll_seed=0 returns the deterministic floor [0..K-1] (paper main).
    enroll_seed>=1 deterministically shuffles [0..9] and takes the first K.
    """
    if not 1 <= K <= TONGJI_IMGS_PER_SESSION_PER_PALM:
        raise ValueError(f"K must be in 1..10, got {K}")
    if enroll_seed == 0:
        return list(range(K))
    rng = np.random.default_rng(enroll_seed)
    shuffled = np.array(range(TONGJI_IMGS_PER_SESSION_PER_PALM))
    rng.shuffle(shuffled)
    return sorted(int(x) for x in shuffled[:K])


def _row_for_image(
    *,
    root: Path,
    session: str,
    sample_id: int,
    palm_id: int,
    subject_id: int,
    subject_split: str,
    sample_role: str,
) -> ManifestRow:
    local_n = palm_id * TONGJI_IMGS_PER_SESSION_PER_PALM + sample_id + 1
    rel = f"{session}/{TONGJI_FILENAME_FMT.format(local_n)}"
    abs_path = root / rel
    return ManifestRow(
        image_path=str(abs_path),
        dataset="tongji",
        subject_id=subject_id,
        palm_id=palm_id,
        identity_id=make_identity_id(
            IdentityNamingTier.TIER2, subject_id, palm_id, HandSide.UNKNOWN.value
        ),
        hand_side=HandSide.UNKNOWN.value,
        session_id=session,
        phase_id="",
        sample_id=sample_id,
        subject_split=subject_split,
        sample_role=sample_role,
        is_enroll=(sample_role == SampleRole.ENROLL.value),
        is_query=(sample_role == SampleRole.QUERY.value),
        roi_status="ok",
    )


def _emit_train_backbone(
    *, root: Path, subject_ids: Sequence[int]
) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    for sid in subject_ids:
        for palm_id in palm_ids_for_subject(sid):
            for session in ("session1", "session2"):
                for sample_id in range(TONGJI_IMGS_PER_SESSION_PER_PALM):
                    role = (
                        SampleRole.VAL_CKPT.value if sample_id == 9
                        else SampleRole.TRAIN.value
                    )
                    rows.append(_row_for_image(
                        root=root, session=session, sample_id=sample_id,
                        palm_id=palm_id, subject_id=sid,
                        subject_split=SubjectSplit.TRAIN_BACKBONE.value,
                        sample_role=role,
                    ))
    return rows


def _emit_base_or_future(
    *,
    root: Path,
    subject_ids: Sequence[int],
    subject_split: str,
    enroll_sample_ids: Sequence[int],
) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    enroll_set = set(enroll_sample_ids)
    for sid in subject_ids:
        for palm_id in palm_ids_for_subject(sid):
            for sample_id in range(TONGJI_IMGS_PER_SESSION_PER_PALM):
                role = (
                    SampleRole.ENROLL.value if sample_id in enroll_set
                    else SampleRole.UNUSED.value
                )
                rows.append(_row_for_image(
                    root=root, session="session1", sample_id=sample_id,
                    palm_id=palm_id, subject_id=sid,
                    subject_split=subject_split, sample_role=role,
                ))
            for sample_id in range(TONGJI_IMGS_PER_SESSION_PER_PALM):
                rows.append(_row_for_image(
                    root=root, session="session2", sample_id=sample_id,
                    palm_id=palm_id, subject_id=sid,
                    subject_split=subject_split,
                    sample_role=SampleRole.QUERY.value,
                ))
    return rows


def _emit_external(
    *,
    root: Path,
    subject_ids: Sequence[int],
    subject_split: str,
) -> List[ManifestRow]:
    """external_dev / external_test: session1 unused, session2 query (all)."""
    rows: List[ManifestRow] = []
    for sid in subject_ids:
        for palm_id in palm_ids_for_subject(sid):
            for sample_id in range(TONGJI_IMGS_PER_SESSION_PER_PALM):
                rows.append(_row_for_image(
                    root=root, session="session1", sample_id=sample_id,
                    palm_id=palm_id, subject_id=sid,
                    subject_split=subject_split,
                    sample_role=SampleRole.UNUSED.value,
                ))
            for sample_id in range(TONGJI_IMGS_PER_SESSION_PER_PALM):
                rows.append(_row_for_image(
                    root=root, session="session2", sample_id=sample_id,
                    palm_id=palm_id, subject_id=sid,
                    subject_split=subject_split,
                    sample_role=SampleRole.QUERY.value,
                ))
    return rows


def build_tongji_manifest(
    root: Path,
    out_dir: Path,
    seed: int = 42,
    K: int = 3,
    enroll_seed: int = 0,
    *,
    file_integrity_check: bool = True,
) -> Path:
    """Build the canonical Tongji manifest CSV + metadata.json.

    Args:
        root: directory containing `session1/` and `session2/` BMPs.
        out_dir: target directory (created if missing).
        seed: split RNG seed (plan default 42).
        K: enrollment images per palm (plan main = 3).
        enroll_seed: which K sample_ids form the enroll baseline.
            0 → deterministic [0..K-1]; ≥1 → deterministic shuffled pick.
        file_integrity_check: pass-through for tests; production keeps True.

    Returns:
        Path to the written manifest.csv.
    """
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_counts: Dict[str, int] = {}
    if file_integrity_check:
        file_counts = assert_tongji_file_integrity(root)

    # Subject pool: 0..299 (full Tongji eligibility under VERIFIED-INFERRED P0a)
    eligible_subject_ids = list(range(TONGJI_NUM_SUBJECTS))
    split: SplitResult = subject_level_split(
        eligible_subject_ids, DATASET_QUOTAS["tongji"], seed=seed,
    )

    enroll_sample_ids = select_enroll_sample_ids(K=K, enroll_seed=enroll_seed)

    rows: List[ManifestRow] = []
    rows += _emit_train_backbone(
        root=root, subject_ids=split.assignments["train_backbone"],
    )
    rows += _emit_base_or_future(
        root=root, subject_ids=split.assignments["base_anchor"],
        subject_split=SubjectSplit.BASE_ANCHOR.value,
        enroll_sample_ids=enroll_sample_ids,
    )
    rows += _emit_base_or_future(
        root=root, subject_ids=split.assignments["future"],
        subject_split=SubjectSplit.FUTURE.value,
        enroll_sample_ids=enroll_sample_ids,
    )
    rows += _emit_external(
        root=root, subject_ids=split.assignments["external_dev"],
        subject_split=SubjectSplit.EXTERNAL_DEV.value,
    )
    rows += _emit_external(
        root=root, subject_ids=split.assignments["external_test"],
        subject_split=SubjectSplit.EXTERNAL_TEST.value,
    )

    expected_total = (
        TONGJI_NUM_PALMS * TONGJI_IMGS_PER_SESSION_PER_PALM * 2
    )
    if len(rows) != expected_total:
        raise AssertionError(
            f"emitted {len(rows)} rows, expected {expected_total} "
            f"(600 palms × 10 imgs × 2 sessions)"
        )

    # Write CSV
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_csv_dict())

    # Build metadata.json
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
        "dataset": "tongji",
        "schema_version": "exp1_canonical_v1",
        "split_seed": seed,
        "K": K,
        "enroll_seed_baseline": enroll_seed,
        "enroll_sample_ids_baseline": enroll_sample_ids,
        "raw_subject_count": TONGJI_NUM_SUBJECTS,
        "eligible_subject_count": TONGJI_NUM_SUBJECTS,
        "excluded_subject_count": 0,
        "exclusion_reason": {},
        "final_split_counts": final_split_counts,
        # IER-1: P0 evidence
        "tongji_p0a_status": "verified_inferred",
        "tongji_subject_pairing_rule": "subject_id = palm_id // 2",
        "tongji_p0a_evidence": TONGJI_P0A_EVIDENCE,
        "tongji_p0b_status": "unverified",
        "tongji_p0b_reason": (
            "public Tongji documentation does not specify which palm_id is "
            "left vs right hand; multiple searches confirmed gap"
        ),
        "tongji_identity_naming": (
            "subject_<id>-a / subject_<id>-b (Tier 2; -a = even palm_id, "
            "-b = odd palm_id; no L/R claim)"
        ),
        "tongji_contralateral_enabled": True,
        "tongji_upo_enabled": False,
        # IER-5: file integrity
        "tongji_filename_normalization": "per_session_local",
        "tongji_palm_id_formula": "palm_id = (local_image_number - 1) // 10",
        "tongji_session_filename_pairing": (
            "same filename across sessions = same palm_id"
        ),
        "tongji_file_integrity_check": (
            "passed" if file_integrity_check else "skipped (test mode)"
        ),
        "tongji_file_counts": file_counts,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Run validators on the just-written manifest before returning
    import pandas as pd  # local import keeps module-import cheap
    df = pd.read_csv(manifest_path, dtype={"session_id": str, "phase_id": str})
    df["session_id"] = df["session_id"].fillna("")
    df["phase_id"] = df["phase_id"].fillna("")

    validate_canonical_columns(df)
    validate_subject_split_disjointness(df)
    validate_palm_identity_consistency(df)
    validate_no_legacy_fields(df)
    validate_final_split_counts(df, metadata)
    validate_train_backbone_size(df, "tongji")
    validate_tongji_p0_metadata(metadata)
    validate_tongji_file_integrity_metadata(metadata)

    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build Tongji canonical manifest.")
    p.add_argument("--root", type=Path, required=True,
                   help="Tongji_ROI directory containing session1/ session2/")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for manifest.csv + metadata.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--enroll-seed", type=int, default=0)
    p.add_argument(
        "--skip-file-integrity", action="store_true",
        help="Skip IER-5 file existence checks (test mode only)",
    )
    args = p.parse_args(argv)

    manifest_path = build_tongji_manifest(
        root=args.root, out_dir=args.out,
        seed=args.seed, K=args.K, enroll_seed=args.enroll_seed,
        file_integrity_check=not args.skip_file_integrity,
    )
    print(f"wrote {manifest_path}")
    print(f"wrote {args.out / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
