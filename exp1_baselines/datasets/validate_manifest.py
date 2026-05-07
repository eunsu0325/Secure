"""Manifest validators (Implementation Enforcement Requirements: IER-1/2/5/6).

These validators run AFTER a builder produces (manifest CSV + metadata.json).
Each function fails-fast with an explicit ValueError on the first violation
so reviewers can trace the exact assertion.

Coverage:
  - IER-1  : Tongji P0a/P0b evidence fields recorded in metadata
  - IER-2  : final_split_counts in metadata equals actual CSV row counts
  - IER-5  : Tongji file-integrity fields recorded in metadata
             (file existence itself is asserted at build time)
  - IER-6  : train_backbone minimum-size guard (50% fail / 80% strong-warn)
  - structural: canonical column ordering, subject_split disjointness,
    identity_id↔palm_id consistency, no legacy 4-way split fields

Usage (typical):

    df = pd.read_csv(manifest_path)
    metadata = json.loads(metadata_path.read_text())
    validate_canonical_columns(df)
    validate_subject_split_disjointness(df)
    validate_palm_identity_consistency(df)
    validate_final_split_counts(df, metadata)
    validate_train_backbone_size(df, dataset_name="tongji")
    if metadata.get("dataset") == "tongji":
        validate_tongji_p0_metadata(metadata)
        validate_tongji_file_integrity_metadata(metadata)
"""
from __future__ import annotations

import warnings
from typing import Mapping

import pandas as pd

from exp1_baselines.datasets.manifest_schema import (
    EVAL_SUBJECT_SPLITS,
    MANIFEST_COLUMNS,
    SUBJECT_SPLIT_VALUES,
)
from exp1_baselines.datasets.split import DATASET_NOMINAL_TRAIN_BACKBONE


# ---------------------------------------------------------------------------
# Structural validators (run before any IER check)
# ---------------------------------------------------------------------------

def validate_canonical_columns(df: pd.DataFrame) -> None:
    """Header order must match MANIFEST_COLUMNS exactly."""
    actual = list(df.columns)
    if actual != MANIFEST_COLUMNS:
        missing = [c for c in MANIFEST_COLUMNS if c not in actual]
        extra = [c for c in actual if c not in MANIFEST_COLUMNS]
        raise ValueError(
            "manifest column mismatch:\n"
            f"  expected: {MANIFEST_COLUMNS}\n"
            f"  actual:   {actual}\n"
            f"  missing:  {missing}\n"
            f"  extra:    {extra}"
        )


def validate_subject_split_disjointness(df: pd.DataFrame) -> None:
    """No subject_id may appear in more than one subject_split."""
    bad = (
        df.groupby("subject_id")["subject_split"].nunique()
        .loc[lambda s: s > 1]
    )
    if not bad.empty:
        raise ValueError(
            f"subject_id appears in multiple subject_splits: "
            f"{bad.to_dict()} (this breaks identity-disjoint guarantees)"
        )

    unknown = sorted(set(df["subject_split"].unique()) - SUBJECT_SPLIT_VALUES)
    if unknown:
        raise ValueError(
            f"unknown subject_split values: {unknown} "
            f"(allowed: {sorted(SUBJECT_SPLIT_VALUES)})"
        )


def validate_palm_identity_consistency(df: pd.DataFrame) -> None:
    """Each palm_id must have a single identity_id and a single subject_id."""
    bad_id = (
        df.groupby("palm_id")["identity_id"].nunique()
        .loc[lambda s: s > 1]
    )
    if not bad_id.empty:
        raise ValueError(
            f"palm_id maps to multiple identity_ids: {bad_id.to_dict()}"
        )

    bad_subj = (
        df.groupby("palm_id")["subject_id"].nunique()
        .loc[lambda s: s > 1]
    )
    if not bad_subj.empty:
        raise ValueError(
            f"palm_id maps to multiple subject_ids: {bad_subj.to_dict()}"
        )


def validate_no_legacy_fields(df: pd.DataFrame) -> None:
    """Reject any legacy 4-way split residue (deprecated pattern check)."""
    deprecated = {
        "threshold_val", "threshold_gallery", "threshold_unknown",
        "dev_gallery", "dev_unknown",
    }
    found_in_split = deprecated & set(df["subject_split"].unique())
    if found_in_split:
        raise ValueError(
            f"deprecated subject_split values present: {sorted(found_in_split)} "
            f"(these are from the legacy 4-way split — use the canonical 5-way)"
        )


# ---------------------------------------------------------------------------
# IER-2: final_split_counts source-of-truth validator
# ---------------------------------------------------------------------------

def validate_final_split_counts(
    df: pd.DataFrame, metadata: Mapping[str, object]
) -> None:
    """metadata['final_split_counts'] must equal actual CSV-derived counts.

    Plan rule: paper tables, reporters, and downstream tooling read counts
    from metadata.json; if metadata disagrees with the CSV, all derived
    numbers become unreliable.
    """
    final = metadata.get("final_split_counts")
    if not isinstance(final, dict):
        raise ValueError(
            "metadata['final_split_counts'] missing or not a dict — required by IER-2"
        )

    expected_splits = {"train_backbone", *EVAL_SUBJECT_SPLITS}
    if set(final.keys()) != expected_splits:
        raise ValueError(
            f"final_split_counts keys mismatch: "
            f"got {sorted(final.keys())}, expected {sorted(expected_splits)}"
        )

    for split_name in expected_splits:
        rows = df[df["subject_split"] == split_name]
        actual_subjects = int(rows["subject_id"].nunique())
        actual_palms = int(rows["palm_id"].nunique())

        meta_entry = final[split_name]
        if not isinstance(meta_entry, dict):
            raise ValueError(
                f"final_split_counts[{split_name!r}] must be a dict, got {meta_entry!r}"
            )
        meta_subjects = int(meta_entry.get("subjects", -1))
        meta_palms = int(meta_entry.get("palms", -1))

        if actual_subjects != meta_subjects:
            raise ValueError(
                f"{split_name}: actual subjects={actual_subjects} != "
                f"final_split_counts.subjects={meta_subjects}"
            )
        if actual_palms != meta_palms:
            raise ValueError(
                f"{split_name}: actual palms={actual_palms} != "
                f"final_split_counts.palms={meta_palms}"
            )


# ---------------------------------------------------------------------------
# IER-6: train_backbone minimum-size guard
# ---------------------------------------------------------------------------

def validate_train_backbone_size(
    df: pd.DataFrame, dataset_name: str
) -> None:
    """Three-region guard for backbone-training pool size.

    Regions (per plan, IER-6):
      - actual < 50% of nominal  → ValueError (training underpowered)
      - 50% <= actual < 80%      → strong warning (caption-level disclosure
                                    required in result tables)
      - 80% <= actual < nominal  → soft note (logged but not warned)
      - actual >= nominal        → silent OK
    """
    nominal = DATASET_NOMINAL_TRAIN_BACKBONE.get(dataset_name)
    if nominal is None:
        raise ValueError(
            f"unknown dataset_name {dataset_name!r}; expected one of "
            f"{sorted(DATASET_NOMINAL_TRAIN_BACKBONE)}"
        )

    actual = int(
        df.loc[df["subject_split"] == "train_backbone", "subject_id"].nunique()
    )

    if actual < 0.5 * nominal:
        raise ValueError(
            f"{dataset_name}: train_backbone has {actual} subjects, "
            f"below 50% of nominal {nominal}. Backbone training would be "
            f"underpowered; protocol-diagnostic claim invalid."
        )
    if actual < 0.8 * nominal:
        warnings.warn(
            f"STRONG WARNING: {dataset_name} train_backbone {actual} "
            f"is 50-80% of nominal {nominal}. Result table caption MUST "
            f"explicitly state 'N_train={actual} eligible subjects after filtering'.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# IER-1: Tongji P0a/P0b evidence in metadata
# ---------------------------------------------------------------------------

REQUIRED_TONGJI_P0_FIELDS = (
    "tongji_p0a_status",
    "tongji_subject_pairing_rule",
    "tongji_p0a_evidence",
    "tongji_p0b_status",
    "tongji_p0b_reason",
    "tongji_identity_naming",
    "tongji_contralateral_enabled",
    "tongji_upo_enabled",
)


def validate_tongji_p0_metadata(metadata: Mapping[str, object]) -> None:
    """Tongji metadata must record P0a/P0b status + canonical Tier-2 values."""
    missing = [f for f in REQUIRED_TONGJI_P0_FIELDS if f not in metadata]
    if missing:
        raise ValueError(
            f"Tongji metadata missing P0 fields (IER-1): {missing}"
        )

    if metadata["tongji_p0a_status"] != "verified_inferred":
        raise ValueError(
            f"tongji_p0a_status must be 'verified_inferred' "
            f"(canonical spec adopts adjacent-pair P0a inference); "
            f"got {metadata['tongji_p0a_status']!r}"
        )
    if metadata["tongji_p0b_status"] != "unverified":
        raise ValueError(
            f"tongji_p0b_status must be 'unverified' (no public L/R label); "
            f"got {metadata['tongji_p0b_status']!r}"
        )
    if metadata["tongji_subject_pairing_rule"] != "subject_id = palm_id // 2":
        raise ValueError(
            f"tongji_subject_pairing_rule must be 'subject_id = palm_id // 2'; "
            f"got {metadata['tongji_subject_pairing_rule']!r}"
        )
    if metadata["tongji_upo_enabled"] is not False:
        raise ValueError(
            f"tongji_upo_enabled must be False (P0b unverified disables UPO); "
            f"got {metadata['tongji_upo_enabled']!r}"
        )

    evidence = metadata["tongji_p0a_evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(
            "tongji_p0a_evidence must be a non-empty list of strings"
        )


# ---------------------------------------------------------------------------
# IER-5: Tongji file-integrity metadata
# ---------------------------------------------------------------------------

REQUIRED_TONGJI_FILE_INTEGRITY_FIELDS = (
    "tongji_filename_normalization",
    "tongji_palm_id_formula",
    "tongji_session_filename_pairing",
    "tongji_file_integrity_check",
)


def validate_tongji_file_integrity_metadata(
    metadata: Mapping[str, object],
) -> None:
    """Tongji metadata must record per-session local numbering + integrity check."""
    missing = [
        f for f in REQUIRED_TONGJI_FILE_INTEGRITY_FIELDS if f not in metadata
    ]
    if missing:
        raise ValueError(
            f"Tongji metadata missing file-integrity fields (IER-5): {missing}"
        )

    if metadata["tongji_filename_normalization"] != "per_session_local":
        raise ValueError(
            f"tongji_filename_normalization must be 'per_session_local'; "
            f"got {metadata['tongji_filename_normalization']!r}"
        )
    if metadata["tongji_palm_id_formula"] != (
        "palm_id = (local_image_number - 1) // 10"
    ):
        raise ValueError(
            f"tongji_palm_id_formula mismatch; "
            f"got {metadata['tongji_palm_id_formula']!r}"
        )
