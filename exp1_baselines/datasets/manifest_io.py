"""Canonical manifest CSV loader (D1f).

Single read path used by `train_arcface.py` (D1g), `extract_embeddings.py`
(D1h), and the score-matrix builder (D2). Replaces the legacy
`identity_split.json` + `sample_split.json` JSON pair.

Read contract:
  - The CSV MUST have exactly the canonical columns in
    `MANIFEST_COLUMNS` order; `validate_canonical_columns` asserts this.
  - `session_id` and `phase_id` are stored as strings (empty "" allowed);
    pandas can NaN-coerce empty cells, so we explicitly fill back to "".
  - `is_enroll` / `is_query` parse as bool; values "true"/"false" only.

Public helpers cover the four common access patterns in the pipeline:
  1. `get_train_records`     — backbone training images
  2. `get_val_ckpt_records`  — image-level holdout for checkpoint selection
  3. `get_eval_records`      — embedding-extraction targets
  4. `get_split_palm_ids`    — palm_ids in a given subject_split
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from exp1_baselines.datasets.manifest_schema import (
    EVAL_SUBJECT_SPLITS,
    MANIFEST_COLUMNS,
    SampleRole,
    SubjectSplit,
)
from exp1_baselines.datasets.validate_manifest import (
    validate_canonical_columns,
    validate_no_legacy_fields,
    validate_palm_identity_consistency,
    validate_subject_split_disjointness,
)


_BOOL_MAP = {"true": True, "false": False, "True": True, "False": False}


def _coerce_bool(series: pd.Series, col: str) -> pd.Series:
    out = series.map(_BOOL_MAP)
    if out.isna().any():
        bad = series[out.isna()].unique().tolist()
        raise ValueError(
            f"manifest column {col!r} has non-boolean values: {bad[:5]}"
        )
    return out.astype(bool)


def load_manifest(path: Path | str, *, validate: bool = True) -> pd.DataFrame:
    """Load and (by default) validate a canonical manifest CSV.

    Empty `session_id` / `phase_id` cells are normalized to "" rather than NaN.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        keep_default_na=False,
        na_values=[],
        dtype={
            "image_path": str, "dataset": str, "identity_id": str,
            "hand_side": str, "session_id": str, "phase_id": str,
            "subject_split": str, "sample_role": str, "roi_status": str,
            "is_enroll": str, "is_query": str,
            "subject_id": "Int64", "palm_id": "Int64", "sample_id": "Int64",
        },
    )
    df["is_enroll"] = _coerce_bool(df["is_enroll"], "is_enroll")
    df["is_query"] = _coerce_bool(df["is_query"], "is_query")
    df["subject_id"] = df["subject_id"].astype("int64")
    df["palm_id"] = df["palm_id"].astype("int64")
    df["sample_id"] = df["sample_id"].astype("int64")

    if validate:
        validate_canonical_columns(df)
        validate_subject_split_disjointness(df)
        validate_palm_identity_consistency(df)
        validate_no_legacy_fields(df)
    return df


def get_train_records(df: pd.DataFrame) -> pd.DataFrame:
    """Backbone training images: subject_split=train_backbone & sample_role=train."""
    mask = (
        (df["subject_split"] == SubjectSplit.TRAIN_BACKBONE.value)
        & (df["sample_role"] == SampleRole.TRAIN.value)
    )
    return df.loc[mask].reset_index(drop=True)


def get_val_ckpt_records(df: pd.DataFrame) -> pd.DataFrame:
    """Image-level holdout for checkpoint selection (NOT a separate subject set)."""
    mask = (
        (df["subject_split"] == SubjectSplit.TRAIN_BACKBONE.value)
        & (df["sample_role"] == SampleRole.VAL_CKPT.value)
    )
    return df.loc[mask].reset_index(drop=True)


def get_eval_records(
    df: pd.DataFrame,
    splits: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Rows whose subject_split is in the evaluation set.

    Default `splits = EVAL_SUBJECT_SPLITS` covers all four eval subsets
    (base_anchor, future, external_dev, external_test). Pass a narrower
    iterable to restrict (e.g. `["base_anchor", "future"]` for known queries).
    Rows with `sample_role=unused` are kept here — the score-matrix builder
    filters by role at construction time.
    """
    if splits is None:
        splits = EVAL_SUBJECT_SPLITS
    splits = list(splits)
    return df.loc[df["subject_split"].isin(splits)].reset_index(drop=True)


def get_split_palm_ids(df: pd.DataFrame, split_name: str) -> List[int]:
    """Sorted list of unique palm_ids belonging to one subject_split."""
    rows = df.loc[df["subject_split"] == split_name, "palm_id"].unique()
    return sorted(int(x) for x in rows)


def get_split_subject_ids(df: pd.DataFrame, split_name: str) -> List[int]:
    rows = df.loc[df["subject_split"] == split_name, "subject_id"].unique()
    return sorted(int(x) for x in rows)


def get_palm_ids_per_split(df: pd.DataFrame) -> dict:
    """All five subject_splits → sorted palm_id lists; convenience wrapper."""
    out = {}
    for split_name in (SubjectSplit.TRAIN_BACKBONE.value, *EVAL_SUBJECT_SPLITS):
        out[split_name] = get_split_palm_ids(df, split_name)
    return out


def filter_by_palm_ids(
    df: pd.DataFrame,
    palm_ids: Sequence[int],
    *,
    session_id: Optional[str] = None,
    sample_role: Optional[str] = None,
) -> pd.DataFrame:
    """Slice manifest rows by palm_id set, optionally further by session/role.

    Used by the score-matrix builder to assemble enroll/query subsets.
    """
    mask = df["palm_id"].isin(list(palm_ids))
    if session_id is not None:
        mask &= df["session_id"] == session_id
    if sample_role is not None:
        mask &= df["sample_role"] == sample_role
    return df.loc[mask].reset_index(drop=True)


__all__ = [
    "load_manifest",
    "get_train_records",
    "get_val_ckpt_records",
    "get_eval_records",
    "get_split_palm_ids",
    "get_split_subject_ids",
    "get_palm_ids_per_split",
    "filter_by_palm_ids",
    "MANIFEST_COLUMNS",
]
