"""Canonical manifest CSV schema for Exp1 (protocol-diagnostic experiment).

Single source of truth for column ordering, value vocabularies, and the
identity-id naming policy. All manifest builders (Tongji/IITD/BJTU) emit rows
that conform to `ManifestRow`; all readers (`manifest_io.py`) parse the same
canonical CSV header in `MANIFEST_COLUMNS` order.

Schema rationale (see plan: Manifest Schema UPDATE):
  - `subject_id`/`palm_id` separate the partitioning unit from the recognition
    class (Tongji: `subject_id = palm_id // 2` under VERIFIED-INFERRED P0a).
  - `identity_id` follows a 3-tier naming policy keyed off P0a/P0b verification
    status — see `IdentityNamingTier` and `make_identity_id`.
  - `session_id`/`phase_id`/`sample_id` together support enroll-seed-driven
    K=3 prototype reproduction without parsing image paths (IER-7).
  - `subject_split` × `sample_role` together drive train/val/eval routing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import List, Sequence


MANIFEST_COLUMNS: List[str] = [
    "image_path",
    "dataset",
    "subject_id",
    "palm_id",
    "identity_id",
    "hand_side",
    "session_id",
    "phase_id",
    "sample_id",
    "subject_split",
    "sample_role",
    "is_enroll",
    "is_query",
    "roi_status",
]


class SubjectSplit(str, Enum):
    TRAIN_BACKBONE = "train_backbone"
    BASE_ANCHOR = "base_anchor"
    FUTURE = "future"
    EXTERNAL_DEV = "external_dev"
    EXTERNAL_TEST = "external_test"


SUBJECT_SPLIT_VALUES = frozenset(s.value for s in SubjectSplit)
EVAL_SUBJECT_SPLITS: List[str] = [
    SubjectSplit.BASE_ANCHOR.value,
    SubjectSplit.FUTURE.value,
    SubjectSplit.EXTERNAL_DEV.value,
    SubjectSplit.EXTERNAL_TEST.value,
]


class SampleRole(str, Enum):
    TRAIN = "train"
    VAL_CKPT = "val_ckpt"
    ENROLL = "enroll"
    QUERY = "query"
    # `candidate` is for non-sessioned datasets (IITD): role assigned at
    # score-matrix time per enroll_seed (IER-7), not at manifest creation.
    CANDIDATE = "candidate"
    UNUSED = "unused"


SAMPLE_ROLE_VALUES = frozenset(s.value for s in SampleRole)


class HandSide(str, Enum):
    LEFT = "l"
    RIGHT = "r"
    UNKNOWN = "unknown"


HAND_SIDE_VALUES = frozenset(s.value for s in HandSide)


class IdentityNamingTier(str, Enum):
    """3-tier identity-id naming policy (plan: Identity Convention).

    TIER1 — P0a✓ + P0b✓ (subject pairing AND L/R label verified):
        identity_id = "subject_<id>-l" / "subject_<id>-r"
        Used by IITD (Right/Left folder) and BJTU-V2 (L/R filename token).

    TIER2 — P0a✓ + P0b✗ (pairing yes, L/R unknown):
        identity_id = "subject_<id>-a" / "subject_<id>-b"
        -a = even palm_id, -b = odd palm_id (no L/R claim).
        Used by Tongji (P0a VERIFIED-INFERRED, P0b UNVERIFIED).

    TIER3 — P0a✗ (no pairing info):
        identity_id = "palm_<palm_id>"
        Fallback only.
    """
    TIER1 = "tier1_subject_lr"
    TIER2 = "tier2_subject_ab"
    TIER3 = "tier3_palm"


class P0aStatus(str, Enum):
    VERIFIED = "verified"
    VERIFIED_INFERRED = "verified_inferred"
    UNVERIFIED = "unverified"


class P0bStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


def make_identity_id(
    tier: IdentityNamingTier,
    subject_id: int,
    palm_id: int,
    hand_side: str,
) -> str:
    """Build canonical identity_id per the 3-tier naming policy.

    Tier 2 uses palm_id parity (-a even / -b odd) — NEVER guesses L/R.
    Tier 1 requires `hand_side` ∈ {l, r}; raises if "unknown".
    """
    if tier is IdentityNamingTier.TIER1:
        if hand_side not in (HandSide.LEFT.value, HandSide.RIGHT.value):
            raise ValueError(
                f"Tier 1 identity_id requires verified hand_side l/r, got "
                f"{hand_side!r} for subject_id={subject_id} palm_id={palm_id}"
            )
        return f"subject_{subject_id}-{hand_side}"
    if tier is IdentityNamingTier.TIER2:
        suffix = "a" if (palm_id % 2 == 0) else "b"
        return f"subject_{subject_id}-{suffix}"
    if tier is IdentityNamingTier.TIER3:
        return f"palm_{palm_id}"
    raise ValueError(f"unknown IdentityNamingTier: {tier!r}")


def assert_canonical_header(header: Sequence[str]) -> None:
    """Validate that a CSV header matches the canonical column order exactly."""
    actual = list(header)
    if actual != MANIFEST_COLUMNS:
        missing = [c for c in MANIFEST_COLUMNS if c not in actual]
        extra = [c for c in actual if c not in MANIFEST_COLUMNS]
        raise ValueError(
            "manifest header mismatch.\n"
            f"  expected: {MANIFEST_COLUMNS}\n"
            f"  actual:   {actual}\n"
            f"  missing:  {missing}\n"
            f"  extra:    {extra}"
        )


@dataclass
class ManifestRow:
    """One image row in a manifest CSV.

    Field types are intentionally narrow: `subject_id`/`palm_id`/`sample_id`
    are ints (not strings) so consumers can do arithmetic / set ops directly;
    `is_enroll`/`is_query` are bools. CSV writers serialize via `to_csv_dict`
    which renders bools as "true"/"false" and empty `session_id`/`phase_id`
    as the empty string (per per-dataset semantics).
    """

    image_path: str
    dataset: str
    subject_id: int
    palm_id: int
    identity_id: str
    hand_side: str       # one of HAND_SIDE_VALUES
    session_id: str      # "" when not applicable (IITD, BJTU)
    phase_id: str        # "" when not applicable (Tongji, IITD)
    sample_id: int
    subject_split: str   # one of SUBJECT_SPLIT_VALUES
    sample_role: str     # one of SAMPLE_ROLE_VALUES
    is_enroll: bool
    is_query: bool
    roi_status: str      # e.g., "ok", "missing", "corrupt" — set by builder

    def __post_init__(self) -> None:
        if self.hand_side not in HAND_SIDE_VALUES:
            raise ValueError(
                f"hand_side {self.hand_side!r} not in {sorted(HAND_SIDE_VALUES)}"
            )
        if self.subject_split not in SUBJECT_SPLIT_VALUES:
            raise ValueError(
                f"subject_split {self.subject_split!r} not in "
                f"{sorted(SUBJECT_SPLIT_VALUES)}"
            )
        if self.sample_role not in SAMPLE_ROLE_VALUES:
            raise ValueError(
                f"sample_role {self.sample_role!r} not in "
                f"{sorted(SAMPLE_ROLE_VALUES)}"
            )
        if self.sample_id < 0:
            raise ValueError(f"sample_id must be >= 0, got {self.sample_id}")
        if self.palm_id < 0:
            raise ValueError(f"palm_id must be >= 0, got {self.palm_id}")
        if self.subject_id < 0:
            raise ValueError(f"subject_id must be >= 0, got {self.subject_id}")

    def to_csv_dict(self) -> dict:
        """Render this row as a dict keyed by canonical column names.

        Bools render as "true"/"false"; ints stringify directly. Empty
        `session_id`/`phase_id` remain empty strings. Order is the canonical
        column order (Python dicts preserve insertion order).
        """
        d = asdict(self)
        return {
            col: (
                "true" if d[col] is True
                else "false" if d[col] is False
                else str(d[col])
            )
            for col in MANIFEST_COLUMNS
        }
