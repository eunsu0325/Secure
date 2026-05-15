"""Subject-level eligibility-first split for Exp1 manifests.

Implements the canonical 5-way subject split (plan: Identity Split + Eligibility
policy):

  evaluation quotas (allocated first, fail-fast if pool too small):
    - base_anchor    : always-enrolled gallery
    - future         : sequentially enrolled palms (Protocol C trajectory)
    - external_dev   : τ calibration probes (never accept-eligible)
    - external_test  : never-enrolled probes (FPIR evaluation)

  train_backbone (residual after evaluation allocation):
    - all remaining eligible subjects → ArcFace backbone training

The split operates on **subject_ids** (the partitioning unit). Manifest builders
expand each subject_id into its constituent palm_ids. Pure-function, no IO; all
randomness derives from `seed` via `np.random.default_rng` for cross-version
bit-stream stability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

import numpy as np


EVAL_QUOTA_KEYS: List[str] = [
    "base_anchor",
    "future",
    "external_dev",
    "external_test",
]


@dataclass(frozen=True)
class SplitResult:
    """Subject-level split assignment, plus diagnostic counts.

    `assignments` is the authoritative output: split_name → sorted list of
    subject_ids. The counts dict mirrors the assignment sizes for convenience
    when populating `metadata.json["final_split_counts"]`.
    """

    assignments: Dict[str, List[int]]
    counts: Dict[str, int]
    seed: int

    def all_subject_ids(self) -> List[int]:
        out: List[int] = []
        for key in ("train_backbone", *EVAL_QUOTA_KEYS):
            out.extend(self.assignments[key])
        return out


def subject_level_split(
    eligible_subject_ids: Sequence[int],
    quotas: Mapping[str, int],
    seed: int = 42,
) -> SplitResult:
    """Allocate eligible subjects into the 5-way canonical split.

    Args:
        eligible_subject_ids: subject_ids that passed the per-dataset
            eligibility filter (e.g., for IITD: subject has BOTH L and R
            palms with ≥5 imgs each). MUST be unique.
        quotas: dict with keys EVAL_QUOTA_KEYS giving the count of subjects
            assigned to each evaluation split. `train_backbone` is NOT in
            quotas — it absorbs the residual.
        seed: RNG seed for deterministic shuffling. Plan default = 42.

    Returns:
        SplitResult with `assignments` keyed by all five split names and
        all subject_id lists sorted ascending.

    Raises:
        ValueError: on any of:
          - duplicate subject_ids
          - negative quota
          - missing/extra keys in `quotas`
          - eligible pool smaller than the sum of evaluation quotas
            (eligibility-first allocation: NEVER silently shrink quotas)
    """
    pool = list(eligible_subject_ids)
    if len(set(pool)) != len(pool):
        raise ValueError("eligible_subject_ids contains duplicates")

    expected = set(EVAL_QUOTA_KEYS)
    given = set(quotas.keys())
    if given != expected:
        missing = expected - given
        extra = given - expected
        raise ValueError(
            f"quotas keys mismatch: expected {sorted(expected)}, "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for k, v in quotas.items():
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"quota {k!r} must be non-negative int, got {v!r}")

    eval_total = sum(quotas[k] for k in EVAL_QUOTA_KEYS)
    if eval_total > len(pool):
        raise ValueError(
            f"eligible pool too small for evaluation quotas: "
            f"eligible={len(pool)} subjects, evaluation_total={eval_total} "
            f"(quotas={dict(quotas)}). Eligibility-first allocation forbids "
            f"silent quota shrinkage — increase the eligible pool or reduce quotas."
        )

    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(pool), dtype=np.int64)
    rng.shuffle(shuffled)

    assignments: Dict[str, List[int]] = {}
    cursor = 0
    for key in EVAL_QUOTA_KEYS:
        n = quotas[key]
        assignments[key] = sorted(int(x) for x in shuffled[cursor : cursor + n])
        cursor += n
    assignments["train_backbone"] = sorted(int(x) for x in shuffled[cursor:])

    all_assigned = [sid for v in assignments.values() for sid in v]
    if len(set(all_assigned)) != len(all_assigned):
        raise AssertionError("internal split error: subject_id appeared in multiple splits")
    if set(all_assigned) != set(pool):
        raise AssertionError("internal split error: assignment != eligible pool")

    counts = {k: len(v) for k, v in assignments.items()}
    return SplitResult(assignments=assignments, counts=counts, seed=seed)


# Per-dataset evaluation quotas (plan: Identity Split table + V19.10 XJTU-UP).
# train_backbone is residual and NOT included here.
DATASET_QUOTAS: Dict[str, Dict[str, int]] = {
    "tongji": {
        "base_anchor": 15,
        "future": 75,
        "external_dev": 25,
        "external_test": 25,
    },
    "iitd": {
        "base_anchor": 15,
        "future": 55,
        "external_dev": 15,
        "external_test": 15,
    },
    "bjtu_v2": {
        "base_anchor": 10,
        "future": 38,
        "external_dev": 10,
        "external_test": 10,
    },
    # V19.10 (2026-05-15) Scenario A: XJTU-UP-Huawei as 3rd main dataset.
    # 100 subjects → 5+25+7+8 = 45 reserved + 55 train_backbone (= 110 palms).
    # N_train_palm = 110 ≫ IER-6 threshold (50); safe.
    "xjtu_up_huawei": {
        "base_anchor": 5,
        "future": 25,
        "external_dev": 7,
        "external_test": 8,
    },
}


# Nominal train_backbone size (assuming full eligibility); used by IER-6
# to compute the 50% / 80% warning thresholds.
DATASET_NOMINAL_TRAIN_BACKBONE: Dict[str, int] = {
    "tongji": 160,
    "iitd": 130,
    "bjtu_v2": 80,
    "xjtu_up_huawei": 55,   # 100 subjects - 45 reserved = 55 train_backbone subjects (110 palms)
}
