"""Shared score matrix for Protocol A/B/C (D2 refactor).

The score matrix is the structural-correctness anchor: any difference between
protocols comes from row/col slicing rules, not from feature/score differences.

Key fixes vs. the pre-D2 version:
  - external_dev is REMOVED from prototype groups (Hard Rule 3 + plan section
    "Score Matrix"): external_dev is query-only (calibration probes), NEVER a
    gallery prototype. Calibration scores are computed against the operational
    gallery (base_anchor + future[:t]), which is sliced from the same matrix.
  - NPZ schema upgraded to D1h canonical fields (session_id / subject_split /
    identity_id / sample_role / phase_id / subject_id). The legacy
    `session` / `split_name` aliases are no longer accepted.
  - Adds `proto_palm_id_to_col` lookup, `slice_score_matrix`, and per-step
    row/col mask helpers used by Protocols B/C and IER-3 assertions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# EmbeddingStore — D1h NPZ reader
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingStore:
    """In-memory embedding store loaded from `extract_embeddings.py` NPZ.

    Field names match the D1h canonical NPZ schema; `subject_split` replaces
    the legacy `split_name`, and `session_id` replaces the legacy `session`.
    """

    embeddings: np.ndarray       # [N, D] L2-normalized
    palm_id: np.ndarray          # [N] int64 — recognition class
    subject_id: np.ndarray       # [N] int64 — partitioning unit
    identity_id: np.ndarray      # [N] str
    session_id: np.ndarray       # [N] str
    phase_id: np.ndarray         # [N] str
    sample_id: np.ndarray        # [N] int64
    subject_split: np.ndarray    # [N] str
    sample_role: np.ndarray      # [N] str
    path: np.ndarray             # [N] str

    @classmethod
    def from_npz(cls, npz_path) -> "EmbeddingStore":
        data = np.load(npz_path, allow_pickle=True)
        required = {
            "embeddings", "palm_id", "subject_id", "identity_id",
            "session_id", "phase_id", "sample_id",
            "subject_split", "sample_role", "path",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(
                f"NPZ {npz_path} missing canonical fields: {sorted(missing)} "
                "(was the embedding file written by the pre-D1h "
                "extract_embeddings.py? regenerate it)"
            )
        return cls(
            embeddings=data["embeddings"].astype(np.float32),
            palm_id=data["palm_id"].astype(np.int64),
            subject_id=data["subject_id"].astype(np.int64),
            identity_id=np.asarray(data["identity_id"]).astype(str),
            session_id=np.asarray(data["session_id"]).astype(str),
            phase_id=np.asarray(data["phase_id"]).astype(str),
            sample_id=data["sample_id"].astype(np.int64),
            subject_split=np.asarray(data["subject_split"]).astype(str),
            sample_role=np.asarray(data["sample_role"]).astype(str),
            path=np.asarray(data["path"]).astype(str),
        )

    def select_indices(
        self,
        *,
        palm_ids: Sequence[int],
        session_id: Optional[str] = None,
        phase_id: Optional[str] = None,
        sample_role: Optional[str] = None,
        sample_ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Indices into `embeddings` matching all provided filters.

        Returns indices sorted by (palm_id, sample_id) for determinism.
        Empty match returns an empty int64 array.
        """
        palm_set = np.asarray(sorted(set(int(x) for x in palm_ids)), dtype=np.int64)
        mask = np.isin(self.palm_id, palm_set)
        if session_id is not None:
            mask &= self.session_id == session_id
        if phase_id is not None:
            mask &= self.phase_id == phase_id
        if sample_role is not None:
            mask &= self.sample_role == sample_role
        if sample_ids is not None:
            sample_set = np.asarray(sorted(set(int(x) for x in sample_ids)), dtype=np.int64)
            mask &= np.isin(self.sample_id, sample_set)
        idx = np.where(mask)[0]
        if idx.size == 0:
            return idx.astype(np.int64)
        order = sorted(
            idx.tolist(),
            key=lambda i: (int(self.palm_id[i]), int(self.sample_id[i])),
        )
        return np.array(order, dtype=np.int64)


# ---------------------------------------------------------------------------
# ScoreMatrix — slicing-friendly cosine matrix
# ---------------------------------------------------------------------------

@dataclass
class ScoreMatrix:
    """Shared cosine score matrix for Protocol A/B/C.

    Layout:
      rows: every session2 / S-phase / candidate-pool query that may be
            evaluated under any protocol — `query_split` records the
            subject_split each row originated from (base_anchor, future,
            external_dev, external_test).
      cols: prototypes for `base_anchor + future` palms ONLY.
            external_dev is NEVER a prototype (Hard Rule 3); external_test
            is NEVER enrolled. proto_split records the originating split.

    Each cell is a cosine similarity (both inputs L2-normalized).
    Slicing per protocol = picking row subset and column subset; the matrix
    itself is invariant to protocol choice.
    """

    scores: np.ndarray              # [N_query, N_proto]
    query_palm_id: np.ndarray       # [N_query] int64
    query_sample_id: np.ndarray     # [N_query] int64
    query_split: np.ndarray         # [N_query] str
    query_session_id: np.ndarray    # [N_query] str
    proto_palm_id: np.ndarray       # [N_proto] int64
    proto_split: np.ndarray         # [N_proto] str

    def query_mask_for_palms(self, palm_ids: Sequence[int]) -> np.ndarray:
        """Bool row mask selecting queries whose palm_id is in `palm_ids`."""
        target = np.asarray(sorted(set(int(x) for x in palm_ids)), dtype=np.int64)
        return np.isin(self.query_palm_id, target)

    def query_mask_for_split(self, split_name: str) -> np.ndarray:
        """Bool row mask selecting queries whose subject_split matches."""
        return self.query_split == split_name

    def proto_mask_for_palms(self, palm_ids: Sequence[int]) -> np.ndarray:
        """Bool col mask selecting prototypes whose palm_id is in `palm_ids`."""
        target = np.asarray(sorted(set(int(x) for x in palm_ids)), dtype=np.int64)
        return np.isin(self.proto_palm_id, target)

    def proto_palm_id_to_col(self) -> Dict[int, int]:
        """Inverse lookup: palm_id -> column index in `scores`."""
        return {int(pid): col for col, pid in enumerate(self.proto_palm_id)}


# ---------------------------------------------------------------------------
# Prototype + score matrix construction
# ---------------------------------------------------------------------------

def select_enroll_sample_ids_per_palm(
    palm_ids: Sequence[int],
    palm_to_pool_size: Dict[int, int],
    K: int,
    enroll_seed: int,
) -> Dict[int, List[int]]:
    """Pick K enroll sample_ids per palm for a non-sessioned dataset (IER-7).

    enroll_seed=0 returns the deterministic floor [0..K-1] for every palm.
    enroll_seed≥1 deterministically shuffles each palm's `[0..pool-1]` using
    a (palm_id, enroll_seed) entropy mix and takes the first K (sorted).

    Raises if any palm's pool < K (eligibility should have caught this).
    """
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K}")
    out: Dict[int, List[int]] = {}
    for pid in palm_ids:
        pid_int = int(pid)
        n = int(palm_to_pool_size.get(pid_int, 0))
        if n < K:
            raise ValueError(
                f"palm {pid_int}: pool size {n} < K={K} (eligibility "
                "filter in manifest builder should have rejected this palm)"
            )
        if enroll_seed == 0:
            out[pid_int] = list(range(K))
            continue
        seed = (pid_int * 1_000_003) ^ (int(enroll_seed) * 2_654_435_761) & 0xFFFF_FFFF
        rng = np.random.default_rng(seed)
        arr = np.arange(n, dtype=np.int64)
        rng.shuffle(arr)
        out[pid_int] = sorted(int(x) for x in arr[:K])
    return out


def build_prototypes(
    store: EmbeddingStore,
    palm_ids: Sequence[int],
    proto_split_name: str,
    *,
    enroll_session_id: Optional[str] = None,
    enroll_phase_id: Optional[str] = None,
    enroll_sample_ids: Optional[Sequence[int]] = None,
    enroll_sample_ids_per_palm: Optional[Dict[int, Sequence[int]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-pool enrollment-session embeddings per palm, then L2-normalize.

    Args:
        store: full embedding store (output of extract_embeddings).
        palm_ids: palms whose prototypes to build (typically base_anchor +
            future combined).
        proto_split_name: string label written into the returned
            `proto_split` array (used for slicing diagnostics).
        enroll_session_id: e.g. "session1" for Tongji; None disables filter.
        enroll_phase_id: e.g. "F" for BJTU; None disables filter.
        enroll_sample_ids: K-image selection (per IER-7 enroll_seed).
            Sessioned datasets pass a global list (e.g. [0,1,2]); non-sessioned
            datasets must call this function per palm with the palm-specific
            selection in a loop.

    Returns:
        protos:        [P, D] float32, L2-normalized
        proto_palm_id: [P] int64, sorted ascending
        proto_split:   [P] object, all entries == proto_split_name
    """
    if enroll_sample_ids is not None and enroll_sample_ids_per_palm is not None:
        raise ValueError(
            "pass either `enroll_sample_ids` (sessioned global) or "
            "`enroll_sample_ids_per_palm` (non-sessioned), not both"
        )

    palm_ids_sorted = sorted(int(x) for x in palm_ids)
    proto_list = []
    proto_labels = []
    for pid in palm_ids_sorted:
        per_palm_ids = (
            enroll_sample_ids_per_palm.get(pid)
            if enroll_sample_ids_per_palm is not None else enroll_sample_ids
        )
        idx = store.select_indices(
            palm_ids=[pid],
            session_id=enroll_session_id,
            phase_id=enroll_phase_id,
            sample_ids=per_palm_ids,
        )
        if idx.size == 0:
            # Skip palms with zero enrollment images; caller should treat
            # this as a hard error if it shouldn't happen.
            continue
        feats = store.embeddings[idx]
        mean = feats.mean(axis=0)
        norm = float(np.linalg.norm(mean)) + 1e-12
        proto_list.append(mean / norm)
        proto_labels.append(int(pid))

    if not proto_list:
        D = int(store.embeddings.shape[1])
        return (
            np.zeros((0, D), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=object),
        )

    if len(proto_labels) != len(palm_ids_sorted):
        missing = sorted(set(palm_ids_sorted) - set(proto_labels))
        raise ValueError(
            f"prototype count mismatch in split {proto_split_name!r}: "
            f"requested {len(palm_ids_sorted)} got {len(proto_labels)} "
            f"(missing palm_ids: {missing[:10]}...)"
        )

    protos = np.stack(proto_list, axis=0).astype(np.float32)
    palm_ids_arr = np.array(proto_labels, dtype=np.int64)
    split_arr = np.array([proto_split_name] * len(proto_labels), dtype=object)
    return protos, palm_ids_arr, split_arr


def build_score_matrix(
    store: EmbeddingStore,
    *,
    base_anchor_ids: Sequence[int],
    future_ids: Sequence[int],
    external_dev_ids: Sequence[int],
    external_test_ids: Sequence[int],
    enroll_session_id: Optional[str] = None,
    enroll_phase_id: Optional[str] = None,
    enroll_sample_ids: Optional[Sequence[int]] = None,
    enroll_sample_ids_per_palm: Optional[Dict[int, Sequence[int]]] = None,
    query_session_id: Optional[str] = None,
    query_phase_id: Optional[str] = None,
    non_sessioned_complement_query: bool = False,
) -> ScoreMatrix:
    """Build the shared cosine score matrix used by Protocols A/B/C.

    Prototypes (cols): base_anchor + future palms ONLY (Hard Rule 3).
    Queries (rows): all queries from the four eval splits, tagged with
                    `query_split` so any protocol can slice by row.

    Two modes (driven by which enroll-selection arg is passed):

    Sessioned (Tongji, BJTU): pass `enroll_sample_ids` (global list) along
        with `enroll_session_id` / `enroll_phase_id` and `query_session_id` /
        `query_phase_id`. Query rows are filtered by sample_role="query".

    Non-sessioned (IITD): pass `enroll_sample_ids_per_palm` (palm_id → list)
        and set `non_sessioned_complement_query=True`. base_anchor / future
        query rows are derived per palm as the COMPLEMENT of that palm's
        enroll selection (sample_role="candidate" filter). external_dev /
        external_test continue to use sample_role="query" rows.
    """
    # ---- Prototypes (NO external_dev / external_test) ----
    proto_groups: List[Tuple[Sequence[int], str]] = [
        (base_anchor_ids, "base_anchor"),
        (future_ids, "future"),
    ]
    proto_chunks, proto_palm_chunks, proto_split_chunks = [], [], []
    for ids, name in proto_groups:
        if not ids:
            continue
        p, pid, sp = build_prototypes(
            store, ids, name,
            enroll_session_id=enroll_session_id,
            enroll_phase_id=enroll_phase_id,
            enroll_sample_ids=enroll_sample_ids,
            enroll_sample_ids_per_palm=enroll_sample_ids_per_palm,
        )
        if p.shape[0] > 0:
            proto_chunks.append(p)
            proto_palm_chunks.append(pid)
            proto_split_chunks.append(sp)

    if proto_chunks:
        protos = np.concatenate(proto_chunks, axis=0)
        proto_palm_id = np.concatenate(proto_palm_chunks)
        proto_split = np.concatenate(proto_split_chunks)
    else:
        D = int(store.embeddings.shape[1])
        protos = np.zeros((0, D), dtype=np.float32)
        proto_palm_id = np.zeros((0,), dtype=np.int64)
        proto_split = np.zeros((0,), dtype=object)

    # ---- Queries (all four eval splits, tagged) ----
    query_chunks, qpalm, qsamp, qsplit, qsession = [], [], [], [], []

    def _append_block(idx: np.ndarray, name: str) -> None:
        if idx.size == 0:
            return
        query_chunks.append(store.embeddings[idx])
        qpalm.append(store.palm_id[idx])
        qsamp.append(store.sample_id[idx])
        qsplit.append(np.array([name] * idx.size, dtype=object))
        qsession.append(store.session_id[idx])

    # base_anchor / future queries
    for ids, name in (
        (base_anchor_ids, "base_anchor"),
        (future_ids, "future"),
    ):
        if not ids:
            continue
        if non_sessioned_complement_query:
            if enroll_sample_ids_per_palm is None:
                raise ValueError(
                    "non_sessioned_complement_query=True requires "
                    "enroll_sample_ids_per_palm to derive the per-palm "
                    "query complement"
                )
            # Per palm: queries = candidate-role rows whose sample_id is
            # NOT in the palm's enroll selection.
            for pid in sorted(int(x) for x in ids):
                enroll_set = set(int(x) for x in
                                  enroll_sample_ids_per_palm.get(pid, []))
                # All candidate sample_ids for this palm
                all_idx = store.select_indices(
                    palm_ids=[pid], sample_role="candidate",
                )
                # Filter to those NOT in enroll_set
                if all_idx.size == 0:
                    continue
                query_idx = np.array([
                    i for i in all_idx
                    if int(store.sample_id[i]) not in enroll_set
                ], dtype=np.int64)
                _append_block(query_idx, name)
        else:
            idx = store.select_indices(
                palm_ids=ids,
                session_id=query_session_id,
                phase_id=query_phase_id,
                sample_role="query",
            )
            _append_block(idx, name)

    # external_dev / external_test queries (always sample_role=query)
    for ids, name in (
        (external_dev_ids, "external_dev"),
        (external_test_ids, "external_test"),
    ):
        if not ids:
            continue
        idx = store.select_indices(
            palm_ids=ids,
            session_id=query_session_id,
            phase_id=query_phase_id,
            sample_role="query",
        )
        _append_block(idx, name)

    if not query_chunks:
        raise RuntimeError(
            "no queries assembled — check identity_split and store contents"
        )
    queries = np.concatenate(query_chunks, axis=0)
    query_palm_id = np.concatenate(qpalm)
    query_sample_id = np.concatenate(qsamp)
    query_split = np.concatenate(qsplit)
    query_session_id_arr = np.concatenate(qsession)

    if protos.shape[0] == 0:
        scores = np.zeros((queries.shape[0], 0), dtype=np.float32)
    else:
        scores = (queries @ protos.T).astype(np.float32)

    return ScoreMatrix(
        scores=scores,
        query_palm_id=query_palm_id.astype(np.int64),
        query_sample_id=query_sample_id.astype(np.int64),
        query_split=query_split,
        query_session_id=query_session_id_arr,
        proto_palm_id=proto_palm_id.astype(np.int64),
        proto_split=proto_split,
    )


# ---------------------------------------------------------------------------
# C-step row / col mask helpers (IER-3 building blocks)
# ---------------------------------------------------------------------------

def enrolled_palm_set(
    base_anchor_ids: Sequence[int],
    future_order: Sequence[int],
    t: int,
) -> List[int]:
    """Palm_ids enrolled at Protocol C step t = base_anchor ∪ future_order[:t]."""
    if t < 0 or t > len(future_order):
        raise ValueError(
            f"step t={t} out of range [0, {len(future_order)}]"
        )
    return sorted(set(int(x) for x in base_anchor_ids)
                  | set(int(x) for x in future_order[:t]))


def not_enrolled_future_set(
    future_order: Sequence[int], t: int,
) -> List[int]:
    """Palms still future-unknown at step t = future_order[t:]."""
    return sorted(int(x) for x in future_order[t:])


def gallery_col_mask(
    sm: ScoreMatrix,
    base_anchor_ids: Sequence[int],
    future_order: Sequence[int],
    t: int,
) -> np.ndarray:
    """Bool col mask selecting prototypes that are enrolled at step t."""
    return sm.proto_mask_for_palms(
        enrolled_palm_set(base_anchor_ids, future_order, t)
    )


def known_query_row_mask(
    sm: ScoreMatrix,
    base_anchor_ids: Sequence[int],
    future_order: Sequence[int],
    t: int,
) -> np.ndarray:
    """Rows from palms enrolled at step t (TPIR/FNIR evaluation cohort)."""
    return sm.query_mask_for_palms(
        enrolled_palm_set(base_anchor_ids, future_order, t)
    )


def future_unknown_query_row_mask(
    sm: ScoreMatrix, future_order: Sequence[int], t: int,
) -> np.ndarray:
    """Rows from palms NOT yet enrolled (Future-FPIR evaluation cohort)."""
    return sm.query_mask_for_palms(not_enrolled_future_set(future_order, t))


def calibration_query_row_mask(
    sm: ScoreMatrix, external_dev_ids: Sequence[int],
) -> np.ndarray:
    """Rows reserved for τ calibration (external_dev only)."""
    return sm.query_mask_for_palms(external_dev_ids)


def external_unknown_query_row_mask(
    sm: ScoreMatrix, external_test_ids: Sequence[int],
) -> np.ndarray:
    """Rows for External-FPIR evaluation (external_test only)."""
    return sm.query_mask_for_palms(external_test_ids)


def slice_score_matrix(
    sm: ScoreMatrix,
    *,
    row_mask: np.ndarray,
    col_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply (row, col) bool masks; return (scores, q_palm, q_sample, q_split, p_palm).

    The companion arrays are sliced consistently so callers can compute
    rank-1 / TPIR / FPIR without re-aligning indices.
    """
    if row_mask.dtype != bool:
        raise TypeError("row_mask must be bool")
    if col_mask.dtype != bool:
        raise TypeError("col_mask must be bool")
    sliced_scores = sm.scores[np.ix_(row_mask, col_mask)]
    return (
        sliced_scores,
        sm.query_palm_id[row_mask],
        sm.query_sample_id[row_mask],
        sm.query_split[row_mask],
        sm.proto_palm_id[col_mask],
    )


# ---------------------------------------------------------------------------
# IER-3 step-consistency assertion (used by protocols)
# ---------------------------------------------------------------------------

def assert_step_consistency(
    *,
    t: int,
    base_anchor_ids: Sequence[int],
    future_order: Sequence[int],
    external_dev_ids: Sequence[int],
    external_test_ids: Sequence[int],
    known_query_pids: Sequence[int],
    future_unknown_query_pids: Sequence[int],
    calibration_query_pids: Sequence[int],
    external_unknown_query_pids: Sequence[int],
    gallery_col_pids: Sequence[int],
) -> None:
    """Assert all per-step row/col palm_id sets match canonical expectations.

    These four invariants catch the most common implementation bugs:
      1. all-future-known or all-future-unknown (regardless of t)
      2. not-yet-enrolled future cols leaking into the gallery
      3. external_dev ↔ external_test mix-up
      4. wrong query split for calibration
    """
    enrolled = set(int(x) for x in base_anchor_ids) | set(int(x) for x in future_order[:t])
    not_enrolled = set(int(x) for x in future_order[t:])

    if set(int(x) for x in known_query_pids) != enrolled:
        raise AssertionError(
            f"step {t}: known query rows mismatch "
            f"(expected enrolled={sorted(enrolled)})"
        )
    if set(int(x) for x in future_unknown_query_pids) != not_enrolled:
        raise AssertionError(
            f"step {t}: future-unknown rows mismatch "
            f"(expected not_enrolled={sorted(not_enrolled)})"
        )
    if set(int(x) for x in calibration_query_pids) != set(int(x) for x in external_dev_ids):
        raise AssertionError(
            f"step {t}: calibration rows must be external_dev only"
        )
    if set(int(x) for x in external_unknown_query_pids) != set(int(x) for x in external_test_ids):
        raise AssertionError(
            f"step {t}: external_test rows mismatch"
        )
    if set(int(x) for x in gallery_col_pids) != enrolled:
        raise AssertionError(
            f"step {t}: gallery cols must equal enrolled set"
        )
    if not_enrolled.isdisjoint(set(int(x) for x in gallery_col_pids)) is False:
        raise AssertionError(
            f"step {t}: not-yet-enrolled future cols leaked into gallery"
        )
