"""Protocol orchestrator (D6 entry point).

Runs Protocols A / B / C-fixed / C-recal at one (order_seed, enroll_seed,
target_fpir) configuration and writes per-protocol JSON files. Paper-scale
multi-seed aggregation should wrap this orchestrator in an outer loop and
collect across (order_seed × enroll_seed) pairs (plan: Seeds section,
order_seed=10, enroll_seed=3 for paper).

Inputs:
  --embeddings  : NPZ written by `extract_embeddings.py` (D1h schema)
  --manifest    : manifest.csv used to derive split palm_ids
  --target_fpir : e.g. 0.05; reliability flags are computed against this
  --t_step      : Protocol C step size (e.g. 10 for Tongji, 5 for BJTU)
  --order_seed  : controls future enrollment order (slicing only)
  --enroll_seed : controls which K imgs form prototypes (re-extracts the
                  score matrix; if 0 uses the manifest baseline directly)

Outputs (under --out):
  protocol_a.json
  protocol_b.json
  protocol_c_fixed.json
  protocol_c_recal.json
  meta.json   (config + sanity check status)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp1_baselines.datasets.manifest_io import (
    get_split_palm_ids, load_manifest,
)
from exp1_baselines.eval.protocols import (
    assert_b_equals_c_recal_endpoint,
    build_t_schedule,
    contralateral_xor1,
    run_protocol_a,
    run_protocol_b,
    run_protocol_c_fixed,
    run_protocol_c_recal,
)
from exp1_baselines.eval.score_matrix import (
    EmbeddingStore,
    build_score_matrix,
    select_enroll_sample_ids_per_palm,
)


def deterministic_future_order(
    future_palm_ids: Sequence[int], order_seed: int,
) -> List[int]:
    """Reproducible future-enrollment order from a seed (slicing only)."""
    palms = sorted(int(x) for x in future_palm_ids)
    if order_seed == 0:
        return palms
    rng = np.random.default_rng(order_seed)
    arr = np.array(palms, dtype=np.int64)
    rng.shuffle(arr)
    return [int(x) for x in arr]


def select_enroll_sample_ids_sessioned(
    K: int, enroll_seed: int, num_samples_per_palm: int = 10,
) -> List[int]:
    """K-sample selection from session1/F-phase pool (Tongji/BJTU policy)."""
    if K <= 0 or K > num_samples_per_palm:
        raise ValueError(f"K must be in 1..{num_samples_per_palm}, got {K}")
    if enroll_seed == 0:
        return list(range(K))
    rng = np.random.default_rng(enroll_seed)
    arr = np.array(range(num_samples_per_palm), dtype=np.int64)
    rng.shuffle(arr)
    return sorted(int(x) for x in arr[:K])


def _to_jsonable(obj):
    """Recursively convert numpy scalars / tuples / NaN to JSON-friendly values."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    return obj


def _palm_to_pool_size_from_store(
    store: EmbeddingStore, palm_ids: Sequence[int],
) -> Dict[int, int]:
    """Count `sample_role=candidate` rows per palm (IITD enrollment pool)."""
    out: Dict[int, int] = {}
    for pid in palm_ids:
        idx = store.select_indices(palm_ids=[int(pid)], sample_role="candidate")
        out[int(pid)] = int(idx.size)
    return out


def run_orchestrator(
    *,
    embeddings_npz: Path,
    manifest_csv: Path,
    out_dir: Path,
    target_fpir: float,
    t_step: int,
    order_seed: int = 0,
    enroll_seed: int = 0,
    K: int = 3,
    enroll_session_id: Optional[str] = "session1",
    enroll_phase_id: Optional[str] = None,
    query_session_id: Optional[str] = "session2",
    query_phase_id: Optional[str] = None,
    sessioned_K_pool: int = 10,
    non_sessioned: bool = False,
) -> Dict[str, object]:
    """Run all four protocols + B==C-recal sanity. Returns a meta dict.

    Set `non_sessioned=True` for IITD-style datasets: enroll selection becomes
    per-palm (deterministic from `enroll_seed` + per-palm pool size), and
    base/future query rows are derived as the per-palm complement of the
    enroll selection. external_dev / external_test still use sample_role="query".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load splits from manifest ----
    df = load_manifest(manifest_csv)
    base_anchor_ids = get_split_palm_ids(df, "base_anchor")
    future_palm_ids = get_split_palm_ids(df, "future")
    external_dev_ids = get_split_palm_ids(df, "external_dev")
    external_test_ids = get_split_palm_ids(df, "external_test")

    if not base_anchor_ids or not future_palm_ids:
        raise RuntimeError(
            "manifest is missing base_anchor or future palm_ids; "
            "did the builder run?"
        )

    future_order = deterministic_future_order(future_palm_ids, order_seed)

    # ---- enroll selection (sessioned global vs non-sessioned per-palm) ----
    store = EmbeddingStore.from_npz(embeddings_npz)
    enroll_sample_ids: Optional[List[int]] = None
    enroll_sample_ids_per_palm: Optional[Dict[int, List[int]]] = None
    palm_to_pool_size: Dict[int, int] = {}

    if non_sessioned:
        all_eval_palms = list(set(base_anchor_ids) | set(future_palm_ids))
        palm_to_pool_size = _palm_to_pool_size_from_store(store, all_eval_palms)
        enroll_sample_ids_per_palm = select_enroll_sample_ids_per_palm(
            palm_ids=all_eval_palms,
            palm_to_pool_size=palm_to_pool_size,
            K=K, enroll_seed=enroll_seed,
        )
    else:
        if enroll_session_id is not None or enroll_phase_id is not None:
            enroll_sample_ids = select_enroll_sample_ids_sessioned(
                K, enroll_seed, sessioned_K_pool,
            )

    # ---- score matrix ----
    sm = build_score_matrix(
        store,
        base_anchor_ids=base_anchor_ids,
        future_ids=future_palm_ids,
        external_dev_ids=external_dev_ids,
        external_test_ids=external_test_ids,
        enroll_session_id=enroll_session_id,
        enroll_phase_id=enroll_phase_id,
        enroll_sample_ids=enroll_sample_ids,
        enroll_sample_ids_per_palm=enroll_sample_ids_per_palm,
        query_session_id=query_session_id,
        query_phase_id=query_phase_id,
        non_sessioned_complement_query=non_sessioned,
    )

    t_max = len(future_order)
    t_schedule = build_t_schedule(t_max, t_step)

    # ---- protocols ----
    a_results = run_protocol_a(
        sm, base_anchor_ids=base_anchor_ids,
        future_order=future_order, t_schedule=t_schedule,
    )
    b_result = run_protocol_b(
        sm, base_anchor_ids=base_anchor_ids,
        future_ids=future_palm_ids,
        external_dev_ids=external_dev_ids,
        external_test_ids=external_test_ids,
        target_fpir=target_fpir,
    )
    c_fixed = run_protocol_c_fixed(
        sm, base_anchor_ids=base_anchor_ids,
        future_order=future_order,
        external_dev_ids=external_dev_ids,
        external_test_ids=external_test_ids,
        target_fpir=target_fpir, t_schedule=t_schedule,
        contralateral_of=contralateral_xor1,
    )
    c_recal = run_protocol_c_recal(
        sm, base_anchor_ids=base_anchor_ids,
        future_order=future_order,
        external_dev_ids=external_dev_ids,
        external_test_ids=external_test_ids,
        target_fpir=target_fpir, t_schedule=t_schedule,
        contralateral_of=contralateral_xor1,
    )

    # ---- B == C-recal endpoint sanity ----
    sanity_status = "ok"
    sanity_error: Optional[str] = None
    try:
        assert_b_equals_c_recal_endpoint(b_result, c_recal[-1], tol=1e-6)
    except AssertionError as e:
        sanity_status = "failed"
        sanity_error = str(e)

    # ---- write outputs ----
    (out_dir / "protocol_a.json").write_text(
        json.dumps(_to_jsonable(a_results), indent=2)
    )
    (out_dir / "protocol_b.json").write_text(
        json.dumps(_to_jsonable(b_result), indent=2)
    )
    (out_dir / "protocol_c_fixed.json").write_text(
        json.dumps(_to_jsonable(c_fixed), indent=2)
    )
    (out_dir / "protocol_c_recal.json").write_text(
        json.dumps(_to_jsonable(c_recal), indent=2)
    )

    meta = {
        "target_fpir": target_fpir,
        "t_step": t_step,
        "t_max": t_max,
        "t_schedule": t_schedule,
        "K": K,
        "order_seed": order_seed,
        "enroll_seed": enroll_seed,
        "non_sessioned": non_sessioned,
        "enroll_sample_ids": enroll_sample_ids,
        "enroll_sample_ids_per_palm": enroll_sample_ids_per_palm,
        "palm_to_pool_size": palm_to_pool_size,
        "enroll_session_id": enroll_session_id,
        "enroll_phase_id": enroll_phase_id,
        "query_session_id": query_session_id,
        "query_phase_id": query_phase_id,
        "split_counts": {
            "base_anchor": len(base_anchor_ids),
            "future": len(future_palm_ids),
            "external_dev": len(external_dev_ids),
            "external_test": len(external_test_ids),
        },
        "future_order": future_order,
        "score_matrix_shape": list(sm.scores.shape),
        "b_vs_c_recal_endpoint_sanity": sanity_status,
        "b_vs_c_recal_endpoint_sanity_error": sanity_error,
    }
    (out_dir / "meta.json").write_text(json.dumps(_to_jsonable(meta), indent=2))
    return meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run Protocols A/B/C and dump JSON.")
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target_fpir", type=float, required=True)
    p.add_argument("--t_step", type=int, required=True)
    p.add_argument("--order_seed", type=int, default=0)
    p.add_argument("--enroll_seed", type=int, default=0)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--enroll_session", type=str, default="session1",
                   help='Session id for enrollment (Tongji "session1"; '
                        '"" or "none" disables filter — for IITD use "none").')
    p.add_argument("--enroll_phase", type=str, default="",
                   help='Phase id for enrollment ("F" for BJTU; empty otherwise).')
    p.add_argument("--query_session", type=str, default="session2")
    p.add_argument("--query_phase", type=str, default="")
    p.add_argument("--sessioned_K_pool", type=int, default=10,
                   help="Total imgs in enrollment session per palm (Tongji=10).")
    p.add_argument("--non_sessioned", action="store_true",
                   help="Use IITD-style per-palm enroll selection + complement queries.")
    args = p.parse_args(argv)

    def _opt(s: str) -> Optional[str]:
        if s == "" or s.lower() == "none":
            return None
        return s

    meta = run_orchestrator(
        embeddings_npz=args.embeddings,
        manifest_csv=args.manifest,
        out_dir=args.out,
        target_fpir=args.target_fpir,
        t_step=args.t_step,
        order_seed=args.order_seed,
        enroll_seed=args.enroll_seed,
        K=args.K,
        enroll_session_id=_opt(args.enroll_session),
        enroll_phase_id=_opt(args.enroll_phase),
        query_session_id=_opt(args.query_session),
        query_phase_id=_opt(args.query_phase),
        sessioned_K_pool=args.sessioned_K_pool,
        non_sessioned=args.non_sessioned,
    )
    print(json.dumps({
        "sanity": meta["b_vs_c_recal_endpoint_sanity"],
        "score_matrix_shape": meta["score_matrix_shape"],
        "out": str(args.out),
    }, indent=2))
    return 0 if meta["b_vs_c_recal_endpoint_sanity"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
