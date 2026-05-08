"""Exp1-S: WebPalm low-FPIR security stress orchestrator.

Implements the three Exp1-S protocol variants on a single in-pool dataset
(Tongji / IITD / BJTU) using WebPalm as the external impostor pool:

  B-S      : static endpoint. τ calibrated once on webpalm_cal vs full gallery.
             Realized FPIR reported once on webpalm_test vs full gallery.
  C-fixed-S: τ calibrated once at gallery_0 = base_anchor on webpalm_cal.
             Realized FPIR reported at every t step under stale τ_0.
  C-recal-S: τ recalibrated every step on webpalm_cal vs current gallery_t.
             Realized FPIR reported at every t step under fresh τ_t.

For each (protocol, target α) we produce:
  - Per-step (or single endpoint) realized FPIR
  - Cluster bootstrap 95% CI on realized FPIR (B=1000, identity-level for
    WebPalm because WebPalm is one-shot per identity; plan §V6 + §V11)
  - Security violation rate (plan §C8): fraction of t steps where the
    bootstrap CI lower bound exceeds α
  - Peak realized FPIR / α; first violation step; AUC above target

The script is per-(dataset, target). Aggregating across (dataset, backbone,
target) for the paper figure happens in a downstream plotter.

Plan locks:
  - α ∈ {1e-3, 1e-4} (plan §V12; 1e-5 noise-dominated)
  - λ tuning forbidden at α=1e-4 (plan §V9; this script does not tune
    anything — it only reports realized FPIR; λ-tuning + mitigation
    happen in `mitigation.py`)
  - WebPalm-test is touched only here for the realized FPIR; λ from
    Phase 4 mitigation is applied via `--apply-tau` (one external τ per α)

Usage:
    python experiments/exp1s_orchestrator.py \
        --inpool-npz   experiments/generated/tongji_full/mfn_tongji_112_embeddings.npz \
        --inpool-manifest experiments/generated/tongji_full/manifest.csv \
        --webpalm-npz  ~/research_data/webpalm/embeddings_mfn112.npz \
        --target-fpir 0.001 0.0001 \
        --t-step 10 \
        --order-seed 0 \
        --enroll-seed 0 \
        --out ~/research_data/webpalm/exp1s/tongji
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp1_baselines.datasets.manifest_io import (
    get_split_palm_ids,
    load_manifest,
)
from exp1_baselines.eval.orchestrator import (
    deterministic_future_order,
    select_enroll_sample_ids_sessioned,
)
from exp1_baselines.eval.protocols import (
    contralateral_xor1,
)
from exp1_baselines.eval.score_matrix import (
    EmbeddingStore,
    build_score_matrix,
    enrolled_palm_set,
    select_enroll_sample_ids_per_palm,
)
from exp1_baselines.eval.thresholding import calibrate_threshold_at_fpir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cos_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B.T


def webpalm_top1_against_gallery(
    wp_emb: np.ndarray, gallery_protos: np.ndarray
) -> np.ndarray:
    """Per-WebPalm-probe top-1 cosine score against current gallery."""
    if gallery_protos.shape[0] == 0:
        return np.zeros((wp_emb.shape[0],), dtype=np.float32)
    return cos_sim_matrix(wp_emb, gallery_protos).max(axis=1)


def cluster_bootstrap_fpir(
    top1: np.ndarray, tau: float, B: int = 1000, seed: int = 42,
) -> Dict[str, float]:
    """Identity-level bootstrap (one identity per probe in WebPalm).

    Returns realized FPIR + 95% CI lo/hi.
    """
    rng = np.random.default_rng(seed)
    n = top1.size
    if n == 0:
        return {"fpir": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": 0}
    realized = float((top1 >= tau).mean())
    fpirs = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        fpirs[b] = float((top1[idx] >= tau).mean())
    return {
        "fpir": realized,
        "ci_lo": float(np.percentile(fpirs, 2.5)),
        "ci_hi": float(np.percentile(fpirs, 97.5)),
        "n": int(n),
    }


def violation_metrics(
    per_step_results: List[Dict[str, object]],
    target: float,
) -> Dict[str, object]:
    """Plan §C8 metrics: violation_rate, peak/target, first_violation, AUC."""
    n_steps = len(per_step_results)
    violations = 0
    first = None
    peak = 0.0
    auc = 0.0
    for s in per_step_results:
        fpir = float(s["fpir"])
        ci_lo = float(s.get("ci_lo", fpir))
        if ci_lo > target:
            violations += 1
            if first is None:
                first = int(s["t"])
        peak = max(peak, fpir)
        if fpir > target:
            auc += fpir - target
    return {
        "violation_rate": violations / max(1, n_steps),
        "peak_fpir": peak,
        "peak_to_target_ratio": peak / target if target > 0 else float("inf"),
        "first_violation_step": first,
        "violation_auc": auc,
        "n_violations": violations,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# Build gallery prototypes (same logic as exp1_baselines but exposed)
# ---------------------------------------------------------------------------

def build_gallery_protos(
    store: EmbeddingStore,
    palm_ids: Sequence[int],
    enroll_session_id: str | None,
    enroll_phase_id: str | None,
    enroll_sample_ids: Sequence[int] | None,
    enroll_sample_ids_per_palm: Dict[int, Sequence[int]] | None,
) -> np.ndarray:
    """Build per-palm prototypes (mean-pool L2-normalized)."""
    protos = []
    palm_ids_sorted = sorted(int(x) for x in palm_ids)
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
            continue
        feats = store.embeddings[idx]
        mean = feats.mean(axis=0)
        norm = float(np.linalg.norm(mean)) + 1e-12
        protos.append(mean / norm)
    if not protos:
        D = int(store.embeddings.shape[1])
        return np.zeros((0, D), dtype=np.float32)
    return np.stack(protos, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inpool-npz", type=Path, required=True)
    p.add_argument("--inpool-manifest", type=Path, required=True)
    p.add_argument("--webpalm-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target-fpir", type=float, nargs="+", default=[1e-3, 1e-4])
    p.add_argument("--t-step", type=int, default=10)
    p.add_argument("--order-seed", type=int, default=0)
    p.add_argument("--enroll-seed", type=int, default=0)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--enroll-session", type=str, default="session1")
    p.add_argument("--enroll-phase", type=str, default="")
    p.add_argument("--query-session", type=str, default="session2")
    p.add_argument("--query-phase", type=str, default="")
    p.add_argument("--non-sessioned", action="store_true")
    p.add_argument("--bootstrap-B", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=42)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    def _opt(s: str) -> str | None:
        return None if (s == "" or s.lower() == "none") else s

    # ----- Load in-pool manifest + embeddings -----
    df = load_manifest(args.inpool_manifest)
    base_anchor_ids = get_split_palm_ids(df, "base_anchor")
    future_palm_ids = get_split_palm_ids(df, "future")
    print(f"[in-pool] base_anchor={len(base_anchor_ids)} "
          f"future={len(future_palm_ids)}")

    inpool_store = EmbeddingStore.from_npz(args.inpool_npz)
    print(f"[in-pool] embeddings: {inpool_store.embeddings.shape}")

    future_order = deterministic_future_order(future_palm_ids, args.order_seed)

    enroll_session_id = _opt(args.enroll_session)
    enroll_phase_id = _opt(args.enroll_phase)

    if args.non_sessioned:
        all_eval_palms = list(set(base_anchor_ids) | set(future_palm_ids))
        palm_to_pool = {}
        for pid in all_eval_palms:
            idx = inpool_store.select_indices(palm_ids=[int(pid)], sample_role="candidate")
            palm_to_pool[int(pid)] = int(idx.size)
        enroll_sample_ids_per_palm = select_enroll_sample_ids_per_palm(
            palm_ids=all_eval_palms, palm_to_pool_size=palm_to_pool,
            K=args.K, enroll_seed=args.enroll_seed,
        )
        enroll_sample_ids = None
    else:
        enroll_sample_ids = (
            select_enroll_sample_ids_sessioned(args.K, args.enroll_seed)
            if enroll_session_id is not None or enroll_phase_id is not None
            else None
        )
        enroll_sample_ids_per_palm = None

    # ----- Load WebPalm embeddings + split assignment -----
    wp = np.load(args.webpalm_npz, allow_pickle=True)
    wp_emb = wp["embeddings"]
    wp_split = wp["split"]
    cal_mask = wp_split == "calibration"
    dev_mask = wp_split == "dev"
    test_mask = wp_split == "test"
    wp_cal = wp_emb[cal_mask]
    wp_dev = wp_emb[dev_mask]
    wp_test = wp_emb[test_mask]
    print(f"[WebPalm] cal={wp_cal.shape[0]} dev={wp_dev.shape[0]} "
          f"test={wp_test.shape[0]}")

    # ----- t schedule -----
    t_max = len(future_order)
    schedule = list(range(0, t_max + 1, args.t_step))
    if schedule[-1] != t_max:
        schedule.append(t_max)
    print(f"[schedule] t in {schedule[:5]}..{schedule[-3:]} (t_max={t_max})")

    # ----- Build galleries at every t (cache prototypes) -----
    gallery_cache: Dict[int, np.ndarray] = {}
    for t in schedule:
        enrolled = enrolled_palm_set(base_anchor_ids, future_order, t)
        gallery_cache[t] = build_gallery_protos(
            inpool_store, enrolled,
            enroll_session_id=enroll_session_id,
            enroll_phase_id=enroll_phase_id,
            enroll_sample_ids=enroll_sample_ids,
            enroll_sample_ids_per_palm=enroll_sample_ids_per_palm,
        )

    # ----- Run B-S, C-fixed-S, C-recal-S for each target α -----
    all_results = {}

    for alpha in args.target_fpir:
        # B-S endpoint
        full_gallery = gallery_cache[t_max]
        cal_top1_full = webpalm_top1_against_gallery(wp_cal, full_gallery)
        tau_B = calibrate_threshold_at_fpir(cal_top1_full, alpha)
        test_top1_full = webpalm_top1_against_gallery(wp_test, full_gallery)
        b_metrics = cluster_bootstrap_fpir(
            test_top1_full, tau_B, B=args.bootstrap_B, seed=args.bootstrap_seed,
        )
        b_result = {
            "protocol": "B-S",
            "alpha": alpha,
            "tau": float(tau_B),
            "gallery_size": int(full_gallery.shape[0]),
            **b_metrics,
        }

        # C-fixed-S: τ at gallery_0 (base_anchor), evaluated at every t
        gallery_0 = gallery_cache[0]
        cal_top1_0 = webpalm_top1_against_gallery(wp_cal, gallery_0)
        tau_fixed = calibrate_threshold_at_fpir(cal_top1_0, alpha)
        c_fixed_steps = []
        for t in schedule:
            G = gallery_cache[t]
            tt = webpalm_top1_against_gallery(wp_test, G)
            metrics = cluster_bootstrap_fpir(
                tt, tau_fixed, B=args.bootstrap_B, seed=args.bootstrap_seed,
            )
            c_fixed_steps.append({
                "t": int(t), "gallery_size": int(G.shape[0]),
                "tau": float(tau_fixed), **metrics,
            })

        # C-recal-S: τ recalibrated each step
        c_recal_steps = []
        for t in schedule:
            G = gallery_cache[t]
            ct = webpalm_top1_against_gallery(wp_cal, G)
            tau_t = calibrate_threshold_at_fpir(ct, alpha)
            tt = webpalm_top1_against_gallery(wp_test, G)
            metrics = cluster_bootstrap_fpir(
                tt, tau_t, B=args.bootstrap_B, seed=args.bootstrap_seed,
            )
            c_recal_steps.append({
                "t": int(t), "gallery_size": int(G.shape[0]),
                "tau": float(tau_t), **metrics,
            })

        # Violation metrics
        c_fixed_vio = violation_metrics(c_fixed_steps, alpha)
        c_recal_vio = violation_metrics(c_recal_steps, alpha)

        all_results[f"alpha_{alpha:.0e}"] = {
            "alpha": alpha,
            "B_S": b_result,
            "C_fixed_S": {"steps": c_fixed_steps, "violation": c_fixed_vio},
            "C_recal_S": {"steps": c_recal_steps, "violation": c_recal_vio},
        }
        print(f"[α={alpha:.0e}] B-S: tau={tau_B:.4f}, FPIR={b_metrics['fpir']:.5f} "
              f"[{b_metrics['ci_lo']:.5f}, {b_metrics['ci_hi']:.5f}]")
        print(f"  C-fixed-S violation_rate={c_fixed_vio['violation_rate']:.2f}, "
              f"peak/α={c_fixed_vio['peak_to_target_ratio']:.2f}x")
        print(f"  C-recal-S violation_rate={c_recal_vio['violation_rate']:.2f}, "
              f"peak/α={c_recal_vio['peak_to_target_ratio']:.2f}x")

    # ----- Save -----
    out_json = args.out / "exp1s_results.json"
    out_json.write_text(json.dumps({
        "schedule": schedule,
        "t_max": t_max,
        "order_seed": args.order_seed,
        "enroll_seed": args.enroll_seed,
        "K": args.K,
        "non_sessioned": args.non_sessioned,
        "bootstrap_B": args.bootstrap_B,
        "bootstrap_seed": args.bootstrap_seed,
        "split_counts": {
            "webpalm_cal": int(wp_cal.shape[0]),
            "webpalm_dev": int(wp_dev.shape[0]),
            "webpalm_test": int(wp_test.shape[0]),
            "base_anchor": len(base_anchor_ids),
            "future": len(future_palm_ids),
        },
        "results": all_results,
    }, indent=2))
    print(f"[ok] {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
