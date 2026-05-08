"""Drift-aware threshold mitigation for sequential open-set palmprint
identification (plan §C2 / §V1 / §V9 / §V10 / §V14a).

The mitigation problem:
  C-recal-S maintains realized FPIR ≤ α by recalibrating τ at every t step
  against the operational gallery. This costs |P_cal| × m(t) similarity
  comparisons per step. As gallery grows the calibration cost grows
  linearly per step.

The proposed method:
  Conservative Gallery-Size-Aware Thresholding. At early gallery sizes
  m_k ∈ M_fit (e.g., {30, 50, 80, 120}), compute empirical thresholds
  τ̂(m_k) via standard tie-aware calibration on `webpalm_calibration`.
  Fit a log-linear point estimator
                    τ̂_pred(m) = â + b̂ · log(m)
  and an empirical bootstrap distribution of `τ̂_pred^{(b)}(m)` across
  identity-level resamples of `webpalm_calibration` (B=1000, seed=42 per
  §V11). The conservative threshold for any deployment gallery size m is
                    τ_safe(m) = clip(Q_q(τ̂_pred^{(b)}(m)), -1, 1)
  where Q_q is the empirical q-quantile of the bootstrap distribution
  (V14a primary form, replaces λ × σ̂_τ form used in v3 §V1).

The safety quantile q is selected on `webpalm_dev` per §V9: the smallest
q in a candidate grid that maintains realized FPIR ≤ α_select on every
dev step. α_select = 1e-3 only (V9 lock); α=1e-4 reuses the same q for
single-shot evaluation on `webpalm_test`.

Usage:
    python -m exp1_baselines.eval.mitigation \
        --inpool-npz       <embeddings.npz> \
        --inpool-manifest  <manifest.csv> \
        --webpalm-npz      ~/research_data/webpalm/embeddings_mfn112.npz \
        --m-fit            30 50 80 120 \
        --t-step           10 \
        --order-seed       0 \
        --enroll-seed      0 \
        --K                3 \
        --B                1000 \
        --out              ~/research_data/webpalm/mitigation/tongji
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from exp1_baselines.datasets.manifest_io import (
    get_split_palm_ids,
    load_manifest,
)
from exp1_baselines.eval.orchestrator import (
    deterministic_future_order,
    select_enroll_sample_ids_sessioned,
)
from exp1_baselines.eval.score_matrix import (
    EmbeddingStore,
    enrolled_palm_set,
    select_enroll_sample_ids_per_palm,
)
from exp1_baselines.eval.thresholding import calibrate_threshold_at_fpir


# ---------------------------------------------------------------------------
# Plan-locked constants
# ---------------------------------------------------------------------------

# V9: λ tuning ban at α=1e-4. Tune at α=1e-3 only.
ALPHA_TUNE = 1e-3
ALPHA_SINGLESHOT = 1e-4

# V14a: candidate quantile grid (replaces λ grid). Maps roughly to one-sided
# normal quantiles for backwards-compatibility cross-reference but is used
# directly as empirical bootstrap quantile.
QUANTILE_GRID = [0.50, 0.75, 0.84, 0.90, 0.95, 0.975, 0.994]
LAMBDA_REFERENCE_TABLE = {
    0.50: 0.0, 0.75: 0.67, 0.84: 1.0, 0.90: 1.28,
    0.95: 1.64, 0.975: 1.96, 0.994: 2.5,
}

# V11: bootstrap reproducibility
BOOTSTRAP_SEED = 42

# V10: cosine bound for clipping
TAU_LOW = -1.0
TAU_HIGH = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cos_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B.T


def top1(wp: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    if gallery.shape[0] == 0:
        return np.zeros((wp.shape[0],), dtype=np.float32)
    return cos_sim(wp, gallery).max(axis=1)


def build_protos(
    store: EmbeddingStore, palm_ids: Sequence[int],
    enroll_session_id: str | None,
    enroll_phase_id: str | None,
    enroll_sample_ids: Sequence[int] | None,
    enroll_sample_ids_per_palm: Dict[int, Sequence[int]] | None,
) -> np.ndarray:
    protos = []
    for pid in sorted(int(x) for x in palm_ids):
        per_palm = (
            enroll_sample_ids_per_palm.get(pid)
            if enroll_sample_ids_per_palm is not None else enroll_sample_ids
        )
        idx = store.select_indices(
            palm_ids=[pid],
            session_id=enroll_session_id,
            phase_id=enroll_phase_id,
            sample_ids=per_palm,
        )
        if idx.size == 0:
            continue
        feats = store.embeddings[idx]
        m = feats.mean(axis=0)
        protos.append(m / (np.linalg.norm(m) + 1e-12))
    if not protos:
        D = int(store.embeddings.shape[1])
        return np.zeros((0, D), dtype=np.float32)
    return np.stack(protos, axis=0).astype(np.float32)


def fit_log_linear(ms: np.ndarray, taus: np.ndarray) -> tuple[float, float]:
    """OLS fit of τ = a + b · log(m). Returns (a, b)."""
    log_m = np.log(ms.astype(np.float64))
    X = np.stack([np.ones_like(log_m), log_m], axis=1)
    coef, *_ = np.linalg.lstsq(X, taus.astype(np.float64), rcond=None)
    return float(coef[0]), float(coef[1])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inpool-npz", type=Path, required=True)
    p.add_argument("--inpool-manifest", type=Path, required=True)
    p.add_argument("--webpalm-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--m-fit", type=int, nargs="+", default=[30, 50, 80, 120],
                   help="Gallery sizes used for fitting τ̂_pred(m)")
    p.add_argument("--t-step", type=int, default=10)
    p.add_argument("--order-seed", type=int, default=0)
    p.add_argument("--enroll-seed", type=int, default=0)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--enroll-session", type=str, default="session1")
    p.add_argument("--enroll-phase", type=str, default="")
    p.add_argument("--non-sessioned", action="store_true")
    p.add_argument("--B", type=int, default=1000)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def _opt(s):
        return None if (s == "" or s.lower() == "none") else s

    # ----- Load -----
    df = load_manifest(args.inpool_manifest)
    base_anchor_ids = get_split_palm_ids(df, "base_anchor")
    future_palm_ids = get_split_palm_ids(df, "future")
    future_order = deterministic_future_order(future_palm_ids, args.order_seed)
    t_max = len(future_order)

    inpool_store = EmbeddingStore.from_npz(args.inpool_npz)

    enroll_session_id = _opt(args.enroll_session)
    enroll_phase_id = _opt(args.enroll_phase)
    if args.non_sessioned:
        all_eval = list(set(base_anchor_ids) | set(future_palm_ids))
        palm_to_pool = {
            int(pid): int(inpool_store.select_indices(
                palm_ids=[int(pid)], sample_role="candidate"
            ).size) for pid in all_eval
        }
        enroll_per_palm = select_enroll_sample_ids_per_palm(
            palm_ids=all_eval, palm_to_pool_size=palm_to_pool,
            K=args.K, enroll_seed=args.enroll_seed,
        )
        enroll_sample_ids = None
    else:
        enroll_sample_ids = (
            select_enroll_sample_ids_sessioned(args.K, args.enroll_seed)
            if enroll_session_id is not None or enroll_phase_id is not None
            else None
        )
        enroll_per_palm = None

    wp = np.load(args.webpalm_npz, allow_pickle=True)
    wp_emb = wp["embeddings"]
    wp_split = wp["split"]
    wp_cal = wp_emb[wp_split == "calibration"]
    wp_dev = wp_emb[wp_split == "dev"]
    wp_test = wp_emb[wp_split == "test"]
    print(f"[in-pool] base_anchor={len(base_anchor_ids)} "
          f"future={len(future_palm_ids)} t_max={t_max}")
    print(f"[WebPalm] cal={wp_cal.shape[0]} dev={wp_dev.shape[0]} "
          f"test={wp_test.shape[0]}")

    # ----- Gallery cache for all m_fit + the full schedule -----
    schedule = list(range(0, t_max + 1, args.t_step))
    if schedule[-1] != t_max:
        schedule.append(t_max)
    # m_fit values must be ≤ t_max + len(base_anchor); requested m larger
    # than that get clamped to the maximum reachable gallery size.
    base_size = len(base_anchor_ids)
    m_fit_filt = [m for m in args.m_fit if m >= base_size]
    print(f"[m-fit] requested m={m_fit_filt} (base_anchor size={base_size})")
    # Map each requested m to a target step t and to the actual gallery size
    # at that step (which may differ if t hit t_max).
    m_request_to_t = {}
    for m in m_fit_filt:
        target_t = min(max(0, m - base_size), t_max)
        m_request_to_t[m] = target_t

    gallery_cache: Dict[int, np.ndarray] = {}
    needed_ts = sorted(set(list(m_request_to_t.values()) + schedule))
    for t in needed_ts:
        enrolled = enrolled_palm_set(base_anchor_ids, future_order, t)
        gallery_cache[t] = build_protos(
            inpool_store, enrolled,
            enroll_session_id=enroll_session_id,
            enroll_phase_id=enroll_phase_id,
            enroll_sample_ids=enroll_sample_ids,
            enroll_sample_ids_per_palm=enroll_per_palm,
        )

    # ----- Step 1: compute τ̂(m_k) for each m_k via tie-aware calibration -----
    # Deduplicate by actual_m: different requested m may map to same gallery
    # if they exceeded t_max.
    actual_m_to_t: Dict[int, int] = {}
    fit_tau_per_m: Dict[int, float] = {}
    for m_req, t in m_request_to_t.items():
        gal = gallery_cache[t]
        actual_m = int(gal.shape[0])
        if actual_m in actual_m_to_t:
            continue
        actual_m_to_t[actual_m] = t
        scores = top1(wp_cal, gal)
        tau = calibrate_threshold_at_fpir(scores, ALPHA_TUNE)
        fit_tau_per_m[actual_m] = float(tau)
    print(f"[fit] τ̂(m_k) at α={ALPHA_TUNE}: {fit_tau_per_m}")

    ms = np.array(sorted(fit_tau_per_m.keys()))
    taus = np.array([fit_tau_per_m[int(m)] for m in ms])
    a_hat, b_hat = fit_log_linear(ms, taus)
    print(f"[fit] τ̂_pred(m) = {a_hat:.4f} + {b_hat:.4f} · log(m)")

    # ----- Step 2: bootstrap τ̂_pred(m) on identity-resamples of cal -----
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_cal = wp_cal.shape[0]
    bootstrap_taus: Dict[int, np.ndarray] = {
        int(m): np.zeros(args.B, dtype=np.float64) for m in ms
    }
    bootstrap_predict_at: Dict[int, np.ndarray] = {}
    for b in range(args.B):
        idx = rng.integers(0, n_cal, size=n_cal)
        cal_b = wp_cal[idx]
        # τ̂(m_k) on resampled cal at every fit point
        taus_b = []
        for m in ms:
            gal = gallery_cache[actual_m_to_t[int(m)]]
            scores_b = top1(cal_b, gal)
            tau_b = calibrate_threshold_at_fpir(scores_b, ALPHA_TUNE)
            taus_b.append(tau_b)
            bootstrap_taus[int(m)][b] = tau_b
        # Refit log-linear on bootstrap pairs
        a_b, b_b = fit_log_linear(ms, np.array(taus_b))
        # Predict at every t in schedule
        for t in schedule:
            actual_m = max(1, gallery_cache[t].shape[0])
            if t not in bootstrap_predict_at:
                bootstrap_predict_at[t] = np.zeros(args.B, dtype=np.float64)
            bootstrap_predict_at[t][b] = a_b + b_b * np.log(actual_m)

    # ----- Step 3: dev-side q selection -----
    # For each candidate q, compute τ_safe(m(t)) = clip(Q_q(predict[t]), -1, 1)
    # and check whether realized FPIR on webpalm_dev <= ALPHA_TUNE at every t.
    def realize_dev_for_q(q: float) -> Dict[int, Dict[str, float]]:
        per_step = {}
        for t in schedule:
            tau_safe = float(np.clip(
                np.percentile(bootstrap_predict_at[t], q * 100),
                TAU_LOW, TAU_HIGH,
            ))
            scores = top1(wp_dev, gallery_cache[t])
            fpir = float((scores >= tau_safe).mean())
            per_step[t] = {"tau_safe": tau_safe, "fpir_dev": fpir}
        return per_step

    selected_q = None
    q_search_log = []
    for q in QUANTILE_GRID:
        dev_steps = realize_dev_for_q(q)
        worst_fpir = max(s["fpir_dev"] for s in dev_steps.values())
        all_under = all(s["fpir_dev"] <= ALPHA_TUNE for s in dev_steps.values())
        q_search_log.append({
            "q": q, "lambda_ref": LAMBDA_REFERENCE_TABLE[q],
            "dev_worst_fpir": worst_fpir,
            "passes_dev_constraint": all_under,
        })
        print(f"[q={q}] dev worst_fpir={worst_fpir:.5f} "
              f"{'PASS' if all_under else 'fail'}")
        if all_under and selected_q is None:
            selected_q = q
            selected_dev = dev_steps

    if selected_q is None:
        print(f"[WARN] no q in {QUANTILE_GRID} satisfies dev constraint at "
              f"α={ALPHA_TUNE}; using maximum q={QUANTILE_GRID[-1]} and flagging mitigation_failure=true")
        selected_q = QUANTILE_GRID[-1]
        selected_dev = realize_dev_for_q(selected_q)
        mitigation_failure_at_alpha_tune = True
    else:
        mitigation_failure_at_alpha_tune = False

    print(f"[selected] q={selected_q} (λ_ref={LAMBDA_REFERENCE_TABLE[selected_q]})")

    # ----- Step 4: single-shot test evaluation at α_tune AND α_singleshot -----
    # The same q/τ_safe are used at both α; α=1e-4 is single-shot per V9.
    test_results = {}
    clip_count = 0
    for alpha_label, alpha in [("alpha_1e-3", ALPHA_TUNE),
                                ("alpha_1e-4", ALPHA_SINGLESHOT)]:
        per_step = []
        for t in schedule:
            tau_predicted = float(np.percentile(
                bootstrap_predict_at[t], selected_q * 100
            ))
            tau_safe = float(np.clip(tau_predicted, TAU_LOW, TAU_HIGH))
            if tau_predicted != tau_safe:
                clip_count += 1
            scores = top1(wp_test, gallery_cache[t])
            n = scores.size
            fa = int((scores >= tau_safe).sum())
            fpir = fa / n if n else float("nan")
            per_step.append({
                "t": int(t),
                "gallery_size": int(gallery_cache[t].shape[0]),
                "tau_predicted": tau_predicted,
                "tau_safe": tau_safe,
                "false_accepts": fa,
                "fpir": fpir,
            })
        # Failure ladder per §V4
        violations = sum(1 for s in per_step if s["fpir"] > alpha)
        peak = max(s["fpir"] for s in per_step) if per_step else 0.0
        mean_fpir = float(np.mean([s["fpir"] for s in per_step]))
        if violations == 0 and mean_fpir <= 0.8 * alpha:
            tier = "pass_strong"
        elif violations == 0:
            tier = "pass_marginal"
        elif violations / len(per_step) < 0.1 and peak < 1.5 * alpha:
            tier = "conditional_pass"
        else:
            tier = "failure"
        test_results[alpha_label] = {
            "alpha": alpha,
            "steps": per_step,
            "violations_n": violations,
            "violations_rate": violations / len(per_step),
            "peak_fpir": peak,
            "peak_to_alpha": peak / alpha if alpha > 0 else float("inf"),
            "mean_fpir": mean_fpir,
            "tier": tier,
        }
        print(f"[test α={alpha:.0e}] tier={tier}, violations="
              f"{violations}/{len(per_step)}, peak/α={peak/alpha:.2f}x")

    # ----- Save -----
    out = {
        "method": "Conservative Gallery-Size-Aware Thresholding (V14a primary form)",
        "alpha_tune": ALPHA_TUNE,
        "alpha_singleshot": ALPHA_SINGLESHOT,
        "m_fit_requested": args.m_fit,
        "m_fit_actual": [int(m) for m in ms],
        "tau_pred_log_linear": {"a_hat": a_hat, "b_hat": b_hat},
        "fit_tau_per_m": fit_tau_per_m,
        "selected_q": selected_q,
        "selected_q_lambda_ref": LAMBDA_REFERENCE_TABLE[selected_q],
        "mitigation_failure_at_alpha_tune": mitigation_failure_at_alpha_tune,
        "q_search_log": q_search_log,
        "dev_per_step_at_selected_q": selected_dev,
        "test_results": test_results,
        "clip_activations": int(clip_count),
        "clip_rate": float(clip_count) / max(1, 2 * len(schedule)),
        "B_bootstrap": args.B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "webpalm_calibration_identity",
        "schedule": schedule,
        "t_max": t_max,
    }
    out_path = args.out / "mitigation_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[ok] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
