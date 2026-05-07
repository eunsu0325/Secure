"""Exp1 plotting (D1h-schema protocol JSONs).

Generates paper-ready figures from one or more protocol-result directories
emitted by `exp1_baselines/eval/orchestrator.py`.

Single-seed mode (one result dir):
    python experiments/plot_exp1.py \
        --results experiments/generated/tongji_full/protocols \
        --out     experiments/generated/tongji_full/figures \
        --tag     tongji_mfn

Multi-seed mode (many result dirs, one per seed combination):
    python experiments/plot_exp1.py \
        --results experiments/generated/tongji_multiseed/seed_*/protocols \
        --out     experiments/generated/tongji_multiseed/figures \
        --tag     tongji_mfn_multiseed

In multi-seed mode the curves are mean ± 95% CI bands across runs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


def _safe_float(v) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_results(results_dir: Path) -> Dict[str, object]:
    """Load all 4 protocol JSONs + meta from one result directory."""
    return {
        "a":      _load_json(results_dir / "protocol_a.json"),
        "b":      _load_json(results_dir / "protocol_b.json"),
        "cfixed": _load_json(results_dir / "protocol_c_fixed.json"),
        "crecal": _load_json(results_dir / "protocol_c_recal.json"),
        "meta":   _load_json(results_dir / "meta.json"),
    }


def _stack_metric(seed_runs: List[Dict[str, object]], protocol_key: str,
                  metric_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_axis, mean_array, ci_halfwidth_array) across runs.

    Each run is expected to be a list of step dicts; the t-axes must match.
    """
    runs_metric = []
    t_axis = None
    for r in seed_runs:
        steps = r[protocol_key]
        ts = np.array([s["t"] for s in steps])
        if t_axis is None:
            t_axis = ts
        elif not np.array_equal(t_axis, ts):
            raise ValueError(
                f"runs disagree on t-axis for {protocol_key}: "
                f"{ts} vs {t_axis}"
            )
        runs_metric.append([_safe_float(s.get(metric_key)) for s in steps])
    arr = np.array(runs_metric, dtype=np.float64)  # [n_runs, n_t]
    mean = np.nanmean(arr, axis=0)
    if arr.shape[0] > 1:
        std = np.nanstd(arr, axis=0, ddof=1)
        n = np.sum(~np.isnan(arr), axis=0)
        ci_half = 1.96 * std / np.sqrt(np.maximum(n, 1))
    else:
        ci_half = np.zeros_like(mean)
    return t_axis, mean, ci_half


# ---------------------------------------------------------------------------
# Paper main figure: 4-panel C trajectory diagnostic
# ---------------------------------------------------------------------------

def plot_main_diagnostic(
    seed_runs: List[Dict[str, object]],
    out_path: Path,
    tag: str,
) -> None:
    """4-panel C trajectory + B endpoint diagnostic.

    Panels:
        (top-left)  TPIR: C-recal vs C-fixed, with B endpoint marker
        (top-right) τ drift: C-recal vs C-fixed (flat) vs B (point)
        (bot-left)  External-FPIR: target-line, C-recal stays near target,
                    C-fixed drifts away
        (bot-right) Future-FPIR: C-recal trajectory (climbs near t=t_max)
    """
    target_fpir = _safe_float(seed_runs[0]["meta"].get("target_fpir"))
    n_runs = len(seed_runs)

    # --- TPIR ---
    t_cr, tpir_cr, ci_cr = _stack_metric(seed_runs, "crecal", "tpir")
    _,     tpir_cf, ci_cf = _stack_metric(seed_runs, "cfixed", "tpir")
    # Protocol B is a single endpoint per run
    b_tpir = np.array([_safe_float(r["b"]["tpir"]) for r in seed_runs])
    b_tpir_mean = float(np.nanmean(b_tpir))

    # --- tau ---
    _, tau_cr, ci_tau_cr = _stack_metric(seed_runs, "crecal", "tau")
    _, tau_cf, ci_tau_cf = _stack_metric(seed_runs, "cfixed", "tau")
    b_tau = np.array([_safe_float(r["b"]["tau"]) for r in seed_runs])
    b_tau_mean = float(np.nanmean(b_tau))

    # --- External FPIR ---
    _, ext_cr, ci_ext_cr = _stack_metric(seed_runs, "crecal", "external_fpir")
    _, ext_cf, ci_ext_cf = _stack_metric(seed_runs, "cfixed", "external_fpir")
    b_ext = np.array([_safe_float(r["b"]["external_fpir"]) for r in seed_runs])
    b_ext_mean = float(np.nanmean(b_ext))

    # --- Future FPIR (C-recal; C-fixed too if available) ---
    _, fut_cr, ci_fut_cr = _stack_metric(seed_runs, "crecal", "future_fpir")
    _, fut_cf, ci_fut_cf = _stack_metric(seed_runs, "cfixed", "future_fpir")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    title_suffix = f" (n={n_runs} seeds)" if n_runs > 1 else ""

    # TPIR
    ax = axes[0, 0]
    ax.plot(t_cr, tpir_cr, "o-", color="C0", label="C-recal")
    ax.fill_between(t_cr, tpir_cr - ci_cr, tpir_cr + ci_cr,
                    alpha=0.20, color="C0")
    ax.plot(t_cr, tpir_cf, "s--", color="C3", label="C-fixed")
    ax.fill_between(t_cr, tpir_cf - ci_cf, tpir_cf + ci_cf,
                    alpha=0.15, color="C3")
    ax.axhline(b_tpir_mean, linestyle=":", color="k",
               label=f"B endpoint = {b_tpir_mean:.3f}")
    ax.set_xlabel("enrollment step t")
    ax.set_ylabel("TPIR")
    ax.set_title(f"{tag} — TPIR trajectory{title_suffix}")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")

    # tau drift
    ax = axes[0, 1]
    ax.plot(t_cr, tau_cr, "o-", color="C0", label="C-recal τ (drift)")
    ax.fill_between(t_cr, tau_cr - ci_tau_cr, tau_cr + ci_tau_cr,
                    alpha=0.20, color="C0")
    ax.plot(t_cr, tau_cf, "s--", color="C3", label="C-fixed τ (one-time)")
    ax.fill_between(t_cr, tau_cf - ci_tau_cf, tau_cf + ci_tau_cf,
                    alpha=0.15, color="C3")
    ax.axhline(b_tau_mean, linestyle=":", color="k",
               label=f"B τ = {b_tau_mean:.3f}")
    ax.set_xlabel("enrollment step t")
    ax.set_ylabel("τ")
    ax.set_title(f"{tag} — Threshold drift{title_suffix}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # External FPIR
    ax = axes[1, 0]
    ax.plot(t_cr, ext_cr, "o-", color="C0", label="C-recal Ext-FPIR")
    ax.fill_between(t_cr, ext_cr - ci_ext_cr, ext_cr + ci_ext_cr,
                    alpha=0.20, color="C0")
    ax.plot(t_cr, ext_cf, "s--", color="C3", label="C-fixed Ext-FPIR (drifts!)")
    ax.fill_between(t_cr, ext_cf - ci_ext_cf, ext_cf + ci_ext_cf,
                    alpha=0.15, color="C3")
    ax.axhline(target_fpir, linestyle=":", color="gray",
               label=f"target = {target_fpir:.2f}")
    ax.axhline(b_ext_mean, linestyle="--", color="k", alpha=0.6,
               label=f"B Ext-FPIR = {b_ext_mean:.3f}")
    ax.set_xlabel("enrollment step t")
    ax.set_ylabel("External-FPIR")
    ax.set_title(f"{tag} — External-FPIR drift{title_suffix}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    # Future FPIR (only meaningful for C-recal; t=t_max is NaN)
    ax = axes[1, 1]
    ax.plot(t_cr, fut_cr, "o-", color="C0", label="C-recal Future-FPIR")
    ax.fill_between(t_cr, fut_cr - ci_fut_cr, fut_cr + ci_fut_cr,
                    alpha=0.20, color="C0")
    ax.plot(t_cr, fut_cf, "s--", color="C3", alpha=0.5,
            label="C-fixed Future-FPIR")
    ax.axhline(target_fpir, linestyle=":", color="gray",
               label=f"target = {target_fpir:.2f}")
    ax.set_xlabel("enrollment step t")
    ax.set_ylabel("Future-FPIR")
    ax.set_title(f"{tag} — Future-unknown FPIR (B has no equivalent)"
                 + title_suffix)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supplementary: Protocol A rank-1 with Wilson CI
# ---------------------------------------------------------------------------

def plot_protocol_a(
    seed_runs: List[Dict[str, object]],
    out_path: Path,
    tag: str,
) -> None:
    n_runs = len(seed_runs)
    t_axis, rank1, ci_h = _stack_metric(seed_runs, "a", "rank1")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_axis, rank1, "o-", color="C2",
            label=f"rank-1 (n={n_runs})" if n_runs > 1 else "rank-1")
    ax.fill_between(t_axis, rank1 - ci_h, rank1 + ci_h, alpha=0.20, color="C2")
    # Per-step Wilson CI (single-seed only)
    if n_runs == 1:
        steps = seed_runs[0]["a"]
        cis = [s.get("rank1_ci95", [None, None]) for s in steps]
        lows = np.array([_safe_float(c[0]) for c in cis])
        highs = np.array([_safe_float(c[1]) for c in cis])
        ax.fill_between(t_axis, lows, highs, alpha=0.15, color="C2",
                        label="Wilson 95% CI")
    ax.set_xlabel("enrollment step t")
    ax.set_ylabel("Protocol A rank-1")
    ax.set_title(f"{tag} — Protocol A (closed-set rank-1)")
    ax.set_ylim(0.5, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Plot Exp1 protocol results")
    p.add_argument("--results", nargs="+", required=True,
                   help="One or more protocol result dirs")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tag", required=True, type=str)
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_runs: List[Dict[str, object]] = []
    for r in args.results:
        rp = Path(r)
        if not rp.is_dir():
            raise FileNotFoundError(f"results dir not found: {rp}")
        seed_runs.append(load_results(rp))

    print(f"loaded {len(seed_runs)} run(s)")

    main_path = out_dir / f"{args.tag}_main_diagnostic.png"
    plot_main_diagnostic(seed_runs, main_path, args.tag)
    print(f"wrote {main_path}")

    a_path = out_dir / f"{args.tag}_protocol_a.png"
    plot_protocol_a(seed_runs, a_path, args.tag)
    print(f"wrote {a_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
