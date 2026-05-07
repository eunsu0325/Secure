"""Multi-dataset overlay figure: TPIR / tau / Ext-FPIR / Future-FPIR.

Plots three datasets' C-recal trajectories side by side on the same axes
so the universality of the static-vs-trajectory effect is visible.

Each entry passed via `--datasets <label>=<glob_to_protocols_dirs>` aggregates
across (order_seed × enroll_seed) within that dataset and draws a single
mean curve with a 95% CI band. Protocol B endpoints (mean across seeds) are
drawn as horizontal reference lines per dataset for the TPIR panel.

Usage:
    python experiments/plot_multi_dataset.py \
        --datasets \
            "Tongji=experiments/generated/tongji_multiseed/order*/protocols" \
            "IITD=experiments/generated/iitd_multiseed/order*/protocols" \
            "BJTU=experiments/generated/bjtu_multiseed/order*/protocols" \
        --out experiments/generated/multi_dataset_figures \
        --tag exp1_multi
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe(v) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _load_runs(globpat: str) -> List[Dict]:
    runs = []
    for d in sorted(glob.glob(globpat)):
        p = Path(d)
        runs.append({
            "a":      json.loads((p / "protocol_a.json").read_text()),
            "b":      json.loads((p / "protocol_b.json").read_text()),
            "cfixed": json.loads((p / "protocol_c_fixed.json").read_text()),
            "crecal": json.loads((p / "protocol_c_recal.json").read_text()),
            "meta":   json.loads((p / "meta.json").read_text()),
        })
    return runs


def _stack(runs: List[Dict], proto: str, key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    runs_metric = []
    t_axis = None
    for r in runs:
        steps = r[proto]
        ts = np.array([s["t"] for s in steps])
        if t_axis is None:
            t_axis = ts
        runs_metric.append([_safe(s.get(key)) for s in steps])
    arr = np.array(runs_metric, dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    if arr.shape[0] > 1:
        std = np.nanstd(arr, axis=0, ddof=1)
        n = np.sum(~np.isnan(arr), axis=0)
        ci_h = 1.96 * std / np.sqrt(np.maximum(n, 1))
    else:
        ci_h = np.zeros_like(mean)
    return t_axis, mean, ci_h


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True,
                   help='label=glob pairs, e.g. "Tongji=path/to/order*/protocols"')
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tag", required=True, type=str)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: Dict[str, List[Dict]] = {}
    for entry in args.datasets:
        if "=" not in entry:
            raise ValueError(f"expected label=glob, got {entry!r}")
        label, gp = entry.split("=", 1)
        runs = _load_runs(gp)
        if not runs:
            print(f"[skip] {label}: no runs at {gp}")
            continue
        datasets[label] = runs
        print(f"[load] {label}: {len(runs)} runs")

    if not datasets:
        print("no datasets loaded")
        return 1

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ["C0", "C2", "C4", "C5"]

    for i, (label, runs) in enumerate(datasets.items()):
        c = colors[i % len(colors)]
        n_runs = len(runs)

        # TPIR
        ax = axes[0, 0]
        t, m, h = _stack(runs, "crecal", "tpir")
        ax.plot(t, m, "o-", color=c, label=f"{label} (n={n_runs})")
        ax.fill_between(t, m - h, m + h, alpha=0.18, color=c)
        # B endpoint
        b_tpir = float(np.nanmean([_safe(r["b"]["tpir"]) for r in runs]))
        ax.axhline(b_tpir, linestyle="--", color=c, alpha=0.5,
                   label=f"{label} B = {b_tpir:.3f}")

        # tau
        ax = axes[0, 1]
        t, m, h = _stack(runs, "crecal", "tau")
        ax.plot(t, m, "o-", color=c, label=f"{label} τ-recal")
        ax.fill_between(t, m - h, m + h, alpha=0.18, color=c)
        t, m, h = _stack(runs, "cfixed", "tau")
        ax.plot(t, m, ":", color=c, alpha=0.6, label=f"{label} τ-fixed")

        # Ext-FPIR
        ax = axes[1, 0]
        t, m, h = _stack(runs, "crecal", "external_fpir")
        ax.plot(t, m, "o-", color=c, label=f"{label} C-recal")
        ax.fill_between(t, m - h, m + h, alpha=0.18, color=c)
        t, m, h = _stack(runs, "cfixed", "external_fpir")
        ax.plot(t, m, ":", color=c, alpha=0.6, label=f"{label} C-fixed (drift)")

        # Future-FPIR
        ax = axes[1, 1]
        t, m, h = _stack(runs, "crecal", "future_fpir")
        ax.plot(t, m, "o-", color=c, label=f"{label}")
        ax.fill_between(t, m - h, m + h, alpha=0.18, color=c)

    target_fpir = float(np.nanmean([
        _safe(r["meta"].get("target_fpir"))
        for runs in datasets.values() for r in runs
    ]))

    axes[0, 0].set_title(f"{args.tag} — C-recal TPIR + B endpoints")
    axes[0, 0].set_xlabel("enrollment step t")
    axes[0, 0].set_ylabel("TPIR")
    axes[0, 0].set_ylim(0.0, 1.02)  # widened to accommodate underfit datasets (e.g., BJTU N_train=45)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="lower left")

    axes[0, 1].set_title(f"{args.tag} — Threshold drift across datasets")
    axes[0, 1].set_xlabel("enrollment step t")
    axes[0, 1].set_ylabel("τ")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(fontsize=8, loc="lower right")

    axes[1, 0].set_title(f"{args.tag} — External-FPIR (C-recal vs C-fixed drift)")
    axes[1, 0].set_xlabel("enrollment step t")
    axes[1, 0].set_ylabel("External-FPIR")
    axes[1, 0].axhline(target_fpir, linestyle="--", color="gray",
                       label=f"target = {target_fpir:.2f}")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(fontsize=8, loc="upper left")

    axes[1, 1].set_title(f"{args.tag} — Future-FPIR (C-recal; B has no equivalent)")
    axes[1, 1].set_xlabel("enrollment step t")
    axes[1, 1].set_ylabel("Future-FPIR")
    axes[1, 1].axhline(target_fpir, linestyle="--", color="gray",
                       label=f"target = {target_fpir:.2f}")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = out_dir / f"{args.tag}_main.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
