"""Plot Exp1-S security-stress trajectories (plan §C8 metrics).

Reads `exp1s_results.json` from `exp1s_orchestrator.py` for each in-pool
dataset and produces:
  - `exp1s_main.png`: 4-panel figure
      (a) realized FPIR trajectory at α=1e-3 (C-fixed-S vs C-recal-S vs B-S)
      (b) realized FPIR trajectory at α=1e-4 (same)
      (c) τ trajectory across t (C-fixed-S flat, C-recal-S adaptive)
      (d) violation rate / peak ratio bar chart per (dataset, α)
  - `exp1s_violation_table.csv`: tabular summary

Uses log y-axis for FPIR panels because the violation can be 1e-3 to 1e-1.

Usage:
    python experiments/plot_exp1s.py \
        --results \
            "Tongji=~/research_data/webpalm/exp1s/tongji/exp1s_results.json" \
            "IITD=~/research_data/webpalm/exp1s/iitd/exp1s_results.json" \
            "BJTU=~/research_data/webpalm/exp1s/bjtu/exp1s_results.json" \
        --out ~/research_data/webpalm/exp1s/figures
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: str) -> Dict[str, object]:
    return json.loads(Path(path).expanduser().read_text())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True,
                   help="label=path pairs, e.g., Tongji=path/exp1s_results.json")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out = args.out.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = {}
    for entry in args.results:
        if "=" not in entry:
            raise ValueError(f"--results expects label=path, got {entry!r}")
        lab, pth = entry.split("=", 1)
        datasets[lab] = _load(pth)
        print(f"[load] {lab}: t_max={datasets[lab]['t_max']}, "
              f"schedule={len(datasets[lab]['schedule'])} steps")

    targets = [1e-3, 1e-4]
    colors = {"Tongji": "C0", "IITD": "C2", "BJTU": "C4"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # ---------- (a) and (b): realized FPIR trajectory ----------
    for col, alpha in enumerate(targets):
        ax = axes[0, col]
        akey = f"alpha_{alpha:.0e}"
        for label, data in datasets.items():
            c = colors.get(label, "C5")
            res = data["results"][akey]
            cf = res["C_fixed_S"]["steps"]
            cr = res["C_recal_S"]["steps"]
            ts = [s["t"] for s in cf]
            cf_fpir = [s["fpir"] for s in cf]
            cr_fpir = [s["fpir"] for s in cr]
            cf_lo = [s["ci_lo"] for s in cf]
            cf_hi = [s["ci_hi"] for s in cf]
            cr_lo = [s["ci_lo"] for s in cr]
            cr_hi = [s["ci_hi"] for s in cr]

            ax.plot(ts, cf_fpir, "s--", color=c, alpha=0.85,
                    label=f"{label} C-fixed-S")
            ax.fill_between(ts, cf_lo, cf_hi, alpha=0.10, color=c)
            ax.plot(ts, cr_fpir, "o-", color=c,
                    label=f"{label} C-recal-S")
            ax.fill_between(ts, cr_lo, cr_hi, alpha=0.18, color=c)

            # B-S endpoint marker
            b_fpir = res["B_S"]["fpir"]
            ax.plot([ts[-1]], [b_fpir], marker="*", markersize=14,
                    color=c, markeredgecolor="black", markeredgewidth=1.0,
                    label=f"{label} B-S endpoint")

        ax.axhline(alpha, linestyle=":", color="red", linewidth=1.5,
                   label=f"target α = {alpha:.0e}")
        ax.set_yscale("log")
        ax.set_xlabel("enrollment step t")
        ax.set_ylabel("realized FPIR")
        ax.set_title(f"Exp1-S realized FPIR vs t (α = {alpha:.0e})")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="upper left", ncol=2)

    # ---------- (c) τ trajectory ----------
    ax = axes[1, 0]
    akey = "alpha_1e-03"
    for label, data in datasets.items():
        c = colors.get(label, "C5")
        cf = data["results"][akey]["C_fixed_S"]["steps"]
        cr = data["results"][akey]["C_recal_S"]["steps"]
        ts = [s["t"] for s in cf]
        ax.plot(ts, [s["tau"] for s in cf], "s--", color=c, alpha=0.7,
                label=f"{label} τ-fixed")
        ax.plot(ts, [s["tau"] for s in cr], "o-", color=c,
                label=f"{label} τ-recal")
    ax.set_xlabel("enrollment step t")
    ax.set_ylabel("threshold τ")
    ax.set_title("Threshold trajectory across datasets (α = 1e-3)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    # ---------- (d) Violation summary bars ----------
    ax = axes[1, 1]
    labels_for_bars: List[str] = []
    cf_violation = []
    cr_violation = []
    cf_peak = []
    cr_peak = []
    bar_colors = []
    for label, data in datasets.items():
        for alpha in targets:
            akey = f"alpha_{alpha:.0e}"
            r = data["results"][akey]
            labels_for_bars.append(f"{label}\nα={alpha:.0e}")
            cf_violation.append(r["C_fixed_S"]["violation"]["violation_rate"])
            cr_violation.append(r["C_recal_S"]["violation"]["violation_rate"])
            cf_peak.append(r["C_fixed_S"]["violation"]["peak_to_target_ratio"])
            cr_peak.append(r["C_recal_S"]["violation"]["peak_to_target_ratio"])
            bar_colors.append(colors.get(label, "C5"))

    x = np.arange(len(labels_for_bars))
    w = 0.35
    ax.bar(x - w / 2, cf_violation, w, label="C-fixed-S violation rate",
           color=[matplotlib.colors.to_rgba(c, 0.5) for c in bar_colors],
           edgecolor=bar_colors, linewidth=1.5)
    ax.bar(x + w / 2, cr_violation, w, label="C-recal-S violation rate",
           color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_for_bars, fontsize=8)
    ax.set_ylabel("violation rate (fraction of t steps)")
    ax.set_title("Security violation rate (CI lower bound > α)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)

    fig.suptitle("Exp1-S WebPalm Security Stress (3 datasets, MFN-112)",
                 fontsize=13)
    fig.tight_layout()
    out_path = args.out / "exp1s_main.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[save] {out_path}")

    # ---------- CSV summary ----------
    csv_path = args.out / "exp1s_violation_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "alpha",
            "B_S_realized_fpir",
            "C_fixed_violation_rate", "C_fixed_peak_over_alpha",
            "C_fixed_first_violation_step",
            "C_recal_violation_rate", "C_recal_peak_over_alpha",
            "C_recal_first_violation_step",
        ])
        for label, data in datasets.items():
            for alpha in targets:
                r = data["results"][f"alpha_{alpha:.0e}"]
                w.writerow([
                    label, f"{alpha:.0e}",
                    f"{r['B_S']['fpir']:.5f}",
                    f"{r['C_fixed_S']['violation']['violation_rate']:.3f}",
                    f"{r['C_fixed_S']['violation']['peak_to_target_ratio']:.2f}",
                    r['C_fixed_S']['violation']['first_violation_step'],
                    f"{r['C_recal_S']['violation']['violation_rate']:.3f}",
                    f"{r['C_recal_S']['violation']['peak_to_target_ratio']:.2f}",
                    r['C_recal_S']['violation']['first_violation_step'],
                ])
    print(f"[save] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
