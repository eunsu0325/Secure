"""Plot Phase 4 mitigation comparison vs C-fixed-S / C-recal-S baselines.

For each (dataset, alpha) panel, overlays realized FPIR trajectories of:
  - C-fixed-S      (stale tau_0)            from exp1s_results.json
  - C-recal-S      (per-step tau_t)         from exp1s_results.json
  - Mitigation     (Conservative GSAT)      from mitigation_results.json

Reference line at target alpha. Log y-axis. Annotations: violation count and
peak/alpha ratio per series.

Usage:
    python experiments/plot_mitigation.py \
        --exp1s    Tongji=~/research_data/webpalm/exp1s/tongji/exp1s_results.json \
                   IITD=~/research_data/webpalm/exp1s/iitd/exp1s_results.json \
                   BJTU=~/research_data/webpalm/exp1s/bjtu/exp1s_results.json \
        --mitig    Tongji=~/research_data/webpalm/mitigation/tongji/mitigation_results.json \
                   IITD=~/research_data/webpalm/mitigation/iitd/mitigation_results.json \
                   BJTU=~/research_data/webpalm/mitigation/bjtu/mitigation_results.json \
        --out      ~/research_data/webpalm/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_pairs(items: List[str]) -> Dict[str, Path]:
    out = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"expected label=path, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = Path(v).expanduser()
    return out


def _load(path: Path) -> Dict:
    return json.loads(path.read_text())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp1s", nargs="+", required=True, help="label=path pairs")
    p.add_argument("--mitig", nargs="+", required=True, help="label=path pairs")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out = args.out.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)

    exp1s = {k: _load(v) for k, v in _parse_pairs(args.exp1s).items()}
    mitig = {k: _load(v) for k, v in _parse_pairs(args.mitig).items()}

    labels = list(exp1s.keys())
    targets = [(1e-3, "alpha_1e-03", "alpha_1e-3"),
               (1e-4, "alpha_1e-04", "alpha_1e-4")]

    fig, axes = plt.subplots(2, len(labels), figsize=(5.5 * len(labels), 9),
                              sharey="row")
    if len(labels) == 1:
        axes = axes.reshape(2, 1)

    for col, label in enumerate(labels):
        ed = exp1s[label]
        md = mitig[label]
        for row, (alpha, key_e, key_m) in enumerate(targets):
            ax = axes[row, col]
            res_e = ed["results"][key_e]
            cf = res_e["C_fixed_S"]["steps"]
            cr = res_e["C_recal_S"]["steps"]
            ts_e = [s["t"] for s in cf]
            cf_fpir = [s["fpir"] for s in cf]
            cf_lo = [s["ci_lo"] for s in cf]
            cf_hi = [s["ci_hi"] for s in cf]
            cr_fpir = [s["fpir"] for s in cr]
            cr_lo = [s["ci_lo"] for s in cr]
            cr_hi = [s["ci_hi"] for s in cr]

            # mitigation steps
            mit = md["test_results"][key_m]["steps"]
            ts_m = [s["t"] for s in mit]
            mit_fpir = [s["fpir"] for s in mit]

            # plot — clip very small values for log axis
            eps = 1e-7
            ax.plot(ts_e, [max(v, eps) for v in cf_fpir], "s--",
                    color="C3", label="C-fixed-S", alpha=0.85)
            ax.fill_between(ts_e, [max(v, eps) for v in cf_lo],
                             [max(v, eps) for v in cf_hi], alpha=0.10, color="C3")
            ax.plot(ts_e, [max(v, eps) for v in cr_fpir], "o-",
                    color="C2", label="C-recal-S")
            ax.fill_between(ts_e, [max(v, eps) for v in cr_lo],
                             [max(v, eps) for v in cr_hi], alpha=0.18, color="C2")
            ax.plot(ts_m, [max(v, eps) for v in mit_fpir], "^-",
                    color="C0", label="Mitigation (GSAT)", linewidth=1.7)

            ax.axhline(alpha, linestyle=":", color="red", linewidth=1.5,
                       label=f"target α = {alpha:.0e}")
            ax.set_yscale("log")
            ax.set_xlabel("enrollment step t")
            if col == 0:
                ax.set_ylabel("realized FPIR (log scale)")
            tier = md["test_results"][key_m]["tier"]
            n_v = md["test_results"][key_m]["violations_n"]
            n_t = len(mit)
            peak_x = md["test_results"][key_m]["peak_to_alpha"]
            ax.set_title(
                f"{label}, α = {alpha:.0e}\n"
                f"GSAT tier={tier}, violations={n_v}/{n_t}, peak/α={peak_x:.2f}×"
            )
            ax.grid(alpha=0.3, which="both")
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="lower right", ncol=1)

    fig.suptitle(
        "Mitigation: Conservative Gallery-Size-Aware Thresholding (V14a) "
        "vs C-fixed-S / C-recal-S",
        fontsize=13,
    )
    fig.tight_layout()
    out_path = args.out / "mitigation_main.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[save] {out_path}")

    # ---- Companion figure: tau trajectory comparison ----
    fig2, axes2 = plt.subplots(1, len(labels), figsize=(5.5 * len(labels), 4.5),
                                sharey=False)
    if len(labels) == 1:
        axes2 = [axes2]

    for col, label in enumerate(labels):
        ax = axes2[col]
        ed = exp1s[label]
        md = mitig[label]
        cf = ed["results"]["alpha_1e-03"]["C_fixed_S"]["steps"]
        cr = ed["results"]["alpha_1e-03"]["C_recal_S"]["steps"]
        mit = md["test_results"]["alpha_1e-3"]["steps"]
        ts_e = [s["t"] for s in cf]
        ax.plot(ts_e, [s["tau"] for s in cf], "s--",
                color="C3", label="τ-fixed (stale)")
        ax.plot(ts_e, [s["tau"] for s in cr], "o-",
                color="C2", label="τ-recal (oracle)")
        ax.plot([s["t"] for s in mit], [s["tau_safe"] for s in mit], "^-",
                color="C0", label=f"τ_safe (q={md['selected_q']})")
        ax.set_xlabel("enrollment step t")
        ax.set_ylabel("threshold τ")
        ax.set_title(f"{label} threshold trajectories (α = 1e-3)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig2.suptitle(
        "Mitigation thresholds vs C-fixed-S / C-recal-S (V14a empirical "
        "bootstrap quantile + V10 cosine clip)",
        fontsize=12,
    )
    fig2.tight_layout()
    out_path2 = args.out / "mitigation_tau.png"
    fig2.savefig(out_path2, dpi=150)
    plt.close(fig2)
    print(f"[save] {out_path2}")

    # ---- CSV summary table for paper ----
    csv_path = args.out / "mitigation_summary.csv"
    with csv_path.open("w") as f:
        f.write(",".join([
            "dataset", "alpha", "method",
            "violations_n", "violations_rate",
            "peak_fpir", "peak_to_alpha", "mean_fpir", "tier",
            "selected_q",
        ]) + "\n")
        for label in labels:
            md = mitig[label]
            ed = exp1s[label]
            for alpha, key_e, key_m in targets:
                # GSAT mitigation row
                tr = md["test_results"][key_m]
                f.write(",".join(str(x) for x in [
                    label, f"{alpha:.0e}", "Mitigation_GSAT",
                    tr["violations_n"], f"{tr['violations_rate']:.3f}",
                    f"{tr['peak_fpir']:.5g}", f"{tr['peak_to_alpha']:.2f}",
                    f"{tr['mean_fpir']:.5g}", tr["tier"],
                    md["selected_q"],
                ]) + "\n")
                # Baseline rows from exp1s
                for proto_key, proto_label in [("C_fixed_S", "C-fixed-S"),
                                                ("C_recal_S", "C-recal-S")]:
                    v = ed["results"][key_e][proto_key]["violation"]
                    steps = ed["results"][key_e][proto_key]["steps"]
                    fpirs = [s["fpir"] for s in steps]
                    f.write(",".join(str(x) for x in [
                        label, f"{alpha:.0e}", proto_label,
                        int(v["violation_rate"] * len(steps)),
                        f"{v['violation_rate']:.3f}",
                        f"{max(fpirs):.5g}",
                        f"{v['peak_to_target_ratio']:.2f}",
                        f"{sum(fpirs)/len(fpirs):.5g}",
                        "n/a", "n/a",
                    ]) + "\n")
    print(f"[save] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
