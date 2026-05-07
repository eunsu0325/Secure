"""Plot Protocol C curves and score distributions.

Generates two figures per model:
  - <tag>_protocol_c_curve.png     : enrollment size t vs. metrics
  - <tag>_score_distributions.png  : known/future/external score histograms

Usage::

    python exp1_baselines/plot_protocol_c.py \\
        --results_dir exp1_baselines/results \\
        --output_dir  exp1_baselines/figures \\
        --tags mfn ir50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _safe(values: List, default=np.nan) -> np.ndarray:
    """Convert a list possibly containing None/null to a float array with NaN."""
    out = []
    for v in values:
        if v is None:
            out.append(default)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(default)
    return np.asarray(out, dtype=np.float64)


def plot_protocol_c_curve(res_c: Dict, output_path: Path, tag: str) -> None:
    steps = res_c["steps"]
    t_vals = np.asarray([s["t"] for s in steps], dtype=np.int64)

    known_acc = _safe([s.get("known_identification_acceptance_at_tau") for s in steps])
    known_misid = _safe([s.get("known_misidentification_acceptance_at_tau") for s in steps])
    known_rank1 = _safe([s.get("known_rank1") for s in steps])

    future_far = _safe([s.get("future_false_accept_rate_at_tau") for s in steps])
    future_rej = _safe([s.get("future_rejection_rate_at_tau") for s in steps])

    external_far = _safe([s.get("external_false_accept_rate_at_tau") for s in steps])
    external_rej = _safe([s.get("external_rejection_rate_at_tau") for s in steps])

    dir_future = _safe([s.get("DIR_at_FPIR1pct_known_vs_future") for s in steps])
    dir_external = _safe([s.get("DIR_at_FPIR1pct_known_vs_external") for s in steps])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(t_vals, known_rank1, marker="o", label="known rank1")
    ax.plot(t_vals, known_acc, marker="s", label="known correct accept @tau")
    ax.plot(t_vals, known_misid, marker="^", label="known misidentification @tau")
    ax.set_xlabel("enrollment size t")
    ax.set_ylabel("rate")
    ax.set_title(f"{tag} — Known outcomes (Protocol C)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(t_vals, future_far, marker="o", label="future FAR @tau")
    ax.plot(t_vals, future_rej, marker="s", label="future rejection @tau")
    ax.set_xlabel("enrollment size t")
    ax.set_ylabel("rate")
    ax.set_title(f"{tag} — Remaining-future outcomes (Protocol C)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(t_vals, external_far, marker="o", label="external FAR @tau")
    ax.plot(t_vals, external_rej, marker="s", label="external rejection @tau")
    ax.set_xlabel("enrollment size t")
    ax.set_ylabel("rate")
    ax.set_title(f"{tag} — External-test outcomes (Protocol C)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t_vals, dir_future, marker="o", label="DIR@FPIR=1% (future)")
    ax.plot(t_vals, dir_external, marker="s", label="DIR@FPIR=1% (external)")
    ax.set_xlabel("enrollment size t")
    ax.set_ylabel("DIR")
    ax.set_title(f"{tag} — Curve-based DIR@FPIR (Protocol C)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_score_distributions(res_c: Dict, output_path: Path, tag: str,
                             t_marks: Optional[List[int]] = None) -> None:
    sd = res_c.get("score_distributions", {})
    if not sd:
        return
    available_t = sorted(int(k.split("_")[1]) for k in sd.keys())
    if t_marks is None:
        t_marks = []
        for cand in [10, 50, 100, 150]:
            if cand in available_t:
                t_marks.append(cand)
        if not t_marks:
            t_marks = [available_t[len(available_t) // 2]]

    fig, axes = plt.subplots(1, len(t_marks), figsize=(4.5 * len(t_marks), 4),
                             sharey=True, squeeze=False)
    bins = np.linspace(-0.2, 1.0, 60)
    for i, t in enumerate(t_marks):
        key = f"t_{t}"
        ax = axes[0, i]
        if key not in sd:
            ax.text(0.5, 0.5, f"no data t={t}", transform=ax.transAxes, ha="center")
            continue
        known = np.asarray(sd[key].get("known_top1_scores", []), dtype=np.float64)
        future = np.asarray(sd[key].get("future_top1_scores", []), dtype=np.float64)
        external = np.asarray(sd[key].get("external_top1_scores", []), dtype=np.float64)
        if known.size > 0:
            ax.hist(known, bins=bins, alpha=0.5, label=f"known (n={known.size})",
                    color="C2", density=True)
        if future.size > 0:
            ax.hist(future, bins=bins, alpha=0.5, label=f"future (n={future.size})",
                    color="C1", density=True)
        if external.size > 0:
            ax.hist(external, bins=bins, alpha=0.5, label=f"external (n={external.size})",
                    color="C3", density=True)
        ax.axvline(res_c.get("tau", float("nan")), color="k", linestyle="--",
                   label=f"tau={res_c.get('tau', float('nan')):.3f}")
        ax.set_title(f"{tag} t={t}")
        ax.set_xlabel("top1 cosine score")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Plot Protocol C results")
    parser.add_argument("--results_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--tags", nargs="+", default=["mfn", "ir50"])
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = (PROJECT_ROOT / results_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for tag in args.tags:
        json_path = results_dir / f"{tag}_protocol_c.json"
        if not json_path.exists():
            print(f"[skip] {json_path} not found")
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            res_c = json.load(f)

        curve_path = output_dir / f"{tag}_protocol_c_curve.png"
        plot_protocol_c_curve(res_c, curve_path, tag)
        print(f"[save] {curve_path}")

        sd_path = output_dir / f"{tag}_score_distributions.png"
        plot_score_distributions(res_c, sd_path, tag)
        print(f"[save] {sd_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
