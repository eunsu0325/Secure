"""Plot multi-backbone (MFN vs CCNet) trajectory comparison (plan §V17.7).

For each (dataset, target_fpir) panel, overlays:
  - MFN-ArcFace B endpoint star + C-recal trajectory + tau drift
  - CCNet-ArcFace B endpoint star + C-recal trajectory + tau drift

Optionally also shows C-fixed-S baseline trajectories for visual contrast.

Inputs are the per-backbone orchestrator outputs in
  experiments/generated/<dataset>_full/protocols/protocol_b.json
  experiments/generated/<dataset>_full/protocols/protocol_c_recal.json
  experiments/generated/<dataset>_full/protocols/protocol_c_fixed.json

For CCNet, the same files appear under
  experiments/generated/<dataset>_full/ccnet_protocols/

Usage:
    python experiments/plot_multi_backbone.py \
        --datasets Tongji=experiments/generated/tongji_full \
                   IITD=experiments/generated/iitd_full \
        --out experiments/figures
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


def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


def _parse_pairs(items: List[str]) -> Dict[str, Path]:
    out = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"expected label=path, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = Path(v).expanduser()
    return out


def _trajectory(c_recal: List[dict], key: str) -> List[float]:
    return [s.get(key) for s in c_recal]


def _t_axis(c_recal: List[dict]) -> List[int]:
    return [s["t"] for s in c_recal]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True,
                   help="label=root_dir pairs, e.g. Tongji=experiments/generated/tongji_full")
    p.add_argument("--mfn-subdir", default="protocols",
                   help="subdir under each dataset root for MFN orchestrator JSONs")
    p.add_argument("--ccnet-subdir", default="ccnet_protocols",
                   help="subdir for CCNet orchestrator JSONs")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    args.out = args.out.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = _parse_pairs(args.datasets)

    # 2 rows × N cols layout: row 0 = TPIR trajectory, row 1 = tau drift
    n = len(datasets)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 8), sharey="row")
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (label, root) in enumerate(datasets.items()):
        mfn_dir = root / args.mfn_subdir
        ccnet_dir = root / args.ccnet_subdir
        mfn_b = _load_json(mfn_dir / "protocol_b.json")
        mfn_cr = _load_json(mfn_dir / "protocol_c_recal.json")
        if isinstance(mfn_cr, dict) and "steps" in mfn_cr:
            mfn_cr = mfn_cr["steps"]
        has_ccnet = (ccnet_dir / "protocol_b.json").exists()
        if has_ccnet:
            cc_b = _load_json(ccnet_dir / "protocol_b.json")
            cc_cr = _load_json(ccnet_dir / "protocol_c_recal.json")
            if isinstance(cc_cr, dict) and "steps" in cc_cr:
                cc_cr = cc_cr["steps"]

        # ---- TPIR trajectory ----
        ax = axes[0, col]
        ts = _t_axis(mfn_cr)
        ax.plot(ts, _trajectory(mfn_cr, "tpir"), "o-", color="C0",
                label="MFN-ArcFace C-recal", linewidth=1.7)
        ax.plot([ts[-1]], [mfn_b["tpir"]], "*", color="C0",
                markersize=16, markeredgecolor="black", markeredgewidth=0.8,
                label="MFN B endpoint")
        if has_ccnet:
            ts_c = _t_axis(cc_cr)
            ax.plot(ts_c, _trajectory(cc_cr, "tpir"), "s-", color="C2",
                    label="CCNet-ArcFace C-recal", linewidth=1.7)
            ax.plot([ts_c[-1]], [cc_b["tpir"]], "*", color="C2",
                    markersize=16, markeredgecolor="black", markeredgewidth=0.8,
                    label="CCNet B endpoint")
        else:
            ax.text(0.5, 0.95, "CCNet results not yet generated",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=9, style="italic", color="gray")
        ax.set_xlabel("enrollment step t")
        if col == 0:
            ax.set_ylabel("TPIR @ α=5%")
        ax.set_title(f"{label} TPIR trajectory")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8, loc="lower left")

        # ---- tau drift ----
        ax = axes[1, col]
        ax.plot(ts, _trajectory(mfn_cr, "tau"), "o-", color="C0",
                label="MFN τ-recal", linewidth=1.7)
        if has_ccnet:
            ax.plot(_t_axis(cc_cr), _trajectory(cc_cr, "tau"),
                    "s-", color="C2", label="CCNet τ-recal", linewidth=1.7)
        ax.set_xlabel("enrollment step t")
        if col == 0:
            ax.set_ylabel("threshold τ")
        ax.set_title(f"{label} threshold drift")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(
        "Multi-backbone protocol-effect comparison (V17.7): "
        "MFN-ArcFace vs CCNet-ArcFace under standardized open-set recipe",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = args.out / "multi_backbone_main.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[save] {out_path}")

    # CSV summary
    csv_path = args.out / "multi_backbone_summary.csv"
    with csv_path.open("w") as f:
        f.write("dataset,backbone,B_endpoint_tpir,C_recal_endpoint_tpir,tau_t0,tau_tmax,gallery_t0,gallery_tmax\n")
        for label, root in datasets.items():
            mfn_dir = root / args.mfn_subdir
            mfn_b = _load_json(mfn_dir / "protocol_b.json")
            mfn_cr = _load_json(mfn_dir / "protocol_c_recal.json")
            if isinstance(mfn_cr, dict) and "steps" in mfn_cr:
                mfn_cr = mfn_cr["steps"]
            f.write(f"{label},MFN-ArcFace,{mfn_b['tpir']:.4f},{mfn_cr[-1]['tpir']:.4f},"
                    f"{mfn_cr[0]['tau']:.4f},{mfn_cr[-1]['tau']:.4f},"
                    f"{mfn_cr[0]['gallery_size']},{mfn_cr[-1]['gallery_size']}\n")
            ccnet_dir = root / args.ccnet_subdir
            if (ccnet_dir / "protocol_b.json").exists():
                cc_b = _load_json(ccnet_dir / "protocol_b.json")
                cc_cr = _load_json(ccnet_dir / "protocol_c_recal.json")
                if isinstance(cc_cr, dict) and "steps" in cc_cr:
                    cc_cr = cc_cr["steps"]
                f.write(f"{label},CCNet-ArcFace,{cc_b['tpir']:.4f},{cc_cr[-1]['tpir']:.4f},"
                        f"{cc_cr[0]['tau']:.4f},{cc_cr[-1]['tau']:.4f},"
                        f"{cc_cr[0]['gallery_size']},{cc_cr[-1]['gallery_size']}\n")
    print(f"[save] {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
