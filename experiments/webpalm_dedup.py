"""WebPalm dedup screening against Tongji/IITD/BJTU galleries (plan §V2).

For each in-pool dataset (Tongji, IITD, BJTU), this script computes the
nearest-neighbor cosine similarity from each WebPalm identity to that
dataset's eval-split prototypes, then flags candidates by the union of
three criteria:

  (a) cosine similarity > 0.5 (permissive fixed threshold)
  (b) top 0.1% nearest-neighbor similarities (relative)
  (c) above the 99.9th percentile of the WebPalm-vs-WebPalm impostor
      similarity distribution (calibrated)

The manual-review cap (plan §V2): per in-pool dataset, retain at most 200
top-similarity candidates from the union. Three datasets → ~600 manual
reviews total. Output is a CSV per dataset listing the candidate WebPalm
ID, the nearest in-pool prototype palm_id, the cosine similarity, and
which criteria flagged it.

A simple HTML grid is also produced for visual review: each row has the
WebPalm raw image (or its ROI thumbnail) and the matched in-pool
prototype, side-by-side.

Usage:
    python experiments/webpalm_dedup.py \
        --webpalm-npz   ~/research_data/webpalm/embeddings_mfn112.npz \
        --inpool-npz    experiments/generated/tongji_full/mfn_tongji_112_embeddings.npz \
                        experiments/generated/iitd_full/mfn_iitd_112_embeddings.npz \
                        experiments/generated/bjtu_full/mfn_bjtu_112_embeddings.npz \
        --inpool-names  tongji iitd bjtu_v2 \
        --webpalm-raw   ~/research_data/webpalm/raw \
        --webpalm-roi   ~/research_data/webpalm/roi_224/224 \
        --out           ~/research_data/webpalm/dedup
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def cos_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between two L2-normalized matrices."""
    return A @ B.T


def topk_per_row(M: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Per-row top-k. Returns (similarities, col_indices), each [N, k]."""
    if k == 1:
        col = M.argmax(axis=1)
        sim = M[np.arange(M.shape[0]), col]
        return sim[:, None], col[:, None]
    idx = np.argpartition(-M, k, axis=1)[:, :k]
    rows = np.arange(M.shape[0])[:, None]
    sim = M[rows, idx]
    order = (-sim).argsort(axis=1)
    return np.take_along_axis(sim, order, axis=1), np.take_along_axis(idx, order, axis=1)


def webpalm_self_impostor_p999(emb: np.ndarray, sample_size: int = 5000,
                                seed: int = 42) -> float:
    """Estimate the 99.9th percentile of WebPalm self-similarity (impostor
    distribution; WebPalm is one-shot so any inter-row similarity is by
    definition impostor)."""
    rng = np.random.default_rng(seed)
    N = emb.shape[0]
    if N <= sample_size:
        idx = np.arange(N)
    else:
        idx = rng.choice(N, sample_size, replace=False)
    sub = emb[idx]
    M = cos_sim_matrix(sub, sub)
    # exclude diagonal
    np.fill_diagonal(M, -np.inf)
    upper = M[np.triu_indices(M.shape[0], k=1)]
    return float(np.percentile(upper, 99.9))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--webpalm-npz", type=Path, required=True)
    p.add_argument("--inpool-npz", type=Path, nargs="+", required=True)
    p.add_argument("--inpool-names", type=str, nargs="+", required=True)
    p.add_argument("--webpalm-raw", type=Path, required=True)
    p.add_argument("--webpalm-roi", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold-fixed", type=float, default=0.5)
    p.add_argument("--top-percent", type=float, default=0.001,
                   help="0.001 = top 0.1%")
    p.add_argument("--manual-review-cap", type=int, default=200)
    args = p.parse_args()

    if len(args.inpool_npz) != len(args.inpool_names):
        sys.exit("--inpool-npz count must match --inpool-names count")
    args.out.mkdir(parents=True, exist_ok=True)

    # Load WebPalm
    wp = np.load(args.webpalm_npz, allow_pickle=True)
    wp_emb = wp["embeddings"]
    wp_fn = wp["filename"]
    print(f"[load] WebPalm: {wp_emb.shape[0]} ids, dim={wp_emb.shape[1]}")

    # Estimate WebPalm self-impostor p99.9 for criterion (c)
    p999 = webpalm_self_impostor_p999(wp_emb)
    print(f"[calib] WebPalm self-impostor 99.9 percentile: {p999:.4f}")

    union_records: List[dict] = []
    per_dataset_summary: Dict[str, dict] = {}

    for npz_path, ds_name in zip(args.inpool_npz, args.inpool_names):
        data = np.load(npz_path, allow_pickle=True)
        ip_emb = data["embeddings"]
        ip_palm = data["palm_id"] if "palm_id" in data.files else np.arange(ip_emb.shape[0])
        ip_split = (
            data["subject_split"] if "subject_split" in data.files
            else np.array(["unknown"] * ip_emb.shape[0])
        )
        # Filter to eval splits only (base_anchor / future / external_dev / external_test)
        eval_mask = np.isin(
            ip_split,
            ["base_anchor", "future", "external_dev", "external_test"],
        )
        ip_emb_eval = ip_emb[eval_mask]
        ip_palm_eval = ip_palm[eval_mask]
        # Aggregate per palm_id (mean-pool then renormalize) so we
        # screen against palm-level prototypes, not per-image embeddings.
        unique_palms, inv = np.unique(ip_palm_eval, return_inverse=True)
        palm_emb = np.zeros((unique_palms.size, ip_emb_eval.shape[1]),
                             dtype=np.float32)
        for i in range(unique_palms.size):
            mask = (inv == i)
            mean = ip_emb_eval[mask].mean(axis=0)
            palm_emb[i] = mean / (np.linalg.norm(mean) + 1e-12)
        print(f"[{ds_name}] {palm_emb.shape[0]} palm prototypes from "
              f"{eval_mask.sum()} eval rows")

        # Per WebPalm id, find nearest in-pool palm prototype
        sims, cols = topk_per_row(cos_sim_matrix(wp_emb, palm_emb), k=1)
        sims = sims[:, 0]
        cols = cols[:, 0]

        # Criteria
        crit_a = sims > args.threshold_fixed
        topk = max(1, int(np.ceil(args.top_percent * sims.size)))
        # top topk indices
        top_idx = np.argpartition(-sims, topk - 1)[:topk]
        crit_b = np.zeros_like(sims, dtype=bool)
        crit_b[top_idx] = True
        crit_c = sims > p999

        union = crit_a | crit_b | crit_c
        union_idx = np.where(union)[0]
        # Sort by similarity descending
        union_idx_sorted = union_idx[np.argsort(-sims[union_idx])]
        # Cap at manual_review_cap
        union_idx_capped = union_idx_sorted[: args.manual_review_cap]

        # Write candidate CSV
        out_csv = args.out / f"candidates_{ds_name}.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "webpalm_id", "nearest_inpool_palm_id", "cosine_similarity",
                "crit_a_above_0.5", "crit_b_top_pct", "crit_c_above_p999",
                "review_priority",
            ])
            for rank, i in enumerate(union_idx_capped):
                w.writerow([
                    str(wp_fn[i]),
                    int(unique_palms[cols[i]]),
                    f"{float(sims[i]):.4f}",
                    bool(crit_a[i]),
                    bool(crit_b[i]),
                    bool(crit_c[i]),
                    rank + 1,
                ])
        print(f"[write] {out_csv} ({len(union_idx_capped)} entries)")

        # Per-dataset summary
        per_dataset_summary[ds_name] = {
            "in_pool_palm_count": int(unique_palms.size),
            "criterion_a_count": int(crit_a.sum()),
            "criterion_b_count": int(crit_b.sum()),
            "criterion_c_count": int(crit_c.sum()),
            "union_count": int(union.sum()),
            "manual_review_count": int(len(union_idx_capped)),
            "p999_webpalm_self": p999,
        }

        # Add to global records (ID dedup will happen at the next step
        # since the same WebPalm ID may appear in multiple per-dataset CSVs)
        for i in union_idx_capped:
            union_records.append({
                "webpalm_id": str(wp_fn[i]),
                "in_pool_dataset": ds_name,
                "in_pool_palm_id": int(unique_palms[cols[i]]),
                "cosine_similarity": float(sims[i]),
                "crit_a": bool(crit_a[i]),
                "crit_b": bool(crit_b[i]),
                "crit_c": bool(crit_c[i]),
            })

    # Global review HTML grid (top by max similarity)
    union_records.sort(key=lambda r: -r["cosine_similarity"])
    html_path = args.out / "manual_review_grid.html"
    with html_path.open("w") as f:
        f.write("<html><head><meta charset=utf-8>"
                "<title>WebPalm dedup manual review</title>"
                "<style>body{font-family:sans-serif;background:#222;color:#eee}"
                "table{border-collapse:collapse;margin:8px}"
                "td{padding:4px;border:1px solid #555;vertical-align:top}"
                "img{max-width:200px;max-height:200px}"
                "</style></head><body>"
                f"<h2>WebPalm dedup candidates ({len(union_records)} entries)</h2>"
                "<table><tr>"
                "<th>rank</th><th>webpalm_id</th><th>WebPalm raw</th>"
                "<th>WebPalm ROI</th><th>in-pool dataset/palm</th>"
                "<th>similarity</th><th>criteria</th><th>decision</th>"
                "</tr>")
        for rank, r in enumerate(union_records[:1000]):  # cap HTML at 1000
            wp_id = r["webpalm_id"]
            crit_str = "".join([
                "A" if r["crit_a"] else "·",
                "B" if r["crit_b"] else "·",
                "C" if r["crit_c"] else "·",
            ])
            # Find raw + ROI image paths
            raw_path = ""
            for ext in (".jpg", ".JPG", ".jpeg", ".png"):
                cand = args.webpalm_raw / f"{wp_id}{ext}"
                if cand.exists():
                    raw_path = str(cand.resolve())
                    break
            roi_path = args.webpalm_roi / wp_id / f"{wp_id}_roi.jpg"
            roi_str = str(roi_path.resolve()) if roi_path.exists() else ""
            f.write("<tr>"
                    f"<td>{rank + 1}</td>"
                    f"<td>{html.escape(wp_id)}</td>"
                    f'<td><img src="file://{html.escape(raw_path)}"></td>'
                    f'<td><img src="file://{html.escape(roi_str)}"></td>'
                    f"<td>{html.escape(r['in_pool_dataset'])}<br>"
                    f"palm_id={r['in_pool_palm_id']}</td>"
                    f"<td>{r['cosine_similarity']:.3f}</td>"
                    f"<td>{crit_str}</td>"
                    "<td>(check to remove)</td>"
                    "</tr>")
        f.write("</table></body></html>")
    print(f"[write] {html_path}")

    # Summary JSON
    summary = {
        "webpalm_total": int(wp_emb.shape[0]),
        "webpalm_self_impostor_p999": p999,
        "per_dataset": per_dataset_summary,
        "global_union_records": len(union_records),
        "criteria_definitions": {
            "a": f"cosine similarity > {args.threshold_fixed}",
            "b": f"top {args.top_percent * 100:.2f}%",
            "c": "above 99.9 percentile of WebPalm self-impostor distribution",
        },
        "manual_review_cap_per_dataset": args.manual_review_cap,
        "next_step": (
            "Manually review candidates in candidates_*.csv "
            "and manual_review_grid.html. Mark confirmed overlaps in a "
            "removal list (e.g., dedup_removed_ids.txt) and pass to "
            "split_webpalm.py via an exclusion file."
        ),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ok] summary at {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
