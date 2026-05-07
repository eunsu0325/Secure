# Exp1 — Open-set palmprint identification protocol diagnostic

This module evaluates how the *static* open-set endpoint protocol (Protocol B)
hides sequential enrollment dynamics — gallery growth, threshold drift, and a
future-unknown state — that the trajectory-based Protocol C reveals.

**Paper claim (locked wording):**
> Static endpoint evaluation hides how TPIR, realized FPIR, and threshold
> behavior evolve as the enrolled gallery grows. Future-FPIR is not claimed
> to be intrinsically harder than External-FPIR; it is a temporally
> meaningful state absent from static endpoint protocols.

This is a **methodology paper** — the proposed mitigation method is the
subject of follow-up experiments and is not part of this codebase.

---

## Datasets

Within-dataset retraining (no cross-domain). Subject-level 5-way split with
seed=42; eligibility-first allocation (evaluation quotas met first; the
residual feeds `train_backbone`).

| Dataset | Subjects | Palms | imgs/palm | Sessions / phases |
|---|---|---|---|---|
| Tongji | 300 | 600 | 10 / session | session1, session2 |
| IITD V1 (Segmented) | ~230 | ~460 | ~5 | none |
| BJTU-V2 (ROI) | 148 | 296 | ~5 + ~5 | F (enroll), S (query) |

Eligibility filters reject subjects missing one hand or with too few images;
final per-split counts are recorded in each `metadata.json` and re-asserted
on every load (IER-2).

## Protocols

- **A — closed-set rank-1** (background diagnostic; no τ).
- **B — static full-gallery endpoint** (single number per dataset/backbone).
- **C-fixed — sequential, τ frozen at gallery_0 = base_anchor**.
- **C-recal — sequential, τ recalibrated against the operational gallery**
  at every step.

All four share the same cosine score matrix; protocol differences come only
from row/column slicing rules (Hard Rule 6). External_dev is the only
calibration probe set (Hard Rule 1) and never has a prototype column
(Hard Rule 3). Future palms not yet enrolled are masked from the gallery
at every C step (Hard Rule 4). The orchestrator asserts `B == C-recal at
t=t_max` (Δ < 1e-6) on every run; on real Tongji the check is exact
(0 difference).

## Repository layout

```
exp1_baselines/
├── backbones/                # MobileFaceNet (main), IR50 (appendix)
├── datasets/                 # canonical 5-way manifest builders
│   ├── manifest_schema.py    # CSV schema + 3-tier identity_id factory
│   ├── split.py              # subject_level_split, eligibility-first
│   ├── validate_manifest.py  # IER-1 / 2 / 5 / 6 validators
│   ├── tongji_manifest.py    # Tongji ROI builder (Tier 2 ID; sessioned)
│   ├── iitd_manifest.py      # IITD Segmented (Tier 1; non-sessioned, candidate role)
│   ├── bjtu_manifest.py      # BJTU-V2 (Tier 1; sessioned by F/S phase)
│   └── manifest_io.py        # canonical CSV reader
├── eval/                     # protocols + metrics
│   ├── score_matrix.py       # cosine score matrix + C-step row/col masks
│   ├── thresholding.py       # tie-aware τ calibration
│   ├── metrics.py            # Wilson CI, deterministic argmax, decomposition
│   ├── protocols.py          # A / B / C-fixed / C-recal + B==C-recal sanity
│   └── orchestrator.py       # one-config runner emitting protocol JSONs
├── train_arcface.py          # MFN/IR50 + ArcFace head, manifest-driven
├── extract_embeddings.py     # canonical D1h NPZ schema for eval splits
└── legacy/                   # archived pre-D1 scripts (do NOT use)

experiments/
├── run_d6_tongji.sh          # end-to-end Tongji pipeline (manifest -> train -> extract -> protocols)
├── run_d6_iitd.sh            # IITD variant (--non_sessioned)
├── run_d6_bjtu.sh            # BJTU-V2 variant (phase F/S)
├── run_multiseed_tongji.py   # 30 (order_seed × enroll_seed) trajectories on a trained ckpt
└── plot_exp1.py              # 4-panel diagnostic + Protocol A figures
```

## Running Tongji end-to-end

```bash
TONGJI_ROOT=/path/to/Tongji_ROI \
DEVICE=mps  \
AMP=false   \
OUT_DIR=experiments/generated/tongji_full \
bash experiments/run_d6_tongji.sh
```

Outputs:
- `OUT_DIR/manifest.csv` + `metadata.json`
- `OUT_DIR/mfn_arcface_tongji_scratch_112.pt` (~5MB)
- `OUT_DIR/mfn_tongji_112_embeddings.npz` (~12MB)
- `OUT_DIR/protocols/{protocol_a,protocol_b,protocol_c_fixed,protocol_c_recal,meta}.json`

The orchestrator verifies `meta.b_vs_c_recal_endpoint_sanity == "ok"`
before exiting.

## Reproducing paper-scale variance bands

The trained checkpoint and embeddings are reused; only the orchestrator
is re-run across `order_seed × enroll_seed = 30` combinations:

```bash
python experiments/run_multiseed_tongji.py    # ~10 seconds for 30 runs
python experiments/plot_exp1.py \
    --results experiments/generated/tongji_multiseed/order*/protocols \
    --out     experiments/generated/tongji_multiseed/figures \
    --tag     tongji_mfn_multiseed
```

## Operating points reported

We report both `target_fpir = 0.05` (reliable: n × target = 25) and
`target_fpir = 0.01` (reliable boundary: n × target = 5) per the plan's
reliability flag rule. Lower targets (e.g. 0.001) require an impostor pool
larger than the in-dataset external_test set and are out of scope for the
in-pool main experiment.

## Hard Rules (enforced in code)

1. τ calibration uses ONLY external_dev queries (never future / external_test).
2. "Known correct accept" = `(top1_id == ground_truth) AND (top1_score >= τ)`.
3. external_test never has prototypes (always unknown).
4. Future cols masked at every C step (not-yet-enrolled palms must not contribute).
5. WebPalm-style external impostor pool not used in this repo (post-MVP only).
6. A / B / C share gallery composition `base_anchor + future_order[:t]` per t.
7. Within-dataset retraining only (no cross-domain).
8. Subject-level split conditional on P0a (subject pairing); palm-level fallback otherwise.
9. K (enrollment) does NOT apply to external_dev / external_test; all their
   query images are queries.
10. Sessioned datasets use the query session only for external probes
    (acquisition consistency).
11. GPDS dataset excluded.

These are asserted by `validate_manifest.py` (manifest-time) and
`eval/protocols.py::evaluate_step` (run-time IER-3).
