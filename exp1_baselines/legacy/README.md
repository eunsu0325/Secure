# Legacy scripts (deprecated; do not use for new work)

These files predate the canonical 5-way subject-level split + new manifest schema
(plan: D1-D5). They are retained for reference and for the historical
`exp1_run.py` flow until that script is also retired.

## What replaces what

| Legacy file (here) | New module | Notes |
|---|---|---|
| `build_tongji_for_baselines_legacy.py` | `exp1_baselines/datasets/tongji_manifest.py` | 4-way -> 5-way split; sample_role; metadata.json; IER-1/2/5/6 validators |
| `eval_protocols_legacy.py` (680 lines) | `exp1_baselines/eval/{score_matrix,thresholding,metrics,protocols,orchestrator}.py` | Modular split; Hard Rule 3 fix (no external_dev prototype); B==C-recal sanity; tie-aware tau; Wilson CI |

## Known issues in legacy code

- `build_tongji_for_baselines_legacy.py` uses the deprecated 4-way split:
  `base_ids / threshold_val_ids / future_ids / external_test_ids` with
  `threshold_gallery_ids` / `threshold_unknown_ids` sub-split.
- `eval_protocols_legacy.py` line ~355 has the v8.5 bug where
  `gallery_ids = enrollment_order[:t]` omits `base_anchor` from Protocol A.
- Legacy `calibrate_tau_at_far` is not tie-aware.

Do not import from these modules.
