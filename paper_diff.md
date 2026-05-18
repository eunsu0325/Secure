# COCONUT Paper-Readiness Diff Log

Phase 1.5 cleanup pass for the COCONUT diagnostic tear-down (Exp2-pre, plan at
`~/.claude/plans/floofy-skipping-blossom.md`). Every entry below cites the
file:line, what changed, why, and which audit finding (A1–A5, B1–B8) it
addresses. Audit findings catalogued from 100% precise read of ~8800 LoC of
`coconut/` + `train_coconut.py` + `config/config.yaml` on 2026-05-19.

**Principle**: changes are bit-exact for normal-case behavior on a fixed seed
(verified via `eval_curve.csv` hash-match) except for the two documented
behavior changes A1 and A5, which only affect previously-broken code paths.

---

## Tier 1 — correctness fixes (must land before Phase 2)

### A1 — train_coconut.py main loop honors `num_experiences`

- **File**: `train_coconut.py:458–580` (main `for exp_id, ... in enumerate(data_stream):` loop)
- **Finding**: `config.training.num_experiences` was only used in label-style
  checks (last-eval marker at L469, last-checkpoint marker at L567, ETA at
  L577). The loop iterated **all** users yielded by `ExperienceStream`. Phase 0b
  verified the bug: setting `num_experiences=5` for a smoke run still
  produced 148 BJTU users worth of work (~80 minutes on L4).
- **Change**: After the per-experience eval/checkpoint block, add
  `if exp_id + 1 >= config_obj.training.num_experiences: break`.
- **Behavior impact**: Smoke runs with `num_experiences=5` now stop after 5
  experiences. Full-config runs (`num_experiences=100` or higher than the
  stream length) are unaffected — the loop exhausts the stream as before.
- **Verification**: Phase 1.5 smoke re-run with `num_experiences=5` must
  produce exactly 5 checkpoints (`checkpoint_exp_5.pth`) and 5 entries in
  `eval_curve.csv`.

### A2 — Remove duplicate `@torch.no_grad()` decorator on `_diagnose_pca`

- **File**: `coconut/training/trainer.py:2460–2462`
- **Finding**: `@torch.no_grad()` decorator applied twice on `_diagnose_pca`.
  Functionally harmless (decorators compose idempotently here) but editorially
  sloppy.
- **Change**: Delete the second decorator line.
- **Behavior impact**: None.

### A3 — Correct "6144D" → "2048D" in 3 sites

- **Files**: `config/config.yaml:71`, `coconut/training/trainer.py:2464`,
  `coconut/training/trainer.py:2631`
- **Finding**: Comments at these sites claim "6144D feature distillation" /
  "6144D 임베딩의 실효 차원" / "config에 따라 6144D 또는 512D 특징 사용".
  Actual CCNet feature dim is 2048 (per `coconut/models/ccnet.py:290`,
  verified via Phase 0b synthetic forward).
- **Change**: Replace "6144D" → "2048D" at all three sites. Add one-line
  comment at `coconut/models/ccnet.py:getFeatureCode` anchoring the dim.
- **Behavior impact**: None (comments only).

### A4 — Remove duplicate `coconut/memory/stream.py`

- **Files**: delete `coconut/memory/stream.py`; edit `coconut/memory/__init__.py`
- **Finding**: `coconut/memory/stream.py` and `coconut/data/stream.py` are
  byte-identical (`diff` produces no output). Both `__init__.py` files export
  `ExperienceStream` from their respective local `stream.py`. Two canonical
  copies of identical code invite drift on future edits.
- **Change**: Delete `coconut/memory/stream.py`. Update
  `coconut/memory/__init__.py` to `from coconut.data.stream import ExperienceStream`.
- **Behavior impact**: None — both old import paths still resolve to the same
  class.

### A6 — numpy-aware JSON encoder in train_coconut.py

- **File**: `train_coconut.py:48-62` (new helper), `train_coconut.py:603, 609, 741` (3 json.dump call sites)
- **Finding**: Smoke verification of T1+T2 run completed all 5 experiences cleanly but crashed at the finalization step `json.dump(training_history, f, indent=4)` with `TypeError: Object of type float32 is not JSON serializable`. `training_history['forgetting_reports']` contains numpy scalars from `ForgettingTracker.get_report()` (which calls `np.mean()`, `np.std()`, etc.). stdlib json can't serialize np.float32 / np.float64. Phase 0b never exposed this because the `num_experiences` break bug (A1) prevented the run from reaching finalization.
- **Change**: Added `_json_default(o)` helper at the top of train_coconut.py (mirroring the pattern already used in trainer.py:1833 for `diag_history.json`). Applied `default=_json_default` to all 3 `json.dump` sites: training_history.json, fpir_drift_log.json, summary.json.
- **Behavior impact**: training_history.json, fpir_drift_log.json, and summary.json now serialize successfully when they contain numpy scalars. Downstream files (`eval_curve.csv`, `det_curve.csv`, t-SNE / DET plots) are unblocked.

### A5 — visualization.py early-exit returns `fig` not `ax`

- **File**: `coconut/evaluation/visualization.py:113–124`
- **Finding**: The "Perfect Separation" early-exit branch of `plot_det_curve`
  returns `ax` (Axes) instead of `fig` (Figure). The function signature is
  annotated `-> plt.Figure`. Callers doing `fig = plot_det_curve(...); fig.savefig(...)`
  would crash in the perfect-separation case. Doesn't affect Phase 2 (never
  perfect) but reviewers running the code on toy data could hit it.
- **Change**: Restructure the function so `fig, ax = plt.subplots(...)` happens
  unconditionally at the top, the if/else only populates `ax`, and `return fig`
  is the single exit point.
- **Behavior impact**: Perfect-separation case now returns `fig` (consistent
  with normal case). All other cases are bit-exact.

---

## Tier 2 — paper-readability fixes

### B2 — Verbose-gate orphan `_analyze_cosine_distribution_epoch`

- **File**: `coconut/training/trainer.py:815–817` (call site), `coconut/training/trainer.py:_analyze_cosine_distribution_epoch` (definition)
- **Finding**: 131-line diagnostic was called every experience in `train_experience` but produced output only when `self.verbose` (default `false`). Wasted feature extraction in paper-canonical runs.
- **Change**: Wrap the call in `if self.verbose:`. Body untouched; can be removed entirely later if no debug-run uses it.
- **Behavior impact**: Paper-canonical run (verbose=false) now skips redundant feature extraction. Verbose run unchanged.

### B3 — Dedup legacy result-dict aliases (`FNIR`, `FPIR_in`, `FRR`, `MisID`, `TRR_unknown`)

- **Files**: `coconut/training/trainer.py:1978–1983` (removed alias block); `train_coconut.py:515, 720–727` (callers now read canonical keys)
- **Finding**: trainer.py wrote 5 short alias keys mirroring the canonical `FNIR@1%FPIR / achieved_FPIR@1% / det_fail@1% / id_fail@1%`. Only `train_coconut.py` summary path read them. `TRR_unknown` had no readers.
- **Change**:
  - `train_coconut.py:515`: `metrics.get('FPIR_in', None)` → `metrics.get('achieved_FPIR@1%', None)`.
  - `train_coconut.py:721–724`: 4 lookups updated to canonical keys (keys in `summary['final_openset']` dict kept short for downstream consumers).
  - `trainer.py:1978–1985`: 5-line alias block removed; `mode` and `score_type` keys preserved.
- **Behavior impact**: `summary['final_openset']` JSON file is bit-exact (same values, same keys). Internal result dict shrinks by 5 keys (no external consumer affected).

### B4 — Document Exp1 backbone factory living in `coconut/models/`

- **No code change in Phase 1.5**.
- **Files affected**: `coconut/models/model_factory.py` (213 LoC), `coconut/models/repvit_wrapper.py` (182 LoC), `coconut/data/transforms.py:get_repvit_transforms`.
- **Finding**: These three files are **Exp1** backbone-building utilities (see their docstrings: "Backbone factory for Exp1", "RepViT wrapper for the Exp1 robustness backbone"). They are not imported by COCONUT trainer or by `coconut/*/__init__.py`. `train_coconut.py:351` constructs CCNet directly via `ccnet(weight=...)`.
- **Why no move now**: Moving them out of `coconut/` would touch `exp1_baselines/` callers and trigger Exp1 churn we don't want during the diagnostic. They are silently dormant from COCONUT's perspective.
- **Paper text obligation**: any paper section that cites COCONUT code must not cite these three Exp1 files; if the paper discusses RepViT it does so via the Exp1 section, citing `coconut/models/repvit_wrapper.py` explicitly as Exp1 infrastructure.

### B5 — PCA diagnostic seed derived from config

- **File**: `coconut/training/trainer.py:_diagnose_pca` (sampling block around L2483)
- **Finding**: Hardcoded `np.random.RandomState(42)` for the buffer subsample. Independent of `self.config.training.seed`. Means PCA diagnostic logs are identical across multi-seed sweeps — minor reproducibility lie.
- **Change**: `seed = getattr(self.config.training, 'seed', 42) + self.experience_count` (matches the `_analyze_cosine_distribution_epoch` pattern).
- **Behavior impact**: PCA diagnostic print output now varies per seed. Final result metrics (eval_curve.csv) unaffected — PCA is a side-channel log.

### B6 — Lift magic numbers (3000/1000/2000) to module-level constants

- **File**: `coconut/training/trainer.py` (4 sites for 3000, 1 site for 1000, 1 site for 2000)
- **Finding**: τ calibration subsample caps (3000 for unknown_dev, 1000 for xdomain negref, 2000 for PCA diagnostic) were buried inline magic ints. Reviewers can't audit "no cherry-pick" without grep.
- **Change**: Added three named constants at module top: `MAX_UNK_CALIB_SAMPLES = 3000`, `MAX_NEGREF_EVAL_SAMPLES = 1000`, `MAX_PCA_DIAG_SAMPLES = 2000`. All 6 inline sites replaced.
- **Behavior impact**: None (refactor only).

### B7 — Remove mid-refactor import comment

- **File**: `coconut/training/trainer.py:16–24` (import block)
- **Finding**: A `# 기존 모듈 import (점진적 마이그레이션 예정)` comment partitioned the import block, signaling unfinished refactor. The two halves can be merged cleanly because `coconut.data` now exports `get_scr_transforms` AND `MemoryDataset` from a single namespace.
- **Change**: Comment removed; `MemoryDataset` and `get_scr_transforms` consolidated onto one `from coconut.data import ...` line.
- **Behavior impact**: None.

### B8 — Repair UTF-8 mojibake in config.yaml comments

- **File**: `config/config.yaml:77–81` (QAR comments)
- **Finding**: `���` (UTF-8 replacement character) appeared in 5 Korean comment locations from a prior file-encoding round-trip accident.
- **Change**: Restored the 5 corrupted bytes to their original Korean characters:
  - `배치 ���성` → `배치 구성`
  - `feature 보���` → `feature 보존`
  - `임���치` → `임계치`
  - `memory_size/100��10` → `memory_size/100 ≈ 10`
- **Verification**: `grep -c $'\xef\xbf\xbd' config/config.yaml` returns 0.
- **Behavior impact**: None (YAML parser ignored mojibake comments).

---

## Tier 3 — deferred (post-paper or post-Phase 4)

### C-GHOST (downgraded from B1) — Physical removal of GHOST legacy code

- **Decision** (2026-05-19): originally tier-2 "remove ~190 LoC GHOST block". Downgraded to tier-3 because:
  1. `NCMClassifier.state_dict` stores `_custom_ghost_*` keys ([ncm.py:73–79](coconut/classifiers/ncm.py#L73)) and `_load_from_state_dict` reads them ([ncm.py:104–109](coconut/classifiers/ncm.py#L104)). Removing these would break loading of any pre-cleanup checkpoint.
  2. All trainer.py GHOST call sites use `getattr(self, 'use_ghost', False)` defensive guards. With `use_ghost: false` (default), no GHOST branch ever executes — the code is already functionally dormant.
  3. Paper text impact is zero either way: `paper_diff.md` documents the deprecation; paper §3 simply does not mention GHOST.
- **Mitigation in Phase 1.5**: `config/config.yaml:121–126` updated to label GHOST as **DEPRECATED** with explicit note that physical removal is deferred.
- **Plan for removal**: after the paper is submitted and prior checkpoints no longer need to load, remove (a) all `getattr(self, 'use_ghost', ...)` branches in trainer.py (~150 LoC), (b) `NCMClassifier._compute_ghost_scores / compute_ghost_max_scores / set_ghost_stats / ghost_*_raw / ghost_enabled` (~80 LoC in ncm.py), (c) `_custom_ghost_*` state_dict serialization, (d) `use_ghost / ghost_n_augment / ghost_shrinkage_min_n` from config.

### C1 — Lift bootstrap iter count and FPIR target list to config

- Files: `coconut/training/trainer.py:1257` (`n_iter=500`), `coconut/training/trainer.py:1796, 1958` (`[1, 5, 10]` FPIR targets).
- Currently fine because all paper runs use these defaults.

### C2 — Lift hardcoded `% 10` checkpoint / t-SNE interval

- File: `train_coconut.py:475, 567, 575`.
- Currently fine because all paper runs save at the same cadence.

### C3 — Split `config.yaml` into `paper_canonical.yaml` + `experimental.yaml`

- PCA shrinkage knobs (`pca_shrinkage_mode`, `pca_shrinkage_lambda`, `pca_k_mode`, `pca_fixed_k`, `mahalanobis_variant`) belong in an experimental file because the paper picks one canonical setting.

### C4 — `getFeatureCode` → `get_feature_code` PEP8 rename

- Affects CCNet + RepViTWrapper + every call site. ~30 minutes work but breaks Exp1 saved checkpoints if not done with backward-compat property.

### C5 — Audit `extract_scores_for_user` orphan path

- File: `coconut/openset/score_extraction.py:extract_scores_for_user` (260 lines).
- Looks underused vs the trainer's main `_evaluate_openset` path. Confirm zero paper callers, then remove.
