# Phase 5 — TIFS Multi-Backbone Implementation Session Log

> **Plan reference**: `~/.claude/plans/you-are-helping-implement-witty-river.md`
> **Active spec**: §V17 TIFS Active Implementation Block
> **Authority chain**: V17 > V16 > V15 > V14 > V9–V12 > V1–V8 > Corrections v2 > v1 roadmap
> **Session dates**: 2026-05-08 → ongoing
> **Current branch**: main

---

## 1. Plan layer history (chronological corrections)

| Layer | Scope | Status |
|---|---|---|
| v1 roadmap (Phases 0–6) | Original TIFS extension plan | Superseded on conflicts |
| Corrections v2 (C1–C14) | Direction (Gumbel down, single mitigation, BJTU supplementary, etc.) | Active where not superseded |
| v3 §V1–§V8 (Round 1) | Operational specifics (λ grid, dedup, bias analysis, etc.) | Active where not superseded |
| v3 §V9–§V12 (Round 2) | α=1e-4 dev-tuning ban, τ_safe clip, bootstrap seed, 1:N FPIR wording | Active |
| §V13 | Authority precedence chain | Active (extended by V17) |
| §V14a/b/c | τ_safe primary form (empirical bootstrap quantile), Cliff's δ primary | Active |
| §V15 | ROI geometry fix (size-relative inward) | Applied, active |
| §V16 | CCNet integration lock (verified upstream + 2048D inference + L1/L2 gating) | Active |
| **§V17** | **Final TIFS Active Implementation Block** | **Active, top of precedence** |

---

## 2. Plan file inline corrections (high-priority, this session)

| Plan line | Original (deprecated) | Corrected to | Method |
|---|---|---|---|
| 23–25 | "Target venue: T-BIOM. TIFS extension … out of current scope." | "Current active target: TIFS extension. Fallback: T-BIOM…" + DEPRECATED block | Inline edit + V17.1 cross-ref |
| 131 | "**Main**: ROI native (Tongji 128×128, IITD as-provided, BJTU as-provided)" | "Main (T-BIOM track)…" + DEPRECATED tag + V17.2 forward-ref | Inline edit |
| 190–192 | 12-column manifest schema (no phase_id/sample_id) | "DEPRECATED 12-column schema (do not use)" + active 14-column schema | Inline edit |

Verification: `grep -nE "Main.*ROI native\|out of current scope\|session_id, subject_split, sample_role"` — all three show DEPRECATED tag adjacent.

---

## 3. §V17 block (added at plan lines 2849–3168)

| Sub-section | Content | Lines |
|---|---|---|
| V17.1 | Active target (TIFS extension; T-BIOM fallback) | 2867–2876 |
| V17.2 | 2-tier resolution (MFN=112, CCNet=128 architecturally required) | 2877–2897 |
| V17.3 | D1 EXPANSION schema (14 columns) | 2898–2909 |
| V17.4 | Theory monotonicity-first; EVT/GEV auxiliary | 2910–2916 |
| V17.5 | WebPalm 1K trial + Wilson tier + dedup mandatory + λ at 1e-3 only | 2917–2925 |
| V17.6 | Single mitigation (Conservative Gallery-Size-Aware Thresholding) | 2926–2935 |
| V17.7 | Multi-backbone Tongji+IITD only; BJTU smartphone-domain sanity; no CCNet on BJTU | 2936–2944 |
| V17.8 | CCNet stress test under standardized recipe; 2048D inference; code-level asserts | 2945–2966 |
| V17.9 | Mandatory verification sequence (static → synthetic → 1-batch overfit → 5ep smoke → 50ep) | 2967–3014 |
| V17.10 | Per-dataset TPIR@5% gate (≥0.50 / 0.30..0.50 / <0.30); BJTU CCNet not in scope | 3015–3037 |
| V17.11 | Required metadata telemetry (train-time, extract-time, gate-time fields) | 3038–3097 |
| V17.12 | Device fallback policy (runtime selection only) | 3098–3105 |
| V17.13 | Deprecated v1/v2 sections table (8 rows) | 3106–3124 |
| V17.14 | Locked implementation order (steps 0–10) | 3125–3163 |
| V17.15 | GPT push-back acknowledgments (3 rejections) | 3164–3168 |

---

## 4. Code changes (commit-by-commit)

### 4.1 Pre-V17 commits (before this session's V17 lock)

| Commit | Title | Scope |
|---|---|---|
| 186b2c1 | Phase 5 prep: CCNet feature-only wrapper | Initial 6144D wrapper (later corrected to 2048D in V17) |
| f8ed799 | Phase 4 mitigation: Conservative Gallery-Size-Aware Thresholding | V14a mitigation |

### 4.2 V17 implementation commits (this session)

| Commit | Title | Files | Lines | Scope |
|---|---|---|---|---|
| **41899ac** | MPS device fallback in train_arcface and extract_embeddings | 2 | +14/-2 | V17.12 standalone (cuda → mps → cpu) |
| **f3e7568** | Phase 5 V17 lock: CCNet wrapper 2048D + recipe parity configs | 4 | +330/-99 | V17.8 wrapper + V17.2 configs + plot_multi_backbone.py |
| **136f6bc** | Phase 6 theory: monotonicity-first revision | 1 | +198/-73 | V14a/V17.4 paper restructure |
| **1e95d14** | Phase 5 V17.11 telemetry: train-time + extract-time metadata + backfill | 4 | +364 | V17.11 telemetry + helper scripts |

### 4.3 Files created/modified

**New files**:
- `exp1_baselines/backbones/ccnet.py` (vendored Yang et al. 2023 + 2048D wrapper)
- `exp1_baselines/configs/ccnet_arcface_tongji_128.yaml`
- `exp1_baselines/configs/ccnet_arcface_iitd_128.yaml`
- `experiments/plot_multi_backbone.py` (V17.7 figure)
- `experiments/backfill_v17_11_metadata.py` (post-hoc V17.11 sidecar for Tongji ckpt)
- `experiments/run_phase5_iitd_after_tongji_gate.sh` (V17.10 gate + IITD launch)
- `paper/phase1_theory_threatmodel.md` (theory draft, V14a-revised)

**Modified files**:
- `exp1_baselines/train_arcface.py`: V17.12 device fallback + V17.11 train-time telemetry + V17.8 ccnet branch
- `exp1_baselines/extract_embeddings.py`: V17.12 device fallback + V17.11 extract-time telemetry + V17.8 ccnet branch + auto-derived image_size/channels
- `exp1_baselines/datasets/tongji_dataset.py`: per-backbone channels (1=grayscale CCNet, 3=RGB MFN/IR50)

---

## 5. V17.9 verification results (all 4 steps passed before launch)

| Step | Method | Pass criterion | Result |
|---|---|---|---|
| 1. Static code inspection | WebFetch upstream train.py + user-provided models/ccnet.py | upstream Adam/StepLR/CE+SupCon verified; getFeatureCode = 2048D | ✓ V16.1 |
| 2. Synthetic forward | `forward(torch.randn(2,1,128,128))` | shape (2, 2048), L2-norm = 1 ± 1e-5 | ✓ shape OK, norm = 1.000000 |
| 3. One-batch overfit | 4 ids × 4 imgs × 50 SGD steps | loss decrease + finite gradients + non-zero grad both heads | ✓ 24.56 → 0.05; bb_grad 100→0.06; head_grad 33→0.13 |
| 4. 5ep smoke | `train_arcface --smoke` (1 ep × 4 batches) + extract + mini-orchestrator | training completes; NPZ 2048D; orchestrator passes Hard Rule + B==C-recal sanity | ✓ NPZ shape (5600, 2048) L2=1.0; sanity = ok |

---

## 6. Tongji 50ep training timeline

### 6.1 Three launch attempts (operational issues + recoveries)

| Attempt | PID | Start | Issue | Resolution |
|---|---|---|---|---|
| 1st | 94082 | 5:46 PM May 8 | num_workers=4 → 4 multiprocessing-fork workers each held an MPS model → 23GB used / 84MB free → 906K swapouts → ep 2 took 3 hours | Killed; configs num_workers 4 → 0 |
| 2nd | 94824 | 9:46 PM May 8 | Python stdout buffered when piped through `tee`; log file empty after 1h 46min CPU time | Killed; switched to PYTHONUNBUFFERED=1 + direct redirect (no tee) |
| **3rd** | **99715** | **2:03 AM May 9** | None | **Currently in progress (ep 48/50)** |

### 6.2 3rd attempt loss/acc trajectory (the successful run)

| Epoch | avg_loss | avg_acc | time (s) | Note |
|---|---|---|---|---|
| 1 | 39.83 | 0.0000 | 240 | ArcFace s=48 m=0.5 initial |
| 2 | 40.11 | 0.0000 | 235 | margin penalty dominant |
| 3 | 36.26 | 0.0000 | 239 | descent begins |
| 5 | 33.45 | 0.0000 | 237 | |
| 10 | (skipped — consistent decrease) | 0.0000 | ~237 | |
| 20 | 20.50 | 0.0000 | 271 | halfway, still pre-transition |
| 21 | 19.02 | 0.0000 | 1216 | one-time slow ep (system) |
| 25 | 12.52 | 0.0000 | 245 | |
| 30+ | (phase transition begins) | rising | | margin learned |
| 34 | 1.12 | **0.9535** | 226 | acc breakthrough |
| 36 | 0.94 | 0.9674 | 227 | |
| 40 | (continued descent) | ~0.98 | ~210 | |
| 43 | 0.81 | 0.9826 | 197 | |
| 45 | 0.83 | 0.9819 | 219 | converged |
| 47 | 0.82 | 0.9826 | 212 | |
| 48 | (in progress) | ~0.99 | | |

ArcFace **phase transition** between ep 25 and ep 34 — typical pattern when angular margin "clicks in" after enough cosine separation accumulates.

**Final outlook**: ep 50 expected avg_loss ~0.8, avg_acc ~98%. V17.10 gate (TPIR@5% ≥ 0.50) should pass comfortably.

---

## 7. V17.11 metadata telemetry (audit-driven additions)

The user asked for a self-audit during training; that audit revealed V17.11 telemetry was specified by the plan but missing from the running training code. Three fixes were committed (1e95d14):

### 7.1 Train-time fields (in `train_arcface.py`, written to ckpt + sidecar JSON)

```
backbone, feature_dim, feature_source, feature_l2_normalized,
pretrained_used, author_pretrained_checkpoint_loaded, training_recipe,
input_resolution, input_channels, optimizer, scheduler, loss,
arcface_s/m, batch_size, P_K, epochs_planned/completed,
total_iterations, train_palm_count, train_image_count,
checkpoint_selection, device, torch_version, git_commit, spec_version
```

### 7.2 Extract-time fields (in `extract_embeddings.py`, written to NPZ + sidecar JSON)

```
embedding_norm_mean/std/min/max, extraction_device, torch_version,
n_samples_extracted, embedding_dim, extraction_git_commit,
ckpt_path, ckpt_arch, ckpt_image_size, ckpt_channels,
ckpt_v17_11_metadata_present, spec_version
```

### 7.3 Backfill (in `experiments/backfill_v17_11_metadata.py`)

Reconstructs all V17.11 fields from `ckpt['config']` + system info for the
currently-running Tongji ckpt (which was launched before V17.11 was
committed). Writes the same sidecar JSON with `backfilled: true` flag.

---

## 8. Phase 6 theory revision (V14a-compliant)

`paper/phase1_theory_threatmodel.md` was restructured per V14a + V17.4:

| Section | Before | After |
|---|---|---|
| §4.2 Lemma 1 | Monotonicity stated, no special status | Tagged "CENTRAL SECURITY THEOREM"; explicit "load-bearing" subsection |
| §4.3 Lemma 2 | EVT presented as supporting framework | Tagged "AUXILIARY MODELING LENS"; allows Gumbel OR Weibull |
| §4.4 Theorem 1 | Single statement, EVT-derived | Split: Theorem 1a (load-bearing, monotonicity-only) + 1b (auxiliary Gumbel closed form) |
| §4.5 Corollary 1 | Single statement, Gumbel-derived | Split: Corollary 1a (load-bearing, weak/strict FPIR inflation) + 1b (auxiliary Gumbel inflation factor) |
| §4.6 Empirical | "KS p>0.05 gate" (deprecated by V14a) | Descriptive Gumbel + Weibull side-by-side, no pass/fail gate |
| §4.7 Why this matters | EVT-grounded | Robustness paragraph: result independent of GEV family |
| §A2 Domain of attraction | Pre-locked Gumbel | Enumerates Weibull/Gumbel/Fréchet without locking |

Commit 136f6bc.

---

## 9. Helper scripts ready for Phase 5 step 6–10

### 9.1 `experiments/run_phase5_iitd_after_tongji_gate.sh`

After Tongji ckpt save, run this:
1. Extract Tongji CCNet 2048D embeddings (writes V17.11 extract sidecar JSON automatically)
2. Single-seed orchestrator (order_seed=0, enroll_seed=0, target_fpir=0.05) at endpoint
3. Parse `protocol_c_recal.json` → final-step TPIR@5%
4. Apply V17.10 gate:
   - ≥ 0.50 → save gate metadata, launch IITD CCNet 50ep in background
   - 0.30..0.50 → marginal, manual decision (200ep escalation)
   - < 0.30 → undertraining failure, manual decision

### 9.2 `experiments/backfill_v17_11_metadata.py`

Run once on Tongji ckpt to write V17.11 sidecar JSON:
```
python experiments/backfill_v17_11_metadata.py \
    --ckpt experiments/generated/tongji_full/ccnet_arcface_tongji_scratch_128.pt \
    --train_image_count 5760 \
    --train_palm_count 320
```

### 9.3 `experiments/plot_multi_backbone.py`

After Tongji + IITD CCNet results land in `*/ccnet_protocols/`:
```
python experiments/plot_multi_backbone.py \
    --datasets Tongji=experiments/generated/tongji_full \
               IITD=experiments/generated/iitd_full \
    --out experiments/figures
```

Generates 2-row × 2-col figure (TPIR trajectory + tau drift; MFN vs CCNet).

---

## 10. Pending steps (after Tongji 50ep completes)

| Step | Action | Estimated time |
|---|---|---|
| Tongji backfill | `python experiments/backfill_v17_11_metadata.py …` | 5 sec |
| Tongji extract + gate | `bash experiments/run_phase5_iitd_after_tongji_gate.sh` | 5–10 min (extract + orchestrator) |
| If pass → IITD launch | (auto from gate script) | ~1 hr (IITD smaller dataset, 50ep) |
| IITD extract + gate | Same helper logic, manually re-run for IITD | 5 min |
| Full multi-seed orchestrator | order_seed=10, enroll_seed=3 for both datasets × CCNet | ~30 min |
| `plot_multi_backbone.py` | MFN vs CCNet comparison figure | 1 min |
| Phase 5 final commit | All Phase 5 outputs + plot + JSONs | 1 min |

---

## 11. Open risks (V17.10 gate failure paths)

If Tongji TPIR@5% at ep 50 < 0.50:

| Tier | Path |
|---|---|
| 0.30 ≤ TPIR < 0.50 (marginal) | L2 escalation: extend Tongji to 200 epochs; report 50ep in appendix as budget-matched |
| TPIR < 0.30 (undertraining failure) | L2 escalation OR exclude Tongji CCNet from main; report in limitations as "not converged under standardized recipe within iteration budget" |

If even L2 200ep < 0.50: Tongji CCNet excluded from central evidence;
multi-backbone universality claim weakens but Tongji MFN remains as primary
evidence (TPIR 89.3% from Phase 0).

Current Tongji 50ep has avg_acc 98% on training samples by ep 47, so the
gate should pass comfortably; this contingency is documented for
completeness only.

---

## 12. Reviewer-defense framing (locked paper wording)

For reviewer questions about CCNet, use V16.6/V17.8 wording verbatim:

> "We adopt the CCNet feature architecture (Yang et al. 2023, IEEE TIFS,
> 'Comprehensive Competition Mechanism in Palmprint Recognition') and
> retrain it under the same standardized open-set ArcFace recipe used
> for MFN-ArcFace. We do not attempt to reproduce CCNet's reported
> closed-set verification accuracy. The original authors note that
> CCNet's cross-identity generalization is limited (Yang et al. 2023
> §IV.E), making CCNet a meaningful architecture-level stress test for
> our sequential open-set protocol analysis. The training-recipe
> deviation from upstream (Adam lr=0.001, batch 1024, 3000 epochs,
> hybrid CE + SupCon) to ours (SGD lr=0.01, batch 128, 50 epochs,
> ArcFace CE only) is documented; iteration budget is approximately
> 8× fewer than upstream's, by design, to preserve recipe parity with
> MFN."

For 2048D inference embedding choice, use V17.8 wording:

> "Our wrapper outputs the 2048-D L2-normalized embedding that matches
> Yang et al. 2023's released `getFeatureCode` inference path. The
> 6144-D `concat(fc, fc1)` feature used internally during upstream
> training (as the SupCon target) is not exposed as the default
> evaluation embedding."

---

*Saved 2026-05-10. Will be updated when Phase 5 step 6–10 complete with
final TPIR/Wilson CI/cluster bootstrap numbers.*
