# Phase 2 Draft: COCONUT Method

**Status**: DRAFT v0.1. Sections 5.2 and 5.3 in detail; 5.1, 5.4–5.6 in outline.
**Target venue**: TIFS (matches phase1).
**Companion sections in paper**: Sections 5 (Method), 6 (Experimental Setup).

---

## Section 5 — Method

### 5.1 System overview

We address the continual open-set palmprint identification problem
formalized in §3. Concretely, the system processes a stream of enrollment
experiences `e_1, e_2, …, e_T`, where each `e_t` introduces a single new
user `u_t` accompanied by `n_u` enrollment images. After each experience
the system must (i) extend its gallery with a prototype for `u_t`, (ii)
recalibrate the operational threshold `τ_t`, and (iii) preserve recognition
of all previously enrolled users `{u_1, …, u_{t−1}}`.

A complete system therefore comprises six components: a feature backbone, a
projection head that maps backbone features to a lower-dimensional metric
space, a metric-learning loss driving inter-user separation in that space,
a memory buffer for replay against catastrophic forgetting, a prototype
classifier with score normalization, and a threshold calibrator. We
describe each in turn.

### 5.2 Backbone and PCA-initialised Projection Head

#### 5.2.1 Backbone

We use the CCNet backbone (Yang et al., 2023) pretrained on the Tongji
palmprint dataset with ArcFace loss (cf. §[Experiment 1] for pretraining
details). CCNet outputs a 2048-dimensional feature vector for each input
image after global pooling. We refer to this raw output as `f ∈ ℝ^{2048}`.

#### 5.2.2 Motivation for dimensionality reduction

The ProxyAnchor paper (Kim et al., 2020, Fig. 5) reports stable Recall@1
across embedding dimensions in the range 32–1024 on Cars-196, with no
results for 2048. Using a 2048-dimensional embedding in the continual
open-set setting compounds three problems:

1. **High-dimensional prototype noise.** Each user supplies only `n_u = 9`
   enrollment samples in our protocol, yielding a sample-to-dimension ratio
   of `9 / 2048 ≈ 0.0044`. Class-mean prototypes computed from so few
   samples are dominated by per-sample noise rather than capturing the
   user's true central tendency.

2. **Score compression in open-set rejection.** Cosine similarity between
   two random unit vectors in `ℝ^D` concentrates around `0` with standard
   deviation `O(1/√D)`. At `D = 2048`, both genuine and impostor score
   distributions are compressed into a narrow band, reducing the margin
   available for FPIR-calibrated thresholding.

3. **Backbone feature redundancy.** ArcFace-trained backbones empirically
   exhibit an effective rank well below their nominal output dimension; the
   top several hundred PCA components typically capture > 99 % of the
   variance of CCNet features on in-domain data.

We therefore introduce a linear projection from `ℝ^{2048}` to `ℝ^D` for
`D ∈ {128, 512}` and treat the choice of `D` as an ablation axis.

#### 5.2.3 Architecture and initialisation

The projection head is a single linear layer `W ∈ ℝ^{D × 2048}` without
bias:

```
h = W f,   h ∈ ℝ^D.
```

We omit the bias term because all downstream consumers (cosine NCM,
ProxyAnchor) L2-normalise their inputs, which absorbs any additive shift.
The projection head is trainable end-to-end alongside the backbone.

**PCA initialisation.** Rather than initialising `W` from a Gaussian
distribution, we initialise it with the top-`D` principal components of
pretrained CCNet features on the in-domain enrollment pool. Let
`F ∈ ℝ^{N × 2048}` be the matrix of CCNet features extracted from `N`
in-domain images. We compute the centred SVD

```
F − μ_F = U Σ V^T,
```

and set `W ← V[:, :D]^T`. With this initialisation the projection at
training start equals the PCA of pretrained features, retaining ≥ 99 % of
the variance in CCNet's effective subspace while reducing dimensionality
to `D`. The projection remains trainable so that downstream metric
learning can rotate the basis toward class-discriminative directions that
PCA, being unsupervised, cannot find on its own.

**Sample requirement.** The number of non-zero singular values in the
centred matrix `F − μ_F` is `min(N − 1, 2048)`. PCA initialisation with
`D = 512` therefore requires `N ≥ 513` samples for the projection
weight to be fully defined; in our protocol we use `N ≈ 450` from
`enroll_file` augmented by a small held-out unknown_dev pool, yielding
`N ≈ 900`, sufficient for `D = 512`. For `D = 1024` the sample budget is
inadequate and we do not include it in our sweep.

**Data leakage.** PCA is unsupervised and uses no class labels; only the
feature distribution shape. We further restrict PCA fitting to
`enroll_file` (the training pool) and never use `unknown_test_file` (the
final FPIR evaluation pool), eliminating any direct evaluation-set
leakage.

#### 5.2.4 Learning rate

We use a single learning rate `η` for the backbone and the projection head
(`projection_lr_ratio = 1`). This is conservative: the backbone is
pretrained and well-conditioned, the projection head begins at a
near-optimal PCA initialisation, and both benefit from the same gentle
update schedule. The proxy parameters of the metric-learning loss
(§5.3) receive a higher learning rate by a factor of 50, as is standard
in the metric learning literature.

### 5.3 Proxy Anchor metric learning for continual enrolment

#### 5.3.1 Background

Proxy Anchor loss (Kim et al., 2020) achieves a favourable trade-off
between pair-based and proxy-based metric learning by treating each
class proxy as an anchor and computing a softplus-LSE loss against all
in-batch embeddings. The canonical form (Eq. (4) in the original paper)
is

```
ℒ_PA(X) = (1 / |P^+|) Σ_{p ∈ P^+} log(1 + Σ_{x ∈ X^+_p} exp(−α (s(x, p) − δ)))
        + (1 / |P|)   Σ_{p ∈ P}   log(1 + Σ_{x ∈ X^−_p} exp( α (s(x, p) + δ))),
```

where `P` is the set of all proxies, `P^+ ⊆ P` is the set of proxies that
have at least one positive sample in the batch `X`, and `s(·, ·)` is
cosine similarity.

#### 5.3.2 Continual extension

In our continual setting `|P|` grows by one at each enrollment experience.
We instantiate a new proxy `p_t` for each new user `u_t` and initialise
it with the L2-normalised mean of `u_t`'s enrollment features (rather than
a random Gaussian as in the original paper). This *feature-mean
initialisation* avoids the cold-start period in which a randomly
initialised proxy is far from its class cluster and produces uninformative
gradients during the first few iterations.

We retain the original hyperparameters `α = 32, δ = 0.1` without
modification.

#### 5.3.3 Negative-term formulation under class-balanced replay

Our class-balanced memory buffer (§5.4) draws `memory_batch_size`
samples per training step in equal allocation across all currently
enrolled users. With `memory_batch_size = 160` and gallery sizes up to
`|P| = 50` in our experiments, every batch contains at least one
positive sample for every proxy, i.e., `P^+ = P` deterministically. Under
this batch composition the canonical Eq. (4) reduces to a form numerically
identical to a legacy variant that sums and averages the negative term
over `P^+` rather than `P`. We confirmed this equivalence both
analytically (the inner sums and the normalising denominators coincide
when `P^+ = P`) and empirically with a controlled ablation at `T = 35`
users in which the two variants produced bit-exact identical trajectories
(Table TODO). We therefore report results under the legacy variant for
backward compatibility with our earlier ablation series; this footnote
should appear in the experimental section.

#### 5.3.4 Deviations from the original Proxy Anchor recipe

For transparency we list the three hyperparameter deviations from Kim et
al. (2020):

| Hyperparameter | Original | Ours | Justification |
|---|---|---|---|
| Proxy LR multiplier | 100× | 50× | Continual proxy addition; 100× induces oscillation in newly added proxies competing with established ones |
| Optimiser | AdamW | Adam | Marginal effect (~0.5 pp); historical choice retained for reproducibility with earlier ablation series |
| Loss scalar `λ_PA` | implicit 1.0 | 0.5 | Vestige of a multi-loss configuration in earlier system versions; with ProxyAnchor as the sole loss this is equivalent to halving the effective proxy learning rate, which together with the 50× multiplier yields a 25× effective rate relative to the backbone |

These deviations are stable across our ablation series and do not
affect the qualitative conclusions of any experiment.

### 5.4 Memory buffer and replay

(Outline; to be detailed.)

- **Class-balanced reservoir buffer** (`coconut/memory/buffer.py`).
  Per-class reservoir of size `memory_size / |gallery_t|` (adaptive),
  yielding equal representation across all enrolled users.
- **Replay schedule.** Each training step draws
  `memory_batch_size = 160` samples balanced across users plus
  `experience_batch_size = 16` samples from the current user.
- **Quality-Aware Replay (QAR).** Users with current cosine score below
  `τ_t + rehab_margin` are selected for additional rehabilitation
  sampling (`rehab_samples_per_user = 4`). This targets users whose
  prototypes are at risk of degradation without inflating overall replay
  cost.

### 5.5 Prototype classifier with S-norm

(Outline.)

- **Cosine NCM.** Each user `u` has prototype `μ_u` equal to the
  L2-normalised mean of their projected features in memory. Match score
  is `s(x, u) = ⟨h_x, μ_u⟩`.
- **S-norm.** Per-class score statistics `(μ̂_u, σ̂_u)` are computed on a
  held-out unknown-dev pool, and the operational score is
  `s'(x, u) = (s(x, u) − μ̂_u) / σ̂_u`. This compensates for per-user
  variation in impostor score distribution caused by prototype crowding.
- **Threshold gate.** A query is accepted if `max_u s'(x, u) ≥ τ_t` and
  the argmax identifies the predicted user.

### 5.6 Threshold calibration

(Outline.)

- After each enrollment experience, the threshold `τ_t` is recalibrated
  on a fixed unknown-dev pool to achieve the target FAR (e.g., `α =
  0.01`). Specifically, `τ_t` is the `1 − α` quantile of the per-class
  maximum S-normalised impostor score on the calibration pool against
  the current gallery `G_t`.

---

## Section 6 — Experimental Setup (outline)

### 6.1 Datasets

- BJTU PalmV2 (ROI) — primary in-domain dataset, 100 users with 9 images each.
- IITD palmprint — cross-domain FPIR evaluation only.
- Tongji palmprint — backbone pretraining only.

### 6.2 Continual protocol

- `num_experiences = 50` for final results, `35` for sweeps.
- 80/20 train/probe split per user.
- `unknown_dev_file` (first 50 % of unknown users) for `τ` calibration.
- `unknown_test_file` (last 50 % of unknown users) for final FPIR
  evaluation — never seen during training or calibration.

### 6.3 Metrics

- **Mean TAR@1 % FPIR over late experiences** (Mean(40..50) for
  `T = 50`, Mean(25..35) for `T = 35`). This is our primary metric;
  it is more robust to single-experience variance than FinalTAR (see
  §6.5).
- **BWT** (backward transfer, positive = previous users improved).
- **Final Forgetting** (mean per-user drop from peak to final).
- **WorstTAR** over the same late window.
- **F@C95** (lowest FPR at which CCR remains at 95 % of closed-set
  accuracy, adapted from Rabinowitz et al., 2025).

### 6.4 Ablations

- **Component ablation** (9 variants): L_naive, L_replay,
  only_{proxy, idl, der, qar, snorm, maha, recal}, no_{proxy, idl, der,
  qar, snorm, maha, recal}, L_minimal, L_minimal_no_supcon, L_full,
  only_proxy_no_supcon. Measured at `T = 50`.
- **Dimensionality sweep** (6 variants, `T = 35`):
  baseline_{legacy, canonical}_2048,
  proj_{legacy, canonical}_512,
  proj_{legacy, canonical}_128.
- **Winner verification**: best dimensionality config at `T = 50`.

### 6.5 FinalTAR vs Mean TAR

We report Mean over late experiences rather than FinalTAR because the
former is a more reliable estimator. As an illustration, L_full at
`T = 100` exhibits FinalTAR = 0.848 due to a single anomalously hard
final user, while the same run's Mean(91..100) = 0.917 — a difference of
6.9 pp that does not reflect any change in system state. Mean reports
remove this single-experience variance.

---

## TODOs before submission

- [ ] Fill Tables in §5.3.4 (deviations) and §6.4 (ablation results) with measured numbers.
- [ ] Complete §5.4 (memory buffer details with exact equations).
- [ ] Complete §5.5 (S-norm formal derivation).
- [ ] Complete §5.6 (threshold calibration formal definition + cost analysis).
- [ ] Add Section 7 (Results) with figures.
- [ ] Cite Kim et al. 2020 (ProxyAnchor), Yang et al. 2023 (CCNet), Movshovitz-Attias et al. 2017 (ProxyNCA), Wang et al. 2018 (CosFace), Deng et al. 2019 (ArcFace), Rabinowitz et al. 2025 (GHOST).
- [ ] Cross-reference §3 (threat model) and §4 (theory) at appropriate points.
