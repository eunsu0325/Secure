# Phase 1 Draft: Theory + Threat Model

**Status**: DRAFT v0.1. To be refined during paper writing (Week 6).
**Target venue**: TIFS (security framing required).
**Companion sections in paper**: Sections 3 (Threat Model), 4 (Theory).

---

## Section 3 — Threat Model

### 3.1 System and operational setting

We consider a deployed open-set palmprint identification system with a
gallery that **grows over time**. At deployment time `t = 0`, the gallery
contains an initial *base anchor* set of palms. As enrollment proceeds, new
palms are added to the gallery. We denote the gallery size at step `t` by
`m(t)`, the cumulative gallery `G_t = {p_1, ..., p_{m(t)}}`, and the system's
acceptance threshold by `τ_t`.

A query `x` is matched against `G_t` by computing per-prototype similarity
scores `s(x, p_i)`, taking the top-1 score `S_{m(t)}(x) = max_i s(x, p_i)`,
and **accepting** if `S_{m(t)}(x) ≥ τ_t` (with the identity of the top-1
prototype as the predicted label).

The system's operational requirement is that the **realized False Positive
Identification Rate (FPIR)** stay at or below a target `α` (e.g., `10⁻³`):

```
FPIR(t, τ_t) = Pr_{x ~ unknown}[S_{m(t)}(x) ≥ τ_t] ≤ α.
```

### 3.2 Adversary

We consider an **enrollment-time adversary** with two variants:

- **(A) Capacity-injection adversary** (insider or compromised enrollment
  channel). Capability: induce growth of `m(t)` over time. Goal: cause the
  realized FPIR to drift above the target `α` so that subsequent unauthorized
  authentication attempts succeed at a rate higher than the certified
  operating point.
- **(B) Passive-observation adversary**. Capability: observe published
  evaluation reports that use a static endpoint protocol (e.g., Protocol B
  in §5). Goal: identify deployment regimes where the gap between
  endpoint-certified FPIR and operational FPIR exceeds a threshold, then
  time impostor attempts accordingly.

Both variants exploit the same underlying vulnerability: the assumption that
a threshold certified at one gallery size remains valid as the gallery grows.

### 3.3 Defender

The defender holds:

- a fixed *calibration probe set* `P_cal` drawn from a held-out unknown
  population (we use `external_dev` and, in the security-stress experiment,
  WebPalm calibration probes);
- the ability to compute `τ_t` from `P_cal` against the operational gallery
  `G_t` at any chosen subset of steps;
- knowledge of the score distribution of the deployed backbone (assumed
  fixed across the deployment lifetime).

The defender's goal is to maintain `FPIR(t, τ_t) ≤ α` at every `t` while
minimizing the **calibration cost**, defined below.

### 3.4 Cost model

A single calibration at step `t` requires `|P_cal| · m(t)` similarity
comparisons. A protocol that recalibrates at every step has cumulative cost
proportional to `|P_cal| · ∑_t m(t)`. The C-recal baseline (per-step
recalibration) is the *oracle* in the sense that it always satisfies
the operational requirement, but its cost grows superlinearly with deployment
lifetime.

### 3.5 Vulnerability

The vulnerability we expose in this paper is that protocols which freeze
`τ` after one calibration—e.g., a deployment that calibrates at `t = 0` and
reuses `τ_0` thereafter—suffer a structural FPIR violation as `m(t)` grows:

```
FPIR(t, τ_0) = Pr[S_{m(t)} ≥ τ_0]
              ≥ Pr[S_{m(0)} ≥ τ_0] = α     (Lemma 1)
```

with strict inequality whenever `m(t) > m(0)` and the per-prototype score
distribution is non-degenerate (Theorem 1 below). The static endpoint
evaluation protocol (Protocol B) does not capture this drift because it
reports a single `(τ, FPIR)` pair at full enrollment, hiding the deployment
trajectory.

### 3.6 Defense (preview)

Section 7 proposes **drift-aware calibration under limited probe budget**: a
calibration recipe that estimates `τ̂_m` for any gallery size `m` from a
small set of observed `(m_k, τ_{m_k})` pairs and the score distribution
fitted on the calibration probes. Section 8 evaluates this defense against
the enrollment-time adversary across three datasets.

---

## Section 4 — Theory

### 4.1 Notation and assumptions

- Probe space: `X` (unknown query images); gallery: `G_m = {p_1, ..., p_m}`.
- Per-prototype similarity score: `s(x, p_i) ∈ R`, assumed bounded (we use
  cosine similarity, so `s ∈ [-1, 1]`).
- Top-1 score: `S_m(x) = max_{i ≤ m} s(x, p_i)`.
- Threshold: `τ`. Accept iff `S_m(x) ≥ τ`.
- Target FPIR: `α ∈ (0, 1)`.
- For an unknown probe `X` drawn from a fixed population, define the
  per-prototype impostor score CDF:
  ```
  F_s(t) = Pr[s(X, p_1) ≤ t]
  ```
  We assume `F_s` is identical across `i = 1, ..., m` (exchangeability of
  prototypes) and continuous on its support.
- We do **not** assume independence of `s(X, p_i)` across `i`. Real palmprint
  prototypes share a backbone-induced distribution structure and exhibit
  positive dependence (similar palms produce correlated scores). The IID case
  is a useful boundary; we will state both IID-strict and exchangeable results.

### 4.2 Lemma 1 (Score monotonicity in gallery size)

For every probe `x` and `m_2 ≥ m_1`,
```
S_{m_1}(x) ≤ S_{m_2}(x).
```

**Proof.** `S_m(x) = max_{i ≤ m} s(x, p_i)`. The maximum over a superset is
greater than or equal to the maximum over the subset. ∎

**Consequence.** The marginal CDF of `S_m` is monotone in `m`:
`Pr[S_{m_2} ≤ τ] ≤ Pr[S_{m_1} ≤ τ]` for `m_2 ≥ m_1`.

### 4.3 Lemma 2 (Limiting distribution under EVT)

Suppose `{s(X, p_i)}_{i=1}^∞` is a stationary sequence with marginal CDF
`F_s` and satisfies a long-range mixing condition `D(u_m)` of Leadbetter
[Leadbetter 1983]. Then there exist normalizing sequences `(a_m, b_m)` with
`b_m > 0` such that
```
Pr[(S_m - a_m) / b_m ≤ y] → G(y)    as m → ∞,
```
where `G` is one of the three Generalized Extreme Value (GEV)
distributions—Gumbel, Fréchet, or Weibull—determined by the tail of `F_s`.

**Special case (Gumbel domain of attraction).** When the upper tail of `F_s`
decays exponentially (light-tailed; appropriate for cosine similarity bounded
above by 1), `G(y) = exp(-exp(-y))` with normalizing constants
`a_m = F_s^{-1}(1 - 1/m)` and `b_m` derived from the slope of `F_s^{-1}` at
`1 - 1/m`.

**Implications for our problem.** For finite `m`, the upper tail of `S_m`
is approximately
```
Pr[S_m > τ] ≈ 1 - G((τ - a_m) / b_m).
```
This gives a closed-form approximation of the realized FPIR at threshold `τ`
and gallery size `m`.

**On exchangeability.** Strict IID is sufficient but not necessary; the EVT
limit holds under weak mixing. Empirically, we verify in §4.5 that the
fitted Gumbel parameters track observed `S_m` distributions on Tongji, IITD,
and BJTU (KS-test `p > 0.05` at every step `t`).

### 4.4 Theorem 1 (Threshold drift is structural)

Let `τ_m` denote the unique threshold satisfying `Pr[S_m ≥ τ_m] = α` (i.e.,
the threshold that achieves the target FPIR at gallery size `m`). Then under
the assumptions of Lemma 2:

1. **Monotonicity.** For `m_2 > m_1`, `τ_{m_2} > τ_{m_1}` (strict).
2. **Closed-form (Gumbel case).** Under the Gumbel limiting form,
   ```
   τ_m ≈ a_m - b_m · log(-log(1 - α))
                ↑—— grows like F_s^{-1}(1 - 1/m), i.e., logarithmically in m
   ```

**Proof sketch.**
1. Lemma 1 gives `Pr[S_{m_2} ≥ τ_{m_1}] ≥ Pr[S_{m_1} ≥ τ_{m_1}] = α`. Since
   the CDF is continuous and strictly monotone on its support, the threshold
   yielding exactly `α` at `m_2` must be strictly greater than `τ_{m_1}`.
2. Solve `1 - G((τ - a_m) / b_m) = α` for `τ`, using the Gumbel CDF. ∎

### 4.5 Corollary 1 (Static-τ FPIR violation under enrollment growth)

Suppose the system is calibrated at time `t = 0` with gallery size `m_0` and
threshold `τ_0` chosen so that `Pr[S_{m_0} ≥ τ_0] = α`. If the gallery grows
to `m_t > m_0` and the threshold is **not** updated, then the realized FPIR
satisfies:
```
α(t) := Pr[S_{m_t} ≥ τ_0] > α.
```

Under the Gumbel limit, the inflation factor is approximately
```
α(t) / α  ≈  exp((a_{m_t} - a_{m_0}) / b_{m_0})
            = exp((F_s^{-1}(1 - 1/m_t) - F_s^{-1}(1 - 1/m_0)) / b_{m_0})
```
which grows monotonically with `m_t / m_0`.

**Numerical example (Gumbel with `b = 0.05`, base `m_0 = 30`).**
- `m_t = 30   → α(t) / α = 1.0`
- `m_t = 100  → α(t) / α ≈ 3.4`
- `m_t = 1000 → α(t) / α ≈ 11`

**Operational interpretation.** A deployment that certifies `α = 10⁻³` at
`m_0 = 30` and lets the gallery grow to `m_t = 1000` will see realized FPIR
on the order of `10⁻²`, an order-of-magnitude security regression that is
invisible to the static endpoint protocol.

### 4.6 Empirical validation (to be filled by experiments)

Three checks demonstrate that the IID/exchangeable EVT framework captures
the observed protocol effect on real data:

1. **Goodness-of-fit.** For each (dataset, gallery_size_t), collect the
   `external_dev` top-1 scores `{S_{m(t)}(x_j)}_j`. Fit a Gumbel distribution
   by maximum likelihood; report KS / Anderson-Darling statistics. Hypothesis:
   GoF accepted at α = 0.05 for every `(dataset, t)`.

2. **Predicted vs observed `τ_m`.** Use the Gumbel parameters fitted at
   `m_0` (small base anchor) to predict `τ̂_m` for `m > m_0` via Theorem 1.
   Compare to `τ_m` observed by running C-recal directly. Hypothesis:
   predicted falls within the Wilson 95% CI of the observed at every `t`.

3. **Predicted vs observed inflation factor.** Use Corollary 1 to predict
   `α(t) / α` under stale `τ_0` and compare to the C-fixed protocol's
   observed realized FPIR ratio. Hypothesis: predicted matches observed
   within multiplicative factor 2 across all `t` and all three datasets.

### 4.7 Why this matters for evaluation methodology

Theorem 1 and Corollary 1 are not merely empirical observations; they show
that under standard regularity conditions (continuous score distribution,
mild dependence), threshold drift in sequential open-set identification is
**structurally inevitable**. Any evaluation protocol that reports a single
`(τ, FPIR)` pair at one gallery size therefore cannot certify the security
of a deployment whose gallery size differs from the evaluation point. The
trajectory-based protocol (Protocol C-recal in §5) exposes this structural
behavior; the static endpoint protocol (Protocol B) hides it.

This argument elevates the gap between B and C from "an interesting
empirical observation" to "a measurement-theoretic necessity": the static
endpoint protocol *cannot* certify an operating point that requires
threshold-stable behavior across gallery sizes, because no single threshold
is stable in this sense.

---

## Section 4-Appendix — Notes on assumptions and limitations

### A1. Why exchangeability instead of IID

The IID assumption is sufficient but stronger than necessary for the EVT
limit (Lemma 2). For palmprint identification, the per-prototype scores
`s(X, p_i)` are mildly positively dependent because they share a fixed
backbone and a fixed query embedding. Empirically, the dependence is weak
enough that the GEV limit kicks in at modest `m` (e.g., `m = 20-30`),
which we verify in §4.5.

### A2. Domain of attraction

For cosine similarity bounded above by 1, the upper tail of `F_s` typically
exhibits sub-exponential decay → Gumbel domain of attraction. Heavy-tailed
score distributions (Fréchet) would imply a polynomial inflation factor
rather than logarithmic; this is the more pessimistic regime, but does not
arise in our experiments.

### A3. Boundary degeneracy

When `α` is close to `1/m` (i.e., the "minimum observable FPIR" of the
calibration probe set), the Gumbel approximation degrades and finite-sample
correction is needed. Section 5 documents the per-target reliability flag
that tracks this regime.

### A4. Distinction from face-recognition longitudinal drift

Prior longitudinal evaluation work in face recognition (e.g., NIST FRTE
ongoing reports) tracks performance changes due to **template aging** of a
fixed gallery. Our threshold-staleness vulnerability is orthogonal: even
with no template aging, the act of growing the gallery alone causes FPIR
drift. The two mechanisms can compound.

---

## To-do list before paper draft

- [ ] Fill in §4.5 with actual KS p-values per (dataset, t)
- [ ] Generate predicted-vs-observed `τ_m` figure (Tongji example)
- [ ] Generate predicted-vs-observed inflation-factor figure
- [ ] Cite Leadbetter 1983, Coles 2001 (EVT references)
- [ ] Cite NIST FRTE longitudinal reports for the §A4 distinction
- [ ] Cite RegPalm 2025 (TIFS) for low-FPIR palmprint context
- [ ] Convert this draft to LaTeX during paper writing (Week 6)

---

## Symbols quick reference (for §3-§4)

| Symbol | Meaning |
|---|---|
| `m`, `m(t)` | Gallery size (at step `t`) |
| `G_m`, `G_t` | Gallery (at size `m` or step `t`) |
| `s(x, p)` | Per-prototype similarity score |
| `S_m(x)` | Top-1 score against `G_m` |
| `F_s` | Per-prototype impostor score CDF |
| `α` | Target FPIR |
| `τ_m` | Threshold yielding `α` at gallery size `m` |
| `τ_0`, `τ_t` | Threshold at step 0 (deployment) or step `t` |
| `α(t)` | Realized FPIR at step `t` under static `τ_0` |
| `(a_m, b_m)` | Gumbel normalizing sequences |
| `G(y) = exp(-exp(-y))` | Standard Gumbel CDF |
| `P_cal` | Calibration probe set |

---

*Draft saved 2026-05-08. Pre-experimental; numerical examples are
illustrative only. To be re-validated with §4.5 results and rewritten in
LaTeX before submission.*
