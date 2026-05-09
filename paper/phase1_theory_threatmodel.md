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

### 4.2 Lemma 1 (Score monotonicity in gallery size) — CENTRAL SECURITY THEOREM

For every probe `x` and `m_2 ≥ m_1`,
```
S_{m_1}(x) ≤ S_{m_2}(x).
```

**Proof.** `S_m(x) = max_{i ≤ m} s(x, p_i)`. The maximum over a superset is
greater than or equal to the maximum over the subset. ∎

**Consequence.** The marginal CDF of `S_m` is monotone in `m`:
`Pr[S_{m_2} ≤ τ] ≤ Pr[S_{m_1} ≤ τ]` for `m_2 ≥ m_1`.

**Why this is the load-bearing result for our security claim.** Lemma 1 is
**assumption-free** beyond the existence of a maximum (which holds for any
finite gallery and any well-defined per-prototype score). It does NOT
require:

- a specific score distribution (Gaussian, sub-exponential, etc.),
- independence or exchangeability across prototypes,
- a particular tail behavior, or
- convergence of any normalized statistic.

Section 4.4 below uses Lemma 1 alone to prove that the realized FPIR under
a static threshold strictly inflates as the gallery grows. The
Extreme-Value-Theory (EVT) framework introduced in §4.3 is an **auxiliary
modeling lens** that gives closed-form predictions of `τ_m` and inflation
factors; it is NOT load-bearing for the security argument.

### 4.3 Lemma 2 (Limiting distribution under EVT) — AUXILIARY MODELING LENS

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
limit holds under weak mixing. The EVT family at the limit (Gumbel,
Fréchet, or Weibull) is determined by the upper tail of `F_s`. Cosine
similarity is bounded above by `1`, so `F_s` has a finite right endpoint;
under sub-exponential decay this corresponds to the Gumbel domain, while
polynomial decay near the endpoint can place the limit in the Weibull
domain. We do **not** lock a single GEV family in this paper: §4.6 reports
empirical fits per dataset.

**Important framing**: this lemma serves only to support the closed-form
modeling in §4.4-Aux and §4.5-Aux. The central security results in §4.4
and §4.5 derive from Lemma 1 alone and do NOT depend on which GEV family
applies, or even on whether the EVT limit holds.

### 4.4 Theorem 1 (Threshold drift is structural)

Let `τ_m` denote the threshold satisfying `Pr[S_m ≥ τ_m] = α` (i.e., the
threshold that achieves the target FPIR at gallery size `m`).

**Theorem 1a (load-bearing — uses only Lemma 1).** Assume `F_s` is continuous
and strictly increasing on its support (no atoms at any candidate
threshold). Then for `m_2 > m_1`,
```
τ_{m_2} > τ_{m_1}    (strict).
```

**Proof.** Lemma 1 gives `Pr[S_{m_2} ≥ τ_{m_1}] ≥ Pr[S_{m_1} ≥ τ_{m_1}] = α`.
Since the CDF of `S_m` is continuous and strictly monotone on its support
(inherited from `F_s` via the max), the threshold yielding exactly `α` at
`m_2` must satisfy `τ_{m_2} > τ_{m_1}`. ∎

**Note**: Theorem 1a uses no distributional assumption beyond continuity
of `F_s`. It is the load-bearing result for the security argument in §4.5
(Corollary 1a). Threshold drift is therefore a **structural property of any
sequential-enrollment system with a continuous, non-degenerate score
distribution** — independent of which EVT family applies, whether the EVT
limit holds at all, or whether the prototypes are independent.

**Theorem 1b (auxiliary closed form — uses Lemma 2 + Gumbel domain).** When
`F_s` lies in the Gumbel domain of attraction with normalizing sequences
`(a_m, b_m)`,
```
τ_m ≈ a_m - b_m · log(-log(1 - α))
             ↑—— grows like F_s^{-1}(1 - 1/m), i.e., logarithmically in m.
```

**Proof.** Solve `1 - G((τ - a_m) / b_m) = α` for `τ`, using the Gumbel CDF
`G(y) = exp(-exp(-y))`. ∎

**Note**: Theorem 1b is a **closed-form approximation** that gives a
quantitative growth rate when the Gumbel limit applies. If the score tail
puts `F_s` in the Weibull or Fréchet domain, the closed form changes (still
monotone, still increasing in `m`, but different growth rate). The Gumbel
formula is reported here for the dataset-specific empirical fits in §4.6
and the mitigation predictor in §7; it is not a load-bearing component of
the central security claim.

### 4.5 Corollary 1 (Static-τ FPIR violation under enrollment growth)

Suppose the system is calibrated at time `t = 0` with gallery size `m_0` and
threshold `τ_0` chosen so that `Pr[S_{m_0} ≥ τ_0] = α`.

**Corollary 1a (load-bearing — uses only Lemma 1).** If the gallery grows
to `m_t > m_0` and the threshold is **not** updated, then for any `F_s` that
is non-degenerate (i.e., assigns positive probability to scores at or above
`τ_0`):
```
α(t) := Pr[S_{m_t} ≥ τ_0]  ≥  Pr[S_{m_0} ≥ τ_0]  =  α,
```
with strict inequality whenever `m_t > m_0` and `Pr[s(X, p) ≥ τ_0] > 0` for
the new prototypes (i.e., the new prototypes contribute non-zero
above-threshold mass).

**Proof.** From Lemma 1, `S_{m_t} \geq S_{m_0}` pointwise, so
`Pr[S_{m_t} ≥ τ_0] \geq Pr[S_{m_0} ≥ τ_0] = α`. Strict inequality follows
when at least one of the added prototypes can attain a score above `τ_0`
with positive probability. ∎

**This is the central security claim**: any sequential-enrollment system
with a continuous, non-degenerate score distribution suffers **at least
weak FPIR inflation under static-τ deployment**, and **strict inflation**
under any non-pathological score distribution. No distributional assumption,
EVT limit, or independence assumption is required.

**Corollary 1b (auxiliary quantitative inflation — uses Lemma 2 + Gumbel
domain).** When `F_s` lies in the Gumbel domain, the inflation factor is
approximately
```
α(t) / α  ≈  exp((a_{m_t} - a_{m_0}) / b_{m_0})
            = exp((F_s^{-1}(1 - 1/m_t) - F_s^{-1}(1 - 1/m_0)) / b_{m_0}).
```

**Numerical example (Gumbel with `b = 0.05`, base `m_0 = 30`).**
- `m_t = 30   → α(t) / α = 1.0`
- `m_t = 100  → α(t) / α ≈ 3.4`
- `m_t = 1000 → α(t) / α ≈ 11`

**Operational interpretation.** Under the Gumbel approximation, a
deployment that certifies `α = 10⁻³` at `m_0 = 30` and lets the gallery
grow to `m_t = 1000` will see realized FPIR on the order of `10⁻²` — an
order-of-magnitude security regression that is invisible to the static
endpoint protocol. If `F_s` is in the Weibull domain, the multiplicative
inflation factor grows more slowly with `m_t / m_0`; if in the Fréchet
domain, more aggressively. Corollary 1a guarantees inflation in **all
three** EVT regimes; Corollary 1b quantifies it specifically for the
Gumbel case.

### 4.6 Empirical analysis (descriptive — no pass/fail gates)

Two empirical exercises support the theoretical results above. Per V14a,
**no goodness-of-fit pass/fail gate is used**: KS or Anderson-Darling
p-values are reported descriptively, not as decision criteria. The
load-bearing security claims (Theorem 1a, Corollary 1a) do not depend on
any GEV fit.

1. **GEV fit (Gumbel and Weibull) per dataset.** For each
   `(dataset, gallery_size_t)`, collect `external_dev` top-1 scores
   `{S_{m(t)}(x_j)}_j`. Fit Gumbel and Weibull distributions by maximum
   likelihood; report log-likelihood, AIC, KS statistic, and AD statistic
   for both fits side-by-side. Per V14a, we do not declare one fit
   "accepted" and the other "rejected"; we report the better-fitting GEV
   family per dataset and continue with closed-form predictions in §4.7
   only when the better fit is reasonable.

2. **Predicted vs observed `τ_m` and inflation factor (Gumbel-domain
   datasets only).** For datasets where Gumbel is the better-fitting family,
   use Gumbel parameters fitted at `m_0` to predict `τ̂_m` for `m > m_0`
   via Theorem 1b, and predict `α̂(t) / α` via Corollary 1b. Compare to
   C-recal observed `τ_m` and C-fixed observed `α(t) / α` respectively.
   Report median absolute error and 95% Wilson CI agreement rate. **No
   pass/fail gate**; the predictor is offered as one modeling tool, with
   explicit caveats when fit quality is marginal.

The structural claim of Theorem 1a / Corollary 1a is independent of
whether either prediction matches well: even if the closed-form formula is
inaccurate, the fact that `τ_m` is strictly increasing and `α(t)` strictly
inflates remains true and can be observed directly from C-recal trajectory
data.

### 4.7 Why this matters for evaluation methodology

The load-bearing results (Theorem 1a, Corollary 1a) require only:

- a continuous, non-degenerate per-prototype score distribution `F_s`, and
- the trivial monotonicity property `S_m = max_{i ≤ m} s(·, p_i)`.

Under these minimal conditions, threshold drift in sequential open-set
identification is **structurally inevitable**. Any evaluation protocol
that reports a single `(τ, FPIR)` pair at one gallery size therefore
cannot certify the security of a deployment whose gallery size differs
from the evaluation point. The trajectory-based protocol (Protocol
C-recal in §5) exposes this structural behavior; the static endpoint
protocol (Protocol B) hides it.

This argument elevates the gap between B and C from "an interesting
empirical observation" to "a measurement-theoretic necessity": the static
endpoint protocol *cannot* certify an operating point that requires
threshold-stable behavior across gallery sizes, because no single
threshold is stable in this sense.

**Robustness of the security claim**: because the load-bearing argument
depends only on monotonicity and non-degeneracy, the result is **robust to
the choice of GEV family** (Gumbel/Fréchet/Weibull) — and indeed robust
to whether the EVT limit applies at all. A reviewer who challenges the
specific Gumbel approximation in §4.4-Aux/§4.5-Aux does not undermine
Theorem 1a/Corollary 1a; the closed-form approximation is auxiliary
modeling, while the security claim itself is structural.

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

For cosine similarity bounded above by `1`, the upper tail of `F_s` has a
finite right endpoint. Within the EVT classification, this corresponds to:

- **Gumbel domain** if the tail decays sub-exponentially toward the
  endpoint (e.g., `1 - F_s(t) ∼ c · exp(-h(t))` with `h` slowly varying),
- **Weibull domain** if the tail decays polynomially toward the endpoint
  (e.g., `1 - F_s(t) ∼ c · (1 - t)^α` with `α > 0`),
- **Fréchet domain** if the upper tail is heavy. Heavy-tailed cosine
  scores are atypical in palmprint embeddings and do not arise in our
  data; we mention this case for completeness.

We do **not** lock the GEV family at the theory level. Per V14a, §4.6
reports per-dataset empirical fit and continues with the better-fitting
family. Crucially, Theorem 1a and Corollary 1a (the load-bearing
security results) are independent of which GEV family is chosen.

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

Per V14a (V17 authority chain), Phase 6 paper writing must:

- [ ] Lead §4 with Lemma 1 + Theorem 1a + Corollary 1a as the load-bearing
      security argument; the EVT/Gumbel apparatus is auxiliary modeling
      lens for §4.6 and §7 mitigation only.
- [ ] §4.6 empirical analysis: report Gumbel + Weibull GEV fits side-by-
      side per dataset (log-likelihood, AIC, KS, AD). **Do NOT** include a
      "KS p > 0.05" pass/fail gate.
- [ ] Generate predicted-vs-observed `τ_m` figure for the better-fitting
      GEV family on each dataset (Gumbel-domain only when fit is
      reasonable).
- [ ] Generate predicted-vs-observed inflation-factor figure with explicit
      caveats about fit quality.
- [ ] Add a robustness paragraph (§4.7) emphasizing that the central
      security claim does not depend on the GEV family.
- [ ] Cite Leadbetter 1983, Coles 2001 (EVT references).
- [ ] Cite NIST FRTE longitudinal reports for the §A4 distinction
      (template aging vs. gallery-growth drift).
- [ ] Cite RegPalm 2025 (TIFS) for low-FPIR palmprint context.
- [ ] Cite Yang et al. 2023 (TIFS, "Comprehensive Competition Mechanism in
      Palmprint Recognition") for CCNet, with explicit "stress-test under
      our standardized open-set recipe; not a reproduction" framing per
      plan §V17.8.
- [ ] Convert this draft to LaTeX during paper writing (Week 6).

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

*Draft saved 2026-05-08; revised 2026-05-09 per plan §V14a + §V17.4
(monotonicity-first theory, EVT/GEV demoted to auxiliary modeling lens,
KS pass/fail gate removed). Pre-experimental; numerical examples are
illustrative only. To be re-validated with §4.6 results and rewritten in
LaTeX before submission.*
