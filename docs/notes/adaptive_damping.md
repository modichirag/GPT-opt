# DAPOpNorm: Diagnostics, Hypotheses, and Adaptive Damping

## Background

The DAPOpNorm update is:

$$W \leftarrow W - \eta \cdot \mathrm{sign}(G C^{-1/2}) C^{-1/2}$$

where $C = XX^T$ is the input activation covariance (tracked via EMA), and $C^{-1/2}$ is
computed via eigendecomposition with relative damping:

$$C_{\mathrm{eff}} = C + \delta \cdot \lambda_{\max}(C) \cdot I$$

## What opnorm Actually Measures

When $\kappa(C_{\mathrm{eff}}) \approx 1/\delta$ by construction (so `C_condition` was removed
as a diagnostic — it is approximately constant regardless of LR):

$$\|C_{\mathrm{eff}}^{-1/2}\|_{\mathrm{op}} = \frac{1}{\sqrt{\lambda_{\min}(C_{\mathrm{eff}})}}
\approx \frac{1}{\sqrt{\delta \cdot \lambda_{\max}(C)}}$$

Large opnorm $\Leftrightarrow$ small $\lambda_{\max}(C)$: the absolute scale of the activation
covariance has collapsed, not that the covariance has become degenerate (shape is unchanged).

### Update Scale and the Implicit Annealing Property

The relative damping formula gives $C_{\mathrm{eff}}^{-1/2} = \lambda_{\max}(C)^{-1/2} \cdot \tilde{C}_{\mathrm{eff}}^{-1/2}$
where $\tilde{C} = C/\lambda_{\max}(C)$. Since $\mathrm{sign}(\cdot)$ is scale-invariant:

$$\text{update} = \mathrm{sign}(G \tilde{C}^{-1/2}) \cdot \lambda_{\max}(C)^{-1/2} \cdot \tilde{C}_{\mathrm{eff}}^{-1/2}$$

The effective update magnitude scales as $\lambda_{\max}(C)^{-1/2}$. At good LR,
$\lambda_{\max}(C)$ grows ~47× during training — so the effective step naturally anneals
as the model develops structured activations. This is **a useful implicit property**, not
a bug: the method automatically reduces its effective step size as training progresses.

## New Diagnostics

Replaced `C_condition` with `C_eigmax` and `C_eigmin` (before damping, per layer per step):

| Diagnostic | What it measures |
|-----------|-----------------|
| `C_inv_sqrt_opnorm` | $1/\sqrt{\lambda_{\min}(C_{\mathrm{eff}})}$ — preconditioner amplification |
| `C_eigmax` | $\lambda_{\max}(C)$ before damping — absolute covariance scale |
| `C_eigmin` | $\lambda_{\min}(C)$ before damping — covariance degeneracy |

For existing runs (which logged `C_condition`), `C_eigmax` can be reconstructed:
$$\lambda_{\max}(C) \approx \frac{\kappa(C_{\mathrm{eff}})}{(1+\delta) \cdot \mathrm{opnorm}^2}$$

## Empirical Findings

### Covariance scale trajectory (d=0.01, fineweb1B)

At lr=0.03 (good): $\lambda_{\max}(C)$ grows 0.11 → 5.15 over training, opnorm decreases.
At lr=0.07/0.15 (bad): $\lambda_{\max}(C)$ peaks around warmup end then collapses to ~0.1–0.2,
opnorm stays large (~17–30). The failure onset is at the warmup boundary (~step 381).

### Damping sweep results (fineweb1B, best val loss per damping)

| Damping $\delta$ | Best val | Best LR | LR range beating Muon (4.2385) |
|-----------------|----------|---------|-------------------------------|
| 0.0             | 4.336    | 0.03    | none                          |
| 0.001           | 4.252    | 0.03    | narrow                        |
| 0.003           | 4.215    | 0.01    | lr=0.01–0.02                  |
| 0.01            | 4.213    | 0.03    | lr=0.01–0.05                  |
| 0.03            | 4.202    | 0.03    | lr=0.01–0.07                  |
| 0.05            | 4.209    | 0.05    | lr=0.03–0.07                  |
| 0.1             | 4.232    | 0.07    | lr=0.03–0.07                  |
| 0.3             | 4.246    | 0.05–0.07 | narrowing (approaching Muon) |

Note: in the limit $\delta \to \infty$, $C_{\mathrm{eff}} \approx \delta\lambda_{\max}(C)I$
and the update reduces to $\mathrm{sign}(G)$ — exactly Muon. DAPOpNorm interpolates between
a fully covariance-adapted method ($\delta=0$) and Muon ($\delta\to\infty$). The optimal
$\delta \approx 0.03$–$0.05$ is still far from the Muon limit.

## The Effective Step Size $\eta_{\mathrm{eff}} = \eta \cdot \overline{\mathrm{opnorm}}$

The key insight from the sweep: the optimal configs all have similar **effective step size**
$\eta_{\mathrm{eff}} = \eta \times \text{mean opnorm over training}$, regardless of the nominal LR or damping:

| Config           | Val loss | $\eta_{\mathrm{eff}}$ (end of training) |
|-----------------|----------|----------------------|
| d=0.03, lr=0.03 | **4.202**| 0.053                |
| d=0.03, lr=0.04 | 4.205    | 0.069                |
| d=0.05, lr=0.05 | 4.209    | 0.069                |
| d=0.01, lr=0.03 | 4.213    | 0.086                |
| d=0.003, lr=0.03| 4.246    | 0.240 ← too large    |

The top configs cluster at $\eta_{\mathrm{eff}}^* \approx 0.05$–$0.07$. Configs with
$\eta_{\mathrm{eff}} \gg \eta_{\mathrm{eff}}^*$ underperform. This is a strong empirical
signal: the optimal operating point corresponds to a target effective step magnitude, not
specific values of $\eta$ or $\delta$ independently.

## Directions for Adaptive Damping

### KFAC's LM approach (for comparison)

KFAC adapts damping via the Levenberg-Marquardt reduction ratio:
$\rho = \Delta f_{\text{actual}} / \Delta f_{\text{predicted}}$. Increase damping if
$\rho < 0.25$, decrease if $\rho > 0.75$. Issues in our setting:
- Requires an extra forward pass per step to measure actual loss decrease
- Noisy in the stochastic mini-batch setting
- The "predicted decrease" for an LMO step is non-trivial to compute
- Thresholds are heuristic

### Proposed: target effective step $\eta_{\mathrm{eff}}^*$

Given the empirical finding that $\eta_{\mathrm{eff}}^* \approx 0.05$–$0.07$, the natural adaptive scheme is:
set $\delta$ each step to maintain $\eta \cdot \|C_{\mathrm{eff}}^{-1/2}\|_{\mathrm{op}} = \eta_{\mathrm{eff}}^*$.

Since $\|C_{\mathrm{eff}}^{-1/2}\|_{\mathrm{op}} \approx 1/\sqrt{\delta \cdot \lambda_{\max}(C)}$:

$$\delta = \frac{\eta^2}{(\eta_{\mathrm{eff}}^*)^2 \cdot \lambda_{\max}(C)}$$

**Properties:**
- $\lambda_{\max}(C)$ is already computed every step (from `C_eigmax` diagnostic)
- Preserves the implicit annealing property: as $\lambda_{\max}(C)$ grows,
  $\delta$ automatically decreases, keeping $\eta_{\mathrm{eff}}$ constant (rather than letting opnorm decrease)
- At bad LR where $\lambda_{\max}(C)$ stays small, $\delta$ automatically increases —
  breaking the runaway feedback loop
- Replaces two hyperparameters $(\eta, \delta)$ with one $(\eta_{\mathrm{eff}}^*)$, making the method robust
  to LR choice

This is cleaner than the LM approach: no extra forward pass, no stochastic noise in the
adaptation signal, and the target ($\eta_{\mathrm{eff}}^*$) has a direct physical interpretation.

### Per-layer vs global $\eta_{\mathrm{eff}}^*$

The formula above can be applied per layer (using each layer's $\lambda_{\max}(C_\ell)$)
with the same global $\eta_{\mathrm{eff}}^*$. This automatically accounts for different activation scales
across layers — analogous to per-layer learning rates in LARS/LAMB, but derived from the
covariance structure rather than gradient norms.

### Nuclear-norm-scaled update

The theoretically correct update (Lemma 2.3 in dap.tex) includes $\|GC_{\mathrm{eff}}^{-1/2}\|_*$ as a
scaling factor — the current method drops this, using only the sign:

$$\text{(full update)} = \eta \cdot \|GC_{\mathrm{eff}}^{-1/2}\|_* \cdot \mathrm{sign}(GC_{\mathrm{eff}}^{-1/2}) \cdot C_{\mathrm{eff}}^{-1/2}$$

With the nuclear-norm scaling, the effective step size becomes:

$$\eta_{\mathrm{eff}} = \eta \cdot \|GC_{\mathrm{eff}}^{-1/2}\|_* \cdot \|C_{\mathrm{eff}}^{-1/2}\|_{\mathrm{op}}$$

In the heavy-damping regime this scales as $\eta \cdot \|G\|_* / (\delta \lambda_{\max}(C))$,
vs the sign-only version's $\eta / \sqrt{\delta \lambda_{\max}(C)}$. The difference is quadratic
vs square-root dependence on damping, making the nuc-norm-scaled version more sensitive to $\delta$.

The nuclear-norm-scaled update is **self-regulating**: as gradients shrink near convergence,
$\|GC_{\mathrm{eff}}^{-1/2}\|_*$ shrinks automatically, and the effective step size decreases with
it. The sign-only version strips this signal — step magnitude is gradient-magnitude-independent
by construction — which is precisely why adaptive damping is more necessary and more useful
for the sign-only design.

Targeting a fixed $\eta_{\mathrm{eff}}^*$ is meaningful for the sign-only version because the step
size genuinely does not depend on gradient magnitude. For the nuclear-norm-scaled version,
$\eta_{\mathrm{eff}}$ naturally varies with gradient norm, so the same fixed-target scheme does not
apply directly; a different control scheme (e.g., gradient-normalized target) would be needed.

**Bottom line:** the adaptive $\delta$ proposal in this doc is specifically motivated by the
sign-only design choice, which trades away the self-regulating property of the full update in
exchange for robustness to gradient scale, but thereby requires external damping control.

## H2 Results and Corrected Analysis

### Experimental Results (fineweb1B, full horizon)

Original H2 adaptive damping ($\eta_{\mathrm{eff}}^* = 0.06$ with scheduled-lr coupling)
was tested on full fineweb1B runs (3814 optimizer steps). Results:

| Config | Best val loss |
|--------|-------------|
| Fixed d=0.03, lr=0.03 (baseline) | **4.202** |
| Adaptive $\eta_{\mathrm{eff}}^*=0.06$, lr=0.03 | 4.236 |
| Adaptive $\eta_{\mathrm{eff}}^*=0.06$, lr=0.01 | worse |

This **falsifies the original H2 formulation**. The failure is not due to adaptive damping
in general, but due to coupling $\delta$ to scheduled lr.

### Root Cause: Scheduled LR in the Formula

The formula $\delta = \eta^2 / ((\eta_{\mathrm{eff}}^*)^2 \lambda_{\max}(C))$ uses `group["lr"]`,
which is the **scheduled** learning rate. With warmup (0→0.03 over 381 steps) and cosine decay
(0.03→0), $\eta \approx 0$ for ~36% of training. This makes $\delta \approx 0$, leaving the
model undamped during warmup and late training.

### Diagnostic Comparison (lr=0.03 fineweb1B)

| Steps | Fixed d=0.03 | Adaptive $\eta_{\mathrm{eff}}^*=0.06$ |
|-------|-------------|--------------------------------------|
| 0–50 | $\delta=0.03$, opnorm≈1.5, $\eta_{\mathrm{eff}}$≈0.002 | $\delta≈10^{-5}$, opnorm≈21, $\eta_{\mathrm{eff}}$≈0.64 |
| 500–1000 | $\delta=0.03$, opnorm≈1.7, $\eta_{\mathrm{eff}}$≈0.05 | $\delta≈0.02$, opnorm≈2.5, $\eta_{\mathrm{eff}}$≈0.08 |
| 2000–3814 | $\delta=0.03$, opnorm≈1.8, $\eta_{\mathrm{eff}}$≈0.03 | $\delta→0$, opnorm≈10, $\eta_{\mathrm{eff}}$≈0.29 |

The fixed-damping baseline keeps opnorm stable at ~1.5–1.8 throughout training. The adaptive
scheme only works in the narrow window where lr is near its peak.

### Corrected Insight: Target Opnorm, Not $\eta_{\mathrm{eff}}$

The stable quantity in the best fixed-damping run is **opnorm** (~1.7), not $\eta_{\mathrm{eff}}$.
The lr schedule handles step size; damping handles preconditioning scale. These should be
independent — coupling them via $\eta_{\mathrm{eff}} = \eta \cdot \mathrm{opnorm}$ caused the
formula to collapse when $\eta$ is near zero.

### Revised Formula

Target opnorm directly:

$$\delta = \frac{1}{\mathrm{opnorm\_target}^2 \cdot \lambda_{\max}(C)}$$

No lr dependence. The damping $\delta$ is determined solely by the covariance scale and the
desired preconditioner amplification.

**Derived target**: From the best fixed run (d=0.03, lr=0.03), opnorm ≈ 1.7. Setting
$\mathrm{opnorm\_target} = 2.0$ gives a conservative target that should reproduce the
baseline behavior.

At step 0 with $\lambda_{\max}(C) \approx 7.8$: $\delta = 1/(4.0 \times 7.8) \approx 0.032$,
which matches the fixed baseline immediately (not $\delta \approx 0$).

## Smoke Test Results (opnorm_target formula, fineweb1B lr=0.03)

These were intentional **prefix comparisons** (10, 381, 762 steps), not full-horizon
3814-step evaluations.

### What was implemented

Removed lr from the adaptive damping formula. Instead of
$\delta = \eta^2 / ((\eta_{\mathrm{eff}}^*)^2 \lambda_{\max}(C))$ (which used the scheduled lr),
the new formula targets the opnorm directly:

$$\delta = \frac{1}{\mathrm{opnorm\_target}^2 \cdot \overline{\lambda_{\max}}}$$

where $\overline{\lambda_{\max}}$ is the arithmetic mean of $\lambda_{\max}(C_\ell)$ across
the 8 DAPOpNorm layers. This is a **relative** damping coefficient — `_compute_C_inv_sqrt`
applies it as $C_{\mathrm{eff}} = C + \delta \cdot \lambda_{\max}(C_\ell) \cdot I$ per layer.

### What worked

- **$\delta$ is lr-independent**: at step 1, $\delta = 0.032$ (matching the fixed d=0.03 baseline)
  even though scheduled lr ≈ 0.00008. The old formula gave $\delta \approx 10^{-5}$ here.
- **$\delta$ is stable throughout warmup**: no collapse to zero.
- **Loss tracks the baseline** through the first 500 steps.

### Prefix outcome

By step 762, the adaptive run (val loss 4.723) fell slightly behind the fixed baseline (4.712).
The adaptive $\delta \approx 0.023$ is lower than the fixed 0.03, producing higher mean opnorm
(~2.6 vs baseline ~1.75).

### Root cause: Jensen's inequality

The formula computes a single global $\delta$ using the **arithmetic mean** of per-layer
$\lambda_{\max}$. But the per-layer opnorm is $1/\sqrt{\delta \cdot \lambda_{\max}(C_\ell)}$,
a **convex** function of $\lambda_{\max}$. By Jensen's inequality:

$$\overline{\mathrm{opnorm}} = \frac{1}{L}\sum_\ell \frac{1}{\sqrt{\delta \cdot \lambda_{\max}(C_\ell)}}
\geq \frac{1}{\sqrt{\delta \cdot \overline{\lambda_{\max}}}}
= \mathrm{opnorm\_target}$$

The mean opnorm is **always above** the target when using the arithmetic mean eigmax. The
larger the spread in per-layer eigmax, the worse the gap. On fineweb1B the per-layer eigmax
spans ~30× (from 1.98 to 31.79 at step 762), giving a Jensen gap of 2.6 vs 2.0.

Empirical verification (step 762):

| Layer | $\lambda_{\max}(C_\ell)$ | opnorm | Ratio to mean |
|-------|-------------------------|--------|---------------|
| 0 (attn c_attn L0) | 17.57 | 1.56 | 1.63× |
| 1 (attn c_proj L0) | 1.98 | 4.65 | 0.18× |
| 2 (mlp c_fc L0) | 12.99 | 1.82 | 1.21× |
| 3 (mlp c_proj L0) | 5.52 | 2.80 | 0.51× |
| 4 (attn c_attn L1) | 8.34 | 2.27 | 0.77× |
| 5 (attn c_proj L1) | 3.26 | 3.61 | 0.30× |
| 6 (mlp c_fc L1) | 31.79 | 1.16 | 2.95× |
| 7 (mlp c_proj L1) | 4.74 | 3.02 | 0.44× |
| **Mean** | **10.77** | **2.60** | — |

Note: $\lambda_{\min}(C_\ell)$ is $O(10^{-3})$ to $O(10^{-5})$ across all layers and steps,
so the approximation $\mathrm{opnorm} \approx 1/\sqrt{\delta \cdot \lambda_{\max}}$ is accurate
to within 1–2%. The under-damping is entirely due to the averaging, not the single-layer formula.

### Fix: correct averaging

To control the **mean** opnorm across layers, we need:

$$\overline{\mathrm{opnorm}} = \frac{1}{L}\sum_\ell \frac{1}{\sqrt{\delta \cdot \lambda_{\max}(C_\ell)}}
= \frac{1}{\sqrt{\delta}} \cdot \frac{1}{L}\sum_\ell \frac{1}{\sqrt{\lambda_{\max}(C_\ell)}}
= \mathrm{opnorm\_target}$$

Solving for $\delta$:

$$\delta = \frac{\left(\frac{1}{L}\sum_\ell \frac{1}{\sqrt{\lambda_{\max}(C_\ell)}}\right)^2}{\mathrm{opnorm\_target}^2}$$

This replaces $1/\overline{\lambda_{\max}}$ with $\overline{1/\sqrt{\lambda_{\max}}}^2$. The
correction factor equals $\overline{1/\sqrt{x}}^2 / (1/\overline{x})$ which is $\geq 1$ by
Jensen, so the corrected $\delta$ is always larger (more damping) than the naive formula.

Empirical verification at step 762:
- Naive (arithmetic mean): $\delta = 0.023$, mean opnorm = 2.60
- Corrected: $\delta = 0.040$, mean opnorm = 2.00 (exact)
- Fixed baseline: $\delta = 0.030$, mean opnorm = 1.75

The corrected formula now gives $\delta$ **above** the baseline at step 762 ($0.040 > 0.030$).
This may over-damp, in which case `opnorm_target` should be tuned upward slightly. But the
formula is now mathematically consistent — the mean opnorm equals the target by construction.

## Per-layer adaptive damping

Instead of computing a single global $\delta$, compute $\delta_\ell$ per layer inside the
eigendecomposition (which is already done per layer). Since we have all eigenvalues of $C_\ell$:

$$\delta_\ell = \max\left(0,\; \frac{1/\mathrm{opnorm\_target}^2 - \lambda_{\min}(C_\ell)}{\lambda_{\max}(C_\ell)}\right)$$

This gives **exact** opnorm = opnorm_target for every layer, every step. No averaging, no
Jensen bias. Each layer gets exactly the damping it needs.

### Smoke test results (762 steps, lr=0.03, opnorm_target=2.0)

| Step | Fixed d=0.03 | Global corrected | Per-layer | Per-layer diff |
|------|-------------|-----------------|-----------|---------------|
| 50   | 6.922       | 7.042           | 7.119     | +0.198        |
| 200  | 5.633       | 5.616           | 5.611     | -0.022        |
| 381  | 5.364       | 5.330           | 5.271     | **-0.094**    |
| 500  | 5.021       | 5.014           | 4.963     | -0.057        |
| 700  | 4.662       | 4.657           | 4.623     | -0.039        |
| 762  | 4.712       | 4.719           | 4.688     | **-0.025**    |

Per-layer beats the baseline from step 200 onward. The early penalty (step 50) is from heavy
damping on layers with tiny eigmax (e.g., c_proj L0 gets $\delta=2.3$ at step 1 because
eigmax=0.108), which recovers quickly as the EMA covariance warms up.

Per-layer deltas at step 1 (showing the 30× spread that global averaging can't capture):

| Layer | $\lambda_{\max}$ | $\delta_\ell$ |
|-------|-----------------|--------------|
| attn c_attn L0 | 5.95 | 0.042 |
| attn c_proj L0 | 0.11 | 2.307 |
| mlp c_fc L0 | 6.98 | 0.036 |
| mlp c_proj L0 | 2.05 | 0.122 |
| attn c_attn L1 | 15.94 | 0.016 |
| attn c_proj L1 | 1.45 | 0.172 |
| mlp c_fc L1 | 25.46 | 0.010 |
| mlp c_proj L1 | 4.31 | 0.058 |

## Full-Horizon Sweep Results (fineweb1B, 3814 steps)

Both methods ran full 3814-step fineweb1B training across lr = {0.01, 0.02, 0.03, 0.05, 0.07}
with opnorm_target=2.0, ema_beta=0.99.

### Comparison table

| lr | Fixed (best δ) | Global Adaptive | Per-Layer Adaptive | Δ global | Δ perlayer |
|----|---------------|----------------|-------------------|----------|-----------|
| 0.01 | 4.215 (δ=0.003) | 4.235 | **4.205** | +0.020 | **-0.010** |
| 0.02 | 4.205 (δ=0.01) | 4.219 | **4.189** | +0.014 | **-0.016** |
| 0.03 | 4.202 (δ=0.03) | 4.202 | **4.189** | +0.000 | **-0.013** |
| 0.05 | 4.209 (δ=0.05) | 4.219 | **4.198** | +0.010 | **-0.011** |
| 0.07 | 4.232 (δ=0.1) | 4.716 | **4.202** | +0.484 | **-0.030** |

Best overall: fixed = 4.202, global adaptive = 4.202, **per-layer adaptive = 4.189**

### Analysis

**Per-layer adaptive damping beats the best hand-tuned fixed damping at every LR tested.**

Key observations:
1. **Per-layer is uniformly better**: improvements of 0.010–0.030 across all LRs, with a single
   hyperparameter (opnorm_target=2.0). No per-LR damping tuning needed.
2. **Per-layer is remarkably flat**: val loss range is only 4.189–4.205 (0.016 spread) across
   a 7× range of learning rates, vs fixed damping's 4.202–4.232 (0.030 spread).
3. **lr=0.07 is the biggest win**: per-layer achieves 4.202 (matching the fixed-damping best
   at lr=0.03), while global adaptive catastrophically fails at 4.716.
4. **Global adaptive is mediocre**: matches fixed damping only at lr=0.03 (where the formula
   was calibrated) and degrades at other LRs. The Jensen correction helps but can't fix the
   fundamental issue of using a single δ for layers with 30× eigmax spread.
5. **Global fails catastrophically at lr=0.07**: likely because a single average δ can't
   simultaneously handle layers with very different covariance scales at high LR.

### Why per-layer works

The per-layer formula $\delta_\ell = \max(0, (1/\mathrm{opnorm\_target}^2 - \lambda_{\min}(C_\ell)) / \lambda_{\max}(C_\ell))$
gives **exact** opnorm control per layer. Each layer gets exactly the damping it needs based on
its own covariance spectrum, with zero averaging bias. The eigendecomposition is already computed
per layer, so there is no additional cost.

### Conclusion

Per-layer adaptive damping with opnorm_target=2.0 achieves the original goal: **near-optimal
damping performance across a range of learning rates with a single hyperparameter**, eliminating
the need to jointly tune (lr, δ). It also finds a better operating point than any fixed δ tested.

## Hypotheses to Test

**H1 (Causality)**: Does opnorm blowup cause training failure, or is it a symptom?
Plot fine-grained opnorm vs loss around the warmup boundary (steps 300–600) to see which
diverges first.
