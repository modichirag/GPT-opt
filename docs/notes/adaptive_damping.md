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

### Experimental Results (fineweb1B)

H2 adaptive damping ($\eta_{\mathrm{eff}}^* = 0.06$) was tested on fineweb1B. Results:

| Config | Best val loss |
|--------|-------------|
| Fixed d=0.03, lr=0.03 (baseline) | **4.202** |
| Adaptive $\eta_{\mathrm{eff}}^*=0.06$, lr=0.03 | 4.236 |
| Adaptive $\eta_{\mathrm{eff}}^*=0.06$, lr=0.01 | worse |

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

## Hypotheses to Test

**H1 (Causality)**: Does opnorm blowup cause training failure, or is it a symptom?
Plot fine-grained opnorm vs loss around the warmup boundary (steps 300–600) to see which
diverges first.

**H2 (Adaptive $\delta$ via target $\eta_{\mathrm{eff}}^*$)**: Implement $\delta_t = \eta^2 / ((\eta_{\mathrm{eff}}^*)^2 \lambda_{\max}(C_t))$
per layer. Test with $\eta_{\mathrm{eff}}^* = 0.06$. Prediction: the method becomes insensitive to LR choice
and matches or beats the best fixed-$\delta$ result automatically.

**H3 (Per-layer breakdown)**: Which weight matrices drive the opnorm blowup at high LR?
The 8 DAPOpNorm-eligible matrices (Q, K, V, O projections, MLP fc1/fc2 across 2 layers)
likely have very different $\lambda_{\max}(C)$ trajectories.
