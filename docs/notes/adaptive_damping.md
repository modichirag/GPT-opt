# DAPOpNorm: Adaptive Damping

## Background

The DAPOpNorm update is:

$$W \leftarrow W - \eta \cdot \mathrm{sign}(G C^{-1/2}) C^{-1/2}$$

where $C = XX^T$ is the input activation covariance (tracked via EMA), and $C^{-1/2}$ is
computed via eigendecomposition with relative damping:

$$C_{\mathrm{eff}} = C + \delta \cdot \lambda_{\max}(C) \cdot I$$

The operator norm of the preconditioner measures its amplification:

$$\|C_{\mathrm{eff}}^{-1/2}\|_{\mathrm{op}} = \frac{1}{\sqrt{\lambda_{\min}(C_{\mathrm{eff}})}}
\approx \frac{1}{\sqrt{\delta \cdot \lambda_{\max}(C)}}$$

Large opnorm means small $\lambda_{\max}(C)$ — the activation covariance scale has collapsed,
not that the covariance shape is degenerate.

## Why Adaptive Damping?

With fixed damping, optimal $\delta$ varies with learning rate: the best (lr, $\delta$) pairs
from the fineweb1B sweep are (0.03, 0.03), (0.05, 0.05), (0.01, 0.003). Tuning $\delta$
jointly with lr is expensive.

The effective step size $\eta_{\mathrm{eff}} = \eta \cdot \mathrm{opnorm}$ clusters at
$\approx 0.05$--$0.07$ across the top configs. This suggests damping should target a specific
preconditioner scale. But the stable quantity in the best fixed-damping run is actually
**opnorm itself** (~1.7), not $\eta_{\mathrm{eff}}$ — because the lr schedule already handles
step size control. Coupling $\delta$ to scheduled lr (as in an earlier failed attempt) causes
$\delta \to 0$ during warmup and cooldown, leaving the model undamped. The correct approach
is to target opnorm directly, independent of the lr schedule.

## Per-Layer Adaptive Damping

Since the eigendecomposition is already computed per layer, we can set $\delta_\ell$ per layer
to achieve exact opnorm control:

$$\delta_\ell = \max\left(0,\; \frac{1/\mathrm{opnorm\_target}^2 - \lambda_{\min}(C_\ell)}{\lambda_{\max}(C_\ell)}\right)$$

This gives $\|C_{\mathrm{eff},\ell}^{-1/2}\|_{\mathrm{op}} = \mathrm{opnorm\_target}$ exactly,
for every layer, every step.

**Why per-layer, not global?** Layers have ~30x spread in $\lambda_{\max}$ (e.g., 1.98 to 31.79
at step 762 on fineweb1B). A single global $\delta$ computed from the mean $\lambda_{\max}$
produces mean opnorm above the target by Jensen's inequality (the mapping
$\lambda_{\max} \mapsto 1/\sqrt{\delta \cdot \lambda_{\max}}$ is convex). A Jensen-corrected
global formula exists but still fails at high lr due to the extreme per-layer spread.

**Properties:**
- **Single hyperparameter**: opnorm_target replaces joint (lr, $\delta$) tuning
- **Zero extra cost**: uses eigenvalues already available from the per-layer eigendecomposition
- **Exact control**: no averaging bias, each layer gets exactly the damping it needs

## Implicit Annealing

At good learning rates, $\lambda_{\max}(C)$ grows ~47x during training as the model develops
structured activations. With the per-layer formula, this growth causes $\delta_\ell$ to decrease
over training, which preserves the implicit step annealing of the fixed-damping method (where
opnorm decreases as $\lambda_{\max}$ grows). The adaptive formula keeps opnorm constant instead,
but the lr schedule still provides annealing.

At bad learning rates where $\lambda_{\max}(C)$ collapses, $\delta_\ell$ increases — breaking
the runaway feedback loop where small $\lambda_{\max}$ causes large opnorm, which causes
large effective steps, which prevent the model from developing structured activations.

## Connection to Muon

In the limit $\delta \to \infty$, $C_{\mathrm{eff}} \approx \delta\lambda_{\max}(C)I$ and the
update reduces to $\mathrm{sign}(G)$ — exactly Muon. DAPOpNorm interpolates between full
covariance adaptation ($\delta = 0$) and Muon ($\delta \to \infty$). The optimal operating point
($\delta \approx 0.03$--$0.05$) is far from either extreme.

## Nuclear Norm Note

The theoretically correct update (Lemma 2.3 in dap.tex) includes a $\|GC_{\mathrm{eff}}^{-1/2}\|_*$
scaling factor:

$$\text{(full update)} = \eta \cdot \|GC_{\mathrm{eff}}^{-1/2}\|_* \cdot \mathrm{sign}(GC_{\mathrm{eff}}^{-1/2}) \cdot C_{\mathrm{eff}}^{-1/2}$$

This nuclear-norm-scaled version is **self-regulating**: as gradients shrink near convergence,
the effective step size decreases automatically. The sign-only version strips this signal —
step magnitude is gradient-magnitude-independent by construction. This is precisely why adaptive
damping is more necessary for the sign-only design: it must externally control preconditioner
scale, since the update has no internal mechanism to self-regulate.

For the sign-only version, targeting a fixed opnorm is well-defined because the step magnitude
depends only on the preconditioner, not gradient scale. For the nuclear-norm-scaled version,
a different control scheme (e.g., gradient-normalized target) would be needed.

## Damping Strategies

DAPOpNorm supports several damping strategies, selected by priority:

### 1. `opnorm_target` + `per_layer_damping` (recommended)

**Unified meaning: target $\|\Delta W\|_{\mathrm{op}} = \mathrm{opnorm\_target}$.**

Per layer, computes $\delta_\ell$ via eigendecomposition so that the *update* operator norm
hits the target. The mechanism depends on whether the mode uses matrix sign:

- **Sign modes** (null, full, sign_only, shampoo_sign, sign_input): The matrix sign absorbs
  $\|G\|_{\mathrm{op}}$, so $\|\Delta W\|_{\mathrm{op}} = \|C^{-1/2}\|_{\mathrm{op}}$
  (one-sided) or $\|C_{\mathrm{out}}^{-1/2}\|_{\mathrm{op}} \cdot \|C_{\mathrm{in}}^{-1/2}\|_{\mathrm{op}}$
  (two-sided). The target is achieved by damping the covariance eigenvalues directly.

- **No-sign modes** (kfac, shampoo): $\|\Delta W\|_{\mathrm{op}} \leq
  \|C_{\mathrm{out}}^{-1/2}\|_{\mathrm{op}} \cdot \|G\|_{\mathrm{op}} \cdot
  \|C_{\mathrm{in}}^{-1/2}\|_{\mathrm{op}}$. To hit the target, each side's opnorm target
  is adjusted to $\sqrt{\mathrm{opnorm\_target} / \|G_{\mathrm{mom}}\|_{\mathrm{op}}}$.
  This requires computing $\|G_{\mathrm{mom}}\|_{\mathrm{op}}$ per layer (one SVD).

### 2. `abs_damping` (standard KFAC/Shampoo damping)

$$C_{\mathrm{eff}} = C + \varepsilon I$$

Adds a fixed absolute constant to all eigenvalues. This is the standard regularization used
in KFAC and Shampoo literature. The damping scale is independent of the covariance spectrum,
so the same $\varepsilon$ may under-damp large-scale layers and over-damp small-scale layers.

### 3. `damping` (relative damping)

$$C_{\mathrm{eff}} = C + \delta \cdot \lambda_{\max}(C) \cdot I$$

Original DAPOpNorm damping. The coefficient $\delta$ is dimensionless — it specifies damping
as a fraction of the spectral radius. Scale-invariant (same $\delta$ works regardless of
activation magnitude), but still a single global value across layers.

### 4. `trace_damping` (trace-scaled damping)

$$C_{\mathrm{eff}} = C + \rho \cdot \frac{\mathrm{tr}(C)}{d} \cdot I$$

Scales damping by the mean eigenvalue $\mathrm{tr}(C)/d$ rather than $\lambda_{\max}(C)$.
More robust to outlier eigenvalues than relative damping: if one eigenvalue dominates
$\lambda_{\max}$ but the bulk of the spectrum is much smaller, relative damping over-damps
(since it scales with the outlier), while trace damping scales with the average.

Motivated by Ishikawa & Karakida 2023, who showed that trace-proportional damping is
width-transferable under $\mu$P. Internally converted to relative form:
$\delta_{\mathrm{equiv}} = \rho \cdot \frac{\mathrm{tr}(C)/d}{\lambda_{\max}(C)}$.

### 5. `precond_only_opnorm`

Only relevant for **kfac** and **shampoo** (no-sign doubly-whitened modes).

When `precond_only_opnorm=True`, `opnorm_target` controls only the preconditioner opnorm
$\|C^{-1/2}\|_{\mathrm{op}}$, *not* the full update opnorm. The gradient magnitude is
ignored. For sign modes this flag has no effect (preconditioner opnorm = update opnorm).

This exists as an ablation baseline to test whether gradient-aware adaptive damping
(strategy 1) improves over preconditioner-only targeting.

### Priority

When multiple damping parameters are set, precedence is:
`opnorm_target` + `per_layer_damping` > `abs_damping` > `trace_damping` > `damping`.

### Summary Table

| Mode | `opnorm_target` targets... | `precond_only_opnorm` changes behavior? |
|------|---------------------------|----------------------------------------|
| null | $\|C^{-1/2}\|_{\mathrm{op}}$ (= update opnorm) | No |
| full | $\prod \|C_i^{-1/2}\|_{\mathrm{op}}$ (≈ update opnorm) | No |
| sign_only | N/A (no damping, sign normalizes) | No |
| sign_input | N/A (no damping, sign normalizes) | No |
| shampoo_sign | N/A (no damping, sign normalizes) | No |
| kfac | $\|\Delta W\|_{\mathrm{op}}$ (accounts for $\|G\|_{\mathrm{op}}$) | Yes: ignores $\|G\|_{\mathrm{op}}$ |
| shampoo | $\|\Delta W\|_{\mathrm{op}}$ (accounts for $\|G\|_{\mathrm{op}}$) | Yes: ignores $\|G\|_{\mathrm{op}}$ |

See [`docs/experiments/dap_opnorm_fineweb1b.md`](../experiments/dap_opnorm_fineweb1b.md) for
experimental results.
