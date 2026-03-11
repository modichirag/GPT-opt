# No-Sign Damping Experiments (GPT-tiny, fineweb1B)

## Motivation

The standard DAPOpNorm update uses the matrix sign function:

$$W \leftarrow W - \eta \cdot \mathrm{sign}(G C^{-1/2}) \cdot C^{-1/2}$$

The sign normalizes singular values to 1, so $\|\mathrm{update}\|_{\mathrm{op}} = \|C^{-1/2}\|_{\mathrm{op}}$ — independent of gradient magnitude. Damping controls update scale purely through the preconditioner.

**Without the sign** (kfac and shampoo modes), the update is the raw whitened gradient:

$$W \leftarrow W - \eta \cdot C_{\mathrm{out}}^{-1/2} \, G \, C_{\mathrm{in}}^{-1/2}$$

Now $\|\mathrm{update}\|_{\mathrm{op}} \approx \|C_{\mathrm{out}}^{-1/2}\| \cdot \|G\|_{\mathrm{op}} \cdot \|C_{\mathrm{in}}^{-1/2}\|$, so gradient magnitude directly influences step size. This changes the damping problem fundamentally: the right damping depends on both the covariance spectrum *and* the gradient scale.

### Why explore no-sign modes?

1. **Theoretical**: Sign discards curvature magnitude information. KFAC-style updates preserve the natural gradient structure $F^{-1} g$, which has convergence guarantees that sign-based updates lack.
2. **Practical**: Sign requires Newton-Schulz iterations (5 steps per layer per optimizer step). Removing sign saves ~30% of optimizer compute.
3. **Empirical**: Eschenhagen et al. 2025 ("Clarifying Shampoo") show that ungrafted Shampoo$^{1/2}$ (no sign, no Adam grafting) matches grafted Shampoo when damping is tuned correctly.

## Setup

GPT-tiny (256d, 2L, 4-head, ~6M params) on fineweb1B. 3814 optimizer steps, warmup 10%, cosine decay, EMA $\beta = 0.99$.

### Two covariance sources

| Mode | $C_{\mathrm{in}}$ | $C_{\mathrm{out}}$ | Sign? |
|------|-------------------|---------------------|-------|
| **kfac** | Activation cov $X^T X$ | Output-gradient cov $S^T S$ | No |
| **shampoo** | Gradient cov $G^T G$ | Gradient cov $G G^T$ | No |

### Three damping strategies

1. **opnorm\_target + per\_layer\_damping** (gradient-aware):
   Per-side target $= \sqrt{\mathrm{opnorm\_target} / \|G\|_{\mathrm{op}}}$, then solve for $\delta_\ell$ per layer to hit this target exactly.

2. **Relative damping**: $C_{\mathrm{eff}} = C + \delta \cdot \lambda_{\max}(C) \cdot I$.
   Scale-invariant but fixed across layers.

3. **Trace-scaled damping** (Ishikawa & Karakida 2023): $C_{\mathrm{eff}} = C + \rho \cdot \frac{\mathrm{tr}(C)}{d} \cdot I$.
   Scales with the mean eigenvalue rather than $\lambda_{\max}$.

## Results: kfac (Phase 1)

### kfac + opnorm_target (per-layer adaptive)

|                | lr=0.005 | lr=0.01 | lr=0.02 | lr=0.04 |
|----------------|----------|---------|---------|---------|
| opnorm=1.0     | 4.833    | 4.788   | 4.715   | --      |
| opnorm=2.0     | 4.765    | 4.682   | **4.617** | --    |
| opnorm=4.0     | 4.698    | 4.662   | 4.921   | --      |

Best: **4.617** (lr=0.02, opnorm\_target=2.0). lr=0.04 runs still pending.

### kfac + relative damping

|                | lr=0.005 | lr=0.01 | lr=0.02 | lr=0.04 |
|----------------|----------|---------|---------|---------|
| damping=0.003  | 7.321    | 7.652   | 7.325   | 7.694   |
| damping=0.01   | 7.295    | 7.229   | 7.338   | 7.684   |
| damping=0.03   | 7.005    | **7.002** | 7.292 | --      |

Best: **7.002** (lr=0.01, damping=0.03). All runs >7.0 — model barely trains.

### kfac + trace-scaled damping

|                | lr=0.005 | lr=0.01 | lr=0.02 | lr=0.04 |
|----------------|----------|---------|---------|---------|
| trace=0.1      | 7.022    | **6.973** | 7.435 | --      |
| trace=0.3      | 7.175    | 7.314   | --      | --      |
| trace=1.0      | 7.075    | 7.162   | --      | --      |

Best: **6.973** (lr=0.01, trace=0.1). Same story — model barely trains.

## Results: shampoo (Phase 2)

Gradient covariances ($G^T G$, $G G^T$) instead of activation/output-gradient covariances.

### shampoo + opnorm_target (per-layer adaptive)

|                | lr=0.005 | lr=0.01 | lr=0.02 | lr=0.04 |
|----------------|----------|---------|---------|---------|
| opnorm=1.0     | 4.692    | 4.504   | 4.363   | 4.357   |
| opnorm=2.0     | 4.609    | 4.447   | **4.330** | 4.341 |
| opnorm=4.0     | 4.519    | 4.343   | 4.368   | 4.562   |

Best: **4.330** (lr=0.02, opnorm\_target=2.0). Much better than kfac (4.617).

### shampoo + relative damping

|                | lr=0.005 | lr=0.01 | lr=0.02 | lr=0.04 |
|----------------|----------|---------|---------|---------|
| damping=0.003  | 7.672    | 7.618   | 8.262   | 9.845   |
| damping=0.01   | 7.705    | **7.550** | 7.619 | 8.692   |
| damping=0.03   | 7.964    | 7.654   | 7.669   | 9.260   |

Best: **7.550** (lr=0.01, damping=0.01). Same failure pattern as kfac.

### shampoo + trace-scaled damping

|                | lr=0.005 | lr=0.01 | lr=0.02 | lr=0.04 |
|----------------|----------|---------|---------|---------|
| trace=0.1      | 7.723    | 7.989   | 8.357   | 8.502   |
| trace=0.3      | 7.752    | 7.653   | 7.803   | 8.383   |
| trace=1.0      | **7.556** | 7.638  | 7.719   | 8.111   |

Best: **7.556** (lr=0.005, trace=1.0). Complete failure.

### EShampoo (eigenvalue-corrected, in progress)

Replaces Kronecker eigenvalue scaling with per-element second moments in the eigenbasis
(Adam-style), following Eschenhagen et al. 2025 ("Purifying Shampoo"). This eliminates
the need for damping tuning — the eigenvalue correction provides automatic scale
adaptation, analogous to how Adam's second moment normalizes gradient magnitudes.

Update rule:
$$\tilde{G} = Q_L^T G \, Q_R, \quad D_t = \beta_2 D_{t-1} + (1 - \beta_2) \tilde{G}^{\odot 2}, \quad W \leftarrow W - \eta \, Q_L (D_t^{\odot -1/2} \odot \tilde{M}) Q_R^T$$

where $Q_L, Q_R$ are eigenvectors of $G G^T, G^T G$ (EMA), $\tilde{M}$ is the
momentum gradient in eigenbasis, and $D_t$ has bias correction.

Results pending (job 6057162).

## Key Findings

### 1. opnorm_target is the only viable damping strategy for both kfac and shampoo

Relative and trace damping completely fail for both covariance sources (val >7.0), while
opnorm\_target reaches 4.617 (kfac) and 4.330 (shampoo). The gradient-aware adaptive
formula is essential: without it, the raw whitened gradient's scale is uncontrolled and
training diverges.

### 2. Shampoo covariances significantly outperform KFAC covariances

| Method | Best val |
|--------|----------|
| DAPOpNorm null (sign, per-layer) | **4.189** |
| DAPOpNorm fixed $\delta$=0.03 | 4.202 |
| Muon | 4.239 |
| shampoo + opnorm\_target=2.0 | 4.330 |
| kfac + opnorm\_target=2.0 | 4.617 |
| shampoo + trace=1.0 | 7.556 |
| kfac + trace=0.1 | 6.973 |
| shampoo + relative damping=0.01 | 7.550 |
| kfac + relative damping=0.03 | 7.002 |

With opnorm\_target damping, shampoo (4.330) beats kfac (4.617) by 0.287 nats,
closing most of the gap to Muon (4.239). The remaining gap is 0.091 nats.

### 3. Shampoo is close to Muon but not yet matching

The 0.091 nat gap (4.330 vs 4.239) suggests implementation improvements may help.
The "Clarifying Shampoo" paper (Eschenhagen et al. 2025) shows Shampoo$^{1/2}$ should
*beat* Muon at scale (Llama 320M: 25.26 vs 25.68 perplexity). Potential causes of
our gap:

- **Step-size control**: opnorm\_target is a heuristic approximation. The papers
  recommend eigenvalue correction (EShampoo/SOAP) or grafting instead.
- **EMA initialization bias**: No bias correction on covariance EMA, so early
  steps are biased toward the initial (noisy) covariance estimate.

EShampoo addresses the first issue directly; results pending.

### 4. Answers to key questions

1. **Does gradient-aware damping outperform fixed damping?** Yes, dramatically.
   opnorm\_target is the only strategy that produces meaningful training for both
   kfac and shampoo. Fixed relative and trace damping fail completely for all
   no-sign modes.

2. **Can no-sign modes match sign mode quality?** Partially — shampoo with
   opnorm\_target (4.330) is within 0.091 nats of Muon (4.239) and 0.141 of
   the best sign mode (4.189). KFAC remains far behind (4.617).

3. **Which covariance source is better?** Gradient covariances (shampoo)
   significantly outperform activation/output-gradient covariances (kfac) in
   no-sign mode: 4.330 vs 4.617.

## File Inventory

- `train_scripts/dap_opnorm_tiny_fineweb1B.sh` — train script
- `param_configs/dap_opnorm_tiny_fineweb1B_kfac_opnorm.json` — kfac × opnorm\_target grid
- `param_configs/dap_opnorm_tiny_fineweb1B_kfac_rel.json` — kfac × relative damping grid
- `param_configs/dap_opnorm_tiny_fineweb1B_kfac_trace.json` — kfac × trace-scaled damping grid
- `param_configs/dap_opnorm_tiny_fineweb1B_shampoo_opnorm.json` — shampoo × opnorm\_target grid
- `param_configs/dap_opnorm_tiny_fineweb1B_shampoo_rel.json` — shampoo × relative damping grid
- `param_configs/dap_opnorm_tiny_fineweb1B_shampoo_trace.json` — shampoo × trace-scaled damping grid
- `param_configs/dap_opnorm_tiny_fineweb1B_eshampoo.json` — EShampoo LR sweep
