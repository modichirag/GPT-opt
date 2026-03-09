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

## Key Findings

### 1. opnorm_target is the only viable damping strategy for kfac

Relative and trace damping completely fail (val >7.0), while opnorm\_target reaches 4.617.
The gradient-aware adaptive formula is essential: without it, the raw whitened gradient's
scale is uncontrolled and training diverges.

### 2. kfac is much worse than sign-mode DAPOpNorm

| Method | Best val |
|--------|----------|
| DAPOpNorm null (sign, per-layer) | **4.189** |
| DAPOpNorm fixed $\delta$=0.03 | 4.202 |
| Muon | 4.239 |
| kfac + opnorm\_target=2.0 | 4.617 |
| kfac + trace=0.1 | 6.973 |
| kfac + relative damping=0.03 | 7.002 |

kfac with the best damping (4.617) is 0.428 nats worse than the best sign mode (4.189)
and 0.378 nats worse than Muon (4.239).

### 3. Answers to key questions

1. **Does gradient-aware damping outperform fixed damping?** Yes, dramatically. opnorm\_target is the only strategy that produces meaningful training. Fixed relative and trace damping fail completely for no-sign modes.

2. **Can no-sign modes match sign mode quality?** No — not on GPT-tiny with kfac. The gap is large (0.428 nats). The matrix sign appears to provide important implicit regularization that raw whitened gradients lack.

3. **Phase 2 (shampoo without sign)**: Given the poor kfac results, shampoo without sign is deprioritized.

## File Inventory

- `train_scripts/dap_opnorm_tiny_fineweb1B.sh` — train script
- `param_configs/dap_opnorm_tiny_fineweb1B_kfac_opnorm.json` — kfac × opnorm\_target grid
- `param_configs/dap_opnorm_tiny_fineweb1B_kfac_rel.json` — kfac × relative damping grid
- `param_configs/dap_opnorm_tiny_fineweb1B_kfac_trace.json` — kfac × trace-scaled damping grid
